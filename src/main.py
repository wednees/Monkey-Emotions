import cv2
from deepface import DeepFace
import os

# Путь к папке с изображениями обезьян
monkey_dir = "monkey_emotions"

# Загружаем изображения обезьян в словарь
monkey_images = {}
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
for emotion in emotions:
    path = os.path.join(monkey_dir, f"{emotion}.jpg")
    if os.path.exists(path):
        monkey_images[emotion] = cv2.imread(path)
    else:
        print(f"Файл не найден: {path}")

# Инициализация камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Анализ эмоции на текущем кадре
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']

        # Получаем соответствующее изображение обезьяны
        monkey_img = monkey_images.get(dominant_emotion)
        if monkey_img is not None:
            # Масштабируем изображение обезьяны (например, до 200x200)
            monkey_resized = cv2.resize(monkey_img, (200, 200))
            # Накладываем в угол кадра
            frame[10:210, 10:210] = monkey_resized

        # Показываем эмоцию в тексте
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Ошибка анализа:", e)

    cv2.imshow("Monkey Emotion Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()