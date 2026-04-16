# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Create comprehensive demo test video.

Includes:
1. Registered faces (should be identified)
2. Extra images (group photos, random people - tests unknown detection)

Usage:
    python -m models.experimental.sface.tests.benchmark.create_demo_video
"""

import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
DEMO_DIR = SCRIPT_DIR.parent.parent / "web_demo"
REGISTERED_FACES_DIR = DEMO_DIR / "server" / "registered_faces"
TEST_VIDEO_DIR = DEMO_DIR / "server" / "test_videos"
EXTRA_IMAGES_DIR = TEST_VIDEO_DIR / "extra_images"


def load_registered_faces():
    """Load all registered face images."""
    people = {}

    for person_dir in REGISTERED_FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        face_path = person_dir / "face.jpg"
        if face_path.exists():
            try:
                img = Image.open(face_path).convert("RGB")
                people[person_dir.name] = np.array(img)
                logger.info(f"  Registered: {person_dir.name} ({img.size[0]}x{img.size[1]})")
            except Exception as e:
                logger.warning(f"  Failed to load {face_path}: {e}")

    return people


def load_extra_images():
    """Load extra test images (group photos, random people)."""
    images = []

    if not EXTRA_IMAGES_DIR.exists():
        logger.info("  No extra images folder found")
        return images

    for img_path in EXTRA_IMAGES_DIR.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append({"name": img_path.stem, "image": np.array(img), "label": "Group/Unknown"})
                logger.info(f"  Extra: {img_path.name} ({img.size[0]}x{img.size[1]})")
            except Exception as e:
                logger.warning(f"  Failed to load {img_path}: {e}")

    return images


def create_frame(image, label, frame_size=(640, 480)):
    """Create a video frame with image centered and label."""
    canvas = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # Dark gray background

    # Resize image to fit
    img_pil = Image.fromarray(image)
    max_h, max_w = frame_size[1] - 80, frame_size[0] - 40
    ratio = min(max_w / img_pil.width, max_h / img_pil.height)
    new_size = (int(img_pil.width * ratio), int(img_pil.height * ratio))
    img_resized = img_pil.resize(new_size, Image.LANCZOS)
    img_np = np.array(img_resized)

    # Center on canvas
    y = (frame_size[1] - img_np.shape[0]) // 2 + 20
    x = (frame_size[0] - img_np.shape[1]) // 2
    canvas[y : y + img_np.shape[0], x : x + img_np.shape[1]] = img_np

    # Add label at top
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return canvas


def create_test_video(registered_faces, extra_images, fps=1, duration_per_frame=3.0):
    """Create comprehensive test video."""
    logger.info("\nCreating test video...")

    TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    num_frames_per_image = int(fps * duration_per_frame)

    # Section 1: Registered faces (should be identified)
    logger.info("  Adding registered faces...")
    for name, image in registered_faces.items():
        label = f"Should detect: {name}"
        frame = create_frame(image, label)
        for _ in range(num_frames_per_image):
            frames.append(frame.copy())

    # Section 2: Extra images (group photos, unknown people)
    logger.info("  Adding extra test images...")
    for item in extra_images:
        label = f"Test: {item['label']}"
        frame = create_frame(item["image"], label)
        for _ in range(num_frames_per_image):
            frames.append(frame.copy())

    if not frames:
        logger.error("No frames to write!")
        return None

    # Write video
    video_path = TEST_VIDEO_DIR / "registered_faces_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

    duration = len(frames) / fps
    logger.info(f"\nCreated: {video_path}")
    logger.info(f"  Duration: {duration:.0f} seconds")
    logger.info(f"  Registered faces: {len(registered_faces)}")
    logger.info(f"  Extra test images: {len(extra_images)}")

    return video_path


def main():
    logger.info("=" * 60)
    logger.info("Creating Comprehensive Demo Test Video")
    logger.info("=" * 60)

    # Load registered faces
    logger.info("\nLoading registered faces...")
    registered_faces = load_registered_faces()

    # Load extra test images
    logger.info("\nLoading extra test images...")
    extra_images = load_extra_images()

    if not registered_faces and not extra_images:
        logger.error("No images found!")
        return

    # Create video
    video_path = create_test_video(registered_faces, extra_images, fps=1, duration_per_frame=3.0)

    if video_path:
        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS!")
        logger.info("=" * 60)
        logger.info(f"\nTest video: {video_path}")
        logger.info("\nExpected results when running detection:")
        logger.info("  - Registered faces → identified with names")
        logger.info("  - Group photos → Jim identified (if present), others Unknown")
        logger.info("  - Random people → all Unknown")


if __name__ == "__main__":
    main()
