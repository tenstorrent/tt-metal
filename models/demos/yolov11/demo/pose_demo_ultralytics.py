#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO11 Pose Estimation Demo using Ultralytics Pretrained Model

This demo uses the official Ultralytics YOLO11n-pose pretrained weights
to perform real pose estimation on images.
"""

import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found!")
    print("Please install it with: pip install ultralytics")
    sys.exit(1)

# COCO Keypoint connections (skeleton)
SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],  # Legs
    [6, 12],
    [7, 13],  # Torso
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],  # Arms
    [6, 7],  # Shoulders
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],  # Face
    [5, 6],
    [5, 7],  # Neck connections
]

# Keypoint names
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def visualize_results(results, output_dir):
    """Visualize and save pose estimation results."""

    for idx, result in enumerate(results):
        # Get the original image
        img = result.orig_img.copy()

        if result.keypoints is None or len(result.keypoints) == 0:
            print(f"  No people detected in image {idx}")
            continue

        # Colors for visualization
        colors = [
            (255, 0, 0),
            (255, 85, 0),
            (255, 170, 0),
            (255, 255, 0),
            (170, 255, 0),
            (85, 255, 0),
            (0, 255, 0),
            (0, 255, 85),
            (0, 255, 170),
            (0, 255, 255),
            (0, 170, 255),
            (0, 85, 255),
            (0, 0, 255),
            (85, 0, 255),
            (170, 0, 255),
            (255, 0, 255),
            (255, 0, 170),
        ]

        # Process each detected person
        for person_idx, keypoints in enumerate(result.keypoints.data):
            if result.boxes is not None and len(result.boxes) > person_idx:
                box = result.boxes.xyxy[person_idx].cpu().numpy()
                conf = result.boxes.conf[person_idx].cpu().numpy()

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Person {person_idx+1}: {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Draw keypoints
            kpts = keypoints.cpu().numpy()
            for kpt_idx, (x, y, conf) in enumerate(kpts):
                if conf > 0.5:  # Only draw visible keypoints
                    color = colors[kpt_idx % len(colors)]
                    cv2.circle(img, (int(x), int(y)), 6, color, -1)
                    cv2.circle(img, (int(x), int(y)), 7, (255, 255, 255), 1)

            # Draw skeleton
            for connection in SKELETON:
                kpt1_idx = connection[0]
                kpt2_idx = connection[1]

                if kpt1_idx < len(kpts) and kpt2_idx < len(kpts):
                    kpt1 = kpts[kpt1_idx]
                    kpt2 = kpts[kpt2_idx]

                    if kpt1[2] > 0.5 and kpt2[2] > 0.5:  # Both keypoints visible
                        pt1 = (int(kpt1[0]), int(kpt1[1]))
                        pt2 = (int(kpt2[0]), int(kpt2[1]))
                        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

        # Save result
        output_path = output_dir / f"pose_result_{idx}.jpg"
        cv2.imwrite(str(output_path), img)
        print(f"  ✓ Saved: {output_path.name}")

        return img


def print_detections(results):
    """Print detailed information about detected poses."""
    for idx, result in enumerate(results):
        if result.keypoints is None or len(result.keypoints) == 0:
            continue

        print(f"\n  Image {idx + 1}:")
        for person_idx, keypoints in enumerate(result.keypoints.data):
            if result.boxes is not None and len(result.boxes) > person_idx:
                conf = result.boxes.conf[person_idx].cpu().numpy()
                print(f"\n    Person {person_idx + 1} (confidence: {conf:.3f}):")

            kpts = keypoints.cpu().numpy()
            for kpt_idx, (x, y, kpt_conf) in enumerate(kpts):
                if kpt_conf > 0.5:
                    print(f"      {KEYPOINT_NAMES[kpt_idx]:15s}: ({x:6.1f}, {y:6.1f}) vis={kpt_conf:.3f}")


def main():
    print("=" * 70)
    print("YOLO11 Pose Estimation Demo (Ultralytics Pretrained)")
    print("=" * 70)

    # Setup paths
    demo_dir = Path(__file__).parent
    images_dir = demo_dir / "images"
    output_dir = demo_dir / "runs" / "pose_ultralytics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(ext)))

    if not image_files:
        print(f"\n❌ No images found in {images_dir}")
        print("Please add some images to the demo/images directory")
        return

    print(f"\n[1/4] Loading YOLO11n-pose pretrained model...")
    print("  (This will download the model on first run)")

    try:
        # Load pretrained YOLO11 pose model
        model = YOLO("yolo11n-pose.pt")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have internet connection (for first-time download)")
        print("  2. Install ultralytics: pip install ultralytics")
        return

    print(f"\n[2/4] Found {len(image_files)} image(s) to process")

    # Process images
    print("\n[3/4] Running pose estimation...")
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")

        # Run inference
        results = model(str(image_path), conf=0.25, iou=0.45, verbose=False)

        # Count detections
        num_people = len(results[0].keypoints) if results[0].keypoints is not None else 0
        print(f"  Detected {num_people} person(s)")

        # Visualize
        visualize_results(results, output_dir)

        # Print details
        print_detections(results)

    print("\n" + "=" * 70)
    print(f"[4/4] ✓ Pose estimation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    print("\nNote: This demo uses the official Ultralytics YOLO11n-pose model")
    print("with pretrained weights for accurate pose estimation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
