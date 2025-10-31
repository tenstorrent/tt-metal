#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of YOLO11 Pose Estimation Model

This script demonstrates how to use the YoloV11Pose model for pose estimation.
The model predicts human body keypoints (17 COCO keypoints) along with bounding boxes.
"""

import torch
from yolov11 import YoloV11Pose


def main():
    # Initialize the model
    model = YoloV11Pose()
    model.eval()

    # Create a dummy input image (batch_size=1, channels=3, height=640, width=640)
    dummy_input = torch.randn(1, 3, 640, 640)

    # Run inference
    with torch.no_grad():
        output = model(dummy_input)

    print("=" * 70)
    print("YOLO11 Pose Estimation Model Output")
    print("=" * 70)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print()
    print("Output format: [batch_size, 56, num_anchors]")
    print("  - Channels 0-3:   Bounding box (x, y, w, h)")
    print("  - Channel 4:      Person confidence score")
    print("  - Channels 5-55:  17 keypoints × 3 (x, y, visibility)")
    print()
    print("COCO Keypoint Format (17 keypoints):")
    keypoints = [
        "0: nose",
        "1: left_eye",
        "2: right_eye",
        "3: left_ear",
        "4: right_ear",
        "5: left_shoulder",
        "6: right_shoulder",
        "7: left_elbow",
        "8: right_elbow",
        "9: left_wrist",
        "10: right_wrist",
        "11: left_hip",
        "12: right_hip",
        "13: left_knee",
        "14: right_knee",
        "15: left_ankle",
        "16: right_ankle",
    ]
    for i in range(0, len(keypoints), 3):
        kpts = keypoints[i : i + 3]
        print("  " + ", ".join(kpts))
    print()
    print("=" * 70)

    # Show output statistics
    print("\nOutput Statistics:")
    print(f"  Min value:  {output.min().item():.4f}")
    print(f"  Max value:  {output.max().item():.4f}")
    print(f"  Mean value: {output.mean().item():.4f}")
    print(f"  Std value:  {output.std().item():.4f}")
    print()

    # Example: Extract predictions for the first anchor
    print("Example - First anchor predictions:")
    first_anchor = output[0, :, 0]
    print(f"  Bounding box (x, y, w, h): {first_anchor[0:4].tolist()}")
    print(f"  Person confidence: {first_anchor[4].item():.4f}")
    print(f"  First keypoint (nose) - x, y, vis: {first_anchor[5:8].tolist()}")
    print()

    print("✓ Model successfully ran pose estimation inference!")
    print()


if __name__ == "__main__":
    main()
