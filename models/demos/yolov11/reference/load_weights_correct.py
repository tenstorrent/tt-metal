#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Load pretrained YOLO11n-pose weights into corrected custom implementation
"""

import torch
from ultralytics import YOLO
from yolov11_pose_correct import YoloV11Pose


def load_weights():
    print("=" * 70)
    print("Loading YOLO11n-Pose Weights into Corrected Implementation")
    print("=" * 70)

    # Load Ultralytics model
    print("\n[1/4] Loading Ultralytics YOLO11n-pose...")
    ultra_model = YOLO("yolo11n-pose.pt")
    ultra_state = ultra_model.model.state_dict()
    print(f"✓ Loaded {len(ultra_state)} parameters")

    # Load our custom model
    print("\n[2/4] Loading custom YoloV11Pose model...")
    custom_model = YoloV11Pose()
    custom_state = custom_model.state_dict()
    print(f"✓ Initialized {len(custom_state)} parameters")

    # Direct weight mapping (layer by layer)
    print("\n[3/4] Mapping weights...")

    new_state = {}
    matched = 0

    # Map weights directly (since architecture matches now)
    for ultra_name, ultra_param in ultra_state.items():
        # Convert ultralytics naming to our naming
        custom_name = ultra_name

        if custom_name in custom_state:
            # Check shape match
            if ultra_param.shape == custom_state[custom_name].shape:
                new_state[custom_name] = ultra_param.clone()
                matched += 1
                if matched <= 20:  # Show first 20
                    print(f"  ✓ {custom_name}: {ultra_param.shape}")
            else:
                print(f"  ⚠ Shape mismatch: {custom_name}")
                print(f"     Ultra: {ultra_param.shape} vs Custom: {custom_state[custom_name].shape}")
        else:
            if matched <= 25:
                print(f"  ⚠ Not found in custom model: {custom_name}")

    # Fill in any missing weights with custom model's initial values
    for name in custom_state:
        if name not in new_state:
            new_state[name] = custom_state[name]

    print(f"\n  Matched: {matched}/{len(ultra_state)} Ultra parameters")
    print(f"  Total custom parameters: {len(custom_state)}")

    # Load weights into model
    try:
        custom_model.load_state_dict(new_state, strict=False)
        print("\n✓ Weights loaded successfully!")
    except Exception as e:
        print(f"\n❌ Error loading weights: {e}")
        return None

    # Save model
    print("\n[4/4] Saving model...")
    output_path = "yolov11_pose_pretrained_correct.pth"
    torch.save(custom_model.state_dict(), output_path)
    print(f"✓ Saved to: {output_path}")

    # Test inference
    print("\n" + "=" * 70)
    print("Testing Inference")
    print("=" * 70)

    custom_model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        output = custom_model(dummy_input)

    print(f"\nInput: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Expected shape: [1, 56, num_anchors]")
    print(f"  - 4 bbox values")
    print(f"  - 1 confidence")
    print(f"  - 51 keypoints (17×3)")

    print("\n" + "=" * 70)
    print("✓ Complete!")
    print("\nTo use in your demo:")
    print("  model = YoloV11Pose()")
    print("  model.load_state_dict(torch.load('yolov11_pose_pretrained_correct.pth'))")
    print("=" * 70)

    return custom_model


if __name__ == "__main__":
    load_weights()
