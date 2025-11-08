#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Compare outputs between our custom model and Ultralytics model
to diagnose architecture issues
"""

import torch
from ultralytics import YOLO
from yolov11_pose_correct import YoloV11Pose


def compare_models():
    print("=" * 70)
    print("Comparing Custom vs Ultralytics Model Outputs")
    print("=" * 70)

    # Load both models
    print("\n[1/4] Loading models...")
    ultra_model = YOLO("yolo11n-pose.pt")
    ultra_model.model.eval()

    custom_model = YoloV11Pose()
    custom_model.load_state_dict(torch.load("yolov11_pose_pretrained_correct.pth"))
    custom_model.eval()

    print("✓ Both models loaded")

    # Create same input
    print("\n[2/4] Creating test input...")
    test_input = torch.randn(1, 3, 640, 640)
    print(f"Input shape: {test_input.shape}")

    # Get outputs
    print("\n[3/4] Running inference...")
    with torch.no_grad():
        ultra_output = ultra_model.model(test_input)
        custom_output = custom_model(test_input)

    # Ultralytics might return a list/tuple
    if isinstance(ultra_output, (list, tuple)):
        print(f"\nUltralytics returns {len(ultra_output)} outputs")
        for i, out in enumerate(ultra_output):
            if isinstance(out, torch.Tensor):
                print(f"  Output {i}: {out.shape}")
        ultra_output = ultra_output[0] if len(ultra_output) > 0 else ultra_output
    else:
        print(f"\nUltralytics output shape: {ultra_output.shape}")

    print(f"Custom output shape: {custom_output.shape}")

    # Compare outputs
    print("\n[4/4] Comparing outputs...")
    print("=" * 70)

    # Extract different components
    if isinstance(ultra_output, torch.Tensor) and ultra_output.shape == custom_output.shape:
        print(f"\n✓ Output shapes MATCH: {custom_output.shape}")

        # Compare bbox (channels 0-3)
        ultra_bbox = ultra_output[0, 0:4, :]
        custom_bbox = custom_output[0, 0:4, :]
        bbox_diff = torch.abs(ultra_bbox - custom_bbox).mean()
        print(f"\nBounding Box (channels 0-3):")
        print(f"  Ultralytics range: [{ultra_bbox.min():.4f}, {ultra_bbox.max():.4f}]")
        print(f"  Custom range:      [{custom_bbox.min():.4f}, {custom_bbox.max():.4f}]")
        print(f"  Mean difference:   {bbox_diff:.6f}")

        # Compare confidence (channel 4)
        ultra_conf = ultra_output[0, 4, :]
        custom_conf = custom_output[0, 4, :]
        conf_diff = torch.abs(ultra_conf - custom_conf).mean()
        print(f"\nConfidence (channel 4):")
        print(f"  Ultralytics range: [{ultra_conf.min():.4f}, {ultra_conf.max():.4f}]")
        print(f"  Custom range:      [{custom_conf.min():.4f}, {custom_conf.max():.4f}]")
        print(f"  Mean difference:   {conf_diff:.6f}")

        # Compare keypoints (channels 5-55)
        ultra_kpts = ultra_output[0, 5:56, :]
        custom_kpts = custom_output[0, 5:56, :]
        kpts_diff = torch.abs(ultra_kpts - custom_kpts).mean()
        print(f"\nKeypoints (channels 5-55):")
        print(f"  Ultralytics range: [{ultra_kpts.min():.4f}, {ultra_kpts.max():.4f}]")
        print(f"  Custom range:      [{custom_kpts.min():.4f}, {custom_kpts.max():.4f}]")
        print(f"  Mean difference:   {kpts_diff:.6f}")

        # Overall comparison
        total_diff = torch.abs(ultra_output - custom_output).mean()
        max_diff = torch.abs(ultra_output - custom_output).max()
        print(f"\nOverall:")
        print(f"  Mean absolute difference: {total_diff:.6f}")
        print(f"  Max absolute difference:  {max_diff:.6f}")

        if total_diff < 0.01:
            print("\n✓✓✓ Models are VERY SIMILAR - outputs match well!")
        elif total_diff < 0.1:
            print("\n✓✓ Models are SIMILAR - small differences")
        elif total_diff < 1.0:
            print("\n⚠ Models are DIFFERENT - moderate differences")
        else:
            print("\n❌ Models are VERY DIFFERENT - large differences!")
            print("   This suggests architecture mismatch or incorrect weights")

        # Check if keypoints are significantly different
        if kpts_diff > bbox_diff * 2:
            print("\n⚠⚠ KEYPOINT OUTPUT DIFFERS MORE THAN BBOX!")
            print("   This could explain the scaled-down keypoint visualization")
            print("   Likely issue: Keypoint head (cv4) architecture mismatch")

    else:
        print(f"\n❌ Output shapes DON'T MATCH!")
        print(f"   Ultralytics: {ultra_output.shape if isinstance(ultra_output, torch.Tensor) else type(ultra_output)}")
        print(f"   Custom:      {custom_output.shape}")
        print("\n   This indicates a fundamental architecture difference!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare_models()
