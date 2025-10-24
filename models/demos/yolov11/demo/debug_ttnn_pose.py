#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Debug script to compare PyTorch vs TTNN pose outputs
"""

import torch

import ttnn
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.reference.yolov11_pose_raw_output import YoloV11PoseRaw
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.pose_postprocessing import decode_pose_keypoints_cpu
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose


def main():
    print("=" * 70)
    print("Debugging TTNN Pose Output")
    print("=" * 70)

    # Open device
    print("\n[1/6] Opening TT device...")
    device = ttnn.open_device(device_id=0)

    # Load models
    print("\n[2/6] Loading models...")

    # PyTorch with decoded keypoints
    torch_model_decoded = YoloV11Pose()
    # PyTorch with raw keypoints
    torch_model_raw = YoloV11PoseRaw()

    weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"
    torch_model_decoded.load_state_dict(torch.load(weights_path, map_location="cpu"))
    torch_model_raw.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

    torch_model_decoded.eval()
    torch_model_raw.eval()

    # Create TTNN model
    print("[3/6] Creating TTNN model...")
    dummy_input = torch.randn(1, 3, 640, 640)
    parameters = create_yolov11_pose_model_parameters(torch_model_raw, dummy_input, device=device)
    ttnn_model = TtnnYoloV11Pose(device, parameters)

    # Create test input
    print("\n[4/6] Creating test input...")
    test_input = torch.randn(1, 3, 640, 640)

    # Run PyTorch decoded
    print("\n[5/6] Running inference...")
    with torch.no_grad():
        torch_output_decoded = torch_model_decoded(test_input)
        torch_output_raw = torch_model_raw(test_input)

    # Run TTNN
    ttnn_input = ttnn.from_torch(test_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_mem_config = ttnn.create_sharded_memory_config(
        [1, 16, 640, 640], ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.HEIGHT
    )
    ttnn_input = ttnn_input.to(device, input_mem_config)

    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output_torch = ttnn.to_torch(ttnn_output, dtype=torch.float32)

    # Get anchors/strides
    anchors = ttnn.to_torch(ttnn_model.pose_head.anchors)
    strides = ttnn.to_torch(ttnn_model.pose_head.strides)

    # Clean up dimensions
    while anchors.dim() > 2:
        anchors = anchors.squeeze(0)
    while strides.dim() > 2:
        strides = strides.squeeze(0)
    if anchors.shape[0] != 2 and anchors.shape[1] == 2:
        anchors = anchors.transpose(0, 1)
    if strides.dim() == 1:
        strides = strides.unsqueeze(0)

    print("\n[6/6] Comparing outputs...")
    print("=" * 70)

    # Compare RAW outputs
    print("\n1. RAW Outputs (before keypoint decoding):")
    print(f"   PyTorch RAW bbox:  [{torch_output_raw[:, 0:4, :].min():.2f}, {torch_output_raw[:, 0:4, :].max():.2f}]")
    print(f"   PyTorch RAW conf:  [{torch_output_raw[:, 4, :].min():.4f}, {torch_output_raw[:, 4, :].max():.4f}]")
    print(f"   PyTorch RAW kpts:  [{torch_output_raw[:, 5:56, :].min():.2f}, {torch_output_raw[:, 5:56, :].max():.2f}]")
    print()
    print(f"   TTNN RAW bbox:     [{ttnn_output_torch[:, 0:4, :].min():.2f}, {ttnn_output_torch[:, 0:4, :].max():.2f}]")
    print(f"   TTNN RAW conf:     [{ttnn_output_torch[:, 4, :].min():.4f}, {ttnn_output_torch[:, 4, :].max():.4f}]")
    print(
        f"   TTNN RAW kpts:     [{ttnn_output_torch[:, 5:56, :].min():.2f}, {ttnn_output_torch[:, 5:56, :].max():.2f}]"
    )

    # Check if RAW outputs match
    bbox_diff = (torch_output_raw[:, 0:4, :] - ttnn_output_torch[:, 0:4, :]).abs().mean()
    conf_diff = (torch_output_raw[:, 4, :] - ttnn_output_torch[:, 4, :]).abs().mean()
    kpts_diff = (torch_output_raw[:, 5:56, :] - ttnn_output_torch[:, 5:56, :]).abs().mean()

    print(f"\n   RAW differences (should be small):")
    print(f"   Bbox diff: {bbox_diff:.6f}")
    print(f"   Conf diff: {conf_diff:.6f}")
    print(f"   Kpts diff: {kpts_diff:.6f}")

    # Decode TTNN keypoints on CPU
    print("\n2. After CPU Keypoint Decoding:")
    try:
        ttnn_output_decoded = decode_pose_keypoints_cpu(ttnn_output_torch, anchors, strides)
        print(
            f"   TTNN decoded kpts: [{ttnn_output_decoded[:, 5:56, :].min():.2f}, {ttnn_output_decoded[:, 5:56, :].max():.2f}]"
        )
        print(
            f"   PyTorch decoded:   [{torch_output_decoded[:, 5:56, :].min():.2f}, {torch_output_decoded[:, 5:56, :].max():.2f}]"
        )

        decoded_diff = (torch_output_decoded[:, 5:56, :] - ttnn_output_decoded[:, 5:56, :]).abs().mean()
        print(f"   Decoded kpts diff: {decoded_diff:.6f}")

        if decoded_diff < 1.0:
            print("\n   ✓✓✓ TTNN and PyTorch match after decoding!")
        else:
            print("\n   ❌ Large difference after decoding - check anchor/stride values")
            print(f"\n   Anchor shape: {anchors.shape}, range: [{anchors.min():.2f}, {anchors.max():.2f}]")
            print(f"   Stride shape: {strides.shape}, range: [{strides.min():.2f}, {strides.max():.2f}]")
    except Exception as e:
        print(f"   ❌ Error during decoding: {e}")
        import traceback

        traceback.print_exc()

    # Close device
    ttnn.close_device(device)

    print("\n" + "=" * 70)
    print("Debug complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
