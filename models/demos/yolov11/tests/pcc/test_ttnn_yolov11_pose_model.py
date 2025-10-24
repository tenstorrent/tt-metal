# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test for complete TTNN YOLO11 Pose Estimation Model

This tests the end-to-end pose estimation model including:
- Backbone (layers 0-10)
- Neck (layers 11-22)
- Pose Head (layer 23)
"""

import pytest
import torch

import ttnn
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.model_preprocessing import create_yolov11_input_tensors
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "resolution",
    [
        ([1, 3, 640, 640]),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weights",
    [
        True,
        # False,  # Uncomment to test with random weights
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11_L1_SMALL_SIZE}], indirect=True)
def test_yolov11_pose_model(device, reset_seeds, resolution, use_pretrained_weights, min_channels=8):
    """
    Test complete YOLO11 Pose model

    Verifies that the TTNN implementation produces the same output as
    the PyTorch reference implementation.

    Args:
        device: TT device
        reset_seeds: Fixture to reset random seeds
        resolution: Input image resolution [batch, channels, height, width]
        use_pretrained_weights: Whether to use pretrained weights
        min_channels: Minimum channels for input padding
    """

    # Create PyTorch model
    torch_model = YoloV11Pose()
    torch_model.eval()

    # Load pretrained weights if requested
    if use_pretrained_weights:
        try:
            weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"
            torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print(f"✓ Loaded pretrained weights from {weights_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
            print("  Run: cd models/demos/yolov11/reference && python3 load_weights_correct.py")
            pytest.skip("Pretrained weights not available")

    # Create input tensors
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=resolution[0],
        input_channels=resolution[1],
        input_height=resolution[2],
        input_width=resolution[3],
        is_sub_module=False,
    )

    # Configure sharded memory for input
    n, c, h, w = ttnn_input.shape
    if c == 3:  # Padding will be applied to min_channels
        c = min_channels
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn_input.to(device, input_mem_config)

    # Run PyTorch forward pass
    torch_output = torch_model(torch_input)

    # Preprocess parameters for TTNN
    parameters = create_yolov11_pose_model_parameters(torch_model, torch_input, device=device)

    # Create TTNN model
    ttnn_model = TtnnYoloV11Pose(device, parameters)

    # Run TTNN forward pass
    ttnn_output = ttnn_model(ttnn_input)

    # Convert TTNN output to PyTorch tensor
    ttnn_output = ttnn.to_torch(ttnn_output)

    # Verify output shape matches
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"TTNN output shape: {ttnn_output.shape}")
    assert (
        torch_output.shape == ttnn_output.shape
    ), f"Output shapes don't match: {torch_output.shape} vs {ttnn_output.shape}"

    # Expected output shape: [batch, 56, 8400]
    # - 4 channels: bbox (x, y, w, h)
    # - 1 channel: person confidence
    # - 51 channels: keypoints (17 × 3)
    assert torch_output.shape[1] == 56, f"Expected 56 output channels, got {torch_output.shape[1]}"

    # Assert outputs match with PCC >= 0.99
    assert_with_pcc(torch_output, ttnn_output, 0.99)

    print("✓ YOLO11 Pose model test passed!")
