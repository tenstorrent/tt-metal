# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test for TTNN Pose Head used in YOLO11 Pose Estimation

This tests the pose detection head (layer 23) which includes:
- cv2: Bounding box regression
- cv3: Person confidence (with DWConv)
- cv4: Keypoints (17 × 3)
"""

import pytest

import ttnn
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11.reference.yolov11_pose_correct import PoseHead as torch_pose_head
from models.demos.yolov11.tt.model_preprocessing import create_yolov11_input_tensors
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters_head
from models.demos.yolov11.tt.ttnn_yolov11_pose import TtnnPoseHead as ttnn_pose_head
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "fwd_input_shape",
    [
        # Input shapes for 3 scales: [batch, channels, height, width]
        [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11_L1_SMALL_SIZE}], indirect=True)
def test_yolo_v11_pose_head(
    device,
    reset_seeds,
    fwd_input_shape,
):
    """
    Test YOLO11 Pose Head

    Verifies that TtnnPoseHead produces the same output as PyTorch PoseHead.
    The pose head takes 3 feature maps (from different scales) and outputs:
    - Bounding boxes (4 channels)
    - Person confidence (1 channel)
    - Keypoints (51 channels = 17 keypoints × 3)
    """

    # Create PyTorch Pose Head module
    torch_module = torch_pose_head()
    torch_module.eval()

    # Create input tensors for 3 scales
    torch_input_1, ttnn_input_1 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[2][0],
        input_channels=fwd_input_shape[2][1],
        input_height=fwd_input_shape[2][2],
        input_width=fwd_input_shape[2][3],
    )

    # Move tensors to device and set layout
    ttnn_input_1 = ttnn.to_device(ttnn_input_1, device=device)
    ttnn_input_1 = ttnn.to_layout(ttnn_input_1, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input_2 = ttnn.to_device(ttnn_input_2, device=device)
    ttnn_input_2 = ttnn.to_layout(ttnn_input_2, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input_3 = ttnn.to_device(ttnn_input_3, device=device)
    ttnn_input_3 = ttnn.to_layout(ttnn_input_3, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run PyTorch forward pass
    torch_output = torch_module(torch_input_1, torch_input_2, torch_input_3)

    # Preprocess parameters for TTNN
    parameters = create_yolov11_pose_model_parameters_head(
        torch_module, torch_input_1, torch_input_2, torch_input_3, device=device
    )

    # Create TTNN Pose Head
    ttnn_module = ttnn_pose_head(device=device, parameter=parameters.model, conv_pt=parameters)

    # Run TTNN forward pass
    ttnn_output = ttnn_module(y1=ttnn_input_1, y2=ttnn_input_2, y3=ttnn_input_3, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)

    # Reshape to match PyTorch output
    ttnn_output = ttnn_output.reshape(torch_output.shape)

    # Assert outputs match with PCC >= 0.99
    assert_with_pcc(torch_output, ttnn_output, 0.99)
