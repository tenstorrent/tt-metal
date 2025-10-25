# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test for TTNN DWConv (Depthwise Convolution) used in YOLO11 Pose
"""

import pytest
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11.reference.yolov11_pose_correct import DWConv as torch_dwconv
from models.demos.yolov11.tt.common import get_mesh_mappers
from models.demos.yolov11.tt.model_preprocessing import create_custom_mesh_preprocessor, create_yolov11_input_tensors
from models.demos.yolov11.tt.ttnn_yolov11_dwconv import TtnnDWConv
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, input_shape",
    [
        # DWConv layers from cv3 (confidence head) in pose model
        (64, 64, 3, 1, 1, [1, 64, 80, 80]),  # Scale 0
        (128, 128, 3, 1, 1, [1, 128, 40, 40]),  # Scale 1
        (256, 256, 3, 1, 1, [1, 256, 20, 20]),  # Scale 2
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11_L1_SMALL_SIZE}], indirect=True)
def test_yolo_v11_dwconv(
    device,
    reset_seeds,
    in_channel,
    out_channel,
    kernel,
    stride,
    padding,
    input_shape,
):
    """Test DWConv layer matches PyTorch reference"""

    # Create PyTorch DWConv module
    torch_module = torch_dwconv(in_channel, out_channel, kernel, stride, padding)
    torch_module.eval()

    # Create input tensors
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=input_shape[0],
        input_channels=input_shape[1],
        input_height=input_shape[2],
        input_width=input_shape[3],
    )

    # Move to device and set layout
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run PyTorch forward
    torch_output = torch_module(torch_input)

    # Preprocess parameters for TTNN
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    # Create TTNN module
    ttnn_module = TtnnDWConv(device=device, parameter=parameters, conv_pt=parameters, is_detect=True)

    # Run TTNN forward
    ttnn_output = ttnn_module(device, ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    # Reshape to match PyTorch output
    ttnn_output = ttnn_output.reshape(torch_output.shape)

    # Assert outputs match with PCC >= 0.99
    assert_with_pcc(torch_output, ttnn_output, 0.99)
