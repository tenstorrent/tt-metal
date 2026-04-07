# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.yolov11s.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11s.reference.yolov11s import C2PSA as torch_c2psa_block
from models.demos.yolov11s.tt.model_preprocessing import create_yolov11_input_tensors, create_yolov11_model_parameters
from models.demos.yolov11s.tt.ttnn_yolov11s_c2psa import TtnnC2PSA as ttnn_c2psa_block
from tests.ttnn.utils_for_testing import assert_with_pcc

# Legacy: cv1 stayed 256-wide so PSABranch saw 128 ch (incompatible with num_heads=8 in Attention); TtnnC2PSA also uses hw=400 (20×20 tokens).
# @pytest.mark.parametrize(
#     "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
#     [
#         (
#             [256, 256, 128, 128, 128, 128, 256],
#             [256, 256, 256, 128, 128, 256, 128],
#             [1, 1, 1, 1, 3, 1, 1],
#             [1, 1, 1, 1, 1, 1, 1],
#             [0, 0, 0, 0, 1, 0, 0],
#             [1, 1, 1, 1, 1, 1, 1],
#             [1, 1, 1, 1, 128, 1, 1],
#             [1, 256, 20, 20],
#         ),
#     ],
# )


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11_L1_SMALL_SIZE}], indirect=True)
def test_yolo_v11_c2psa_block(device, reset_seeds):
    if is_wormhole_b0():
        in_channel = [256, 256, 128, 128, 128, 128, 256]
        out_channel = [256, 256, 256, 128, 128, 256, 128]
        kernel = [1, 1, 1, 1, 3, 1, 1]
        stride = [1, 1, 1, 1, 1, 1, 1]
        padding = [0, 0, 0, 0, 1, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]
        groups = [1, 1, 1, 1, 128, 1, 1]
        fwd_input_shape = [1, 256, 20, 20]
    elif is_blackhole():
        in_channel = [256, 512, 256, 256, 256, 256, 512]
        out_channel = [512, 256, 512, 256, 256, 512, 256]
        kernel = [1, 1, 1, 1, 3, 1, 1]
        stride = [1, 1, 1, 1, 1, 1, 1]
        padding = [0, 0, 0, 0, 1, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]
        groups = [1, 1, 1, 1, 256, 1, 1]
        fwd_input_shape = [1, 256, 20, 20]
    else:
        pytest.skip("YOLOv11s C2PSA PCC: Wormhole B0 or Blackhole only.")

    torch_module = torch_c2psa_block(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_c2psa_block(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
