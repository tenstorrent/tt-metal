# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.yolov11s.common import YOLOV11S_L1_SMALL_SIZE
from models.demos.yolov11s.reference.yolov11s import C3k as torch_c3k
from models.demos.yolov11s.tt.model_preprocessing import create_yolov11s_input_tensors, create_yolov11s_model_parameters
from models.demos.yolov11s.tt.ttnn_yolov11s_c3k import TtnnC3K as ttnn_c3k
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups, use_shard_concat, fwd_input_shape",
    [
        # Compact C3k: two bottlenecks (3×3), concat k2+x2 → cv3.
        # Per Bottleneck: cv2.in_channels must equal cv1.out_channels → in[4]==out[3], in[6]==out[5].
        (
            [32, 32, 64, 32, 16, 32, 16],
            [32, 32, 64, 16, 32, 16, 32],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            True,
            [1, 32, 80, 80],
        ),
        # Same topology, interleaved concat path (no sharded_concat).
        (
            [32, 32, 64, 32, 16, 32, 16],
            [32, 32, 64, 16, 32, 16, 32],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            False,
            [1, 32, 40, 40],
        ),
        # YOLOv11s inner C3k slice (from C3k2 block before Detect): 256-wide, 20×20.
        (
            [256, 256, 256, 128, 128, 128, 128],
            [128, 128, 256, 128, 128, 128, 128],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            True,
            [1, 256, 20, 20],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE}], indirect=True)
def test_yolo_v11_c3k(
    device,
    reset_seeds,
    in_channel,
    out_channel,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    use_shard_concat,
    fwd_input_shape,
):
    torch_module = torch_c3k(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11s_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11s_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_c3k(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(device=device, x=ttnn_input, use_shard_concat=use_shard_concat)
    ttnn_output = ttnn.to_torch(ttnn_output)
    expected_flat = torch_output.shape[2] * torch_output.shape[3]
    if ttnn_output.shape[2] > expected_flat:
        ttnn_output = ttnn_output[:, :, :expected_flat, :]
    ttnn_output = ttnn_output.permute(0, 3, 1, 2).reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
