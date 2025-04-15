# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_blackhole


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}

    return tensor_map


def randomize_tensor(tensor_map, tensor_shape):
    tensor_shape = tuple(tensor_shape)
    if tensor_shape in tensor_map.keys():
        torch_tensor = tensor_map[tensor_shape]
    else:
        torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    return torch_tensor


def run_avg_pool2d(
    device, tensor_map, input_shape, kernel_size, stride, padding, dilation, ceil_mode, count_include_pad, shard_scheme
):
    ## Test setup for both.
    in_n, in_c, in_h, in_w = input_shape
    torch.manual_seed(0)
    torch_input = randomize_tensor(tensor_map, input_shape)

    ## Test setup for Actual.
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    ## Get Expected output.
    torch_output = torch.nn.functional.avg_pool2d(
        torch_input,
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    ## Get Actual output
    ttnn_output = ttnn.avg_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        applied_shard_scheme=shard_scheme,
    )

    ## Test teardown for Actual.
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
    )
    ttnn_output = ttnn.permute(ttnn_output, (0, 3, 1, 2))  # N, C, H, W
    ttnn_output = ttnn.to_torch(ttnn_output)

    ## Assertion
    assert_with_pcc(torch_output, ttnn_output, 0.99)
    allclose = torch.allclose(ttnn_output, torch_output, rtol=0.02)
    assert allclose, " Reference and output tensor are not close"


@skip_for_blackhole("Nigthly CI tests failing, ticket #20492")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        # Case: Normal compute & Normal reader kernel.
        [1, 32, 16, 16],
        [1, 512, 112, 32],
        [1, 512, 16, 16],
        [1, 800, 16, 16],
        [2, 32, 16, 16],
        [2, 512, 112, 32],
        [2, 512, 16, 16],
        [2, 800, 16, 16],
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        # Case: Normal compute & Normal reader kernel.
        (2, 2),
        (3, 3),
        # Case: Large compute & Large reader kernel.
        (5, 5),
        (9, 9),
    ),
)
@pytest.mark.parametrize(
    "stride",
    (
        (1, 1),
        (2, 2),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (0, 0),
        (1, 1),
        (2, 2),
        (4, 4),
    ),
)
@pytest.mark.parametrize("dilation", ((1, 1),))
@pytest.mark.parametrize(
    "ceil_mode",
    [
        False,
    ],
)
@pytest.mark.parametrize(
    "count_include_pad",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_run_avg_pool2d(
    device, tensor_map, input_shape, kernel_size, stride, padding, dilation, ceil_mode, count_include_pad, shard_scheme
):
    if any(p > k // 2 for p, k in zip(padding, kernel_size)):
        pytest.skip(
            "Known issue with this combination of parameters - RuntimeError: pad should be at most half of kernel size."
        )
    run_avg_pool2d(
        device,
        tensor_map,
        input_shape,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        shard_scheme=shard_scheme,
    )
