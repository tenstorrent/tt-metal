# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for conv2d_3x3: one dispatch per shape, meant for `run_safe_pytest.sh --profile`.
Correctness is still checked (PCC), so a profile of a wrong kernel is not mistaken for a number."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

# ((H, W, C_in), C_out, padding)
PERF_SHAPES = [
    ((32, 64, 128), 128, 1),
    ((16, 16, 64), 128, 1),
    ((8, 16, 512), 256, 1),
    ((8, 16, 2048), 256, 1),
    ((64, 128, 128), 128, 1),
    ((128, 256, 64), 64, 1),
]


def _shape_id(case):
    (H, W, c_in), c_out, padding = case
    return f"H{H}_W{W}_Ci{c_in}_Co{c_out}_p{padding}"


@pytest.mark.parametrize("case", PERF_SHAPES, ids=[_shape_id(c) for c in PERF_SHAPES])
def test_conv2d_3x3_perf(case, device):
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding = case
    kwargs = dict(
        stride=1,
        padding=padding,
        dilation=1,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = conv2d_3x3(tt_input, tt_weight, input_height=H, input_width=W, bias=tt_bias, padding=padding)

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=padding)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    assert_with_pcc(ref, ttnn.to_torch(tt_output).float(), 0.995)
