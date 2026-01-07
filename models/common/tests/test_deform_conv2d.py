# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger
from models.common.utility_functions import deform_conv2d as deform_conv2d_ttnn

from torchvision.ops import deform_conv2d as deform_conv2d_torch


def torch_ref_deform_conv2d(x, offset, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, offset_groups=1):
    # Convert NHWC -> NCHW
    x = x.permute(0, 3, 1, 2).contiguous()
    weight = weight.permute(3, 2, 0, 1).contiguous()  # (C_out, C_in, kH, kW)
    offset = offset.permute(0, 3, 1, 2).contiguous()

    output = deform_conv2d_torch(x, offset, weight, stride=stride, padding=padding, dilation=dilation)

    return output.permute(0, 2, 3, 1).contiguous()


@pytest.mark.parametrize(
    "B, C_in, C_out, H, W, kH, kW, stride, padding, dilation, groups, offset_groups, memory_config",
    [
        (1, 16, 16, 32, 32, 3, 3, 1, 0, 1, 4, 1, ttnn.DRAM_MEMORY_CONFIG),
        (3, 16, 16, 256, 256, 3, 3, 1, 0, 1, 4, 1, ttnn.DRAM_MEMORY_CONFIG),
        (2, 64, 64, 256, 256, 3, 3, 1, 0, 1, 4, 1, ttnn.DRAM_MEMORY_CONFIG),
        (2, 256, 256, 28, 28, 3, 3, 1, 0, 2, 8, 4, ttnn.L1_MEMORY_CONFIG),
        (6, 512, 512, 16, 44, 3, 3, 1, 1, 1, 4, 1, ttnn.DRAM_MEMORY_CONFIG),
        (4, 3, 5, 128, 128, 3, 3, 2, 0, 1, 1, 1, ttnn.L1_MEMORY_CONFIG),
        (2, 3, 8, 64, 64, 3, 3, 3, 0, 1, 1, 1, ttnn.DRAM_MEMORY_CONFIG),
        (2, 3, 8, 64, 64, 3, 3, 1, 1, 1, 1, 1, ttnn.L1_MEMORY_CONFIG),
        (1, 8, 16, 128, 128, 5, 5, 1, 2, 1, 1, 1, ttnn.DRAM_MEMORY_CONFIG),
        (2, 3, 4, 64, 64, 7, 7, 1, 3, 1, 1, 1, ttnn.L1_MEMORY_CONFIG),
        (1, 3, 5, 64, 64, 3, 3, 1, 0, 2, 1, 1, ttnn.DRAM_MEMORY_CONFIG),
        (1, 3, 5, 64, 64, 3, 3, 1, 1, 2, 1, 1, ttnn.L1_MEMORY_CONFIG),
        (2, 3, 16, 256, 256, 3, 3, 1, 0, 1, 1, 1, ttnn.DRAM_MEMORY_CONFIG),
        (1, 1, 8, 128, 128, 11, 11, 1, 0, 1, 1, 1, ttnn.DRAM_MEMORY_CONFIG),
        (4, 64, 128, 32, 32, 3, 3, 1, 0, 1, 1, 1, ttnn.DRAM_MEMORY_CONFIG),
        (1, 32, 32, 16, 16, 3, 3, 1, 0, 1, 1, 1, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_deform_conv(
    device, B, C_in, C_out, H, W, kH, kW, stride, padding, dilation, groups, offset_groups, memory_config
):
    # Input tensors
    input_nhwc = torch.rand(B, H, W, C_in)

    weight_nhwc = torch.rand(kH, kW, C_in // groups, C_out)

    H_out = ((H + 2 * padding - dilation * (kH - 1) - 1) // stride) + 1
    W_out = ((W + 2 * padding - dilation * (kW - 1) - 1) // stride) + 1
    offset_nhwc = torch.rand(B, H_out, W_out, 2 * kH * kW * offset_groups) * 2 - 1

    out_ref = torch_ref_deform_conv2d(
        input_nhwc,
        offset_nhwc,
        weight_nhwc,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        offset_groups=offset_groups,
    )

    tt_input = ttnn.to_device(
        ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), device, memory_config
    )
    tt_weight = ttnn.to_device(
        ttnn.from_torch(weight_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), device, memory_config
    )
    tt_offset = ttnn.to_device(
        ttnn.from_torch(offset_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), device, memory_config
    )

    out_ttnn = deform_conv2d_ttnn(
        tt_input,
        tt_weight,
        tt_offset,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        offset_groups=offset_groups,
        device=device,
    )

    out_torch = ttnn.to_torch(out_ttnn).to(torch.float32)
    out_ref = out_ref.to(torch.float32)

    # Assertions
    assert out_ref.shape == out_torch.shape, "Shape mismatch between TTNN and Torch output"

    pcc_passed, pcc_message = assert_with_pcc(out_ref, out_torch, pcc=0.99)
    logger.info(f"PCC: {pcc_message}")
    assert pcc_passed, f"Test failed with PCC below threshold"
