# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
FP8_E4M3 tilize + untilize roundtrip on Blackhole.

This is the first step toward an end-to-end FP8 matmul benchmark. It validates
the data-movement path that gets an FP8 tensor onto the device and into a
tile-layout form a matmul kernel can consume, then back out for inspection.

Pipeline exercised:
    torch.float32 (host, RM)
      -> ttnn.from_torch(dtype=fp8_e4m3, layout=ROW_MAJOR)   # FP8_E4M3 RM on device   (PR #43775)
      -> ttnn.tilize(dtype=bfloat8_b)                         # BFLOAT8_B TILE          (PR #44307)
      -> ttnn.untilize()                                       # BFLOAT16 RM (auto)     (existing untilize lowers BFP8 -> bf16)
      -> ttnn.to_torch()                                       # torch.bfloat16

PCC reflects the compounded FP8 + BFP8 quantization, not just one of them.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 32, 256),
        (1, 1, 128, 128),
        (1, 1, 256, 1024),
        (1, 1, 800, 7 * 1024),
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_tilize_untilize_fp8(device, shape, mem_config):
    if not is_blackhole():
        pytest.skip("FP8_E4M3 requires Blackhole hardware")

    torch.manual_seed(0)
    # fp8_e4m3 max representable is ~448. Uniform [0,1) sits comfortably inside that range
    # and avoids saturation. ttnn.from_torch for fp8_e4m3 requires a torch.float32 input
    # (per PR #43775).
    torch_input = torch.rand(shape, dtype=torch.float32)

    tt_fp8_rm = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mem_config,
    )
    assert tt_fp8_rm.dtype == ttnn.fp8_e4m3
    assert tt_fp8_rm.layout == ttnn.ROW_MAJOR_LAYOUT

    tt_bfp8_tile = ttnn.tilize(tt_fp8_rm, dtype=ttnn.bfloat8_b)
    assert tt_bfp8_tile.dtype == ttnn.bfloat8_b
    assert tt_bfp8_tile.layout == ttnn.TILE_LAYOUT

    # ttnn.untilize has no dtype arg: when the input is BFLOAT8_B it auto-lowers
    # to BFLOAT16 in the row-major output (see untilize_device_operation.cpp).
    tt_bf16_rm = ttnn.untilize(tt_bfp8_tile)
    assert tt_bf16_rm.dtype == ttnn.bfloat16
    assert tt_bf16_rm.layout == ttnn.ROW_MAJOR_LAYOUT

    torch_output = ttnn.to_torch(tt_bf16_rm).to(torch.float32)

    # Threshold accommodates the FP8 quantization on creation plus the BFP8
    # block-shared-exponent quantization through tilize. For uniform [0,1) data
    # this typically lands well above 0.99 in practice; tighten when known.
    assert_with_pcc(torch_input, torch_output, 0.99)
