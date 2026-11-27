# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance pins for float32 / bfloat8_b and per-input L1 / mixed placement with DRAM or L1 output.

Every check is bit-exact on logical values. bfloat8_b references are built from the device's own
rendering of each input (to_torch of the input tensor), so whole-page moves must reproduce them.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_concat import g6_concat, g6_concat_plan

BF16, FP32, BFP8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
TORCH_DTYPE = {BF16: torch.bfloat16, FP32: torch.float32, BFP8: torch.bfloat16}
MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}

STRADDLE_H = [(1, 2, 31, 64), (1, 2, 33, 64), (1, 2, 7, 64)]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(7)


def _bits(t):
    t = t.contiguous().to(torch.float32)
    return t.view(torch.int32)


def run_case(shapes, dim, dtype, buffers, device, *, out_buffer="dram", expected_regime=None):
    host = [torch.randn(s, dtype=torch.float32).to(TORCH_DTYPE[dtype]) for s in shapes]
    tt_inputs = [
        ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=MEMORY_CONFIG[b])
        for t, b in zip(host, buffers)
    ]
    reference_inputs = [ttnn.to_torch(t) for t in tt_inputs] if dtype == BFP8 else host
    expected = torch.cat(reference_inputs, dim=dim)

    memory_config = None if out_buffer == "dram" else MEMORY_CONFIG[out_buffer]
    if expected_regime is not None:
        assert g6_concat_plan(tt_inputs, dim, memory_config=memory_config)["regime"] == expected_regime
    output = g6_concat(tt_inputs, dim, memory_config=memory_config)

    assert output.dtype == dtype
    assert output.layout == ttnn.TILE_LAYOUT
    mc = output.memory_config()
    assert mc.buffer_type == MEMORY_CONFIG[out_buffer].buffer_type
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(output)
    assert list(got.shape) == list(expected.shape)
    mismatch = _bits(got) != _bits(expected)
    bad = int(mismatch.sum())
    assert bad == 0, f"{bad} of {expected.numel()} elements differ in bits; first at {mismatch.nonzero()[:4].tolist()}"


@pytest.mark.parametrize(
    "dtype, buffers, out_buffer",
    [
        pytest.param(FP32, ["dram"] * 3, "dram", id="fp32_dram"),
        pytest.param(FP32, ["l1", "dram", "l1"], "l1", id="fp32_mixed_to_l1"),
        pytest.param(BF16, ["l1"] * 3, "dram", id="bf16_l1"),
        pytest.param(BF16, ["dram", "l1", "dram"], "dram", id="bf16_mixed_dram_first"),
        pytest.param(BF16, ["l1", "dram", "l1"], "l1", id="bf16_mixed_l1_first_to_l1"),
    ],
)
def test_row_gather_dtypes_placement(dtype, buffers, out_buffer, device):
    """h join with odd row shifts: DRAM pieces may stage, L1 and fp32 pieces are always direct."""
    run_case(STRADDLE_H, 2, dtype, buffers, device, out_buffer=out_buffer, expected_regime="row_gather")


@pytest.mark.parametrize(
    "shapes, dim, dtype, buffers, out_buffer",
    [
        pytest.param([(1, 2, 64, 64), (1, 2, 64, 96)], -1, BFP8, ["dram"] * 2, "dram", id="bfp8_w_dram"),
        pytest.param([(2, 3, 64, 128), (1, 3, 64, 128)], 0, BFP8, ["dram"] * 2, "dram", id="bfp8_outer_dram"),
        pytest.param([(1, 4, 64, 256)] * 5, 3, BFP8, ["dram"] * 4 + ["l1"], "l1", id="bfp8_mixed_to_l1"),
        pytest.param([(1, 8, 64, 32), (1, 8, 64, 128)], 3, BFP8, ["l1"] * 2, "l1", id="bfp8_l1_to_l1"),
        pytest.param([(1, 6, 31, 64), (1, 6, 31, 192)], -1, FP32, ["dram"] * 2, "dram", id="fp32_trimmed_w"),
        pytest.param([(1, 6, 31, 64), (1, 6, 31, 192)], -1, BF16, ["dram", "l1"], "dram", id="bf16_mixed_trimmed_w"),
        pytest.param([(1, 1, 32, 64), (1, 1, 32, 192)], -1, BF16, ["dram"] * 2, "l1", id="bf16_dram_to_l1"),
        pytest.param([(1, 1, 64, 128), (1, 1, 33, 128)], 2, FP32, ["l1", "dram"], "l1", id="fp32_mixed_h_page_copy"),
    ],
)
def test_page_copy_dtypes_placement(shapes, dim, dtype, buffers, out_buffer, device):
    run_case(shapes, dim, dtype, buffers, device, out_buffer=out_buffer, expected_regime="page_copy")


def test_mixed_placement_beyond_32_inputs(device):
    """Inputs 32..39 select their accessor from the mask's second word."""
    buffers = ["dram" if k % 3 else "l1" for k in range(40)]
    run_case([(1, 1, 32, 32)] * 40, -1, BF16, buffers, device, out_buffer="dram", expected_regime="page_copy")


def test_plan_reports_l1_output_bytes(device):
    shapes = [(1, 1, 64, 64), (1, 1, 64, 192)]
    tt_inputs = [
        ttnn.from_torch(torch.randn(s).to(torch.bfloat16), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device)
        for s in shapes
    ]
    dram_plan = g6_concat_plan(tt_inputs, -1)
    l1_plan = g6_concat_plan(tt_inputs, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert dram_plan["out_l1_per_bank"] == 0
    assert l1_plan["out_l1_per_bank"] > 0
    assert l1_plan["l1_total"] == dram_plan["l1_total"] + l1_plan["out_l1_per_bank"]
