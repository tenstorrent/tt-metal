# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ROW_MAJOR acceptance pins (page_copy_row_major: page = stick) and the TILE x ROW_MAJOR layout matrix.

Every check is bit-exact on logical values. Inputs and output share one layout: the op converts
nothing.
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.g6_concat import g6_concat, g6_concat_plan

BF16, FP32 = ttnn.bfloat16, ttnn.float32
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
TORCH_DTYPE = {BF16: torch.bfloat16, FP32: torch.float32}
MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(11)


def _bits(t):
    return t.contiguous().to(torch.float32).view(torch.int32)


def _inputs(shapes, dtype, layout, buffers, device):
    host = [torch.randn(s, dtype=torch.float32).to(TORCH_DTYPE[dtype]) for s in shapes]
    tt_inputs = [
        ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=MEMORY_CONFIG[b])
        for t, b in zip(host, buffers)
    ]
    return host, tt_inputs


def run_case(shapes, dim, dtype, layout, buffers, device, *, out_buffer="dram", expected_regime=None):
    host, tt_inputs = _inputs(shapes, dtype, layout, buffers, device)
    expected = torch.cat(host, dim=dim)

    memory_config = None if out_buffer == "dram" else MEMORY_CONFIG[out_buffer]
    if expected_regime is not None:
        assert g6_concat_plan(tt_inputs, dim, memory_config=memory_config)["regime"] == expected_regime
    output = g6_concat(tt_inputs, dim, memory_config=memory_config)

    assert output.dtype == dtype
    assert output.layout == layout
    mc = output.memory_config()
    assert mc.buffer_type == MEMORY_CONFIG[out_buffer].buffer_type
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(output)
    assert list(got.shape) == list(expected.shape)
    mismatch = _bits(got) != _bits(expected)
    bad = int(mismatch.sum())
    assert bad == 0, f"{bad} of {expected.numel()} elements differ in bits; first at {mismatch.nonzero()[:4].tolist()}"


@pytest.mark.parametrize(
    "shapes, dim, dtype, buffers, out_buffer",
    [
        # Outer joins (rank-3 dim 0: Qwen 28 / 33 shapes).
        pytest.param([(2, 3, 2560), (14, 3, 2560)], 0, BF16, ["dram"] * 2, "dram", id="outer_bf16"),
        pytest.param([(1, 3, 2560)] * 16, 0, BF16, ["dram"] * 16, "dram", id="outer_bf16_16_inputs"),
        pytest.param([(2, 3, 5, 64), (2, 4, 5, 64)], 1, BF16, ["dram"] * 2, "dram", id="outer_rank4_dim1"),
        # h joins (rank-3 dim 1: Qwen 77 / 141 shapes): any row offset, no tile rows to straddle.
        pytest.param([(16, 2, 2560), (16, 1, 2560)], 1, BF16, ["dram"] * 2, "dram", id="h_bf16"),
        pytest.param([(2, 3, 5, 64), (2, 3, 7, 64), (2, 3, 1, 64)], 2, BF16, ["dram"] * 3, "dram", id="h_rank4"),
        # float32: a stick is W * 4 bytes.
        pytest.param([(3, 5, 96), (2, 5, 96)], 0, FP32, ["dram"] * 2, "dram", id="outer_fp32"),
        pytest.param([(2, 7, 2560), (2, 33, 2560)], 1, FP32, ["dram"] * 2, "dram", id="h_fp32"),
        # Placement (Qwen 159: DRAM + L1 inputs), L1 output, all-L1.
        pytest.param([(1, 2, 2560), (1, 1, 2560)], 1, BF16, ["dram", "l1"], "dram", id="h_mixed_qwen159"),
        pytest.param(
            [(4, 3, 256), (1, 3, 256), (2, 3, 256)], 0, BF16, ["l1", "dram", "l1"], "l1", id="outer_mixed_to_l1"
        ),
        pytest.param([(2, 9, 128), (2, 3, 128)], 1, FP32, ["l1"] * 2, "l1", id="h_fp32_l1_to_l1"),
        # Several blocks per core, ragged last block: 6600 sticks of 5120 B, 60 per core, block cap 25.
        pytest.param([(1, 3000, 2560), (1, 3600, 2560)], 1, BF16, ["dram"] * 2, "dram", id="h_multi_block"),
        # A stick wider than BLOCK_BYTES_CAP: block of one page, transfers above the NoC burst size.
        pytest.param([(2, 1, 98304), (1, 1, 98304)], 0, BF16, ["dram"] * 2, "dram", id="outer_wide_stick"),
    ],
)
def test_g6_concat_row_major(shapes, dim, dtype, buffers, out_buffer, device):
    run_case(shapes, dim, dtype, RM, buffers, device, out_buffer=out_buffer, expected_regime="page_copy_row_major")


def test_g6_concat_row_major_plan(device):
    """Page = stick: 16 * 3 output sticks split over 48 cores, one page per block, no staging."""
    _, tt_inputs = _inputs([(1, 3, 2560)] * 16, BF16, RM, ["dram"] * 16, device)
    plan = g6_concat_plan(tt_inputs, 0)
    assert plan["regime"] == "page_copy_row_major"
    assert plan["num_cores"] == 48
    assert plan["block_pages"] == 1
    assert plan["cb_output_pages_bytes"] == plan["depth"] * 2560 * 2
    assert plan["cb_gather_staging_bytes"] == 0
    assert [c["pages"] for c in plan["cores"][:2]] == [(0, 1), (1, 2)]


def test_g6_concat_row_major_last_dim_excluded(device, expect_error):
    """stick_gather_row_major is deferred: a ROW_MAJOR join along the last dim is refused."""
    _, tt_inputs = _inputs([(1, 3, 64), (1, 3, 128)], BF16, RM, ["dram"] * 2, device)
    with expect_error(ExcludedCell, "stick_gather_row_major|concat_dim"):
        g6_concat(tt_inputs, -1)


@pytest.mark.parametrize("dtype", [pytest.param(BF16, id="bf16"), pytest.param(FP32, id="fp32")])
@pytest.mark.parametrize(
    "shapes, dim",
    [
        pytest.param([(1, 1, 32, 32)] * 2, 2, id="32x32_aligned_small_h"),
        pytest.param([(1, 1, 32, 64), (1, 1, 64, 64)], 2, id="32x64_aligned_h"),
        pytest.param([(1, 1, 64, 128)] * 2, 1, id="64x128_aligned_outer"),
        pytest.param([(1, 1, 256, 512), (1, 1, 128, 512)], 2, id="256x512_aligned_large_h"),
        pytest.param([(1, 1, 48, 32), (1, 1, 16, 32)], 2, id="48x32_H_non_aligned_h"),
        pytest.param([(1, 1, 48, 96)] * 2, 0, id="48x96_H_non_aligned_outer"),
        pytest.param([(2, 3, 5, 64), (2, 3, 7, 64)], 2, id="5x64_H_non_aligned_h"),
        pytest.param([(1, 1, 96, 64), (1, 1, 31, 64)], -2, id="96x64_H_non_aligned_h_last_partial"),
    ],
)
@pytest.mark.parametrize("layout", [pytest.param(TILE, id="tile_to_tile"), pytest.param(RM, id="rm_to_rm")])
def test_g6_concat_layout_matrix(layout, shapes, dim, dtype, device):
    # W stays a multiple of 32: w_non_aligned has no cell in TARGET for either layout. Block formats
    # have no ROW_MAJOR form, so bfloat8_b is not on this matrix.
    run_case(shapes, dim, dtype, layout, ["dram"] * len(shapes), device)
