# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-pinned tests for tilize (op_design.md §5.3).

A regime that only triggers on some grids passes on one device and fails on
another, so the *selected* regime and the *derived blocking* are asserted
directly rather than inferred from a numeric result.

Two things are pinned here:

1. Reader regime (R_ALIGNED vs R_PAD) — keyed on the pad region actually being
   non-empty, NOT on the pad_mode string. `pad_value=` on an already-aligned
   input MUST take the aligned reader.
2. Work distribution — in particular the grid-fill gate: the wide/short
   `[1,1,32,16384]` shape (NT_H = 1) must still light the whole grid via the W
   split. A scheme that collapses to one core there is a failed work split even
   though every correctness test still passes.

DO NOT DELETE — these are the only checks that see the blocking decisions.
"""

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize, validate
from ttnn.operations.tilize.tilize_program_descriptor import R_ALIGNED, R_PAD, derive_blocking


def _plan(device, shape, **kwargs):
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return validate(tt_input, **kwargs)


def _regime(plan):
    return R_PAD if plan.has_pad_region else R_ALIGNED


@pytest.mark.parametrize(
    "shape, kwargs, expected",
    [
        pytest.param([1, 1, 64, 128], {}, R_ALIGNED, id="hot_path"),
        pytest.param([1, 1, 64, 64], {"pad_value": 0.0}, R_ALIGNED, id="auto_pad_on_aligned_is_a_noop"),
        pytest.param([1, 1, 50, 50], {"pad_value": 0.0}, R_PAD, id="auto_w_and_h_tail"),
        pytest.param(
            [1, 1, 50, 50],
            {"output_padded_shape": [1, 1, 128, 128], "pad_value": -18.0},
            R_PAD,
            id="explicit_whole_pad_tiles",
        ),
        pytest.param([1, 1, 32, 50], {"pad_value": 3.5}, R_PAD, id="w_tail_only"),
    ],
)
def test_reader_regime_selection(device, shape, kwargs, expected):
    assert _regime(_plan(device, shape, **kwargs)) == expected


def test_grid_fill_on_wide_short_shape(device):
    """`[1,1,32,16384]`: NT_H = 1 < NUM_CORES — the W split must fill the grid.

    This is the mandatory grid-fill gate: a pure height split runs this shape on
    ONE core. `n_chunks` must rise to at least the core count.
    """
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y

    nt_h, wt = 1, 16384 // 32
    in_tile_bytes = out_tile_bytes = 32 * 32 * 2  # bf16
    wt_chunk, n_chunks, num_blocks = derive_blocking(nt_h, wt, in_tile_bytes, out_tile_bytes, num_cores, 2)

    assert num_blocks >= num_cores, f"only {num_blocks} blocks for {num_cores} cores — the grid collapses"
    assert wt_chunk * n_chunks == wt, "WT_CHUNK must divide WT exactly (one compute kernel, no cliff width)"
    assert wt_chunk > 1, "W chunking must stay COARSE, not collapse to the minimal unit"


def test_tall_shape_keeps_the_pure_height_split(device):
    """`NT_H >= NUM_CORES` => the wide-shape machinery is inert (n_chunks == 1)."""
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y

    nt_h, wt = 2048 // 32 * 32, 2048 // 32  # [1,1,65536,2048]-ish: NT_H >> cores
    in_tile_bytes = out_tile_bytes = 32 * 32 * 2
    wt_chunk, n_chunks, _ = derive_blocking(nt_h, wt, in_tile_bytes, out_tile_bytes, num_cores, 2)

    assert n_chunks == 1 and wt_chunk == wt


def test_cb_footprint_is_bounded_in_w(device):
    """No CB is a function of a whole-op dimension: L1 stays under the budget as W grows."""
    from ttnn.operations.tilize.tilize_program_descriptor import CB_L1_BUDGET

    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    in_tile_bytes = out_tile_bytes = 32 * 32 * 4  # fp32 — the fatter page

    for w in (64, 2048, 16384, 65536):
        wt_chunk, _, _ = derive_blocking(1, w // 32, in_tile_bytes, out_tile_bytes, num_cores, 2)
        l1 = 2 * wt_chunk * (in_tile_bytes + out_tile_bytes)
        assert l1 <= CB_L1_BUDGET, f"W={w}: CB footprint {l1} exceeds the budget"


def test_wide_short_shape_is_correct_end_to_end(device):
    """The grid-fill shape must also be numerically right (a real device run)."""
    torch_input = torch.randn([1, 1, 32, 4096]).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(tilize(tt_input)).to(torch.float32)
    assert torch.equal(out, torch_input.to(torch.float32))
