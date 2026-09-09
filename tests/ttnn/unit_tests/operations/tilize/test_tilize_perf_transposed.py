# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6 perf harness — the transposed and rough-`C` geometries.

DO NOT DELETE. This is the measurement harness behind op_requirements.md's
"Refinement 6 — Speed up the transposed / rough-`C` geometries". Two targets:

  1. `[1,1,16384,32]` — the transposed twin of Refinement 3's `[1,1,32,16384]`.
     Same 512 output tiles, but read as 64 B sticks instead of 512 B ones, so
     the reader issues 8x the NoC commands for the same payload.
  2. `[1,1,1,50304]` (`pad_mode="auto"`) — a rough `C = 1572 = 2^2*3*131`, where
     the divisor constraint on `block_width_tiles` lands the split far below the
     transaction target.

Run for numbers (device kernel ns per row, in the parametrize order):
    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_perf_transposed.py

Correctness needs a plain (non-profile) run — the Tracy wrapper masks the exit
code. Every case asserts BIT IDENTITY (`torch.equal`): tilize does no arithmetic.
"""

import pytest
import torch

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

TALL_NARROW = (1, 1, 16384, 32)  # item 1 — R=512 x C=1
ROUGH_C = (1, 1, 1, 50304)  # item 2 — C=1572, H=1 (pad_mode="auto")

# One representative per distinct kernel path (Refinement 3's guard set).
GUARD_SHAPES = [
    ((1, 1, 2048, 64), "grid2d_full_width"),
    ((1, 1, 32, 2048), "grid2d_width_chunked"),
    ((1, 1, 32, 16384), "attention"),
    ((1, 1, 2048, 2048), "square_large"),
    ((1, 1, 1024, 1024), "square_mid"),
]


def _plan_line(label, shape, plan, grid):
    return (
        f"\n[perf {label}] {shape}: R={plan.tensor_row_blocks} C={plan.tensor_col_tiles} "
        f"bw={plan.block_width_tiles} chunks={plan.num_w_chunks} groups={plan.num_row_groups} "
        f"blocks={plan.num_blocks_total} cores={len(plan.assignment)}/{grid.x * grid.y} "
        f"read={plan.block_row_bytes}B "
        f"in_cb={plan.input_cb_pages}p out_cb={plan.output_cb_pages}p "
        f"L1/core={plan.l1_per_core_bytes // 1024}KiB"
    )


def _run_identity(device, shape, label, pad=False):
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pad_value = 0.0 if pad else None
    tt_output = tilize(tt_input, pad_value=pad_value)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid, pad_value=pad_value)
    print(_plan_line(label, shape, plan, grid))
    got = ttnn.to_torch(tt_output)
    if pad:
        sl = tuple(slice(0, d) for d in shape)
        assert torch.equal(got[sl], torch_input), f"{label}: tilize is not bit-identical"
    else:
        assert torch.equal(got, torch_input), f"{label}: tilize is not bit-identical"
    return plan


def test_perf_tall_narrow(device):
    """Item 1 — the transposed perf-focus shape. 64 B reads, 512 tiles."""
    plan = _run_identity(device, TALL_NARROW, "tall_narrow")
    assert plan.tensor_row_blocks == 512 and plan.tensor_col_tiles == 1
    assert len(plan.assignment) == 64, "the number is only meaningful at 64/64 cores"


def test_perf_rough_c(device):
    """Item 2 — the rough-`C` production form. C = 1572 = 2^2 * 3 * 131."""
    plan = _run_identity(device, ROUGH_C, "rough_c", pad=True)
    assert plan.tensor_col_tiles == 1572
    assert len(plan.assignment) == 64, "the number is only meaningful at 64/64 cores"


@pytest.mark.parametrize("shape,label", GUARD_SHAPES, ids=[g[1] for g in GUARD_SHAPES])
def test_perf_guard_set(device, shape, label):
    _run_identity(device, shape, label)
