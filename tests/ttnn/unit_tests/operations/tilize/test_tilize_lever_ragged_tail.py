# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lever measurement: the RAGGED COLUMN TAIL (Refinement 6, item 2).

DO NOT DELETE. `RAGGED_COLUMN_TAIL` lets the column cut take the design's own
extent (`ceil(C / target)`) and hand the leftover `C % bw` columns to a SECOND
core range with its own `block_width_tiles`. At `False` the extent falls back to
the coarsest DIVISOR of `C`, which is byte-identical to Refinement 5.

`C = 1572 = 2^2 * 3 * 131` is the production rough-`C` form (`[1,1,1,50304]`,
a `LOOSE_CASES` shape): the coarsest divisor at or below the 64-core target of
24 is **12**, so the divisor rule lands 131 chunks of 12 where the target was 63
of 25.

`PIPELINE_WAVES_PER_CORE` is swept alongside it because the two interact — with
a ragged tail available the wave rule can now reach chunk counts the divisor
rule could not, so which wave depth wins is a measurement.

Run for numbers (device kernel ns per row, in the parametrize order):
    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_ragged_tail.py

Every variant asserts BIT IDENTITY.
"""

import pytest
import torch

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

# Same `C = 1572`, once padded (H=1, so the reads are mostly the in-kernel fill
# and the WRITE side dominates) and once tile-aligned (H=32, real DRAM reads).
CASES = [
    ((1, 1, 1, 50304), True, "rough_c_padded"),
    ((1, 1, 32, 50304), False, "rough_c_aligned"),
]

# `None` = the divisor rule (the tail knob off); an int = PIPELINE_WAVES_PER_CORE
# with the tail on. On `C = 1572` at 64 cores these give bw = 12 / 25 / 13 / 9.
VARIANTS = [None, 1, 2, 3]


def _run(device, shape, pad, waves, monkeypatch):
    monkeypatch.setattr(pd, "RAGGED_COLUMN_TAIL", waves is not None)
    if waves is not None:
        monkeypatch.setattr(pd, "PIPELINE_WAVES_PER_CORE", waves)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})

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
    tail = plan.tail_group
    print(
        f"\n[ragged waves={waves}] {shape}: bw={plan.block_width_tiles} chunks={plan.num_w_chunks} "
        f"blocks={plan.num_blocks_total} read={plan.block_row_bytes}B "
        f"tail_bw={0 if tail is None else tail.block_width_tiles} "
        f"cores={plan.num_cores_used}/{grid.x * grid.y} L1/core={plan.l1_per_core_bytes // 1024}KiB"
    )
    got = ttnn.to_torch(tt_output)
    if pad:
        got = got[tuple(slice(0, d) for d in shape)]
    assert torch.equal(got, torch_input), f"waves={waves}: not bit-identical"
    return plan


@pytest.mark.parametrize("shape,pad,label", CASES, ids=[c[2] for c in CASES])
@pytest.mark.parametrize("waves", VARIANTS, ids=[("divisor" if w is None else f"w{w}") for w in VARIANTS])
def test_ragged_tail(device, shape, pad, label, waves, monkeypatch):
    plan = _run(device, shape, pad, waves, monkeypatch)
    if waves is None:
        assert plan.tail_group is None, "the divisor rule must leave no tail"
        assert plan.tensor_col_tiles % plan.block_width_tiles == 0
