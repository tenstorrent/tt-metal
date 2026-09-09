# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lever measurement: `block_width_tiles` on the perf-focus shape.

DO NOT DELETE. This is the harness behind op_design.md's headline perf lamp
("Grid-synchronization / transaction-size"), which asks a question the default
cannot answer by itself: on `[1,1,32,16384]` (R=1, C=512) the column extent is
what sets BOTH the core count and the reader transaction size, and the two pull
opposite ways —

    cores reached = C / block_width_tiles   (R == 1, so num_row_groups == 1)
    bytes per read = block_width_tiles * 32 * elem

so 64 cores @ bw=8 means 512 B reads, while 2 KiB reads mean 16 cores. The
production default is chosen by `derive_plan` (fill the grid first, then take
the coarsest block that fits); this file forces the other points so the choice
is measured rather than asserted.

Every variant is CORRECTNESS-checked — the point of forcing the knob is that a
wrong value must fail loudly rather than quietly under-occupy.

Run for numbers:
    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_block_width.py
The variants run in the `BLOCK_WIDTHS` order below, so the profiler CSV rows
come out in that order. Correctness needs a plain (non-profile) run — the Tracy
wrapper masks pytest's exit code.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

# The perf-focus (attention output projection) shape: R=1, C=512.
PERF_SHAPE = (1, 1, 32, 16384)

# Divisors of C=512. 8 is what derive_plan picks on a 64-core grid.
BLOCK_WIDTHS = [4, 8, 16, 32]


@pytest.mark.parametrize("forced_bw", BLOCK_WIDTHS, ids=[f"bw{w}" for w in BLOCK_WIDTHS])
def test_block_width_lever(device, forced_bw, monkeypatch):
    """Force one value of the column extent, then check the result is still
    exact and report the (cores, bytes-per-read) pair the value implies."""
    grid = device.compute_with_storage_grid_size()

    # Force the extent by overriding the width search only. Everything
    # downstream — num_w_chunks, num_row_groups, the CB page counts,
    # write_rows_per_barrier, the core assignment — still derives from it, so
    # this measures the KNOB and not a hand-built second code path.
    #
    # `RAGGED_COLUMN_TAIL=False` puts the solve back on `_largest_divisor_at_most`
    # (Refinement 6 made the ragged `ceil` the default and only falls back to the
    # divisor). It changes nothing about what is measured here: C = 512 and every
    # forced width is a divisor of it, so the plan carries no tail either way.
    monkeypatch.setattr(pd, "RAGGED_COLUMN_TAIL", False)
    monkeypatch.setattr(pd, "_largest_divisor_at_most", lambda n, limit: forced_bw)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})

    torch.manual_seed(7)
    torch_input = torch.randn(PERF_SHAPE, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)

    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    assert plan.block_width_tiles == forced_bw
    print(
        f"\n[lever block_width_tiles={forced_bw}] cores={plan.num_cores_used}/{grid.x * grid.y}, "
        f"num_w_chunks={plan.num_w_chunks}, blocks_total={plan.num_blocks_total}, "
        f"read={plan.block_row_bytes} B/stick, wrpb={plan.write_rows_per_barrier}, "
        f"L1/core={plan.l1_per_core_bytes // 1024} KiB"
    )

    assert_with_pcc(torch_input.float(), ttnn.to_torch(tt_output).float(), 0.995)
