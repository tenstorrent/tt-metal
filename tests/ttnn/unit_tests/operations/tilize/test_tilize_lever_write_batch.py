# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lever measurement: `WRITE_BATCH_MIN_TILES` on a narrow tensor.

DO NOT DELETE. This is the harness behind op_design.md's "Write batch depth"
perf lamp. The writer's transaction unit is ONE output tile page, so on a narrow
tensor (`C == 1`, e.g. `[1,1,16384,32]`) a per-tile-row barrier puts exactly one
2 KiB write in flight — the named trap. `write_rows_per_barrier` exists to fix
that:

    write_rows_per_barrier = max(1, ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))

which means WRITE_BATCH_MIN_TILES == 1 reproduces the trap, and the knob is
inert (== 1) on any block already that wide. `[1,1,16384,32]` forces
block_width_tiles == 1, so on that shape this constant IS the whole knob — which
is exactly why the measurement belongs here and not on the wide shape.

Every variant is CORRECTNESS-checked: the write batch changes the CB capacity
and the pop quantum, and both must stay wrap-safe at every setting.

Run for numbers:
    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_write_batch.py
Variants run in the `WRITE_BATCHES` order, so the CSV rows come out in that
order. Correctness needs a plain (non-profile) run.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

# R=512, C=1 — the transposed counterpart of the perf-focus shape at an
# identical tile count, and the geometry that pins block_width_tiles to 1.
NARROW_SHAPE = (1, 1, 16384, 32)
# Second narrow geometry from the design's lamp text.
NARROW_SHAPE_2 = (1, 1, 2048, 64)

WRITE_BATCHES = [1, 2, 4, 8, 16]  # 8 is the production default; 1 is the trap


def _run(device, shape, write_batch, monkeypatch):
    monkeypatch.setattr(pd, "WRITE_BATCH_MIN_TILES", write_batch)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})
    grid = device.compute_with_storage_grid_size()

    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)

    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    print(
        f"\n[lever WRITE_BATCH_MIN_TILES={write_batch}] {shape}: "
        f"bw={plan.block_width_tiles} wrpb={plan.write_rows_per_barrier} "
        f"-> {plan.write_rows_per_barrier * plan.block_width_tiles} tile-page writes "
        f"in flight per barrier, cores={len(plan.assignment)}/{grid.x * grid.y}, "
        f"out_cb={plan.output_cb_pages} pages, L1/core={plan.l1_per_core_bytes // 1024} KiB"
    )
    assert_with_pcc(torch_input.float(), ttnn.to_torch(tt_output).float(), 0.995)
    return plan


@pytest.mark.parametrize("write_batch", WRITE_BATCHES, ids=[f"wb{w}" for w in WRITE_BATCHES])
def test_write_batch_lever_tall_narrow(device, write_batch, monkeypatch):
    plan = _run(device, NARROW_SHAPE, write_batch, monkeypatch)
    # On C == 1 the constant passes through to the knob one-for-one.
    assert plan.block_width_tiles == 1
    assert plan.write_rows_per_barrier == write_batch


@pytest.mark.parametrize("write_batch", WRITE_BATCHES, ids=[f"wb{w}" for w in WRITE_BATCHES])
def test_write_batch_lever_narrow_2(device, write_batch, monkeypatch):
    _run(device, NARROW_SHAPE_2, write_batch, monkeypatch)
