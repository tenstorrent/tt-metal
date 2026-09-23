# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 device-ns harness (run with run_safe_pytest.sh --profile).

Knob variants come from the env var TILIZE_R3_VARIANTS, a JSON dict
{name: {KNOB: value, ...}} applied to tilize_program_descriptor via monkeypatch;
shapes from TILIZE_R3_SHAPES ("1x1x16384x64,..."). Each (shape, variant) cell
dispatches TILIZE_R3_REPS times (default 1: device kernel time has no warm-up).
Correctness is asserted on every cell, so a plain run is also a knob-matrix check.
"""

import json
import os

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

VARIANTS = json.loads(os.environ.get("TILIZE_R3_VARIANTS", '{"default": {}}'))
SHAPES = [tuple(int(d) for d in s.split("x")) for s in os.environ.get("TILIZE_R3_SHAPES", "1x1x16384x64").split(",")]
REPS = int(os.environ.get("TILIZE_R3_REPS", "1"))


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_r3(device, monkeypatch, shape, variant):
    for k, v in VARIANTS[variant].items():
        if k == "_GRID":  # probe only: restrict the row split to a smaller grid
            orig = ttnn.split_work_to_cores
            monkeypatch.setattr(
                ttnn,
                "split_work_to_cores",
                lambda g, n, row_wise=False, v=v: orig(ttnn.CoreCoord(*v), n, row_wise=row_wise),
            )
            continue
        monkeypatch.setattr(pd, k, v)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(REPS):
        out = tilize(t)
    assert torch.equal(ttnn.to_torch(out), x)


# Config-spanning guard set: one representative per kernel path x layout x placement at
# default knobs (run with --profile and compare device-kernel ns before / after a change).
def _hs(grid_x, grid_y, shard):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


GUARDS = {
    # id: (shape, input layout tile_h or None, in memory config, out memory config, output tile_h)
    "narrow_dram": ((1, 1, 16384, 64), None, "dram", None, 32),
    "wide_dram": ((1, 1, 8192, 256), None, "dram", None, 32),
    "l1_interleaved": ((1, 1, 4096, 64), None, "l1", "l1", 32),
    "sharded_resident": ((1, 1, 2048, 512), None, "hs", "hs", 32),
    "sharded_accessor": ((1, 1, 2048, 512), None, "hs", "dram", 32),
    "tiny_tile16": ((1, 1, 16384, 64), None, "dram", None, 16),
    "retile_32_to_16": ((1, 1, 16384, 64), 32, "dram", None, 16),
    # Refinement 5 / 6 additions: 2-D split, low_l1, narrow (64-byte) sticks, 2 KiB sticks (gated off)
    "grid_2d_short_wide": ((1, 1, 32, 2048), None, "dram", None, 32),
    "low_l1_narrow": ((1, 1, 16384, 64), None, "dram", None, 32, True),
    "narrow32_dram": ((1, 1, 16384, 32), None, "dram", None, 32),
    "wide1024_dram": ((1, 1, 2048, 1024), None, "dram", None, 32),
    # Refinement 8 co-read (one position per core): tiny DRAM work, and an L1 source past the DRAM gate
    "tiny_one_position": ((1, 1, 128, 64), None, "dram", None, 32),
    "l1_one_position": ((1, 1, 2048, 256), None, "l1", "dram", 32),
}


def _mc(name):
    return {
        None: None,
        "dram": ttnn.DRAM_MEMORY_CONFIG,
        "l1": ttnn.L1_MEMORY_CONFIG,
        "hs": _hs(8, 8, (32, 512)),
    }[name]


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
@pytest.mark.parametrize("guard", list(GUARDS), ids=list(GUARDS))
def test_r3_guard(device, monkeypatch, guard, variant):
    for k, v in VARIANTS[variant].items():
        monkeypatch.setattr(pd, k, v)
    shape, in_tile_h, in_mc, out_mc, tile_h, *low_l1 = GUARDS[guard]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    layout_kwargs = (
        dict(layout=ttnn.ROW_MAJOR_LAYOUT)
        if in_tile_h is None
        else dict(layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([in_tile_h, 32]))
    )
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, memory_config=_mc(in_mc), **layout_kwargs)
    kwargs = {} if tile_h == 32 else dict(tile=ttnn.Tile([tile_h, 32]))
    if low_l1:
        kwargs["low_l1"] = True
    for _ in range(REPS):
        out = tilize(t, memory_config=_mc(out_mc), **kwargs)
    assert torch.equal(ttnn.to_torch(out), x)
