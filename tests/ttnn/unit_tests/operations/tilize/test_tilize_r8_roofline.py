# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 8 empirical DRAM roofline (run with run_safe_pytest.sh --profile).

A native TTNN copy moves the same bytes as tilize on the perf shape: a TILE-to-TILE clone reads
and writes 2 KiB tile pages; a ROW_MAJOR clone reads and writes the sticks. Their device-kernel
ns bound what an interleaved-DRAM read + write mix of this volume costs on this box. Not a tilize
test: it pins the measurement that the Refinement 8 outcome is judged against.
"""

import os

import pytest
import torch
import ttnn

SHAPES = [tuple(int(d) for d in s.split("x")) for s in os.environ.get("TILIZE_R8_SHAPES", "1x1x16384x64").split(",")]


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_dram_copy_roofline(device, shape, layout):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(out), x)


# Co-read (CO_READ_SHARE) gate sweep: one tile-row per Tensix core, the input in L1, output DRAM.
# `l1`: TensorMemoryLayout::INTERLEAVED (streamed: co-read engages). `hs`: HEIGHT_SHARDED 32-row
# shards on 64 cores, which the op consumes resident (zero-copy, no stick reads), so co-read never
# engages there: it is the run-to-run noise control for the same volume. Knob variants via
# TILIZE_R3_VARIANTS as in test_tilize_r3_perf.py.
import json

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

VARIANTS = json.loads(os.environ.get("TILIZE_R3_VARIANTS", '{"default": {}}'))
L1_WIDTHS = [int(w) for w in os.environ.get("TILIZE_R8_WIDTHS", "64,256,512,1024").split(",")]


def _in_mc(kind, width):
    if kind == "l1":
        return ttnn.L1_MEMORY_CONFIG
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, width), ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
@pytest.mark.parametrize("width", L1_WIDTHS)
@pytest.mark.parametrize("kind", ["l1", "hs"])
def test_co_read_l1_source(device, monkeypatch, kind, width, variant):
    for k, v in VARIANTS[variant].items():
        monkeypatch.setattr(pd, k, v)
    torch.manual_seed(0)
    x = torch.randn((1, 1, 2048, width), dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_in_mc(kind, width)
    )
    out = tilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(out), x)
