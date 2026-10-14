# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf-focus shapes (feature_spec LOOSE_CASES, bf16 interleaved DRAM). Run under --profile.

Knob sweeps: TILIZE_PERF_KNOBS='{"OUT_DEPTH_TILE_ROWS": 4}' overrides program-descriptor knobs;
TILIZE_PERF_ITERS=N runs the op N times (N profiled samples).
TILIZE_PERF_ROW_WEIGHTS=path.json sets per-core row-split weights (logical core coords).
"""

import json
import os

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 16384, 64),
        (1, 1, 16384, 32),
        (1, 1, 32768, 64),
        (1, 1, 128, 64),
        (1, 1, 3520, 64),
        (1, 1, 32, 2048),  # short_wide (2-D split)
        (1, 1, 2048, 2048),  # square_large (2-D split)
        (1, 1, 32, 8192),  # short_wide, column-blocked per core
    ],
    ids=lambda s: "x".join(map(str, s[2:])),
)
def test_tilize_perf(device, monkeypatch, shape):
    for name, value in json.loads(os.environ.get("TILIZE_PERF_KNOBS", "{}")).items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)
    weights_path = os.environ.get("TILIZE_PERF_ROW_WEIGHTS")  # JSON {"x,y": weight} (logical coords)
    if weights_path:
        table = json.load(open(weights_path))

        def weight(dev, core):
            return table[f"{core.x},{core.y}"]

        monkeypatch.setattr(pd, "ROW_SPLIT_WEIGHTS", weight)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(int(os.environ.get("TILIZE_PERF_ITERS", "1")) - 1):  # extra profiled samples
        tilize(tt_in).deallocate()
    out = tilize(tt_in)

    assert torch.equal(ttnn.to_torch(out), x)


# Sharded feature_spec LOOSE_CASES (bf16), guard set for the perf passes.
def _hs(grid, shape, orient=ttnn.ShardOrientation.ROW_MAJOR, scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid - 1, grid - 1))])
    return ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shape, orient))


CASES = {
    "hs_in_dram_out": ((2048, 512), lambda: _hs(8, (32, 512)), lambda: ttnn.DRAM_MEMORY_CONFIG),
    "dram_in_hs_out": ((2048, 512), lambda: ttnn.DRAM_MEMORY_CONFIG, lambda: _hs(8, (32, 512))),
    "hs_same": ((2048, 512), lambda: _hs(8, (32, 512)), lambda: _hs(8, (32, 512))),
    "bs_same": (
        (512, 512),
        lambda: _hs(4, (128, 128), ttnn.ShardOrientation.COL_MAJOR, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        lambda: _hs(4, (128, 128), ttnn.ShardOrientation.COL_MAJOR, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_tilize_perf_sharded(device, monkeypatch, case):
    for name, value in json.loads(os.environ.get("TILIZE_PERF_KNOBS", "{}")).items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)
    (h, w), in_mc, out_mc = CASES[case]
    x = torch.randn((1, 1, h, w), dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc())
    for _ in range(int(os.environ.get("TILIZE_PERF_ITERS", "1")) - 1):
        tilize(tt_in, memory_config=out_mc()).deallocate()
    out = tilize(tt_in, memory_config=out_mc())
    assert torch.equal(ttnn.to_torch(out), x)
