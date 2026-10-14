# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf-focus shapes (feature_spec LOOSE_CASES, bf16 interleaved DRAM). Run under --profile.

Knob sweeps: TILIZE_PERF_KNOBS='{"OUT_DEPTH_TILE_ROWS": 4}' overrides program-descriptor knobs;
TILIZE_PERF_ITERS=N runs the op N times (N profiled samples).
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
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(int(os.environ.get("TILIZE_PERF_ITERS", "1")) - 1):  # extra profiled samples
        tilize(tt_in).deallocate()
    out = tilize(tt_in)

    assert torch.equal(ttnn.to_torch(out), x)
