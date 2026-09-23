# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Knob sweep for device-ns measurement (run with run_safe_pytest.sh --profile).

Each (shape, knob) cell dispatches REPS times; the profiler CSV rows come out in
collection order, REPS rows per cell. Correctness is still asserted.
"""
import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

REPS = 10
SHAPES = [
    (1, 1, 16384, 64),
    (1, 1, 16384, 32),
    (1, 1, 32768, 64),
    (1, 1, 128, 64),
    (1, 1, 2048, 64),
    (1, 1, 8192, 256),
    (1, 1, 16384, 512),
    (4, 3, 256, 96),
]
KNOBS = {"quantum1": dict(QUANTUM_MIN_TILES=1), "default": dict()}


@pytest.mark.parametrize("knob", list(KNOBS), ids=list(KNOBS))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_sweep(device, monkeypatch, shape, knob):
    for k, v in KNOBS[knob].items():
        monkeypatch.setattr(pd, k, v)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(REPS):
        out = tilize(t)
    assert torch.equal(ttnn.to_torch(out), x)
