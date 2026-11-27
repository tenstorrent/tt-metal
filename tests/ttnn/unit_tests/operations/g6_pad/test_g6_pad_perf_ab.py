# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Knob A/B for profiling (run with run_safe_pytest.sh --profile). Correctness asserted, perf measured."""

import os

import pytest
import torch
import ttnn

import sys

from ttnn.operations.g6_pad import g6_pad as _g6_pad_fn

g6 = sys.modules[_g6_pad_fn.__module__]

CASES = [
    ((5, 12, 128, 128), ((0, 3), (0, 0), (0, 0), (0, 0)), 0, ttnn.float32),  # p1 (PERF FOCUS), new_storage
    ((1, 1, 2, 62080), ((0, 0), (0, 0), (0, 30), (0, 0)), 0.0),  # p2
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 8), (0, 0)), 0.0),  # p3
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0),  # p4
    ((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.0),
    ((1, 196, 128), ((0, 60), (0, 0)), 0.0),
    ((1, 1, 196, 196), ((0, 60), (0, 60)), -1.7014118346046923e38),
    # Refinement 1 (new_storage_rows): ROW_MAJOR guard cases (5th field = layout)
    ((5, 8192), ((0, 3), (0, 0)), 0, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ((5, 3, 2560), ((0, 3), (0, 0), (0, 0)), 0, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 1, 196, 196), ((0, 60), (0, 60)), 0, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    # Refinement 2 guard set: golden case 63 (uint32 rank 2) and case 299 (L1, extension)
    ((1, 196), ((0, 60),), 0, ttnn.uint32, ttnn.TILE_LAYOUT),
    ((1, 1, 2560), ((0, 0), (0, 31), (0, 0)), 0.0, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG),
]


def _knobs():
    k = {}
    for item in os.environ.get("G6_KNOBS", "").split(","):
        if item:
            name, v = item.split("=")
            k[name] = int(v)
    return k


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda c: "x".join(map(str, c[0]))
    + f"-h{c[1][-2][1] if len(c[1]) > 1 else c[1][-1][1]}"
    + ("-rm" if len(c) > 4 and c[4] == ttnn.ROW_MAJOR_LAYOUT else "")
    + ("-l1" if len(c) > 5 else ""),
)
def test_ab(device, case, monkeypatch):
    monkeypatch.setattr(g6, "KNOBS", {**g6.KNOBS, **_knobs()})
    shape, padding, value = case[:3]
    dtype = case[3] if len(case) > 3 else ttnn.bfloat16
    layout = case[4] if len(case) > 4 else ttnn.TILE_LAYOUT
    mem = case[5] if len(case) > 5 else ttnn.DRAM_MEMORY_CONFIG
    torch.manual_seed(0)
    if dtype in (ttnn.uint32, ttnn.int32):
        torch_dtype, view = torch.int32, torch.int32
        x = torch.randint(0, 1 << 30, shape, dtype=torch.int32)
    else:
        torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
        view = torch.int32 if dtype == ttnn.float32 else torch.int16
        x = (torch.randn(shape) * 100).to(torch_dtype)
    pad = []
    for lo, hi in reversed(padding):
        pad += [lo, hi]
    exp = torch.nn.functional.pad(x, pad, value=value)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mem)
    out = g6.g6_pad(xt, padding, value)
    assert torch.equal(ttnn.to_torch(out).contiguous().view(view), exp.contiguous().view(view))
