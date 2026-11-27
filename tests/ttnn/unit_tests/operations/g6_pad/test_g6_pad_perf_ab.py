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
    ((1, 1, 2, 62080), ((0, 0), (0, 0), (0, 30), (0, 0)), 0.0),  # p2
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 8), (0, 0)), 0.0),  # p3
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0),  # p4
    ((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.0),
    ((1, 196, 128), ((0, 60), (0, 0)), 0.0),
    ((1, 1, 196, 196), ((0, 60), (0, 60)), -1.7014118346046923e38),
]


def _knobs():
    k = {}
    for item in os.environ.get("G6_KNOBS", "").split(","):
        if item:
            name, v = item.split("=")
            k[name] = int(v)
    return k


@pytest.mark.parametrize("case", CASES, ids=lambda c: "x".join(map(str, c[0])) + f"-h{c[1][-2][1]}")
def test_ab(device, case, monkeypatch):
    monkeypatch.setattr(g6, "KNOBS", {**g6.KNOBS, **_knobs()})
    shape, padding, value = case
    torch.manual_seed(0)
    x = (torch.randn(shape) * 100).to(torch.bfloat16)
    pad = []
    for lo, hi in reversed(padding):
        pad += [lo, hi]
    exp = torch.nn.functional.pad(x, pad, value=value)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = g6.g6_pad(xt, padding, value)
    assert torch.equal(ttnn.to_torch(out).view(torch.int16), exp.view(torch.int16))
