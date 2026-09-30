# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf A/B probe for mhc_pre host knobs (Refinement 4): one call per (shape, dtype, knob setting).

Run under run_safe_pytest.sh --profile; the CSV lists one GenericOp per parametrization, in collection
order. Correctness is only smoke-checked here (the acceptance suite owns it).
"""

import pytest
import torch
import ttnn

import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd
from ttnn.operations.mhc_pre import mhc_pre

SHAPES = [
    (1, 1, 640, 4 * 1792),
    (1, 1, 640, 4 * 7168),
    (1, 1, 1280, 4 * 4096),
    (1, 1, 4096, 4 * 1792),
]


@pytest.mark.parametrize("narrow", [False, True], ids=["fullrow", "narrow"])
@pytest.mark.parametrize("x_dtype", [ttnn.bfloat16, ttnn.float32], ids=["xbf16", "xf32"])
@pytest.mark.parametrize("x_shape", SHAPES, ids=lambda s: "X" + "x".join(map(str, s)))
def test_mhc_pre_perf_sweep(device, monkeypatch, x_shape, x_dtype, narrow):
    monkeypatch.setattr(pd, "NARROW_GROUPS", narrow)
    torch.manual_seed(0)
    nc = x_shape[-1]
    x = torch.randn(x_shape, dtype=torch.float32)
    w = torch.randn((nc, 24), dtype=torch.float32) / nc**0.5
    b = torch.randn((1, 24), dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=x_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y, post, comb = mhc_pre(tx, tw, tb, scale=(1.0, 1.0, 1.0))
    for t in (y, post, comb):
        assert torch.isfinite(ttnn.to_torch(t)).all()
