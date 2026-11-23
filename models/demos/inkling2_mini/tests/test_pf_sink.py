# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill attention-sink logits (tt/fused/pf_sink.py) vs ttnn.permute(_cols(qgrs, 6528, 6552), (0, 3, 2, 1)):
bit-equal (logical values and tile padding), plus the time of both at T = 8448."""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_sink import PfSink
from models.demos.inkling2_mini.tt.prefill_model import _cols


def timed(dev, fn, reps=5):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


@pytest.mark.parametrize("t", [256, 2048, 8448])
def test_pf_sink(device, t):
    g = torch.Generator().manual_seed(t)
    qgrs = ttnn.from_torch(torch.randn(1, 1, t, 6560, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device)  # fmt: skip
    ref = lambda: ttnn.permute(_cols(qgrs, 6528, 6552), (0, 3, 2, 1))  # noqa: E731
    mod = PfSink(device)
    a, b = ref(), mod(qgrs, 6528)
    assert tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype, (a.shape, b.shape)
    assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
    if t == 8448:
        print(f"PF_SINK_MS t={t} ops {timed(device, ref):.3f} fused {timed(device, lambda: mod(qgrs, 6528)):.3f}")
