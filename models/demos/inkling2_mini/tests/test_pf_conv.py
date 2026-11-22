# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill fused causal conv (tt/fused/pf_conv.py) vs Inkling2Prefill._conv and its two call-site forms: outputs and
the advanced tail bit-equal (modes bf16 / f32 / resid / hres, T = 256 and 2048), plus the time of both at T = 2048."""

import math
import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_conv import PfConv
from models.demos.inkling2_mini.tt.prefill_model import _f32, _Tail


def ref_conv(taps, x, tail):  # Inkling2Prefill._conv
    _, lags = tail.window(x)
    acc = None
    for r, xl in enumerate(lags):
        term = ttnn.multiply(xl, taps[r], dtype=ttnn.float32)
        acc = term if acc is None else ttnn.add(acc, term)
    return ttnn.typecast(acc, ttnn.bfloat16)


def ref(mode, taps, x, tail, resid):
    y = ref_conv(taps, x, tail)
    if mode == "f32":
        return _f32(y)
    if mode == "resid":
        return ttnn.typecast(ttnn.add(resid, _f32(y)), ttnn.bfloat16)
    if mode == "hres":
        return ttnn.add(_f32(resid), _f32(y))
    return y


def timed(dev, fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


@pytest.mark.parametrize("t", [256, 2048])
@pytest.mark.parametrize("mode", ["bf16", "f32", "resid", "hres"])
def test_pf_conv(device, t, mode):
    dev = device
    g = torch.Generator().manual_seed(t)
    put = lambda a, dt: ttnn.from_torch(a, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    w = torch.randn(4, 2304, generator=g) * 0.3
    w[0] += 1
    w /= math.sqrt(5)
    taps = [put(w[r].view(1, 1, 1, -1), ttnn.float32) for r in range(4)]
    x = put(torch.randn(1, 1, t, 2304, generator=g), ttnn.float32)  # fp32 in: both round to bf16 first
    tail0 = torch.randn(1, 1, 32, 2304, generator=g)
    resid = None
    if mode in ("resid", "hres"):  # hres: the bf16 residual stream h
        resid = put(torch.randn(1, 1, t, 2304, generator=g) * 2, ttnn.float32 if mode == "resid" else ttnn.bfloat16)
    ta = _Tail(put(tail0, ttnn.bfloat16))
    tb = _Tail(ta.t)
    pc = PfConv(dev)
    a = ttnn.to_torch(ref(mode, taps, x, ta, resid))
    b = ttnn.to_torch(pc(taps, x, tb, mode, resid))
    eq, teq = torch.equal(a, b), torch.equal(ttnn.to_torch(ta.t), ttnn.to_torch(tb.t))
    print(
        f"PF_CONV t={t} mode={mode} equal={eq} tail_equal={teq} maxdiff={float((a.float() - b.float()).abs().max()):.3g}"
    )
    assert eq and teq
    if t == 2048:
        ms_ref = timed(dev, lambda: ref(mode, taps, x, ta, resid))
        ms_new = timed(dev, lambda: pc(taps, x, tb, mode, resid))
        print(f"PF_CONV_MS t={t} mode={mode} ops {ms_ref:.2f} fused {ms_new:.2f}")
