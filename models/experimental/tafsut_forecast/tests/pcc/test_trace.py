# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fresh inputs through captured traces (skill §1 C, D2 item 4): every replay against eager on inputs the trace was
not captured with, at several shapes (paper 1024 x 2048 and 1 x 32768 included), shapes interleaved so that each
new shape releases and recaptures the others. ttnn's "Allocating device buffers is potentially unsafe" warning
(an allocation while a trace is alive) fails the test."""

import pytest
import torch

from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden
from models.experimental.tafsut_forecast.tests.helpers import pcc
from models.experimental.tafsut_forecast.tt.model import TtTafsut
from models.experimental.tafsut_forecast.tt.trace import TracedForward

TRACE = [{"l1_small_size": 16384, "trace_region_size": 256 * 1024**2}]
UNSAFE = "potentially unsafe"


def series(rows, length, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    walk = torch.randn(rows, length, generator=g).cumsum(-1) * 0.1
    season = torch.sin(2 * torch.pi * t / (24 + 7 * torch.rand(rows, 1, generator=g)))
    return 10 + 3 * season + walk


@pytest.mark.parametrize("device_params", TRACE, indirect=True)
def test_trace_vs_eager(device, capfd):
    tt = TtTafsut(device)
    shapes = [(1024, 2048), (1, 32768), (16, 8192)]
    inputs = [(s, series(*s, seed=10 * i + j)) for j in range(2) for i, s in enumerate(shapes)]
    eager = [tt.forward(tt.prepare_inputs(x)) for _, x in inputs]  # before any trace exists
    traced = TracedForward(tt)
    order = [0, 1, 2, 3, 4, 5, 0, 4]  # 3-5: fresh values through captures made on 0-2; 0, 4 replayed again
    for n in order:
        (rows, ctx), x = inputs[n]
        got = traced(tt.prepare_inputs(x))
        p, err = pcc(got, eager[n]), float((got - eager[n]).abs().max())
        print(f"[trace] {rows} x {ctx} input {n}: vs eager pcc {p:.7f} max_err {err:.3e} equal {torch.equal(got, eager[n])}")
        assert torch.isfinite(got).all() and p > 0.99999 and err < 1e-2
    traced.release()
    out = capfd.readouterr()
    assert UNSAFE not in out.out + out.err, "an allocation happened while a trace was alive"


@pytest.mark.parametrize("device_params", TRACE, indirect=True)
def test_trace_fresh_inputs(device, capfd):
    tt = TtTafsut(device)
    traced = TracedForward(tt)
    a, b = load_golden("model16"), load_golden("fallback")  # same shape (16 x 2048), different values
    h = load_golden("horizon4")  # 4 x 2048 -> 1100: two shapes, the second captured after the first
    for g in (a, b, h, a, h, b):  # every capture replayed on values it was not captured with
        heads = []
        q = traced.forecast(g["x"], g["horizon"], heads=heads)
        for k, f in enumerate(g["forwards"]):
            p = pcc(heads[k], f["head"])
            print(f"[trace] rows {g['x'].shape[0]} forward {k}: head pcc {p:.6f}")
            assert p > 0.9995
        assert pcc(q, g["quantiles"]) > 0.999
        assert torch.isfinite(q).all()
    traced.release()
    out = capfd.readouterr()
    assert UNSAFE not in out.out + out.err, "an allocation happened while a trace was alive"
