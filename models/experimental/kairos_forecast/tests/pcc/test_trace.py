# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace replay on fresh inputs (skill §1 C, PLAN v1.21): four batch sizes (2, 8, 16 and paper's 512 rows) captured
after all buffers exist, each replayed on inputs it was not captured with, interleaved, and compared with the eager
forward and the golden. Paper is captured on a transformed paper context (rows reversed, x 1.5 + 0.1) and then
replayed on the golden paper input, scored against the golden head (D3 item 6). ttnn's "active trace" allocation
warning fails the test (tests/conftest.py), so the eager references run before the first capture."""
import pytest
import torch

from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tests.pcc.test_dataflow import GOLD, fronts
from models.experimental.kairos_forecast.tests.pcc.test_model import pcc
from models.experimental.kairos_forecast.tests.perf.suite import load
from models.experimental.kairos_forecast.tt.common import Precision
from models.experimental.kairos_forecast.tt.generator import _cat_fronts
from models.experimental.kairos_forecast.tt.host_frontend import front_end
from models.experimental.kairos_forecast.tt.model import TtKairos
from models.experimental.kairos_forecast.tt.router_state import RouterState
from models.experimental.kairos_forecast.tt.trace import TracedKairos


def golden_pair(case, call=0):
    """The two flip halves of one test-golden call as one batch, with the golden head."""
    fs = fronts(torch.load(GOLD / f"{case}.pt", weights_only=False))
    a, b = fs[2 * call], fs[2 * call + 1]
    return _cat_fronts(a, b), torch.cat([a["head"], b["head"]])


def bench_pair(case, n, W, transform=None, gold=False):
    """The flip halves of the first n series of a bench golden (from its bias0), optionally transformed, with the
    golden head when gold (only meaningful for the whole untransformed H <= 128 case)."""
    rec = load(case)
    router = RouterState(rec["bias0"])
    x = rec["context"][:n]
    x = transform(x) if transform else x
    fa, _ = front_end(x, router, W)
    fb, _ = front_end(-x, router, W)
    return _cat_fronts(fa, fb), (torch.cat(rec["head"]) if gold else None)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("prec", ["accuracy", "performance"])
def test_trace_fresh_inputs(device, prec):
    W = DF.load_weights()
    m = TtKairos(device, W, getattr(Precision, prec)())
    capture = [golden_pair("single_gift")[0], golden_pair("ragged4")[0], bench_pair("paper", 8, W)[0]] + [None] * 2 + [
        bench_pair("paper", 256, W, lambda x: x.flip(0) * 1.5 + 0.1)[0]]
    fresh = [  # (B, front, golden head or None): none is the input its trace was captured with
        (2, *bench_pair("single", 1, W)),
        (8, *golden_pair("bias_5call", 3)),
        (16, *bench_pair("ragged", 8, W)),
        (2, *golden_pair("single_gift")),  # back to a captured input after the others ran
        (8, *bench_pair("long_horizon", 4, W)),
        (512, *bench_pair("paper", 256, W, gold=True)),  # the scored paper replay, captured on another input
    ]
    eagers = [m.forward(f) for _B, f, _g in fresh]  # eager first: nothing may allocate once a trace exists
    tr = TracedKairos(m)
    try:
        for (B, f, gold), eager, cap in zip(fresh, eagers, capture, strict=True):
            if cap is not None:  # capture on a different input of the same batch, then replay the fresh one
                tr.forward(cap)
            got = tr.forward(f)
            p, d = pcc(got, eager), float((got - eager).abs().max())
            g = pcc(got, gold) if gold is not None else float("nan")
            print(f"trace {prec} B={B}: replay vs eager pcc {p:.7f} max|d| {d:.3e}; vs golden {g:.6f}")
            assert p > 0.99999 and d < 1e-2, (B, p, d)
            if gold is not None:
                assert g >= (0.999 if prec == "accuracy" else 0.995)
        print(f"trace {prec}: {tr.captures} captures")
    finally:
        tr.release()
