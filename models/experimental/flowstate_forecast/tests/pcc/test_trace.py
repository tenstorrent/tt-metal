# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-input trace test (skill §1 C, PLAN §9 v1.21): two shapes' buffers are allocated, then both traces are
captured; new values go through each trace and are compared with the eager forward on the same values.
Positive control (v1.13): replaying without uploading the new values (stale input) must fail the same check.
The job log must not contain ttnn's "potentially unsafe ... active trace" allocation warning (checked by the job
wrapper, see doc/optimize/iter_00/trace_fresh_inputs.txt)."""

import pytest
import torch

from models.experimental.flowstate_forecast.reference import weights
from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS
from models.experimental.flowstate_forecast.tests.pcc import metrics
from models.experimental.flowstate_forecast.tt.model import FlowStateTT
from models.experimental.flowstate_forecast.tt.trace import TracedForward

CASES = ("model16", "monthly")  # s = 1, 16 series; s = 2, 3 series, NaN-free


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024**2}], indirect=True)
def test_trace_fresh_inputs(device):
    tt = FlowStateTT(weights.load(), device, "accuracy")
    gold = {c: torch.load(GOLDENS / f"{c}.pt", weights_only=True) for c in CASES}
    gen = torch.Generator().manual_seed(7)
    inputs, eager = {}, {}
    for c, g in gold.items():  # eager references first: nothing is allocated once a trace is live (v1.21)
        x0 = g["inputs"].clone()
        fresh = x0 * (1 + 0.3 * torch.rand(x0.shape, generator=gen)) + 0.5 * torch.randn(x0.shape, generator=gen)
        inputs[c] = {"golden": x0, "fresh": fresh}
        eager[c] = {k: tt(x.clone(), g["s"], g["horizon"])["combined_normalized"] for k, x in inputs[c].items()}
    keys, traced = {}, {}
    for c, g in gold.items():  # every shape's input buffer before any capture
        traced[c] = TracedForward(tt)
        _, meta = tt.prepare(g["inputs"].clone(), g["s"], g["horizon"], host_only=True)
        keys[c] = traced[c].add(meta, g["s"])

    def replay(c, x, upload=True):
        g = gold[c]
        h, m = tt.prepare(x.clone(), g["s"], g["horizon"], host_only=True)
        if upload:
            traced[c].upload(h, keys[c])
        return tt.finish(traced[c].execute(keys[c]), m, g["horizon"])["combined_normalized"]

    # one trace live at a time: capturing a second trace while one is live allocates under an active trace,
    # which ttnn warns about and PLAN v1.21 counts as a failure
    for c in CASES:
        traced[c].capture()
        for label in ("golden", "fresh"):
            rep = replay(c, inputs[c][label])
            p = metrics.pearson(eager[c][label], rep)
            err = (eager[c][label] - rep).abs().max().item()
            print(f"trace {c} {label}: pcc vs eager {p:.8f} max abs {err:.3e}")
            assert p > 0.99999 and err < 1e-3, (c, label, p, err)
        # control: the trace still holds the fresh values; replaying it without uploading the golden inputs
        # must fail the same check against the golden inputs' eager output
        err = (eager[c]["golden"] - replay(c, inputs[c]["golden"], upload=False)).abs().max().item()
        print(f"trace {c} control (stale input): max abs {err:.3e}")
        assert err > 1e-3, (c, err)
        traced[c].release()
