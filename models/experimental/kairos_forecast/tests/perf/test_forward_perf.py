# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bench: traced generate() on every bench shape and precision, scored and timed; one record per <shape>-<precision>.

    KAIROS_PERF_JSON=<iter_dir>/bench.json pytest models/experimental/kairos_forecast/tests/perf/test_forward_perf.py
    BENCH_LAYERS=2+2   reduced encoder + decoder layers: timing only (no accuracy keys), for screening; written to
                       bench_reduced.json next to $KAIROS_PERF_JSON
    BENCH_RUNS=5       timed generate() runs after one warm-up

Timing per forward (the PLAN G1 split, each part separately, the device part after a 0.3 s idle so no host work
overlaps it): prepare_s (host front-end + input build), upload_s (copy into the trace inputs), device_s (trace
replay + synchronize), post_s (read-back + quantiles). steady_s is the wall time of one whole generate().
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tests.perf.check_gates import session_stamp
from models.experimental.kairos_forecast.tests.perf.suite import GIFT_CASES, load, run_case
from models.experimental.kairos_forecast.tt.common import Precision
from models.experimental.kairos_forecast.tt.generator import KairosGenerator
from models.experimental.kairos_forecast.tt.model import TtKairos
from models.experimental.kairos_forecast.tt.router_state import RouterState
from models.experimental.kairos_forecast.tt.trace import TracedKairos, _leaves

STAMP = session_stamp()  # v1.23: taken once, at collection (the session start); every record carries it

LAYERS = tuple(int(x) for x in os.environ.get("BENCH_LAYERS", "6+6").split("+"))
RUNS = int(os.environ.get("BENCH_RUNS", "5"))
SHAPES = os.environ.get("BENCH_SHAPES", ",".join(GIFT_CASES)).split(",")
TARGET = ("paper", "accuracy")


class TimedTrace(TracedKairos):
    """TracedKairos with the per-forward split recorded in self.splits (host_frontend time is added by the caller)."""

    def __init__(self, model, idle=0.3):
        super().__init__(model)
        self.idle, self.splits = idle, []

    def forward(self, front):
        import ttnn

        t0 = time.perf_counter()
        self._switch(front)
        host = self.model.prepare_host(front)
        t1 = time.perf_counter()
        dev = {k: v for k, v in self.inputs[self.B].items() if k not in ("reg", "dec_in", "causal")}
        for h, d in zip(_leaves(host), _leaves(dev), strict=True):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.synchronize_device(self.device)
        t2 = time.perf_counter()
        if self.idle:
            time.sleep(self.idle)
        t3 = time.perf_counter()
        out = self.run()
        ttnn.synchronize_device(self.device)
        t4 = time.perf_counter()
        res = ttnn.to_torch(out).float()
        res = res.view(self.B, -1, res.shape[-1])[:, :2]
        t5 = time.perf_counter()
        self.splits.append({"prepare_s": t1 - t0, "upload_s": t2 - t1, "device_s": t4 - t3, "post_s": t5 - t4,
                            "rows": self.B})
        return res


def _record(entry):
    """Full-depth records go to $KAIROS_PERF_JSON; reduced-depth (timing only) ones to bench_reduced.json beside it,
    so check_gates never sees a record without the gate metrics under a gated name."""
    path = os.environ.get("KAIROS_PERF_JSON")
    if not path:
        return
    out = Path(path)
    if entry.get("layers", [6, 6]) != [6, 6]:
        out = out.with_name("bench_reduced.json")
    records = json.loads(out.read_text()) if out.is_file() else []
    records.append(entry)
    out.write_text(json.dumps(records, indent=1) + "\n")


@pytest.fixture(scope="module")
def W():
    return DF.load_weights()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("precision", ["accuracy", "performance"])
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_perf(device, W, shape, precision):
    model = TtKairos(device, W, getattr(Precision, precision)(), layers=LAYERS)
    full = LAYERS == (6, 6)
    timed = TimedTrace(model, idle=0.0)
    rec = load(shape)
    # warm-up (captures the trace) and, at full depth, the scored generate() records on the traced path
    if full:
        recs, _ = run_case(timed, shape, precision)
    else:
        KairosGenerator(timed, RouterState(rec["bias0"])).generate(rec["context"], rec["horizon"], rec["gift"],
                                                                     rec["gift"])
    steady, runs = [], []
    timed.idle = 0.3
    for _ in range(RUNS):
        timed.splits = []
        gen = KairosGenerator(timed, RouterState(rec["bias0"]))
        t0 = time.perf_counter()
        gen.generate(rec["context"], rec["horizon"], rec["gift"], rec["gift"])
        steady.append(time.perf_counter() - t0 - timed.idle * len(timed.splits))
        runs.append({k: sum(s[k] for s in timed.splits) for k in ("prepare_s", "upload_s", "device_s", "post_s")})
    timed.release()
    dev = [r["device_s"] for r in runs]
    entry = {
        "test": "generate",
        "name": shape,
        "precision": precision,
        "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"),
        "layers": list(LAYERS),
        "forwards": len(timed.splits),
        "rows_per_forward": [s["rows"] for s in timed.splits],
        "steady_s": statistics.median(steady),
        "device_s": statistics.median(dev),
        "device_ms": 1e3 * statistics.median(dev),
        "device_s_min": min(dev),
        "split_runs": runs,
        "steady_runs": steady,
        **STAMP,
    }
    if full:
        for r in recs:  # main record carries the generate() metrics; a rollout's _tf record is written alongside
            if r["name"] == shape:
                entry.update({k: v for k, v in r.items() if k not in ("name", "precision", "e2e_s")})
            else:
                _record({**r, "test": "teacher_forced", **STAMP})
    print(f"[PERF] {shape}/{precision} layers {LAYERS}: device {entry['device_ms']:.2f} ms (min "
          f"{1e3 * entry['device_s_min']:.2f}), steady {entry['steady_s'] * 1e3:.1f} ms, "
          f"split {json.dumps({k: round(statistics.median(r[k] for r in runs) * 1e3, 2) for k in runs[0]})}"
          + (f", pcc_normalized {entry['pcc_normalized']:.6f}" if full else ""), flush=True)
    _record(entry)
