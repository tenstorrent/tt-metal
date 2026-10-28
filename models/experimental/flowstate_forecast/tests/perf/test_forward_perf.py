# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: one FlowState forward (traced device part) per bench shape and precision, on one Blackhole chip.

Shapes (CONTRACT §1): paper 64 x 4038 -> 64 at s = 1 (the target, accuracy), long_context 16 x 16360 -> 48 at
s = 0.25, long_horizon 64 x 3382 -> 720, scale_q 256 x 675 -> 8 at s = 6, single 1 x 2048 -> 64, single_long
1 x 16360 -> 48. Inputs from the bench goldens. device_s: median of 5 trace replays, each after a 0.3 s idle;
steady_s: median of 5 end-to-end calls (host front end, upload, replay, host back end).
``MODEL_PERF_JSON=<path>`` appends one record per case.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.flowstate_forecast.reference import weights
from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS
from models.experimental.flowstate_forecast.tests.pcc import metrics as M
from models.experimental.flowstate_forecast.tt.model import FlowStateTT
from models.experimental.flowstate_forecast.tt.trace import TracedForward

ITERS = 5
SETTLE_S = 0.3
SHAPES = ("paper", "long_context", "long_horizon", "scale_q", "single", "single_long")
CASES = [pytest.param(s, p, id=f"{s}-{p}") for s in SHAPES for p in ("accuracy", "performance")]


def _timed(device, fn):
    t0 = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(device)
    return out, time.perf_counter() - t0


def _record(entry):
    path = os.environ.get("MODEL_PERF_JSON")
    if path:
        p = Path(path)
        rec = json.loads(p.read_text()) if p.is_file() else []
        rec.append(entry)
        p.write_text(json.dumps(rec, indent=2) + "\n")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024**2}], indirect=True)
@pytest.mark.parametrize("shape,precision", CASES)
def test_forward_perf(device, shape, precision):
    g = torch.load(GOLDENS / f"bench_{shape}.pt", weights_only=True)
    x, s, H = g["inputs"], g["s"], g["horizon"]
    tt = FlowStateTT(weights.load(), device, precision)
    traced = TracedForward(tt)
    host, meta = tt.prepare(x.clone(), s, H, host_only=True)
    k = traced.add(meta, s)
    traced.capture()

    def run():
        h, m = tt.prepare(x.clone(), s, H, host_only=True)
        traced.upload(h, k)
        return tt.finish(traced.execute(k), m, H)

    _, cold_s = _timed(device, run)
    steady = [_timed(device, run)[1] for _ in range(ITERS)]
    splits = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        h, m = tt.prepare(x.clone(), s, H, host_only=True)
        prepare_s = time.perf_counter() - t0
        _, upload_s = _timed(device, lambda: traced.upload(h, k))
        time.sleep(SETTLE_S)
        dec, device_s = _timed(device, lambda: traced.execute(k))
        t0 = time.perf_counter()
        out = tt.finish(dec, m, H)
        post_s = time.perf_counter() - t0
        splits.append((prepare_s, upload_s, device_s, post_s))
    prepare_s, upload_s, device_s, post_s = (statistics.median(c) for c in zip(*splits))
    met = M.gate_metrics(
        out["combined_normalized"],
        g["combined_normalized"],
        out["quantile_outputs"],
        g["quantile_outputs"],
        out["prediction_outputs"],
        g["prediction_outputs"],
    )
    gh = M.head(g["combined_normalized"])
    gq = M.head(g["quantile_outputs"])
    col = M.gate_metrics(M.collapsed(gh), gh, M.collapsed(gq), gq)
    if dump := os.environ.get("MODEL_PERF_DUMP"):
        Path(dump).mkdir(parents=True, exist_ok=True)
        torch.save(out, Path(dump) / f"{shape}-{precision}.pt")
    traced.release()
    print(
        f"\n[PERF] {shape} / {precision}: {meta['B']} x {x.shape[1]} -> {H}, s={s}, L={meta['L']}, Bp={meta['Bp']}"
        f"\n  device_s {device_s:.4f}  steady_s {statistics.median(steady):.4f}  cold_s {cold_s:.2f}"
        f"\n  split: prepare {prepare_s:.4f} upload {upload_s:.4f} post {post_s:.4f}  cpu_s {g['cpu_s']:.1f}"
        f"\n  metrics: {M.fmt(met)}\n  collapsed: {M.fmt(col)}"
    )
    _record(
        {
            "test": "forward",
            "name": shape,
            "precision": precision,
            "rows": meta["B"],
            "device_rows": meta["Bp"],
            "context": int(x.shape[1]),
            "horizon": H,
            "s": s,
            "L": meta["L"],
            "cold_s": cold_s,
            "steady_s": statistics.median(steady),
            "steady_runs": steady,
            "device_s": device_s,
            "prepare_s": prepare_s,
            "upload_s": upload_s,
            "postprocess_s": post_s,
            "split_runs": [dict(zip(("prepare_s", "upload_s", "device_s", "postprocess_s"), r)) for r in splits],
            **met,
            **{f"collapsed_{k}": v for k, v in col.items()},
            "cpu_s": g["cpu_s"],
            "speedup_vs_cpu": g["cpu_s"] / statistics.median(steady),
            "chip": os.environ.get("TTQ_CHIP_SERIAL"),
            "traced": True,
        }
    )
    assert met["pcc_normalized"] >= (0.999 if precision == "accuracy" else 0.995) and met["pcc_final"] >= 0.98
