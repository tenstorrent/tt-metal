# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: TTNN TimesFM-3 forward (one single-pass CPM decode) against the CPU reference goldens.

Shapes (inputs, head outputs and quantiles from ``tests/golden/make_goldens.py``):
    paper          1024 univariate series x 2048 -> 64      (66 tokens per row, read-out token 63)
    long_context   32 x 15360 -> 1024                       (512 tokens per row, 31 stitched read-outs)
    multivariate   128 items x 8 variates x 2048 -> 64      (1024 rows, variate attention in every layer)
    gift_e2e       64 series x U[256, 2048] -> 48, GIFT-Eval settings (symmetric averaging, make_positive)

Each case is ``<shape>-<precision>``; the contract's target and guards are
``-k "paper or (long_context and accuracy) or (multivariate and accuracy)"``.

``MODEL_PERF_JSON=<path>`` appends one record per case: ``steady_s`` is the median of 5 back-to-back
end-to-end calls (prepare_inputs -> upload_inputs -> forward_device -> postprocess); ``device_s`` and the
host split are medians of 5 more calls, each forward after a 0.3 s idle. ``cpu_s`` is the reference's time
for the shape, recorded with the golden.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import subprocess

import pytest
import torch

from models.experimental.timesfm3_forecast.tests.golden_io import (
    final_metrics,
    gate_failures,
    head_metrics,
    load_golden,
)

STEADY_ITERS = 5
DEVICE_SETTLE_S = 0.3
SHAPES = {
    "paper": "bench_paper",
    "long_context": "bench_long_context",
    "multivariate": "bench_multivariate",
    "gift_e2e": "gift64_h48",
}
# Sanity only (a catastrophic failure stops the run); the gates are judged from the record (``floor_ok``,
# ``tests/perf/check_gates.py``).
SANITY_PCC = 0.99
CASES = [(s, p) for s in SHAPES for p in ("accuracy", "performance")]


def _sync_time(device, fn):
    import ttnn

    start = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(device)
    return out, time.perf_counter() - start


def _git_state() -> dict:
    """PLAN v1.16: the commit and the model's dirty files, so check_gates.py can reject uncommitted evidence."""
    here = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", *a], cwd=here, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty": git("status", "--porcelain", "--", "tt", "tests")}


def _record(entry: dict) -> None:
    path = os.environ.get("MODEL_PERF_JSON")
    if not path:
        return
    out = Path(path)
    records = json.loads(out.read_text()) if out.is_file() else []
    out.write_text(json.dumps([*records, entry], indent=2) + "\n")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("shape,precision", CASES, ids=[f"{s}-{p}" for s, p in CASES])
def test_forward_perf(device, shape, precision):
    import ttnn

    from models.experimental.timesfm3_forecast.tt.common import TtPrecision
    from models.experimental.timesfm3_forecast.tt.model import TtTimesFM3

    g = load_golden(SHAPES[shape])
    c = g["contexts"]  # a list of tensors when the lengths differ (gift_e2e)
    contexts = [t.numpy() for t in c] if isinstance(c, list) else list(c.numpy())
    horizon, settings = g["horizon"], g.get("settings", {})
    tt = TtTimesFM3.from_checkpoint(device, TtPrecision.by_name(precision))

    def run():
        prepared = tt.prepare_inputs(contexts, horizon, **settings)
        inputs = tt.upload_inputs(prepared)
        out = tt.forward_device(inputs)
        head, quantiles = tt.postprocess(out, prepared)
        ttnn.deallocate(out)
        tt.deallocate_inputs(inputs)
        return head, quantiles

    _, cold_s = _sync_time(device, run)
    steady = [_sync_time(device, run)[1] for _ in range(STEADY_ITERS)]

    splits = []
    for _ in range(STEADY_ITERS):
        start = time.perf_counter()
        prepared = tt.prepare_inputs(contexts, horizon, **settings)
        prepare_s = time.perf_counter() - start
        inputs, upload_s = _sync_time(device, lambda: tt.upload_inputs(prepared))
        time.sleep(DEVICE_SETTLE_S)
        out, device_s = _sync_time(device, lambda: tt.forward_device(inputs))
        start = time.perf_counter()
        head, quantiles = tt.postprocess(out, prepared)
        postprocess_s = time.perf_counter() - start
        ttnn.deallocate(out)
        tt.deallocate_inputs(inputs)
        splits.append((prepare_s, upload_s, device_s, postprocess_s))
    prepare_s, upload_s, device_s, postprocess_s = (statistics.median(c) for c in zip(*splits))

    metrics = {**head_metrics(g["head_readout"], head), **final_metrics(g["quantiles"], torch.from_numpy(quantiles))}
    pcc_normalized = metrics["pcc_normalized"]
    failures = gate_failures(metrics, precision, shape)
    rows, tokens = prepared.rows, prepared.num_patches
    print(
        f"\n[PERF] forward {shape} / {precision}: {rows} rows x {tokens} tokens"
        f"\n  cold {cold_s:.3f} s, steady {statistics.median(steady):.4f} s ({', '.join(f'{s:.4f}' for s in steady)})"
        f"\n  split: prepare {prepare_s:.4f} upload {upload_s:.4f} device {device_s:.4f} postprocess {postprocess_s:.4f}"
        f"\n  cpu {g.get('cpu_s', float('nan')):.2f} s ({g.get('threads')} threads); "
        + ", ".join(f"{k} {v:.6f}" for k, v in metrics.items())
        + f"\n  gates: {'ok' if not failures else failures}"
    )
    _record(
        {
            "test": "forward",
            "name": shape,
            "precision": precision,
            "rows": rows,
            "tokens": tokens,
            "cold_s": cold_s,
            "steady_s": statistics.median(steady),
            "steady_runs": steady,
            "device_s": device_s,
            "prepare_s": prepare_s,
            "upload_s": upload_s,
            "postprocess_s": postprocess_s,
            "split_runs": [dict(zip(("prepare_s", "upload_s", "device_s", "postprocess_s"), s)) for s in splits],
            **metrics,
            "floor_ok": not failures,
            "gate_failures": {k: v[0] for k, v in failures.items()},
            "policy": tt.precision.describe(),
            "cpu_s": g.get("cpu_s"),
            "chip": os.environ.get("TTQ_CHIP_SERIAL"),
            **_git_state(),
        }
    )
    assert pcc_normalized >= SANITY_PCC, f"{shape}/{precision}: pcc_normalized {pcc_normalized}"
