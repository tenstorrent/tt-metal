# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: one TTNN t0-gamma forward (``_single_pass``) against the fp32 CPU reference, on one chip.

Cases are ``<shape>-<precision>``; the contract's target is ``paper-accuracy``, its guards ``long_context-accuracy``,
``multivariate-accuracy`` and ``paper-performance``:
    paper         1024 univariate series x 2048 -> 64      66 patches (64 context + 2 withheld), 2 read out
    long_context  32 x 8192 -> 1024                        288 patches, 32 read out
    multivariate  128 series x 8 variates x 2048 -> 64     group attention over 8 rows

Inputs, the reference head output at the read-out patches, raw21 and the CPU time come from the goldens
(``tests/golden/make_goldens.py``), so no CPU forward runs here.

Timing: ``cold_s`` is the first end-to-end call (compile); ``steady_s`` the median of 5 back-to-back end-to-end
calls (prepare, upload, forward, postprocess); the host / device split is the median of 5 runs of each stage,
with a 0.3 s idle before each timed ``forward_device`` (device time depends on the duty cycle).

``MODEL_PERF_JSON=<path>`` appends one record per case to a JSON list (``model_bringup_optim`` contract).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest

from models.experimental.t0_forecast.tests.helpers import FLOORS as GATE_FLOORS
from models.experimental.t0_forecast.tests.helpers import (
    collapsed,
    final_metrics,
    golden,
    meets,
    quantile_metrics,
    weights,
)

STEADY_ITERS = 5
DEVICE_SETTLE_S = 0.3
SHAPES = {
    # name: (golden file, series, variates, context, horizon)
    "paper": ("bench_paper.pt", 1024, 1, 2048, 64),
    "long_context": ("bench_long_context.pt", 32, 1, 8192, 1024),
    "multivariate": ("bench_multivariate.pt", 128, 8, 2048, 64),
}
CASES = [
    "paper-accuracy",
    "long_context-accuracy",
    "multivariate-accuracy",
    "paper-performance",
    "long_context-performance",
    "multivariate-performance",
]


def _sync_time(device, fn):
    import ttnn

    start = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(device)
    return out, time.perf_counter() - start


def _record(entry: dict) -> None:
    path = os.environ.get("MODEL_PERF_JSON")
    if not path:
        return
    out = Path(path)
    records = json.loads(out.read_text()) if out.is_file() else []
    records.append(entry)
    out.write_text(json.dumps(records, indent=2) + "\n")


_MODELS: dict = {}


@pytest.fixture(scope="module")
def device():
    import ttnn

    dev = ttnn.open_device(device_id=0)
    yield dev
    _MODELS.clear()
    ttnn.close_device(dev)


def _tt(device, precision_name: str):
    from models.experimental.t0_forecast.tt.common import TtT0Precision
    from models.experimental.t0_forecast.tt.model import TtT0

    if precision_name not in _MODELS:
        _MODELS.clear()  # one precision's weights on the chip at a time
        _MODELS[precision_name] = TtT0(device, weights(), TtT0Precision.named(precision_name))
    return _MODELS[precision_name]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("case", CASES)
def test_forward_perf(device, case):
    import ttnn

    from models.experimental.t0_forecast.tt.common import pad_to_tile

    shape_name, precision_name = case.split("-")
    file, series, variates, context_len, horizon = SHAPES[shape_name]
    g = golden(file)
    context = g["context"]
    tt = _tt(device, precision_name)

    def run():
        prepared = tt.prepare_inputs(context, horizon)
        inputs = tt.upload_inputs(prepared)
        out = tt.forward_device(inputs)
        final = tt.postprocess(out, prepared)
        ttnn.deallocate(out)
        tt.deallocate_inputs(inputs)
        return final

    _, cold_s = _sync_time(device, run)
    steady = [_sync_time(device, run)[1] for _ in range(STEADY_ITERS)]
    steady_s = statistics.median(steady)

    splits = []
    for _ in range(STEADY_ITERS):
        start = time.perf_counter()
        prepared = tt.prepare_inputs(context, horizon)
        prepare_s = time.perf_counter() - start
        inputs, upload_s = _sync_time(device, lambda: tt.upload_inputs(prepared))
        time.sleep(DEVICE_SETTLE_S)
        out, device_s = _sync_time(device, lambda: tt.forward_device(inputs))
        start = time.perf_counter()
        normalized = tt.normalized(out, prepared)
        final = tt.postprocess(out, prepared)
        postprocess_s = time.perf_counter() - start
        ttnn.deallocate(out)
        tt.deallocate_inputs(inputs)
        splits.append((prepare_s, upload_s, device_s, postprocess_s))
    prepare_s, upload_s, device_s, postprocess_s = (statistics.median(c) for c in zip(*splits))

    metrics = {**quantile_metrics(g["head_readout"], normalized), **final_metrics(g["raw21"], final)}
    collapsed_metrics = {
        **quantile_metrics(g["head_readout"], collapsed(g["head_readout"])),
        **final_metrics(g["raw21"], collapsed(g["raw21"])),
    }
    kinds = g.get("kinds")
    if kinds is not None:
        metrics["pcc_row_min_kind"] = kinds[metrics["pcc_row_min_row"] // variates]
    pcc_norm, pcc_final = metrics["pcc_normalized"], metrics["pcc_final"]
    floors = GATE_FLOORS[precision_name]
    rows = series * variates
    print(
        f"\n[PERF] forward {shape_name} / {precision_name}"
        f"\n  rows x context -> horizon: {rows} x {context_len} -> {horizon} ({series} series x {variates} variates)"
        f"\n  patches:      {prepared.num_patches} per row, {pad_to_tile(prepared.num_patches)} padded"
        f"\n  cold_s:       {cold_s:.3f}"
        f"\n  steady_s:     {steady_s:.4f}  (median of {STEADY_ITERS}: {', '.join(f'{s:.4f}' for s in steady)})"
        f"\n  device_s:     {device_s:.4f}  ({DEVICE_SETTLE_S} s idle before each)"
        f"\n  split:        prepare {prepare_s:.4f}  upload {upload_s:.4f}  postprocess {postprocess_s:.4f}"
        f"\n  cpu_s:        {g['cpu_s']:.2f}  (fp32 reference, 8 threads)  speedup {g['cpu_s'] / steady_s:.1f}x"
        f"\n  pcc:          {pcc_norm:.6f} normalized, {pcc_final:.6f} final, {metrics['pcc_final_spread']:.6f} final spread"
        f" (floors {floors})"
        f"\n  spread / row: {metrics['pcc_spread']:.6f} spread, {metrics['pcc_row_p01']:.6f} row p01, "
        f"{metrics['pcc_row_min']:.6f} row min (collapsed output: {collapsed_metrics['pcc_normalized']:.6f} / "
        f"{collapsed_metrics['pcc_spread']:.6f} / {collapsed_metrics['pcc_row_p01']:.6f} / {collapsed_metrics['pcc_row_min']:.6f}; final "
        f"{collapsed_metrics['pcc_final']:.6f}, final spread {collapsed_metrics['pcc_final_spread']:.6f})"
        f"\n  row min:      row {metrics['pcc_row_min_row']} ({metrics.get('pcc_row_min_kind')}); active rows: p01 "
        f"{metrics['pcc_row_p01_active']:.6f}, min {metrics['pcc_row_min_active']:.6f} ({metrics['rows_excluded']} excluded); "
        f"spread ratio p01 {metrics['spread_ratio_p01']:.4f} p99 {metrics['spread_ratio_p99']:.4f}"
    )
    _record(
        {
            "test": "forward",
            "name": shape_name,
            "precision": precision_name,
            "rows": rows,
            "series": series,
            "variates": variates,
            "context": context_len,
            "horizon": horizon,
            "num_patches": prepared.num_patches,
            "padded_seq": pad_to_tile(prepared.num_patches),
            "cold_s": cold_s,
            "steady_s": steady_s,
            "steady_runs": steady,
            "device_s": device_s,
            "prepare_s": prepare_s,
            "upload_s": upload_s,
            "postprocess_s": postprocess_s,
            "split_runs": [dict(zip(("prepare_s", "upload_s", "device_s", "postprocess_s"), s)) for s in splits],
            "pcc_normalized": pcc_norm,
            **{k: v for k, v in metrics.items() if k != "pcc_normalized"},
            "collapsed": collapsed_metrics,
            "floor_ok": meets(metrics, floors),
            "cpu_s": g["cpu_s"],
            "speedup": g["cpu_s"] / steady_s,
            "chip": os.environ.get("TTQ_CHIP_SERIAL"),
        }
    )
    # A sanity bound keeps the harness running; the contract floors are checked from the record.
    assert pcc_norm >= 0.98 and pcc_final >= 0.95, f"{case}: pcc {pcc_norm}, {pcc_final}"


GIFT_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("precision_name", ["accuracy", "performance"])
def test_e2e_gift_perf(device, precision_name):
    """D4 item 1: end-to-end time of ``predict`` at the GIFT-Eval settings (trim, symmetric averaging = two forwards,
    clip, snap; the 9 GIFT levels) on the paper input. Timing only: the accuracy of these settings is gated by
    tests/pcc/test_model.py::test_e2e_gift_settings."""
    file, series, variates, context_len, horizon = SHAPES["paper"]
    context = golden(file)["context"]
    tt = _tt(device, precision_name)
    calls = []

    def network(p):
        calls.append(1)
        return tt.network(p)

    settings = dict(trim_leading_missing=True, symmetric_averaging=True, clip_nonnegative=True, snap_integers=True)
    from models.experimental.t0_forecast.tt import host

    run = lambda: host.predict(tt.config, network, context, horizon, GIFT_LEVELS, **settings)
    _, cold_s = _sync_time(device, run)
    calls.clear()
    steady = [_sync_time(device, run)[1] for _ in range(STEADY_ITERS)]
    steady_s = statistics.median(steady)
    forwards = len(calls) // STEADY_ITERS
    print(f"\n[e2e] paper_gift-{precision_name}: steady {steady_s:.4f} s ({forwards} forwards), cold {cold_s:.4f} s")
    _record(
        {
            "test": "e2e_gift",
            "name": "paper_gift",
            "precision": precision_name,
            "series": series,
            "context": context_len,
            "horizon": horizon,
            "levels": list(GIFT_LEVELS),
            "settings": settings,
            "forwards": forwards,
            "cold_s": cold_s,
            "steady_s": steady_s,
            "steady_runs": steady,
            "chip": os.environ.get("TTQ_CHIP_SERIAL"),
        }
    )
