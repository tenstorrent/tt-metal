# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: one TiRex-2 forward pass (``tta_sign_flip=False``, ``pad_context=True``) on one Blackhole chip.

Shapes (CONTRACT sections 1 and 2), parametrized as ``<shape>-<precision>``:
    paper          1024 univariate series x 2048 -> 64 (target in ``accuracy``; guard in ``performance``)
    long_context   32 series x 8192 -> 928 (29 read-out tokens, no constant left pad)
    multivariate   128 groups x 8 rows (1 target + 7 past covariates) x 2048 -> 64

Inputs, the upstream head output at the read-out tokens, ``q_one`` and the CPU time come from the fp32 goldens
(``tests/golden/make_goldens.py``). The forward runs as a captured trace (the op-level sLSTM loop is ~50k ops).

``MODEL_PERF_JSON=<path>`` appends one record per case (keys per model_bringup_optim reference.md, plus
``pcc_final``, ``median_err`` and the host split). ``TIREX2_PAD_SKIP=0`` forces the full path (R1-1 baseline).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import golden, have_goldens
from models.experimental.tirex2_forecast.tests.pcc import metrics as M
from models.experimental.tirex2_forecast.tests.pcc.helpers import host_weights

STEADY_ITERS = 5
DEVICE_SETTLE_S = 0.3  # idle before each timed device-only forward (duty-cycle drift, see the skill)
TRACE_REGION = 3 * 1024**3
SHAPES = ("paper", "long_context", "multivariate")
CASES = [pytest.param(s, p, id=f"{s}-{p}") for s in SHAPES for p in ("accuracy", "performance")]


def shape_inputs(name: str):
    """``(golden, context [rows, T], horizon, group_ids, target_mask)`` of a bench shape."""
    g = golden(f"bench_{name}")
    if name == "multivariate":
        groups, per, T = g["context"].shape
        ctx = g["context"].reshape(-1, T)
        gid = torch.arange(groups).repeat_interleave(per)
        tm = (torch.arange(groups * per) % per) == 0
    else:
        ctx, gid, tm = g["context"], None, None
    return g, ctx, g["horizon"], gid, tm


def readout_head(tt, out, prep) -> torch.Tensor:
    """Head output at the read-out tokens ``[rows, n, 288]`` (source row order) from the device window."""
    import ttnn

    from models.experimental.tirex2_forecast.tt.model import WINDOW

    win = ttnn.to_torch(out).float().reshape(prep.device_rows, WINDOW[1] - WINDOW[0], -1)
    tok = [t - WINDOW[0] for t in df.readout_tokens(prep.horizon)]
    head = torch.empty(prep.rows, len(tok), win.shape[-1])
    valid = prep.order >= 0
    head[prep.order[valid]] = win[valid][:, tok]
    return head


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


@pytest.mark.skipif(not have_goldens(), reason="goldens not generated")
@pytest.mark.timeout(2700)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("shape,precision", CASES)
def test_forward_perf(device, shape, precision):
    from models.experimental.tirex2_forecast.tt.common import Precision
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2
    from models.experimental.tirex2_forecast.tt.trace import TracedForward

    g, ctx, horizon, gid, tm = shape_inputs(shape)
    pad_skip = os.environ.get("TIREX2_PAD_SKIP", "1") != "0"
    tt = TtTiRex2(
        device, host_weights(), Precision.by_name(precision), variate=shape == "multivariate", pad_skip=pad_skip
    )
    traced = TracedForward(tt)

    def run():
        prep = tt.prepare_inputs(ctx, horizon, gid, tm)
        out = traced(prep)
        return prep, out, tt.postprocess(out, prep)

    (_, _, _), cold_s = _sync_time(device, run)
    steady = [_sync_time(device, run)[1] for _ in range(STEADY_ITERS)]

    splits = []
    for _ in range(STEADY_ITERS):
        start = time.perf_counter()
        prep = tt.prepare_inputs(ctx, horizon, gid, tm)
        prepare_s = time.perf_counter() - start
        _, upload_s = _sync_time(device, lambda: traced.upload(prep))
        time.sleep(DEVICE_SETTLE_S)
        out, device_s = _sync_time(device, lambda: traced.execute(prep))
        start = time.perf_counter()
        q = tt.postprocess(out, prep)
        postprocess_s = time.perf_counter() - start
        splits.append((prepare_s, upload_s, device_s, postprocess_s))
    prepare_s, upload_s, device_s, postprocess_s = (statistics.median(c) for c in zip(*splits))

    head = readout_head(tt, out, prep)
    if dump := os.environ.get("MODEL_PERF_DUMP"):
        Path(dump).mkdir(parents=True, exist_ok=True)
        torch.save({"head": head, "q": q}, Path(dump) / f"{shape}-{precision}.pt")
    metrics = M.gate_metrics(head, g["head_out"], q, g["q_one"], prep.scale[prep.target_mask])
    collapsed = M.collapsed_metrics(g["head_out"], g["q_one"], prep.loc, prep.scale, horizon, prep.target_mask)
    traced.release()
    tt.release()

    print(
        f"\n[PERF] forward {shape} / {precision}: {prep.rows} rows ({prep.device_rows} on device) x "
        f"{ctx.shape[-1]} -> {horizon}"
        f"\n  cold_s:    {cold_s:.3f}"
        f"\n  steady_s:  {statistics.median(steady):.3f}  ({', '.join(f'{s:.3f}' for s in steady)})"
        f"\n  device_s:  {device_s:.4f}  (median of {STEADY_ITERS}, {DEVICE_SETTLE_S} s idle before each)"
        f"\n  split:     prepare {prepare_s:.4f}  upload {upload_s:.4f}  postprocess {postprocess_s:.4f}"
        f"\n  cpu_s:     {g['cpu_s']:.2f}  ({g['threads']} threads)"
        f"\n  metrics:   {M.fmt(metrics)}"
        f"\n  collapsed: {M.fmt(collapsed)}  (quantiles collapsed to their mean; must fail every floor)"
    )
    _record(
        {
            "test": "forward",
            "name": shape,
            "precision": precision,
            "rows": prep.rows,
            "device_rows": prep.device_rows,
            "context": int(ctx.shape[-1]),
            "horizon": horizon,
            "pad_cut": prep.cut,
            "cold_s": cold_s,
            "steady_s": statistics.median(steady),
            "steady_runs": steady,
            "device_s": device_s,
            "prepare_s": prepare_s,
            "upload_s": upload_s,
            "postprocess_s": postprocess_s,
            "split_runs": [dict(zip(("prepare_s", "upload_s", "device_s", "postprocess_s"), s)) for s in splits],
            **metrics,
            **{f"collapsed_{k}": v for k, v in collapsed.items()},
            "cpu_s": g["cpu_s"],
            "speedup_vs_cpu": g["cpu_s"] / statistics.median(steady),
            "chip": os.environ.get("TTQ_CHIP_SERIAL"),
            "traced": True,
        }
    )
    # every gates.json floor for this shape and precision (D3 condition 3, D4); a missing metric is a miss
    from models.experimental.tirex2_forecast.tests.perf.check_gates import DEFAULT_GATES, _floors, passes

    floors = _floors(json.loads(DEFAULT_GATES.read_text()), shape, precision)
    failed = {k: (metrics.get(k), f) for k, f in floors.items() if k not in metrics or not passes(k, metrics[k], f)}
    assert not failed, f"{shape}/{precision}: floors failed {failed}"
