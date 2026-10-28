# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: Tafsut forecasts on one Blackhole chip, traced, parametrized ``<shape>-<precision>``.

Shapes (CONTRACT §1, §2): ``paper`` (target, 1024 x 2048 -> 64), ``long_context`` (32 x 32768 -> 64),
``long_horizon`` (512 x 2048 -> 2048: two forwards; forward 2 gated teacher-forced), ``gift_ragged`` (32, GIFT batch
construction -> 48) and ``gift_ragged_odd`` (16, batch max 4931 -> bucket 8192).

``MODEL_PERF_JSON=<path>`` appends one record per case. ``BENCH_LAYERS=<n>`` screens with n layers (device time
only, no metrics; the reduced-layer read-out skips the cut). ``BENCH_SHAPES=a,b`` limits the shapes.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tests.golden.goldens import GOLDEN_DIR, load_golden
from models.experimental.tafsut_forecast.tests.pcc import metrics as M

STEADY_ITERS = 5
DEVICE_SETTLE_S = 0.3
TRACE_REGION = 512 * 1024**2
SHAPES = ("paper", "long_context", "long_horizon", "gift_ragged", "gift_ragged_odd")
if os.environ.get("BENCH_SHAPES"):
    SHAPES = tuple(os.environ["BENCH_SHAPES"].split(","))
CASES = [pytest.param(s, p, id=f"{s}-{p}") for s in SHAPES for p in ("accuracy", "performance")]


from models.experimental.tafsut_forecast.tests.perf.check_gates import session_stamp as _stamp

_stamp()  # git stamp, porcelain and started_at at session start (collection), before any device work


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
    from models.experimental.tafsut_forecast.tests.perf.check_gates import session_stamp

    entry.update(session_stamp())  # taken once, at the first test module's import (v1.23)
    out = Path(path)
    records = json.loads(out.read_text()) if out.is_file() else []
    records.append(entry)
    out.write_text(json.dumps(records, indent=2) + "\n")


def forward_inputs(g: dict) -> list:
    """Host contexts of each forward: forward 2 teacher-forced on the reference's forward-1 median."""
    xs = [g["x"]]
    if g["horizon"] > 1024:
        xs.append(torch.cat([g["x"], g["quantiles"][:, :1024, df.MEDIAN_IDX]], -1)[:, -df.MAX_CTX :])
    return xs


def gate_case(g: dict, heads: list, q: torch.Tensor, k: int) -> tuple[dict, dict]:
    """Metrics and collapsed-output metrics of forward ``k`` (head at its read-out tokens; final quantiles)."""
    H = g["horizon"]
    n = -(-min(H - 1024 * k, 1024) // df.PATCH)
    gh = g["forwards"][k]["head"][:, :n]
    prep = df.prepare_inputs(forward_inputs(g)[k])
    gq = g["quantiles"].transpose(1, 2)
    met = M.gate_metrics(heads[k][:, :n].float(), gh, q.transpose(1, 2), gq, prep.scale)
    # collapsed control: forward k's golden head collapsed, then the same post-processing
    c = M.collapsed(gh)
    qc = df.postprocess(c, prep, min(H - 1024 * k, 1024)).transpose(1, 2)
    sl = slice(1024 * k, 1024 * k + qc.shape[-1])
    col = M.gate_metrics(c, gh, qc, gq[:, :, sl], prep.scale)
    return met, col


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("shape,precision", CASES)
def test_forward_perf(device, shape, precision):
    from models.experimental.tafsut_forecast.tt.common import Precision
    from models.experimental.tafsut_forecast.tt.model import NL, TtTafsut
    from models.experimental.tafsut_forecast.tt.trace import TracedForward

    layers = int(os.environ.get("BENCH_LAYERS", NL))
    g = load_golden(shape)
    tt = TtTafsut(device, precision=Precision.by_name(precision), n_layers=layers)
    traced = TracedForward(tt)
    xs = forward_inputs(g)
    H = g["horizon"]

    def run():
        return traced.forecast(g["x"], H)

    _, cold_s = _sync_time(device, run)
    steady = [_sync_time(device, run)[1] for _ in range(STEADY_ITERS)]
    splits, heads = [], []
    for _ in range(STEADY_ITERS):
        tot = [0.0, 0.0, 0.0, 0.0]
        heads = []
        for k, x in enumerate(xs):
            start = time.perf_counter()
            prep = tt.prepare_inputs(x)
            tot[0] += time.perf_counter() - start
            _, up = _sync_time(device, lambda: traced.upload(prep))
            time.sleep(DEVICE_SETTLE_S)
            out, dev = _sync_time(device, lambda: traced.execute(prep))
            start = time.perf_counter()
            import ttnn

            head = ttnn.to_torch(out).float()
            tt.postprocess(head, prep, min(H - 1024 * k, 1024))
            tot[1] += up
            tot[2] += dev
            tot[3] += time.perf_counter() - start
            heads.append(head)
        splits.append(tot)
    prepare_s, upload_s, device_s, postprocess_s = (statistics.median(c) for c in zip(*splits))
    rec = {
        "test": "forward",
        "name": shape,
        "precision": precision,
        "rows": int(g["x"].shape[0]),
        "context": int(g["x"].shape[-1]),
        "horizon": H,
        "forwards": len(xs),
        "layers": layers,
        "cold_s": cold_s,
        "steady_s": statistics.median(steady),
        "steady_runs": steady,
        "device_s": device_s,
        "device_ms": device_s * 1e3,
        "prepare_s": prepare_s,
        "upload_s": upload_s,
        "postprocess_s": postprocess_s,
        "split_runs": [dict(zip(("prepare_s", "upload_s", "device_s", "postprocess_s"), s)) for s in splits],
        "chip": os.environ.get("TTQ_CHIP_SERIAL"),
        "traced": True,
    }
    msg = ""
    if layers == NL:
        q = traced.forecast(g["x"], H)  # end to end (forward 2 fed the device's own median)
        manifest = json.loads((GOLDEN_DIR / "MANIFEST.json").read_text())
        rec["cpu_s"] = manifest["timings_s"].get(shape)
        rec["speedup_vs_cpu"] = rec["cpu_s"] / rec["steady_s"] if rec["cpu_s"] else None
        met, col = gate_case(g, heads, q, 0)
        rec.update(met)
        rec.update({f"collapsed_{k}": v for k, v in col.items() if not isinstance(v, (list, dict))})
        msg = f"\n  metrics:   {M.fmt(met)}\n  collapsed: {M.fmt(col)}"
        if len(xs) > 1:  # forward 2, teacher-forced (CONTRACT §3, D0 amendment 5)
            met2, col2 = gate_case(g, heads, q, 1)
            rec.update({f"f2_{k}": v for k, v in met2.items() if not k.startswith("pcc_final")})
            msg += f"\n  forward 2 (teacher-forced): {M.fmt(met2)}"
        if os.environ.get("TAFSUT_DUMP"):  # raw heads for row-level analysis (D3 item 1)
            torch.save(
                {"heads": heads, "q": q}, os.path.join(os.environ["TAFSUT_DUMP"], f"bench-{shape}-{precision}.pt")
            )
    traced.release()
    print(
        f"\n[PERF] forward {shape} / {precision} ({layers} layers): {rec['rows']} x {rec['context']} -> {H}"
        f"\n  cold_s {cold_s:.3f}  steady_s {rec['steady_s']:.4f}  device_ms {device_s * 1e3:.2f}"
        f"  (runs {', '.join(f'{s[2] * 1e3:.2f}' for s in splits)})"
        f"\n  split: prepare {prepare_s:.4f} upload {upload_s:.4f} postprocess {postprocess_s:.4f}" + msg
    )
    _record(rec)
    if layers == NL:
        from models.experimental.tafsut_forecast.tests.perf.check_gates import DEFAULT_GATES, _floors, passes

        floors = _floors(json.loads(DEFAULT_GATES.read_text()), shape, precision)
        failed = {k: (rec.get(k), f) for k, f in floors.items() if rec.get(k) is not None and not passes(k, rec[k], f)}
        assert not failed, f"{shape}/{precision}: floors failed {failed}"
        caught = [
            k
            for k, f in floors.items()
            if rec.get(f"collapsed_{k}") is not None and not passes(k, rec[f"collapsed_{k}"], f)
        ]
        assert caught, "the collapsed output passes every floor"
