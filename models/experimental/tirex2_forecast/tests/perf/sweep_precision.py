# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Precision-policy sweep on the paper bench shape: every gate metric and the traced device time per policy.

    TIREX2_SWEEP_OUT=<dir> python -m pytest .../tests/perf/sweep_precision.py -s        (under ttq --kind bench)
    TIREX2_SWEEP_POLICIES="accuracy;accuracy+hifi4=all+y32"                           (default: the list below)

Each policy (``Precision.by_name`` syntax: ``hifi4=<roles|all>``, ``bf16=<roles|all>``, ``y32``, ``acc16``) is
built, traced and replayed on the 1024 paper rows (realistic mix); metrics are computed on all rows and on rows
0..63 (the suite's ``paper64``), and device time is the median of 3 replays after a 0.3 s idle, all policies
back to back in one process. Writes ``precision_sweep.json`` / ``.md`` and, per policy, the head output and
forecast (``dumps/<policy>.pt``, for the worst-row analysis).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

from models.experimental.tirex2_forecast.tests.golden.goldens import golden, have_goldens
from models.experimental.tirex2_forecast.tests.pcc import metrics as M
from models.experimental.tirex2_forecast.tests.pcc.helpers import host_weights
from models.experimental.tirex2_forecast.tests.perf.test_forward_perf import DEVICE_SETTLE_S, readout_head

DEFAULT = (
    "accuracy;accuracy+hifi4=all;accuracy+y32;accuracy+hifi4=all+y32;accuracy+acc16;accuracy+hifi4=embed,head;"
    "performance;performance+hifi4=w_in;performance+hifi4=w_in+bf16=w_in;performance+bf16=ffn;"
    "performance+bf16=head+hifi4=head;performance+hifi4=w_in+bf16=w_in,ffn,head;performance+y32;performance+acc16"
)


def worst_rows(g, head, q, n=5):
    """The ``n`` worst rows by median error and by active per-row PCC, with their series kind."""
    kinds = g.get("kinds")
    names = g.get("kind_names")
    kind = (lambda r: names[int(kinds[r])]) if kinds is not None else (lambda r: "?")
    med = ((q[:, 4] - g["q_one"][:, 4]).abs().amax(-1) / g["scale"].reshape(-1)).double()
    rows = M.row_pccs(g["head_out"], head)
    act = M.active_rows(g["head_out"])
    rows_act = torch.where(act, rows, torch.full_like(rows, 2.0))
    std = g["head_out"].double().reshape(len(rows), -1).std(1)
    return {
        "by_median_err": [
            {"row": int(r), "kind": kind(r), "median_err": float(med[r]), "row_pcc": float(rows[r]), "band_std": float(std[r])}
            for r in med.argsort(descending=True)[:n]
        ],
        "by_active_row_pcc": [
            {"row": int(r), "kind": kind(r), "row_pcc": float(rows[r]), "median_err": float(med[r]), "band_std": float(std[r])}
            for r in rows_act.argsort()[:n]
        ],
    }


@pytest.mark.skipif(not have_goldens(), reason="goldens not generated")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 3 * 1024**3}], indirect=True)
def test_sweep_precision(device):
    import ttnn

    from models.experimental.tirex2_forecast.tt.common import Precision
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2
    from models.experimental.tirex2_forecast.tt.trace import TracedForward

    out_dir = Path(os.environ.get("TIREX2_SWEEP_OUT", "/tmp/tirex2_sweep"))
    (out_dir / "dumps").mkdir(parents=True, exist_ok=True)
    policies = [p for p in os.environ.get("TIREX2_SWEEP_POLICIES", DEFAULT).split(";") if p]
    g = golden("bench_paper")
    ctx, horizon = g["context"], g["horizon"]
    sub = slice(0, 64)
    results = []
    for name in policies:
        precision = Precision.by_name(name)
        tt = TtTiRex2(device, host_weights(), precision, variate=False)
        traced = TracedForward(tt)
        prep = tt.prepare_inputs(ctx, horizon)
        out = traced(prep)
        ttnn.synchronize_device(device)
        times = []
        for _ in range(3):
            traced.upload(prep)
            ttnn.synchronize_device(device)
            time.sleep(DEVICE_SETTLE_S)
            start = time.perf_counter()
            out = traced.execute(prep)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - start)
        head = readout_head(tt, out, prep)
        q = tt.postprocess(out, prep)
        traced.release()
        tt.release()
        full = M.gate_metrics(head, g["head_out"], q, g["q_one"], prep.scale)
        p64 = M.gate_metrics(head[sub], g["head_out"][sub], q[sub], g["q_one"][sub], prep.scale[sub])
        rec = {
            "policy": name,
            "device_s": statistics.median(times),
            "device_runs": times,
            "paper": full,
            "paper64": p64,
            "worst": worst_rows(g, head, q),
            "chip": os.environ.get("TTQ_CHIP_SERIAL"),
        }
        torch.save({"head": head, "q": q}, out_dir / "dumps" / f"{name}.pt")
        results.append(rec)
        print(f"\n[SWEEP] {name}: device {rec['device_s'] * 1e3:.1f} ms\n  paper:   {M.fmt(full)}\n  paper64: {M.fmt(p64)}")
        (out_dir / "precision_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    keys = ("pcc_normalized", "pcc_spread", "pcc_row_min_active", "spread_ratio_p01", "spread_ratio_p99",
            "pcc_final_spread", "median_err")
    lines = [
        "# Precision sweep, paper (1024 rows, realistic mix), traced device time back to back on one chip",
        "",
        "| policy | device ms | " + " | ".join(keys) + " | paper64 row_min_active | paper64 median_err |",
        "|---|---|" + "---|" * len(keys) + "---|---|",
    ]
    for r in results:
        cells = " | ".join(f"{r['paper'][k]:.4f}" if k != "median_err" else f"{r['paper'][k]:.3f}" for k in keys)
        lines.append(
            f"| {r['policy']} | {r['device_s'] * 1e3:.1f} | {cells} | {r['paper64']['pcc_row_min_active']:.4f} | "
            f"{r['paper64']['median_err']:.3f} |"
        )
    (out_dir / "precision_sweep.md").write_text("\n".join(lines) + "\n")
