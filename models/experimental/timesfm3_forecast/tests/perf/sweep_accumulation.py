# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Precision sweep (D1 items 1, 4, 5): fp32 dest accumulation and fidelity per role, and the performance-policy
variants, with every gate metric, the per-row spread ratio, named rows and device time. One chip, run under ttq:

    ttq run --model timesfm3 --part 0 --kind bench -- python \
        models/experimental/timesfm3_forecast/tests/perf/sweep_accumulation.py <out_dir> [--only accuracy|performance]

Writes ``<out_dir>/accumulation_sweep.json`` and ``.md``. Device time: median of 3 ``forward_device`` calls,
each after a 0.3 s idle, on the bench shapes. Layer 19 (stack output) PCC on the multi-series goldens (X1).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import torch

from models.experimental.timesfm3_forecast.tests.golden_io import (
    active_rows,
    final_metrics,
    gate_failures,
    head_metrics,
    load_golden,
    pcc,
)

ACC = {
    "A0 hifi2, no fp32 acc (round 0)": dict(),
    "A1 hifi2, fp32 acc (matmul + sdpa)": dict(matmul_fp32_acc=True, sdpa_fp32_acc=True),
    "A2 hifi4, fp32 acc (matmul + sdpa)": dict(
        matmul_fidelity="hifi4", matmul_fp32_acc=True, sdpa_fidelity="hifi4", sdpa_fp32_acc=True
    ),
    "A3 hifi4, no fp32 acc": dict(matmul_fidelity="hifi4", sdpa_fidelity="hifi4"),
    "A4 hifi2, fp32 acc matmul only": dict(matmul_fp32_acc=True),
    "A5 hifi2, fp32 acc sdpa only": dict(sdpa_fp32_acc=True),
}
PERF = {
    "P0 bf8 + lofi (round 0)": dict(),
    "P1 bf16 + hifi2": dict(bf8_weights=False, matmul_fidelity="hifi2", sdpa_fidelity="hifi2"),
    "P2 bf8 except ffn + lofi": dict(bf16_roles=("ffn",)),
    "P3 bf16 + lofi": dict(bf8_weights=False),
    "P4 bf8 + hifi2": dict(matmul_fidelity="hifi2", sdpa_fidelity="hifi2"),
    "P0a bf8 + lofi, fp32 acc": dict(matmul_fp32_acc=True, sdpa_fp32_acc=True),
    "P2a bf8 except ffn + lofi, fp32 acc": dict(bf16_roles=("ffn",), matmul_fp32_acc=True, sdpa_fp32_acc=True),
    "P3a bf16 + lofi, fp32 acc": dict(bf8_weights=False, matmul_fp32_acc=True, sdpa_fp32_acc=True),
    "P4a bf8 + hifi2, fp32 acc": dict(
        matmul_fidelity="hifi2", sdpa_fidelity="hifi2", matmul_fp32_acc=True, sdpa_fp32_acc=True
    ),
}
# D3 condition 1 (R1-0): the performance-policy candidates, all with fp32 accumulation.
FP32 = dict(matmul_fp32_acc=True, sdpa_fp32_acc=True)
PERF_R10 = {
    "P2a bf8 except ffn + lofi, fp32 acc (interim)": dict(bf16_roles=("ffn",), **FP32),
    "P5a bf8 except ffn + hifi2, fp32 acc": dict(
        bf16_roles=("ffn",), matmul_fidelity="hifi2", sdpa_fidelity="hifi2", **FP32
    ),
    "P5b bf8 except ffn, lofi on bf8, hifi2 on ffn + sdpa": dict(
        bf16_roles=("ffn",), matmul_fidelity="lofi", bf16_role_fidelity="hifi2", sdpa_fidelity="hifi2", **FP32
    ),
    "P5c bf8 except ffn, lofi on bf8 + sdpa, hifi2 on ffn": dict(
        bf16_roles=("ffn",), matmul_fidelity="lofi", bf16_role_fidelity="hifi2", sdpa_fidelity="lofi", **FP32
    ),
    "P5d bf8 except ffn + head, lofi on bf8, hifi2 on ffn + head + sdpa": dict(
        bf16_roles=("ffn", "head"), matmul_fidelity="lofi", bf16_role_fidelity="hifi2", sdpa_fidelity="hifi2", **FP32
    ),
}
_ALL = ("tokenizer", "seq_qkv", "seq_wo", "var_qkv", "var_wo", "w_ov", "ffn", "head")
_but = lambda *bf8: tuple(r for r in _ALL if r not in bf8)  # noqa: E731
PERF_R10B = {
    "P6a bf16, lofi except ffn (hifi2), sdpa hifi2": dict(
        bf8_weights=False, bf16_roles=("ffn",), matmul_fidelity="lofi", bf16_role_fidelity="hifi2",
        sdpa_fidelity="hifi2", **FP32
    ),
    "P6b bf8 qkv only, hifi2": dict(bf16_roles=_but("seq_qkv", "var_qkv"), matmul_fidelity="hifi2", sdpa_fidelity="hifi2", **FP32),
    "P6c bf8 qkv + wo, hifi2": dict(
        bf16_roles=_but("seq_qkv", "var_qkv", "seq_wo", "var_wo", "w_ov"), matmul_fidelity="hifi2",
        sdpa_fidelity="hifi2", **FP32
    ),
    "P6d bf16, hifi2, sdpa lofi": dict(bf8_weights=False, matmul_fidelity="hifi2", sdpa_fidelity="lofi", **FP32),
}
BENCH = {"paper": "bench_paper", "long_context": "bench_long_context", "multivariate": "bench_multivariate"}
SUITE = ["full16_2048_h64", "full16_2048_h128", "mv4x8_2048_h64", "gift64_h48"]
NAMED_ROWS = [458, 483, 806, 226]  # paper rows named in D1


def contexts(g):
    c = g["contexts"]
    return [t.numpy() for t in c] if isinstance(c, list) else list(c.numpy())


def row_stats(ref, got, rows):
    e = ref.double().reshape(ref.shape[0], -1, 9)
    d = got.double().reshape(got.shape[0], -1, 9)
    es, ds = e - e.mean(-1, keepdim=True), d - d.mean(-1, keepdim=True)
    out = {}
    for r in rows:
        er, dr = e[r].flatten(), d[r].flatten()
        out[str(r)] = {
            "spread_ratio": float(ds[r].norm() / es[r].norm()),
            "pcc": pcc(er, dr),
            "err_std": float((dr - er).std()),
        }
    return out


def run_case(tt, name, *, timed: bool, layers: bool, named=False):
    import ttnn

    g = load_golden(name)
    hi = tt.prepare_inputs(contexts(g), g["horizon"], **g.get("settings", {}))
    inputs = tt.upload_inputs(hi)
    device_ms = None
    if timed:
        ts = []
        for _ in range(3):
            time.sleep(0.3)
            t0 = time.perf_counter()
            o = tt.forward_device(inputs)
            ttnn.synchronize_device(tt.device)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(o)
        device_ms = 1e3 * statistics.median(ts)
    capture = {} if layers else None
    o = tt.forward_device(inputs, capture)
    head, q = tt.postprocess(o, hi)
    ttnn.deallocate(o)
    tt.deallocate_inputs(inputs)
    ref_q = g["quantiles"]
    ref_q = ref_q[None] if ref_q.dim() == 2 else ref_q
    m = {
        **head_metrics(g["head_readout"], head, active_rows(g["head_readout"])),
        **final_metrics(ref_q, torch.from_numpy(q)),
    }
    rec = {"metrics": m, "device_ms": device_ms}
    if layers:
        valid = hi.token_valid[..., None].expand(-1, -1, 1280)
        ref19 = g["layers"]["layer19"].reshape(hi.rows, hi.num_patches, -1)
        rec["layer19_pcc"] = pcc(ref19[valid], capture["layer19"][valid])
    if named:
        rec["named_rows"] = row_stats(g["head_readout"], head, NAMED_ROWS)
    return rec


def main():
    import ttnn

    from models.experimental.timesfm3_forecast.tt.common import TtPrecision
    from models.experimental.timesfm3_forecast.tt.model import TtTimesFM3, load_checkpoint_weights

    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--only", choices=["accuracy", "performance", "r1-0", "r1-0b"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "accumulation_sweep.json"
    results = json.loads(out_json.read_text()) if out_json.is_file() else {}
    device = ttnn.open_device(device_id=0)
    try:
        weights = load_checkpoint_weights()
        plans = []
        if args.only in (None, "accuracy"):
            plans += [("accuracy", k, v) for k, v in ACC.items()]
        if args.only in (None, "performance"):
            plans += [("performance", k, v) for k, v in PERF.items()]
        if args.only == "r1-0":
            plans += [("performance", k, v) for k, v in PERF_R10.items()]
        if args.only == "r1-0b":
            plans += [("performance", k, v) for k, v in PERF_R10B.items()]
        for base, label, overrides in plans:
            precision = dataclasses.replace(TtPrecision.by_name(base), **overrides)
            tt = TtTimesFM3(device, weights, precision)
            entry = {"base": base, "policy": precision.describe(), "cases": {}}
            shapes = (
                BENCH.items()
                if base == "accuracy"
                else [("paper", "bench_paper"), ("multivariate", "bench_multivariate")]
            )
            for shape, name in shapes:
                entry["cases"][shape] = run_case(tt, name, timed=True, layers=False, named=shape == "paper")
            if base == "accuracy":
                for name in SUITE:
                    entry["cases"][name] = run_case(tt, name, timed=False, layers=name != "gift64_h48")
            for case in entry["cases"].values():
                case["gate_failures"] = {k: v[0] for k, v in gate_failures(case["metrics"], base).items()}
            results[label] = entry
            print(
                f"[SWEEP] {label}: "
                + json.dumps(
                    {
                        k: (c["device_ms"], c["metrics"]["spread_ratio_min"], c["metrics"]["pcc_row_min_active"])
                        for k, c in entry["cases"].items()
                    }
                ),
                flush=True,
            )
            del tt
            out_json.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        ttnn.close_device(device)
    write_md(results, args.out_dir / "accumulation_sweep.md")


def write_md(results, path: Path):
    cols = [
        "pcc_normalized",
        "pcc_spread",
        "pcc_row_p01_active",
        "pcc_row_min_active",
        "spread_ratio_min",
        "spread_ratio_p01",
        "spread_ratio_p99",
        "spread_ratio_max",
        "pcc_final_spread",
    ]
    lines = [
        "# Precision sweep (fp32 accumulation, fidelity, performance policy)",
        "",
        "Device ms: median of 3 forward_device calls after a 0.3 s idle. Metrics: `golden_io.head_metrics` / "
        "`final_metrics`; `spread_ratio` per active row of the golden.",
        "",
    ]
    for label, e in results.items():
        lines += [
            f"## {label}",
            "",
            f"policy: `{json.dumps(e['policy'])}`",
            "",
            "| case | device ms | " + " | ".join(cols) + " | layer19 | gates |",
            "|" + "---|" * (len(cols) + 4),
        ]
        for case, c in e["cases"].items():
            m = c["metrics"]
            dev = "" if c["device_ms"] is None else f"{c['device_ms']:.1f}"
            l19 = "" if "layer19_pcc" not in c else f"{c['layer19_pcc']:.5f}"
            gates = "ok" if not c["gate_failures"] else ", ".join(c["gate_failures"])
            lines.append(f"| {case} | {dev} | " + " | ".join(f"{m[k]:.5f}" for k in cols) + f" | {l19} | {gates} |")
        if "named_rows" in e["cases"].get("paper", {}):
            nr = e["cases"]["paper"]["named_rows"]
            lines += [
                "",
                "paper named rows (spread ratio / pcc / err std): "
                + "; ".join(
                    f"{r}: {v['spread_ratio']:.3f} / {v['pcc']:.4f} / {v['err_std']:.4f}" for r, v in nr.items()
                ),
            ]
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
