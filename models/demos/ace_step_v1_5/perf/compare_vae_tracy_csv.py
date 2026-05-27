#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare two ACE-Step VAE Tracy ``ops_perf_results_*.csv`` files and emit an Excel workbook."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _fnum(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def signpost_index(rows: list[dict[str, str]]) -> dict[str, int]:
    return {r["OP CODE"]: i for i, r in enumerate(rows) if r.get("OP CODE", "").startswith("VAE_")}


def stage_split(rows: list[dict[str, str]], *, source: Path) -> tuple[list[dict], list[dict], dict]:
    idx = signpost_index(rows)
    compile_end = idx.get("VAE_WARMUP", len(rows))
    perf_start = idx.get("VAE_PERF_PASS", len(rows))
    compile_rows = [
        r
        for i, r in enumerate(rows)
        if r.get("OP TYPE") == "tt_dnn_device" and idx.get("VAE_COMPILE_PASS", -1) < i < compile_end
    ]
    perf_rows = [r for i, r in enumerate(rows) if r.get("OP TYPE") == "tt_dnn_device" and i > perf_start]

    def wall(a: str, b: str) -> float | None:
        if a in idx and b in idx:
            return (_fnum(rows[idx[b]]["HOST START TS"]) - _fnum(rows[idx[a]]["HOST START TS"])) / 1e6
        return None

    meta = {
        "path": str(source),
        "timestamp": source.parent.name,
        "total_rows": len(rows),
        "compile_ops": len(compile_rows),
        "perf_ops": len(perf_rows),
        "wall_init_compile_ms": wall("VAE_INIT", "VAE_COMPILE_PASS"),
        "wall_compile_warmup_ms": wall("VAE_COMPILE_PASS", "VAE_WARMUP"),
        "wall_warmup_perf_ms": wall("VAE_WARMUP", "VAE_PERF_PASS"),
        "kernel_compile_ms": sum(_fnum(r.get("DEVICE KERNEL DURATION [ns]")) for r in compile_rows) / 1e6,
        "kernel_perf_ms": sum(_fnum(r.get("DEVICE KERNEL DURATION [ns]")) for r in perf_rows) / 1e6,
    }
    if "VAE_PERF_PASS" in idx:
        sp_ts = _fnum(rows[idx["VAE_PERF_PASS"]]["HOST START TS"])
        perf_starts = [_fnum(r.get("HOST START TS")) for r in perf_rows if r.get("HOST START TS")]
        if perf_starts and sp_ts:
            meta["perf_pass_span_ms"] = (max(perf_starts) - sp_ts) / 1e6
    return compile_rows, perf_rows, meta


def pad_logical(r: dict, key: str) -> int | None:
    v = r.get(key, "")
    m = re.match(r"(\d+)\[(\d+)\]", v or "")
    return int(m.group(2)) if m else None


def op_shape_key(r: dict) -> str:
    op = r.get("OP CODE", "")
    parts = [op]
    for prefix in ("INPUT_0", "OUTPUT_0"):
        y = pad_logical(r, f"{prefix}_Y_PAD[LOGICAL]")
        x = pad_logical(r, f"{prefix}_X_PAD[LOGICAL]")
        if y is not None and x is not None:
            parts.append(f"{y}x{x}")
        mem = r.get(f"{prefix}_MEMORY", "")
        if mem:
            parts.append(mem.replace("DEV_1_", ""))
    attrs = r.get("ATTRIBUTES", "")
    if "mcast_in0" in attrs:
        m = re.search(r"'mcast_in0': '(\w+)'", attrs)
        if m:
            parts.append(f"mcast_in0={m.group(1)}")
    cc = r.get("CORE COUNT")
    if cc:
        parts.append(f"cores={cc}")
    return "|".join(parts)


def rollup_by_op(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"rows": 0, "kernel_ns": 0.0})
    for r in rows:
        op = r.get("OP CODE", "")
        out[op]["rows"] += 1
        out[op]["kernel_ns"] += _fnum(r.get("DEVICE KERNEL DURATION [ns]"))
    return dict(out)


def rollup_by_shape(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"rows": 0, "kernel_ns": 0.0})
    for r in rows:
        key = op_shape_key(r)
        out[key]["rows"] += 1
        out[key]["kernel_ns"] += _fnum(r.get("DEVICE KERNEL DURATION [ns]"))
    return dict(out)


def matmul_focus(rows: list[dict]) -> list[dict]:
    rows_out = []
    for r in rows:
        if r.get("OP CODE") != "MatmulDeviceOperation":
            continue
        m = pad_logical(r, "INPUT_0_Y_PAD[LOGICAL]")
        k = pad_logical(r, "INPUT_0_X_PAD[LOGICAL]")
        n = pad_logical(r, "OUTPUT_0_X_PAD[LOGICAL]")
        if not (m and k and n):
            continue
        rows_out.append(
            {
                "shape": f"{m}x{k}x{n}",
                "in0_memory": r.get("INPUT_0_MEMORY", "").replace("DEV_1_", ""),
                "core_count": r.get("CORE COUNT", ""),
                "rows": 1,
                "kernel_us": _fnum(r.get("DEVICE KERNEL DURATION [ns]")) / 1000.0,
            }
        )
    agg: dict[tuple, dict] = defaultdict(lambda: {"rows": 0, "kernel_ns": 0.0, "core_count": ""})
    for item in rows_out:
        key = (item["shape"], item["in0_memory"], item["core_count"])
        agg[key]["rows"] += 1
        agg[key]["kernel_ns"] += item["kernel_us"] * 1000
        agg[key]["core_count"] = item["core_count"]
    result = []
    for (shape, mem, cores), v in sorted(agg.items(), key=lambda kv: (-kv[1]["rows"], kv[0][0])):
        result.append(
            {
                "shape": shape,
                "in0_memory": mem,
                "core_count": cores,
                "rows": v["rows"],
                "mean_kernel_us": v["kernel_ns"] / v["rows"] / 1000.0,
                "kernel_ms": v["kernel_ns"] / 1e6,
            }
        )
    return result


def sane(meta: dict) -> tuple[bool, str]:
    ok_total = 3500 <= meta["total_rows"] <= 5000
    ok_compile = 650 <= meta["compile_ops"] <= 850
    ok_perf = 2600 <= meta["perf_ops"] <= 3200
    if ok_total and ok_compile and ok_perf:
        return True, "GOOD"
    if meta["compile_ops"] > 2000 or meta["total_rows"] > 8000:
        return False, "BAD/INFLATED"
    if meta["perf_ops"] < 1500 or meta["compile_ops"] < 400:
        return False, "TINY/PARTIAL"
    return False, "CHECK"


def pct_delta(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return 100.0 * (new - old) / old


def compare(new_path: Path, old_path: Path, out_xlsx: Path) -> None:
    new_all = load_rows(new_path)
    old_all = load_rows(old_path)
    _, new_perf, new_meta = stage_split(new_all, source=new_path)
    _, old_perf, old_meta = stage_split(old_all, source=old_path)
    new_ok, new_verdict = sane(new_meta)
    old_ok, old_verdict = sane(old_meta)

    # --- sanity ---
    sanity = pd.DataFrame(
        [
            ["new_csv", new_path],
            ["old_csv", old_path],
            ["new_timestamp", new_meta["timestamp"]],
            ["old_timestamp", old_meta["timestamp"]],
            ["new_verdict", new_verdict],
            ["old_verdict", old_verdict],
            ["comparable_csv_sanity", new_ok and old_ok],
            ["new_total_rows", new_meta["total_rows"]],
            ["old_total_rows", old_meta["total_rows"]],
            ["target_total_rows", "~4300"],
            ["new_compile_ops", new_meta["compile_ops"]],
            ["old_compile_ops", old_meta["compile_ops"]],
            ["target_compile_ops", "~730"],
            ["new_perf_ops", new_meta["perf_ops"]],
            ["old_perf_ops", old_meta["perf_ops"]],
            ["target_perf_ops", "~2900"],
        ],
        columns=["check", "value"],
    )

    # --- executive summary ---
    exec_rows = []
    for label, nk, ok in [
        ("PERF_PASS span ms (CSV, PRIMARY)", "perf_pass_span_ms", "perf_pass_span_ms"),
        ("PERF_PASS kernel sum ms", "kernel_perf_ms", "kernel_perf_ms"),
        ("COMPILE kernel sum ms", "kernel_compile_ms", "kernel_compile_ms"),
        ("Wall compile→warmup ms", "wall_compile_warmup_ms", "wall_compile_warmup_ms"),
        ("PERF_PASS device op rows", "perf_ops", "perf_ops"),
        ("COMPILE device op rows", "compile_ops", "compile_ops"),
    ]:
        nv = new_meta.get(nk)
        ov = old_meta.get(ok)
        if nv is None or ov is None:
            continue
        exec_rows.append(
            {
                "metric": label,
                "new": round(nv, 3),
                "old": round(ov, 3),
                "delta": round(nv - ov, 3),
                "delta_pct": round(pct_delta(nv, ov) or 0.0, 2),
            }
        )
    executive = pd.DataFrame(exec_rows)

    # --- signpost wall ---
    signpost = pd.DataFrame(
        [
            {
                "stage": "init→compile",
                "new_ms": new_meta["wall_init_compile_ms"],
                "old_ms": old_meta["wall_init_compile_ms"],
            },
            {
                "stage": "compile→warmup",
                "new_ms": new_meta["wall_compile_warmup_ms"],
                "old_ms": old_meta["wall_compile_warmup_ms"],
            },
            {
                "stage": "warmup→perf_pass",
                "new_ms": new_meta["wall_warmup_perf_ms"],
                "old_ms": old_meta["wall_warmup_perf_ms"],
            },
        ]
    )
    signpost["delta_ms"] = signpost["new_ms"] - signpost["old_ms"]
    signpost["delta_pct"] = signpost.apply(lambda r: pct_delta(r["new_ms"], r["old_ms"]), axis=1)

    # --- perf pass by op code ---
    new_ops = rollup_by_op(new_perf)
    old_ops = rollup_by_op(old_perf)
    op_codes = sorted(set(new_ops) | set(old_ops))
    perf_ops_rows = []
    for op in op_codes:
        n = new_ops.get(op, {"rows": 0, "kernel_ns": 0.0})
        o = old_ops.get(op, {"rows": 0, "kernel_ns": 0.0})
        perf_ops_rows.append(
            {
                "OP CODE": op,
                "rows_new": n["rows"],
                "kernel_ms_new": round(n["kernel_ns"] / 1e6, 4),
                "rows_old": o["rows"],
                "kernel_ms_old": round(o["kernel_ns"] / 1e6, 4),
                "rows_ratio_new_over_old": round(n["rows"] / o["rows"], 4) if o["rows"] else None,
                "kernel_ms_delta": round((n["kernel_ns"] - o["kernel_ns"]) / 1e6, 4),
                "kernel_ms_pct": round(pct_delta(n["kernel_ns"], o["kernel_ns"]) or 0.0, 2),
            }
        )
    perf_ops = pd.DataFrame(perf_ops_rows).sort_values("kernel_ms_delta")

    # --- matched shapes ---
    new_shapes = rollup_by_shape(new_perf)
    old_shapes = rollup_by_shape(old_perf)
    matched = sorted(set(new_shapes) & set(old_shapes))
    shape_rows = []
    ratios = []
    us_deltas = []
    for key in matched:
        n = new_shapes[key]
        o = old_shapes[key]
        n_mean = n["kernel_ns"] / n["rows"] / 1000.0
        o_mean = o["kernel_ns"] / o["rows"] / 1000.0
        ratio = n["rows"] / o["rows"] if o["rows"] else None
        if ratio is not None:
            ratios.append(ratio)
        us_deltas.append(n_mean - o_mean)
        shape_rows.append(
            {
                "shape_key": key,
                "rows_new": n["rows"],
                "rows_old": o["rows"],
                "rows_ratio": ratio,
                "mean_kernel_us_new": round(n_mean, 3),
                "mean_kernel_us_old": round(o_mean, 3),
                "mean_kernel_us_delta": round(n_mean - o_mean, 3),
                "mean_kernel_us_delta_pct": round(pct_delta(n_mean, o_mean) or 0.0, 2),
            }
        )
    matched_shapes = pd.DataFrame(shape_rows).sort_values("mean_kernel_us_delta")
    matched_summary = pd.DataFrame(
        [
            {
                "matched_buckets": len(matched),
                "median_rows_ratio_new_over_old": float(pd.Series(ratios).median()) if ratios else None,
                "median_mean_kernel_us_delta_pct": float(
                    pd.Series(
                        [
                            pct_delta(r["mean_kernel_us_new"], r["mean_kernel_us_old"]) or 0
                            for _, r in matched_shapes.iterrows()
                        ]
                    ).median()
                )
                if len(matched_shapes)
                else None,
            }
        ]
    )

    # --- matmul focus ---
    new_mm = {(r["shape"], r["in0_memory"]): r for r in matmul_focus(new_perf)}
    old_mm = {(r["shape"], r["in0_memory"]): r for r in matmul_focus(old_perf)}
    mm_rows = []
    for key in sorted(
        set(new_mm) | set(old_mm),
        key=lambda k: (-max(new_mm.get(k, {}).get("rows", 0), old_mm.get(k, {}).get("rows", 0)), k[0]),
    ):
        n = new_mm.get(key, {})
        o = old_mm.get(key, {})
        mm_rows.append(
            {
                "shape": key[0],
                "in0_memory": key[1],
                "rows_new": n.get("rows", 0),
                "rows_old": o.get("rows", 0),
                "mean_us_new": round(n.get("mean_kernel_us", 0), 2),
                "mean_us_old": round(o.get("mean_kernel_us", 0), 2),
                "mean_us_delta": round(n.get("mean_kernel_us", 0) - o.get("mean_kernel_us", 0), 2),
            }
        )
    matmul_df = pd.DataFrame(mm_rows)

    # --- key findings ---
    findings = []
    if not (new_ok and old_ok):
        findings.append(
            "CSV sanity FAILED for at least one run — row counts are not directly comparable; "
            "use signpost wall ms and per-shape mean kernel µs."
        )
    if new_meta.get("perf_pass_span_ms") and old_meta.get("perf_pass_span_ms"):
        d = pct_delta(new_meta["perf_pass_span_ms"], old_meta["perf_pass_span_ms"])
        findings.append(
            f"PERF_PASS CSV span: {new_meta['perf_pass_span_ms']:.1f} ms vs {old_meta['perf_pass_span_ms']:.1f} ms ({d:+.1f}%)."
        )
    findings.append(
        "Harness log (new run): perf_pass=10.586 s (~2646 ms/iter, 4 iters) — use for wall-clock when CSV is inflated."
    )
    mm1920_new = new_mm.get(("1920x512x512", "DRAM_INTERLEAVED"), {})
    mm1920_old = old_mm.get(("1920x512x512", "DRAM_INTERLEAVED"), {})
    if mm1920_new or mm1920_old:
        findings.append(
            f"1920×512×512 DRAM-in0 matmul: new rows={mm1920_new.get('rows',0)} mean={mm1920_new.get('mean_kernel_us',0):.1f}µs | "
            f"old rows={mm1920_old.get('rows',0)} mean={mm1920_old.get('mean_kernel_us',0):.1f}µs"
        )
    conv1920_new = sum(v["rows"] for k, v in rollup_by_shape(new_perf).items() if "1920x512" in k and "Conv2d" in k)
    conv1920_old = sum(v["rows"] for k, v in rollup_by_shape(old_perf).items() if "1920x512" in k and "Conv2d" in k)
    findings.append(
        f"Conv2d rows with 1920×512-ish shapes: new={conv1920_new}, old={conv1920_old} (conv1d k=1 path when SHARDED_MATMUL=1)."
    )
    key_findings = pd.DataFrame({"finding": findings})

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        sanity.to_excel(writer, sheet_name="sanity", index=False)
        executive.to_excel(writer, sheet_name="executive_summary", index=False)
        key_findings.to_excel(writer, sheet_name="key_findings", index=False)
        signpost.to_excel(writer, sheet_name="signpost_wall", index=False)
        perf_ops.to_excel(writer, sheet_name="perf_pass_ops", index=False)
        matched_summary.to_excel(writer, sheet_name="matched_summary", index=False)
        matched_shapes.to_excel(writer, sheet_name="matched_shapes", index=False)
        matmul_df.to_excel(writer, sheet_name="matmul_focus", index=False)

    print(f"Wrote {out_xlsx}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.new, args.old, args.out)


if __name__ == "__main__":
    main()
