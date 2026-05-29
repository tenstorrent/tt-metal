# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare ACE-Step 5 Hz LM Tracy ``ops_perf_results_*.csv`` reports.

Usage (from repo root, venv active)::

    python models/demos/ace_step_v1_5/perf/compare_llm_tracy_csvs.py

    python models/demos/ace_step_v1_5/perf/compare_llm_tracy_csvs.py \\
        --baseline generated/profiler/reports/2026_05_29_05_43_11/ops_perf_results_2026_05_29_05_43_11.csv \\
        --current  generated/profiler/reports/2026_05_29_09_39_49/ops_perf_results_2026_05_29_09_39_49.csv

Metrics are scoped to ops **after** ``LLM_PREFILL_DECODE_COMPILE`` (compile pass + warmup + perf
iterations). When ``LLM_PERF_PASS`` / ``LLM_WARMUP`` signposts exist, an additional perf-only
window is reported.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

LAYOUT_OPS = frozenset(
    {
        "ReshardDeviceOperation",
        "ConcatDeviceOperation",
        "InterleavedToShardedDeviceOperation",
        "ShardedToInterleavedDeviceOperation",
    }
)

P2_FOCUS_OPS = frozenset(
    LAYOUT_OPS
    | {
        "LayerNormDeviceOperation",
        "NLPConcatHeadsDecodeDeviceOperation",
        "SdpaDecodeDeviceOperation",
        "NLPCreateQKVHeadsDecodeDeviceOperation",
        "UntilizeDeviceOperation",
        "InterleavedToShardedDeviceOperation",
        "ShardedToInterleavedDeviceOperation",
    }
)


def _int(v: str | None) -> int:
    if not v or not str(v).strip():
        return 0
    try:
        return int(float(v))
    except ValueError:
        return 0


def _pct(delta: float, base: float) -> str:
    if base == 0:
        return "n/a" if delta == 0 else "+inf%"
    return f"{100.0 * delta / base:+.1f}%"


@dataclass
class OpAgg:
    calls: int = 0
    op_to_op_ns: int = 0
    kernel_ns: int = 0
    host_ns: int = 0

    def add_row(self, row: dict[str, str]) -> None:
        self.calls += 1
        self.op_to_op_ns += _int(row.get("OP TO OP LATENCY [ns]"))
        self.kernel_ns += _int(row.get("DEVICE KERNEL DURATION [ns]"))
        self.host_ns += _int(row.get("HOST DURATION [ns]"))

    @property
    def o2o_ms(self) -> float:
        return self.op_to_op_ns / 1e6

    @property
    def kernel_ms(self) -> float:
        return self.kernel_ns / 1e6


@dataclass
class BucketAgg:
    layout: OpAgg = field(default_factory=OpAgg)
    ln_32x8: OpAgg = field(default_factory=OpAgg)
    ln_32x16: OpAgg = field(default_factory=OpAgg)
    untilize_lm: OpAgg = field(default_factory=OpAgg)
    matmul: OpAgg = field(default_factory=OpAgg)
    other_dnn: OpAgg = field(default_factory=OpAgg)


@dataclass
class ReportSummary:
    label: str
    path: Path
    signposts: dict[str, int]
    compile_s: float
    post_wall_s: float
    perf_wall_s: float | None
    total_dnn: OpAgg
    buckets: BucketAgg
    by_op: dict[str, OpAgg]
    program_cache_hit_pct: float


def _bucket_row(row: dict[str, str]) -> str:
    op = row.get("OP CODE", "")
    if op in LAYOUT_OPS:
        return "layout"
    if op == "MatmulDeviceOperation":
        return "matmul"
    if op == "UntilizeDeviceOperation" and "217216" in row.get("INPUT_0_X_PAD[LOGICAL]", ""):
        return "untilize_lm"
    if op == "LayerNormDeviceOperation":
        y, x = row.get("INPUT_0_Y_PAD[LOGICAL]", ""), row.get("INPUT_0_X_PAD[LOGICAL]", "")
        if "32" in y and "8" in x:
            return "ln_32x8"
        if "32" in y and "16" in x:
            return "ln_32x16"
    return "other_dnn"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _signposts(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        if r.get("OP TYPE") == "signpost" and r.get("OP CODE") and r.get("HOST START TS"):
            out[r["OP CODE"]] = _int(r["HOST START TS"])
    return out


def _filter_window(rows: list[dict[str, str]], start_ts: int, end_ts: int | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        if r.get("OP TYPE") != "tt_dnn_device":
            continue
        ts = _int(r.get("HOST START TS"))
        if ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        out.append(r)
    return out


def summarize_report(path: Path, *, label: str | None = None) -> ReportSummary:
    rows = _load_rows(path)
    sp = _signposts(rows)
    if "LLM_PREFILL_DECODE_COMPILE" not in sp and "LLM_PREFILL_COMPILE" not in sp:
        raise ValueError(f"{path}: missing LLM_PREFILL_DECODE_COMPILE / LLM_PREFILL_COMPILE signpost")
    compile_ts = sp.get("LLM_PREFILL_DECODE_COMPILE") or sp["LLM_PREFILL_COMPILE"]
    init_ts = sp.get("LLM_INIT", compile_ts)

    perf_start = sp.get("LLM_PERF_PASS")
    perf_end = sp.get("LLM_HANDLER_PERF_PASS_END") or sp.get("LLM_PERF_PASS_END")
    if perf_start and not perf_end:
        after = [v for k, v in sp.items() if v > perf_start]
        perf_end = max(after) if after else None

    post_rows = _filter_window(rows, compile_ts, None)
    perf_rows = _filter_window(rows, perf_start, perf_end) if perf_start else []

    def build_agg(dnn_rows: list[dict[str, str]]) -> tuple[OpAgg, BucketAgg, dict[str, OpAgg], float]:
        total = OpAgg()
        buckets = BucketAgg()
        by_op: dict[str, OpAgg] = defaultdict(OpAgg)
        hits = 0
        for r in dnn_rows:
            total.add_row(r)
            bname = _bucket_row(r)
            bucket_obj = getattr(buckets, bname)
            bucket_obj.add_row(r)
            by_op[r["OP CODE"]].add_row(r)
            if str(r.get("PROGRAM CACHE HIT", "")).lower() == "true":
                hits += 1
        hit_pct = 100.0 * hits / len(dnn_rows) if dnn_rows else 0.0
        return total, buckets, dict(by_op), hit_pct

    total, buckets, by_op, hit_pct = build_agg(post_rows)

    ts = [_int(r["HOST START TS"]) for r in post_rows if r.get("HOST START TS")]
    post_wall = (max(ts) - min(ts)) / 1e9 if ts else 0.0

    perf_wall: float | None = None
    if perf_rows:
        pts = [_int(r["HOST START TS"]) for r in perf_rows if r.get("HOST START TS")]
        if pts:
            perf_wall = (max(pts) - min(pts)) / 1e9

    return ReportSummary(
        label=label or path.parent.name,
        path=path,
        signposts=sp,
        compile_s=(compile_ts - init_ts) / 1e9,
        post_wall_s=post_wall,
        perf_wall_s=perf_wall,
        total_dnn=total,
        buckets=buckets,
        by_op=by_op,
        program_cache_hit_pct=hit_pct,
    )


def _discover_today_reports(reports_root: Path) -> dict[str, Path]:
    """Return earliest and latest ``ops_perf_results`` CSV for today's date folders."""
    if not reports_root.is_dir():
        return {}
    dated = sorted(p for p in reports_root.iterdir() if p.is_dir())
    csvs = []
    for d in dated:
        matches = sorted(d.glob("ops_perf_results_*.csv"))
        if matches:
            csvs.append((d.name, matches[-1]))
    if not csvs:
        return {}
    out = {"first": csvs[0][1], "latest": csvs[-1][1]}
    if len(csvs) >= 2:
        out["p1_ref"] = csvs[-2][1]  # second-to-last as intermediate reference
    return out


def _print_report(s: ReportSummary) -> None:
    print(f"\n--- {s.label} ({s.path}) ---")
    print(f"  signposts: {', '.join(k for k in sorted(s.signposts))}")
    print(f"  compile phase:     {s.compile_s:.2f} s")
    print(f"  post-compile wall: {s.post_wall_s:.2f} s")
    if s.perf_wall_s is not None:
        print(f"  perf-pass wall:    {s.perf_wall_s:.2f} s")
    print(
        f"  dnn calls: {s.total_dnn.calls:,}  o2o: {s.total_dnn.o2o_ms:.1f} ms"
        f"  kernel: {s.total_dnn.kernel_ms:.1f} ms  cache hit: {s.program_cache_hit_pct:.1f}%"
    )
    for name in ("layout", "ln_32x8", "ln_32x16", "untilize_lm", "matmul"):
        b: OpAgg = getattr(s.buckets, name)
        if b.calls:
            print(f"    {name:14s} calls={b.calls:6,}  o2o={b.o2o_ms:8.1f} ms  kernel={b.kernel_ms:7.2f} ms")


def _print_delta(title: str, ref: ReportSummary, cur: ReportSummary) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"  reference : {ref.label}")
    print(f"  current   : {cur.label}")
    print(f"{'=' * 72}")

    d_compile = cur.compile_s - ref.compile_s
    d_wall = cur.post_wall_s - ref.post_wall_s
    d_calls = cur.total_dnn.calls - ref.total_dnn.calls
    d_o2o = cur.total_dnn.o2o_ms - ref.total_dnn.o2o_ms
    d_kernel = cur.total_dnn.kernel_ms - ref.total_dnn.kernel_ms

    print(f"  compile phase:       {d_compile:+.2f} s ({_pct(d_compile, ref.compile_s)})")
    print(f"  post-compile wall:   {d_wall:+.2f} s ({_pct(d_wall, ref.post_wall_s)})")
    print(f"  total dnn calls:     {d_calls:+,} ({_pct(d_calls, ref.total_dnn.calls)})")
    print(f"  total dnn op-to-op:  {d_o2o:+.1f} ms ({_pct(d_o2o, ref.total_dnn.o2o_ms)})")
    print(f"  total dnn kernel:    {d_kernel:+.2f} ms ({_pct(d_kernel, ref.total_dnn.kernel_ms)})")

    print("\n  Bucket deltas (calls / op-to-op ms saved):")
    for name in ("layout", "ln_32x8", "ln_32x16", "untilize_lm", "matmul"):
        r: OpAgg = getattr(ref.buckets, name)
        c: OpAgg = getattr(cur.buckets, name)
        dc = c.calls - r.calls
        do2o = c.o2o_ms - r.o2o_ms
        saved = "saved" if do2o < 0 else "added"
        print(f"    {name:14s} calls {dc:+6,}  o2o {do2o:+8.1f} ms ({_pct(do2o, r.o2o_ms)})  [{saved}]")

    print("\n  P2-focus op deltas (top |kernel| or |calls| change):")
    deltas: list[tuple[int, int, float, str, int, float, float]] = []
    all_ops = set(ref.by_op) | set(cur.by_op)
    for op in all_ops:
        if op not in P2_FOCUS_OPS:
            continue
        r, c = ref.by_op.get(op, OpAgg()), cur.by_op.get(op, OpAgg())
        deltas.append(
            (
                abs(c.calls - r.calls),
                abs(c.op_to_op_ns - r.op_to_op_ns),
                (c.o2o_ms - r.o2o_ms),
                op,
                c.calls - r.calls,
                c.o2o_ms,
                r.o2o_ms,
            )
        )
    deltas.sort(key=lambda x: (x[1], x[0]), reverse=True)
    for _, _, do2o, op, dc, co2o, ro2o in deltas[:12]:
        print(f"    {op:45s} calls {dc:+5d}  o2o {do2o:+7.2f} ms  (cur {co2o:.2f} vs ref {ro2o:.2f})")


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[4]
    reports_root = repo_root / "generated" / "profiler" / "reports"

    parser = argparse.ArgumentParser(description="Compare ACE 5 Hz LM Tracy CSV reports.")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline CSV (default: earliest ops_perf_results under reports/).",
    )
    parser.add_argument(
        "--current",
        type=Path,
        help="Current CSV (default: latest ops_perf_results under reports/).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional middle reference CSV (e.g. P1 run) for incremental delta.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=reports_root,
        help=f"Profiler reports root (default: {reports_root}).",
    )
    args = parser.parse_args(argv)

    discovered = _discover_today_reports(args.reports_dir)
    baseline_path = args.baseline or discovered.get("first")
    current_path = args.current or discovered.get("latest")
    reference_path = args.reference or discovered.get("p1_ref")

    if baseline_path is None or current_path is None:
        print("Could not find ops_perf_results CSVs. Run Tracy first or pass --baseline/--current.", file=sys.stderr)
        return 1
    if not baseline_path.is_file() or not current_path.is_file():
        print("Baseline or current CSV path does not exist.", file=sys.stderr)
        return 1

    baseline = summarize_report(baseline_path.resolve(), label=f"baseline ({baseline_path.parent.name})")
    current = summarize_report(current_path.resolve(), label=f"current ({current_path.parent.name})")

    print("ACE-Step 5 Hz LM Tracy CSV comparison")
    print(f"reports dir: {args.reports_dir.resolve()}")
    _print_report(baseline)
    _print_report(current)
    _print_delta("TIME / OPS GAIN: current vs baseline (first → latest)", baseline, current)

    if reference_path and reference_path.resolve() != baseline_path.resolve() and reference_path.is_file():
        reference = summarize_report(reference_path.resolve(), label=f"reference ({reference_path.parent.name})")
        _print_report(reference)
        _print_delta("INCREMENTAL: current vs reference (P1 → P2/P3)", reference, current)

    approx_total_ref = baseline.compile_s + baseline.post_wall_s
    approx_total_cur = current.compile_s + current.post_wall_s
    print(f"\nApprox end-to-end (compile + post-compile wall):")
    print(f"  baseline: {approx_total_ref:.2f} s")
    print(f"  current:  {approx_total_cur:.2f} s")
    print(
        f"  gain:     {approx_total_cur - approx_total_ref:+.2f} s ({_pct(approx_total_cur - approx_total_ref, approx_total_ref)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
