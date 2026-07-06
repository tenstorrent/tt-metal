#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Driver for test_tt_moe_decode.py::test_tt_moe_decode_perf.

For each selected (config x mesh x fabric) case it runs the perf test ONCE under tracy in its own
profiled session, finds the ops_perf_results CSV that run produced, and summarizes the per-op device
times for the ops that make up `TTMoEDecode.forward`.

Why one tracy run per case: the ops-perf CSV is filtered only by op code / device columns, and every
case shares the same op codes. Mixing configs / meshes / fabrics into one CSV would blend the rows,
so each combination is profiled separately -> its own CSV.

How the timings are extracted (see `analyze_csv`): under metal trace tracy writes TWO rows per
captured op -- a trace-CAPTURE record (METAL TRACE ID empty, DEVICE KERNEL DURATION blank) and a
REPLAY record (METAL TRACE ID set, real device time). We keep only rows with a real duration (drops
the capture rows), keep the replay rows (drops the eager compile-run rows, which have no trace id),
then drop the first replay iteration per device (the untimed warmup `execute_trace` the harness runs
before the signposted loop) and AVERAGE the remaining timed iterations. Every op left is one that ran
inside `forward` (the trace contains nothing else).

Usage:
    python run_moe_decode_perf.py <config_id|all> [filters]

    # all (non-skipped) configs on the default 8x4 / fabric_1D_ring:
    python run_moe_decode_perf.py all
    # a single config, seeing the plan only:
    python run_moe_decode_perf.py deepseek_v3 --dry-run
    # sweep two meshes for one config:
    python run_moe_decode_perf.py glm5 --mesh 8x4,16x4
    # just re-analyze an existing CSV (no device run):
    python run_moe_decode_perf.py --analyze generated/profiler/reports/.../ops_perf_results_*.csv

Filters take comma-separated subsets of the pytest ids; omit a filter for its default:
    --mesh    {16x4,16x1,8x4,8x1}   (default: 8x4)
    --fabric  {fabric_1D_ring,fabric_1D}  (default: fabric_1D_ring)
"""
import argparse
import csv
import shlex
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TEST_FILE = THIS_DIR / "test_tt_moe_decode.py"
# THIS_DIR = models/common/tests/modules/moe → parents[2] = models/common
CONFIGS_DIR = THIS_DIR.parents[2] / "modules" / "moe" / "configs"
TEST_NODE = "test_tt_moe_decode_perf"

# Mirrors SKIP_LIST in test_tt_moe_decode.py — these hard-fail/crash, so skip by default.
SKIP_CONFIGS = {"ling_1t", "mistral_large_3", "deepseek_v4_pro"}

MESH_IDS = ["16x4", "16x1", "8x4", "8x1"]
FABRIC_IDS = ["fabric_1D_ring", "fabric_1D"]

DUR_COL = "DEVICE KERNEL DURATION [ns]"
TRACE_COL = "METAL TRACE ID"
OPCODE_COL = "OP CODE"
OPTYPE_COL = "OP TYPE"
DEVICE_COL = "DEVICE ID"


def find_repo_root(start: Path) -> Path:
    p = start
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return start


def config_ids() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))


def k_expr(config: str, all_configs: list[str], mesh: str, fabric: str) -> str:
    """pytest -k expression pinning exactly one case.

    -k does substring-AND on the node id (order-independent), so we guard against ids that are
    substrings of other ids: a config stem that prefixes another stem (deepseek_v3 vs
    deepseek_v3_single_glx) and fabric_1D vs fabric_1D_ring. `TEST_NODE` also disambiguates the perf
    test from `test_tt_moe_decode` (the perf id is not a substring of the correctness id).
    """
    parts = [TEST_NODE, config, mesh]
    parts += [f"not {other}" for other in all_configs if other != config and config in other]
    if fabric == "fabric_1D":
        parts += ["fabric_1D", "not fabric_1D_ring"]
    else:
        parts.append(fabric)
    return " and ".join(parts)


def find_new_csv(reports_dir: Path, slug: str, since: float):
    """Newest ops_perf_results CSV touched since `since`, preferring ones whose path carries the slug."""
    cands = [p for p in reports_dir.glob("**/ops_perf_results_*.csv") if p.stat().st_mtime >= since - 1.0]
    cands.sort(key=lambda p: p.stat().st_mtime)
    slug_hits = [p for p in cands if slug in p.name or slug in str(p.parent)]
    chosen = slug_hits or cands
    return chosen[-1] if chosen else None


def analyze_csv(path: str, warmup_iters: int = 1):
    """Return (per_op, total_avg_us) for the forward ops in an ops_perf CSV.

    per_op maps OP CODE -> dict(iters, avg_us, min_us, max_us). `avg_us` is the crit-path mean: for
    each timed iteration take the max device-kernel time across devices (the slowest device gates the
    step), then average those across iterations. total_avg_us is the sum over ops ≈ the per-forward
    device time.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    def dur_ns(r):
        return r.get(DUR_COL, "").strip()

    # tt_dnn_device rows only (drops signpost / host markers) when the column is present.
    if rows and OPTYPE_COL in rows[0]:
        rows = [r for r in rows if r.get(OPTYPE_COL, "").strip() == "tt_dnn_device"]

    real = [r for r in rows if dur_ns(r) not in ("", "-")]  # drops blank trace-CAPTURE rows
    replay = [r for r in real if r.get(TRACE_COL, "").strip() not in ("", "-")]  # trace REPLAY rows only
    use = replay if replay else real  # trace run -> replay rows; untraced -> all real rows
    traced = bool(replay)

    # op -> device -> ordered list of per-invocation durations (us)
    per = defaultdict(lambda: defaultdict(list))
    for r in use:
        per[r[OPCODE_COL]][int(r[DEVICE_COL])].append(float(dur_ns(r)) / 1000.0)

    per_op = {}
    for op, bydev in per.items():
        # Drop the untimed warmup execute(s): the first `warmup_iters` replay iterations per device.
        # (Trace-capture rows were already removed above by the non-blank duration filter.)
        if traced and warmup_iters:
            bydev = {d: v[warmup_iters:] for d, v in bydev.items()}
        bydev = {d: v for d, v in bydev.items() if v}
        if not bydev:
            continue
        n = min(len(v) for v in bydev.values())
        aligned = {d: v[-n:] for d, v in bydev.items()}  # tail-align across devices
        crit = [max(aligned[d][i] for d in aligned) for i in range(n)]
        per_op[op] = {
            "iters": n,
            "avg_us": statistics.mean(crit),
            "min_us": min(crit),
            "max_us": max(crit),
        }

    total_avg_us = sum(v["avg_us"] for v in per_op.values())
    return per_op, total_avg_us, traced


def print_report(path: str, per_op: dict, total_avg_us: float, traced: bool, title: str = ""):
    mode = "TRACE (replay rows, warmup dropped)" if traced else "UNTRACED (all real rows)"
    lines = [
        title or f"forward per-op device times — {path}",
        f"mode: {mode}   ops: {len(per_op)}   forward avg (sum of op crit-path means): {total_avg_us:.2f} us",
        f"{'OP CODE':<48}{'iters':>6}{'avg_us':>11}{'min_us':>10}{'max_us':>10}{'pct':>8}",
        "-" * 91,
    ]
    for op, r in sorted(per_op.items(), key=lambda kv: kv[1]["avg_us"], reverse=True):
        pct = 100.0 * r["avg_us"] / total_avg_us if total_avg_us else 0.0
        lines.append(f"{op:<48}{r['iters']:>6}{r['avg_us']:>11.2f}{r['min_us']:>10.2f}{r['max_us']:>10.2f}{pct:>7.1f}%")
    lines.append("-" * 91)
    lines.append(f"{'TOTAL (forward)':<48}{'':>6}{total_avg_us:>11.2f}")
    print("\n".join(lines))


def write_summary_csv(out_path: Path, per_op: dict, total_avg_us: float):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["OP CODE", "iters", "avg_us", "min_us", "max_us", "pct"])
        for op, r in sorted(per_op.items(), key=lambda kv: kv[1]["avg_us"], reverse=True):
            pct = 100.0 * r["avg_us"] / total_avg_us if total_avg_us else 0.0
            w.writerow([op, r["iters"], f"{r['avg_us']:.3f}", f"{r['min_us']:.3f}", f"{r['max_us']:.3f}", f"{pct:.2f}"])
        w.writerow(["TOTAL", "", f"{total_avg_us:.3f}", "", "", ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="config id (yaml stem) from the configs dir, or 'all'")
    ap.add_argument("--mesh", help=f"comma subset of {MESH_IDS} (default: 8x4)")
    ap.add_argument("--fabric", help=f"comma subset of {FABRIC_IDS} (default: fabric_1D_ring)")
    ap.add_argument("--warmup-iters", type=int, default=1, help="replay iterations to drop as warmup (default 1)")
    ap.add_argument("--no-profile", action="store_true", help="omit tracy -p (op profiling); faster, no PM ideal")
    ap.add_argument("--include-skipped", action="store_true", help="also run configs in the test's SKIP_LIST")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and commands, run nothing")
    ap.add_argument("--analyze", metavar="CSV", help="analyze an existing ops_perf CSV and exit (no device run)")
    args = ap.parse_args()

    # Analyze-only mode: skip everything device-related.
    if args.analyze:
        per_op, total, traced = analyze_csv(args.analyze, warmup_iters=args.warmup_iters)
        if not per_op:
            print(f"no forward ops found in {args.analyze}", file=sys.stderr)
            return 1
        print_report(args.analyze, per_op, total, traced)
        return 0

    if not args.config:
        ap.error("config id (or 'all') is required unless --analyze is given")

    def axis(val, allowed, name, default):
        if not val:
            return default
        picked = [v.strip() for v in val.split(",") if v.strip()]
        bad = [v for v in picked if v not in allowed]
        if bad:
            ap.error(f"--{name}: {bad} not in {allowed}")
        return picked

    all_configs = config_ids()
    if not all_configs:
        ap.error(f"no yaml configs found in {CONFIGS_DIR}")
    runnable = [c for c in all_configs if args.include_skipped or c not in SKIP_CONFIGS]

    if args.config == "all":
        configs = runnable
    elif args.config in all_configs:
        configs = [args.config]
    else:
        ap.error(f"unknown config '{args.config}'; choose from {all_configs} or 'all'")

    meshes = axis(args.mesh, MESH_IDS, "mesh", ["8x4"])
    fabrics = axis(args.fabric, FABRIC_IDS, "fabric", ["fabric_1D_ring"])

    repo_root = find_repo_root(THIS_DIR)
    reports_dir = repo_root / "generated" / "profiler" / "reports"
    py = sys.executable

    combos = [(c, m, f) for c in configs for m in meshes for f in fabrics]
    print(f"# {len(combos)} case(s): configs={configs} mesh={meshes} fabric={fabrics}")
    print(f"# reports dir: {reports_dir}\n")

    summary = []  # (slug, status, total_us, top_op)
    for i, (config, mesh, fabric) in enumerate(combos, 1):
        slug = f"moe_decode_{config}_{mesh}_{fabric}"
        expr = k_expr(config, all_configs, mesh, fabric)
        prof = [] if args.no_profile else ["-p"]
        # tracy runs the trailing command through one shell round-trip, so the space-containing -k value
        # (and the test path) must be pre-quoted (shlex.quote) or they get word-split -> "0 selected".
        tracy_cmd = [py, "-m", "tracy", "-r", *prof, "-n", slug, "-m",
                     "pytest", shlex.quote(str(TEST_FILE)), "-k", shlex.quote(expr), "-s"]  # fmt: skip

        print(f"=== [{i}/{len(combos)}] {slug} ===")
        print(f"    -k: {expr}")
        print(f"    $ {' '.join(tracy_cmd)}")
        if args.dry_run:
            summary.append((slug, "DRY", None, None))
            print()
            continue

        since = time.time()
        rc = subprocess.run(tracy_cmd, cwd=repo_root).returncode
        if rc != 0:
            print(f"    !! tracy/pytest exited {rc}\n")
            summary.append((slug, f"RUN-FAIL({rc})", None, None))
            continue

        csv_path = find_new_csv(reports_dir, slug, since)
        if csv_path is None:
            print("    !! no ops_perf CSV found (case may have skipped on this hardware)\n")
            summary.append((slug, "NO-CSV", None, None))
            continue

        per_op, total, traced = analyze_csv(str(csv_path), warmup_iters=args.warmup_iters)
        if not per_op:
            print(f"    !! no forward ops in {csv_path}\n")
            summary.append((slug, "NO-OPS", None, None))
            continue

        print_report(str(csv_path), per_op, total, traced, title=f"{slug}  ({csv_path.name})")
        out_csv = csv_path.parent / f"{slug}_forward_op_times.csv"
        write_summary_csv(out_csv, per_op, total)
        print(f"    per-op summary -> {out_csv}\n")
        top_op = max(per_op.items(), key=lambda kv: kv[1]["avg_us"])[0]
        summary.append((slug, "OK", total, top_op))

    print("\n================================ SUMMARY ================================")
    wslug = max([len(s[0]) for s in summary] + [len("case")])
    print(f"  {'case':<{wslug}}  {'fwd_avg_us':>11}  {'top_op':<40}  status")
    for slug, status, total, top_op in summary:
        tot = f"{total:.2f}" if isinstance(total, float) else "-"
        print(f"  {slug:<{wslug}}  {tot:>11}  {(top_op or '-'):<40}  {'' if status == 'OK' else status}")
    fails = [s for s in summary if s[1] not in ("OK", "DRY")]
    print(f"\n{len(summary)} case(s), {len(fails)} problem(s).")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
