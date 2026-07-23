#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Driver for test_mla_ccl_perf.py: for a given MLA CCL op (or `all`), run EACH parametrized test case
under tracy in its own profiled session, then run analyze_trace_csv.py on the CSV that run produced.

Why one tracy run per case: analyze_trace_csv.py filters a CSV only by OP CODE, so every case for an
op shares the same op code. Mixing dtypes / trace vs no-trace / persist / mesh sizes into a single CSV
blends the rows (and the analyzer silently drops the no-trace rows whenever any trace-replay rows are
present). So each (dtype x trace x persist x mesh) combination is profiled separately -> its own CSV.

Each case is selected with a pytest `-k` expression (substring-AND on the node id, order-independent),
profiled with `-n <slug>` so tracy nests the report under generated/profiler/reports/<slug>/<ts>/, and
then summarized. The op code is derived from the op's kind (ag/rs) in MLA_CCL_OPS.

Usage:
    python run_mla_ccl_perf.py <op_id|all> [filters]

    # all cases for the o_proj reduce-scatter:
    python run_mla_ccl_perf.py o_proj_rs
    # just the bf8, trace, 8x4 cases for kv_ag (both persist variants):
    python run_mla_ccl_perf.py kv_ag --dtype bf8 --trace trace --mesh 8x4
    # see the plan without running anything:
    python run_mla_ccl_perf.py all --dry-run

op ids come from MLA_CCL_OPS: q_a_proj_rs q_ag kv_ag o_proj_rs kv576_h64_ag kv512_h64_ag d128_h32_ag
Filters take comma-separated subsets of the pytest ids; omit a filter to sweep that whole axis:
    --dtype   {bf16,bf8}
    --ag-impl {ttnn_ag,async}   # all_gather impl; ignored for reduce_scatter (forced to ttnn_ag)
    --mesh    {1x8,8x4}
    --trace   {no_trace,trace}
    --persist {alloc_out,persist_out_no_bar}

The op code handed to analyze_trace_csv.py depends on (kind, ag_impl): reduce_scatter ->
ReduceScatterMinimalAsyncDeviceOperation; all_gather -> AllGatherDeviceOperation (ttnn_ag) or
AllGatherAsyncDeviceOperation (async).
"""
import argparse
import ast
import itertools
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TEST_FILE = THIS_DIR / "test_mla_ccl_perf.py"
ANALYZER = THIS_DIR / "analyze_trace_csv.py"

# pytest ids for each parametrized axis -- MUST match the ids= in test_mla_ccl_perf's decorators.
DTYPE_IDS = ["bf16", "bf8"]
AG_IMPL_IDS = ["ttnn_ag", "async"]  # all_gather impl; ignored (skipped) for reduce_scatter
TRACE_IDS = ["no_trace", "trace"]
PERSIST_IDS = ["alloc_out", "persist_out_no_bar"]
MESH_IDS = ["1x8", "8x4"]

# Device op code (the analyze_trace_csv.py filter) per (kind, ag_impl). rs ignores ag_impl.
OP_CODE = {
    ("rs", "ttnn_ag"): "ReduceScatterMinimalAsyncDeviceOperation",
    ("ag", "ttnn_ag"): "AllGatherDeviceOperation",  # new top-level ttnn.all_gather
    ("ag", "async"): "AllGatherAsyncDeviceOperation",  # ttnn.experimental.all_gather_async
}


def find_repo_root(start: Path) -> Path:
    p = start
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return start


def load_op_kinds(test_file: Path) -> dict:
    """Parse MLA_CCL_OPS from the test file via AST (no ttnn import) -> {ccl_id: kind}."""
    tree = ast.parse(test_file.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MLA_CCL_OPS" for t in node.targets
        ):
            # each entry is (ccl_id, kind, dim, feat, n1); ccl_id and kind are string literals.
            return {elt.elts[0].value: elt.elts[1].value for elt in node.value.elts}
    raise RuntimeError(f"MLA_CCL_OPS not found in {test_file}")


def k_expr(op: str, dtype: str, ag_impl: str, trace: str, persist: str, mesh: str) -> str:
    """pytest -k expression pinning exactly one case. Substring-AND is order-independent; `not no_trace`
    is required for the trace case because the no_trace id contains the substring `trace`. The ag_impl
    ids (ttnn_ag, async) are mutually disjoint substrings, so no guard is needed there."""
    parts = ["test_mla_ccl_perf", op, dtype, ag_impl, persist, mesh]
    parts += ["trace", "not no_trace"] if trace == "trace" else ["no_trace"]
    return " and ".join(parts)


def find_new_csv(reports_dir: Path, slug: str, since: float):
    """Newest ops_perf_results CSV touched since `since`, preferring ones whose name carries the slug."""
    cands = [p for p in reports_dir.glob("**/ops_perf_results_*.csv") if p.stat().st_mtime >= since - 1.0]
    cands.sort(key=lambda p: p.stat().st_mtime)
    slug_hits = [p for p in cands if slug in p.name or slug in str(p.parent)]
    chosen = slug_hits or cands
    return chosen[-1] if chosen else None


def parse_metrics(stdout: str):
    """Pull (mean_us, pm_ideal_us, util_pct) out of analyze_trace_csv.py's output. Any missing -> None.
    mean = crit-path mean; util = % vs crit-path mean; pm_ideal absent unless profiled with op profiling."""

    def grab(pat):
        m = re.search(pat, stdout)
        return float(m.group(1)) if m else None

    return (
        grab(r"CRIT-PATH.*?mean\s+([\d.]+)"),
        grab(r"PM IDEAL:\s+([\d.]+)us"),
        grab(r"UTILIZATION:\s+([\d.]+)%\s+vs crit-path mean"),
    )


def main():
    op_kinds = load_op_kinds(TEST_FILE)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", help="op id from MLA_CCL_OPS, or 'all'")
    ap.add_argument("--dtype", help=f"comma subset of {DTYPE_IDS}")
    ap.add_argument("--ag-impl", dest="ag_impl", help=f"comma subset of {AG_IMPL_IDS} (all_gather ops only)")
    ap.add_argument("--trace", help=f"comma subset of {TRACE_IDS}")
    ap.add_argument("--persist", help=f"comma subset of {PERSIST_IDS}")
    ap.add_argument("--mesh", help=f"comma subset of {MESH_IDS}")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and commands, run nothing")
    args = ap.parse_args()

    def axis(val, allowed, name):
        if not val:
            return list(allowed)
        picked = [v.strip() for v in val.split(",") if v.strip()]
        bad = [v for v in picked if v not in allowed]
        if bad:
            ap.error(f"--{name}: {bad} not in {allowed}")
        return picked

    dtypes = axis(args.dtype, DTYPE_IDS, "dtype")
    ag_impls = axis(args.ag_impl, AG_IMPL_IDS, "ag-impl")
    traces = axis(args.trace, TRACE_IDS, "trace")
    persists = axis(args.persist, PERSIST_IDS, "persist")
    meshes = axis(args.mesh, MESH_IDS, "mesh")

    if args.op == "all":
        ops = list(op_kinds)
    elif args.op in op_kinds:
        ops = [args.op]
    else:
        ap.error(f"unknown op '{args.op}'; choose from {list(op_kinds)} or 'all'")

    repo_root = find_repo_root(THIS_DIR)
    reports_dir = repo_root / "generated" / "profiler" / "reports"
    py = sys.executable

    # reduce_scatter ignores ag_impl (the `async` case is skipped in the test), so collapse it to
    # ttnn_ag for rs ops -- avoids emitting runs that would just skip.
    combos = [
        (op, dt, ai, tr, pe, me)
        for op in ops
        for ai in (ag_impls if op_kinds[op] == "ag" else ["ttnn_ag"])
        for dt, tr, pe, me in itertools.product(dtypes, traces, persists, meshes)
    ]
    print(
        f"# {len(combos)} case(s) across ops={ops} dtype={dtypes} ag_impl={ag_impls} "
        f"trace={traces} persist={persists} mesh={meshes}"
    )
    print(f"# reports dir: {reports_dir}\n")

    summary = []  # (slug, status, util_line, csv)
    for i, (op, dt, ai, tr, pe, me) in enumerate(combos, 1):
        kind = op_kinds[op]
        op_code = OP_CODE[(kind, ai)]
        slug = f"{op}_{dt}_{ai}_{tr}_{pe}_{me}"
        expr = k_expr(op, dt, ai, tr, pe, me)
        # tracy runs the trailing command via subprocess.Popen([" ".join(args)], shell=True) -- a single
        # unquoted shell round-trip -- so the space-containing -k value (and the path) must be pre-quoted
        # or they get word-split (-> "collected 0 items"). shlex.quote survives the one shell hop.
        tracy_cmd = [py, "-m", "tracy", "-r", "-p", "-n", slug, "-m",
                     "pytest", shlex.quote(str(TEST_FILE)), "-k", shlex.quote(expr)]  # fmt: skip
        analyze_cmd = [py, str(ANALYZER), "<csv>", op_code]

        # Per-case summary row. label is the case minus the dtype/ag columns (those are separate).
        row = {
            "dtype": dt,
            "ag": ai if kind == "ag" else "-",
            "label": f"{op}/{tr}/{pe}/{me}",
            "mean": None,
            "pm": None,
            "util": None,
            "status": "",
        }

        print(f"=== [{i}/{len(combos)}] {slug}  (op_code={op_code}) ===")
        print(f"    -k: {expr}")
        print(f"    $ {' '.join(tracy_cmd)}")
        if args.dry_run:
            print(f"    $ {py} {ANALYZER.name} <newest csv> {op_code}\n")
            row["status"] = "DRY"
            summary.append(row)
            continue

        since = time.time()
        rc = subprocess.run(tracy_cmd, cwd=repo_root).returncode
        if rc != 0:
            print(f"    !! tracy/pytest exited {rc}; skipping analyze\n")
            row["status"] = f"RUN-FAIL({rc})"
            summary.append(row)
            continue

        csv = find_new_csv(reports_dir, slug, since)
        if csv is None:
            print("    !! no ops_perf CSV found for this run\n")
            row["status"] = "NO-CSV"
            summary.append(row)
            continue

        analyze_cmd[2] = str(csv)
        print(f"    $ {py} {ANALYZER.name} {csv} {op_code}")
        res = subprocess.run(analyze_cmd, cwd=repo_root, capture_output=True, text=True)
        sys.stdout.write(res.stdout)
        if res.stderr:
            sys.stderr.write(res.stderr)
        row["mean"], row["pm"], row["util"] = parse_metrics(res.stdout)
        row["status"] = "OK" if res.returncode == 0 else f"ANALYZE-FAIL({res.returncode})"
        summary.append(row)
        print()

    # Per-case table: dtype | ag | case label | mean(us) | PM IDEAL(us) | util(%).
    def fmt(v, suffix=""):
        return f"{v:.1f}{suffix}" if isinstance(v, float) else "-"

    wlabel = max([len(r["label"]) for r in summary] + [len("case")])
    print("\n================================ SUMMARY ================================")
    print(f"  {'dtype':<5}  {'ag':<8}  {'case':<{wlabel}}  {'mean(us)':>9}  {'PM_IDEAL':>9}  {'util%':>6}  status")
    for r in summary:
        ok = r["status"] in ("OK", "DRY")
        print(
            f"  {r['dtype']:<5}  {r['ag']:<8}  {r['label']:<{wlabel}}  "
            f"{fmt(r['mean']):>9}  {fmt(r['pm']):>9}  {fmt(r['util']):>6}  "
            f"{'' if r['status'] == 'OK' else r['status']}"
        )
    fails = [r for r in summary if r["status"] not in ("OK", "DRY")]
    print(f"\n{len(summary)} case(s), {len(fails)} problem(s).")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
