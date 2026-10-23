#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_gather perf sweep on a BH (1,8) box: run sweep_all_gather_bh.py under tracy, reduce, compare branches.

  run      --out DIR [--branch LABEL] [--topos ring,line] [--sweeps size,page,holdout] [--columns ...]
           [--cells PATTERNS] [--iters N] [--check] [--repo PATH]
           One tracy process per (topology, sweep) into DIR/<branch>/<sweep>-<topo>/, each reduced to cells.csv,
           all of them to DIR/<branch>/sweep.csv. --check runs the cells once, bit-exact, without tracy.
  analyze  RUN_DIR [--branch LABEL]       reduce one tracy run directory again
  compare  CSV [CSV ...] --new LABEL [--bar 2] [--pick 3]
           auto column of LABEL vs the best other branch per cell; and, within LABEL, auto vs the best forced column.

Columns: "auto" leaves TTNN_ALL_GATHER_FORCE unset, the others force an algorithm. Builds without the override
ignore it, so --columns defaults to auto there (detected from the source tree) and only auto compares across
branches. The harness is copied into a --repo that lacks it for the run and removed afterwards, so the same files
measure any branch. For an A/B/A comparison run this branch, the baseline(s), then this branch again with
--branch <label>b; compare treats each label as one branch.

Metric, per cell: per traced replay the max over devices of DEVICE KERNEL DURATION (summed over the replay's ops
when the call runs several, i.e. composite), then the median / p10 / p90 over replays. A replay with any duration
over 1e8 ns is a BH profiler wall-clock desync artefact: dropped whole (all devices) and counted. GB/s per
link-direction = (N-1)/N * output bytes / K / t with K = links (line) or 2 * links (ring), GB = 1e9 B: every device
receives (N-1)/N of the output, through one direction on a line end and both on a ring.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HARNESS_REL = Path("tests/ttnn/multidevice_perf_tests/sweep_all_gather_bh.py")
FORCED = ["uc_n2", "uc_n3", "mc"]
DESYNC_NS = 1e8
DUR = "DEVICE KERNEL DURATION [ns]"
SESSION = "METAL TRACE REPLAY SESSION ID"
CSV_COLUMNS = [
    "branch",
    "topology",
    "sweep",
    "shape",
    "layout",
    "dtype",
    "page_bytes",
    "bytes",
    "algo_column",
    "us_median",
    "us_p10",
    "us_p90",
    "gbps_per_link_dir",
    # extras
    "cell",
    "dim",
    "mem_in",
    "mem_out",
    "links",
    "replays",
    "dropped",
    "choice",
    "ops",
]


def git_branch(repo):
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip().split("/")[-1] or "unknown"


def has_override(repo):
    src = Path(repo) / "ttnn/cpp/ttnn/operations/ccl/all_gather"
    return any("TTNN_ALL_GATHER_FORCE" in p.read_text(errors="ignore") for p in src.rglob("*.[ch]pp"))


# ---------------------------------------------------------------------------------------------------- reduction


def _ops_csv(run_dir):
    found = sorted(Path(run_dir).glob("**/ops_perf_results_*.csv"))
    if not found:
        sys.exit(f"no ops_perf_results_*.csv under {run_dir}")
    return found[-1]


def _attr(attrs, key):
    m = re.search(rf"'{key}': '([^']*)'", attrs)
    return m.group(1) if m else ""


def _links(attrs):
    nums = [int(x) for x in re.findall(r"\d+", _attr(attrs, "num_links") or _attr(attrs, "axis_num_links"))]
    return max(nums) if nums and max(nums) > 0 else 2


def traced_blocks(ops):
    """Each traced replay block, in run order: the maximal runs of rows carrying a replay session id. Cells are
    separated by their own untraced compile call, so there is one block per measured cell."""
    traced = ops[SESSION].notna()
    run_id = (traced != traced.shift()).cumsum()
    return [g for _, g in ops[traced].groupby(run_id[traced], sort=True)]


def reduce_block(block, devices):
    per = block.groupby([SESSION, "DEVICE ID"])[DUR].agg(["sum", "max"])
    bad = per.groupby(level=0)["max"].max() > DESYNC_NS
    full = per.groupby(level=0)["sum"].count() == devices
    keep = full & ~bad
    t_us = per["sum"].groupby(level=0).max()[keep] / 1000.0
    return t_us, int((~keep).sum())


def analyze(run_dir, branch):
    run_dir = Path(run_dir)
    cells = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    measured = [c for c in cells if c["status"] == "ok"]
    ops = pd.read_csv(_ops_csv(run_dir), low_memory=False)
    ops = ops[ops["OP TYPE"] == "tt_dnn_device"]
    blocks = traced_blocks(ops)
    if len(blocks) != len(measured):
        sys.exit(f"{run_dir}: {len(blocks)} traced blocks in the ops CSV but {len(measured)} measured cells")
    rows = []
    for c, block in zip(measured, blocks):
        t_us, dropped = reduce_block(block, c["devices"])
        if dropped:
            print(f"!! {c['cell']} {c['column']}: {dropped} replay(s) dropped (desync or missing devices)")
        attrs = str(block["ATTRIBUTES"].iloc[0])
        links = _links(attrs)
        n = c["devices"]
        k = links * (2 if c["topology"] == "ring" else 1)
        med = float(np.median(t_us)) if len(t_us) else float("nan")
        rows.append(
            dict(
                branch=branch,
                topology=c["topology"],
                sweep=c["sweep"],
                shape="x".join(map(str, c["shape"])),
                layout=c["layout"],
                dtype=c["dtype"],
                page_bytes=c["page_bytes"],
                bytes=c["bytes"],
                algo_column=c["column"],
                us_median=med,
                us_p10=float(np.percentile(t_us, 10)) if len(t_us) else float("nan"),
                us_p90=float(np.percentile(t_us, 90)) if len(t_us) else float("nan"),
                gbps_per_link_dir=(n - 1) / n * c["bytes"] / k / (med * 1e3),
                cell=c["cell"],
                dim=c["dim"],
                mem_in=c["mem_in"],
                mem_out=c["mem_out"],
                links=links,
                replays=len(t_us),
                dropped=dropped,
                choice=_attr(attrs, "choice"),
                ops="+".join(sorted({re.sub(r"DeviceOperation.*$", "", o) for o in block["OP CODE"]})),
            )
        )
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(run_dir / "cells.csv", index=False)
    errors = [f"{c['cell']} {c['column']}" for c in cells if c["status"] != "ok"]
    if errors:
        print(f"!! {run_dir.name}: not measured (error at the compile call): {errors}")
    return df


# ---------------------------------------------------------------------------------------------------- run


def run(args):
    repo = Path(args.repo).resolve()
    branch = args.branch or git_branch(repo)
    columns = args.columns or ("auto," + ",".join(FORCED) if has_override(repo) else "auto")
    print(f"branch {branch}: columns {columns}")
    harness = repo / HARNESS_REL
    copied = not harness.exists()
    if copied:
        shutil.copy(HERE / HARNESS_REL.name, harness)
    env = dict(os.environ, TT_METAL_HOME=str(repo), PYTHONPATH=str(repo), AG_COLUMNS=columns)
    env.pop("TTNN_ALL_GATHER_FORCE", None)
    for key, value in (("AG_CELLS", args.cells), ("AG_ITERS", args.iters)):
        if value:
            env[key] = str(value)
    test = f"{HARNESS_REL}::test_all_gather_sweep"
    frames, failed = [], []
    try:
        for topo in args.topos.split(","):
            for sweep in args.sweeps.split(","):
                run_env = dict(env, AG_TOPO=topo, AG_SWEEPS=sweep)
                if args.check:
                    cmd = [sys.executable, "-m", "pytest", test, "-x"]
                    ok = subprocess.run(cmd, cwd=repo, env=dict(run_env, AG_CHECK="1")).returncode == 0
                    print(f"== check {sweep} {topo}: {'ok' if ok else 'FAILED'}")
                    failed += [] if ok else [f"{sweep}-{topo}"]
                    continue
                run_dir = Path(args.out).resolve() / branch / f"{sweep}-{topo}"
                if run_dir.exists():
                    sys.exit(f"{run_dir} exists; move it away first (never mix two runs in one directory)")
                run_dir.mkdir(parents=True)
                run_env["AG_MANIFEST"] = str(run_dir / "manifest.jsonl")
                cmd = [sys.executable, "-m", "tracy", "-r", "-p", "-v", "-o", str(run_dir), "-m", "pytest", test]
                with open(run_dir / "run.log", "w") as log:
                    rc = subprocess.run(cmd, cwd=repo, env=run_env, stdout=log, stderr=subprocess.STDOUT).returncode
                print(f"== {sweep} {topo}: pytest exit {rc} ({run_dir / 'run.log'})")
                if not (run_dir / "manifest.jsonl").exists():
                    failed.append(f"{sweep}-{topo}")
                    continue
                failed += [] if rc == 0 else [f"{sweep}-{topo}"]
                frames.append(analyze(run_dir, branch))
    finally:
        if copied:
            harness.unlink()
    if frames:
        out = Path(args.out).resolve() / branch / "sweep.csv"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"wrote {out}")
    if failed:
        sys.exit(f"failed or partial runs: {failed} (a hang is the user's to clear; do not reset the board)")


# ---------------------------------------------------------------------------------------------------- compare


def compare(args):
    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    key = ["topology", "cell"]
    auto = df[df["algo_column"] == "auto"].pivot_table(index=key, columns="branch", values="us_median")
    if args.new not in auto:
        sys.exit(f"no auto rows for {args.new}; branches: {list(auto.columns)}")
    bases = [b for b in auto.columns if b != args.new]
    pd.set_option("display.width", 200)
    if bases:
        auto["best_base"] = auto[bases].min(axis=1)
        auto["delta_pct"] = 100.0 * (auto[args.new] / auto["best_base"] - 1.0)
        print(f"auto: {args.new} vs the best of {bases} per cell (us; delta > 0 is slower)")
        print(auto.round(2).to_string())
        bad = auto[auto["delta_pct"] > args.bar]
        print(
            f"\nregression bar (no cell > {args.bar}% slower than the best baseline): {'FAIL' if len(bad) else 'PASS'}"
        )
        for (topo, cell), r in bad.iterrows():
            print(f"  slower: {topo} {cell} {r['delta_pct']:+.1f}%")
        for b in bases:
            both = auto[[args.new, b]].dropna()
            print(
                f"geomean speedup vs {b}: {np.exp(np.log(both[b] / both[args.new]).mean()):.3f}x over {len(both)} cells"
            )
    new = df[df["branch"] == args.new].pivot_table(index=key, columns="algo_column", values="us_median")
    forced = [c for c in new.columns if c != "auto"]
    if forced and "auto" in new:
        new["best_forced"] = new[forced].min(axis=1)
        new["pick_pct"] = 100.0 * (new["auto"] / new["best_forced"] - 1.0)
        miss = new[new["pick_pct"] > args.pick]
        print(f"\n{args.new}: auto vs the best forced column ({forced}); misses (> {args.pick}%): {len(miss)}")
        print((miss if len(miss) else new).round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--repo", default=str(HERE.parents[2]))
    r.add_argument("--branch", help="label (default: the repo's branch name)")
    r.add_argument("--topos", default="ring,line")
    r.add_argument("--sweeps", default="size,page,holdout")
    r.add_argument("--columns", help="default: auto plus uc_n2, uc_n3, mc where the override exists")
    r.add_argument("--cells", help="comma fnmatch patterns over cell ids, e.g. 'size/*,holdout/l1_*'")
    r.add_argument("--iters", type=int)
    r.add_argument("--check", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("run_dir")
    a.add_argument("--branch")
    c = sub.add_parser("compare")
    c.add_argument("csv", nargs="+")
    c.add_argument("--new", required=True)
    c.add_argument("--bar", type=float, default=2.0, help="regression bar, percent")
    c.add_argument("--pick", type=float, default=3.0, help="auto vs best forced bar, percent")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    elif args.cmd == "analyze":
        branch = args.branch or Path(args.run_dir).resolve().parent.name
        print(analyze(args.run_dir, branch).to_string(index=False))
    else:
        compare(args)


if __name__ == "__main__":
    main()
