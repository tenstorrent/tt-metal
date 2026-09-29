#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LLK SFPU report: before/after perf and accuracy of the SFPU ops a change touches.

    cli.py run --arch wormhole --head refs/pull/57421/head [--base <ref>] [--ops tanh,exp]

Run it from the tt-llk test venv, on a machine with the target device.
See README.md in this directory.
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accuracy  # noqa: E402
import detect  # noqa: E402
import overlay  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402

import perf  # noqa: E402

#: Report-only thresholds, the LLK perf gate's (#53752): BH 2%, WH 8%, and 30
#: cycles per loop on top.
THRESHOLDS = {"blackhole": 0.02, "wormhole": 0.08, "min_cycles": 30.0}

#: Detected ops beyond this are listed, not measured (the /test cap is 8 too).
MAX_OPS = 8


def _sides(plan, work, mode):
    """``merge-base``: device code of the merge-base vs the PR head, as they are.
    ``rebase``: the tool revision vs the tool revision plus the PR's device diff.
    Either way the host Python is the tool's."""
    if mode == "merge-base":
        base = runner.Side("base", plan.base_sha, work / "base", work / "build-base")
        head = runner.Side("head", plan.head_sha, work / "head", work / "build-head")
        overlay.build_side(plan, base.tree, at=plan.base_sha)
        overlay.build_side(plan, head.tree, at=plan.head_sha)
    else:
        base = runner.Side("base", plan.tool_sha, work / "base", work / "build-base")
        head = runner.Side("head", plan.head_sha, work / "head", work / "build-head")
        overlay.build_side(plan, base.tree)
        overlay.build_side(plan, head.tree, overlay.device_diff(plan))
    return base, head


def cmd_detect(args):
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    plan = overlay.make_plan(args.repo, args.head, args.base, args.main_ref)
    print(
        f"tool {plan.tool_sha[:12]}  base {plan.base_sha[:12]}  head {plan.head_sha[:12]}"
    )
    print(f"device files taken from the PR: {len(plan.applied)}")
    for p in plan.applied:
        print(f"  {p}")
    base, head = _sides(plan, work, args.mode)
    found = detect.changed_ops(
        base, head, args.arch, work / "detect.log", jobs=args.jobs
    )
    print(json.dumps(found, indent=2))
    (work / "detect.json").write_text(json.dumps(found, indent=2))


def _board():
    try:
        out = subprocess.run(
            ["tt-smi", "-ls"], capture_output=True, text=True, timeout=60
        ).stdout
        for line in out.splitlines():
            for name in ("n150", "n300", "p100", "p150", "p300"):
                if name in line.lower():
                    return name
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown board"


def _merge_base_age(plan):
    ts = int(overlay.git(plan.repo, "show", "-s", "--format=%ct", plan.base_sha))
    return int((time.time() - ts) // 86400)


def _math_op_names():
    sys.path.insert(0, str(runner.PYTHON_TESTS))
    from helpers.llk_params import MathOperation

    return {m.name.lower(): m.name for m in MathOperation}


def _sfpu_type_to_op(sfpu_types):
    """detect.py reports SfpuType names (``tanh``); the harness selects MathOperation
    names (``Tanh``). Most map by case; the rest are listed as not covered."""
    names = _math_op_names()
    mapped, unmapped = [], []
    for t in sfpu_types:
        key = t.lower().replace("_", "")
        hit = next((v for k, v in names.items() if k.replace("_", "") == key), None)
        (mapped if hit else unmapped).append(hit or t)
    return mapped, unmapped


def cmd_run(args):
    t0 = time.time()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    log = work / "run.log"
    log.write_text("")
    plan = overlay.make_plan(args.repo, args.head, args.base, args.main_ref)
    print(
        f"tool {plan.tool_sha[:12]}  base {plan.base_sha[:12]}  head {plan.head_sha[:12]}  mode {args.mode}"
    )
    base, head = _sides(plan, work, args.mode)

    notes = []
    if args.ops:
        ops = [o.strip() for o in args.ops.split(",") if o.strip()]
        why = "requested in the command"
        not_covered = []
        found = None
    else:
        found = detect.changed_ops(base, head, args.arch, log, jobs=args.jobs)
        changed = sorted({o for fam in found.values() for o in fam["changed"]})
        ops, not_covered = _sfpu_type_to_op(
            [c for c in changed if not c.startswith("typecast")]
        )
        if any(c.startswith("typecast") for c in changed):
            ops.append("Typecast")
        why = "auto-detected: their compiled SFPU code differs between the two sides"
        if len(ops) > MAX_OPS:
            notes.append(
                f"{len(ops)} ops changed; measured the first {MAX_OPS}. Measure others with "
                f"`/llk-sfpu-test {' '.join(o.lower() for o in ops[MAX_OPS:MAX_OPS + 4])} ...`."
            )
            not_covered += ops[MAX_OPS:]
            ops = ops[:MAX_OPS]
    print(f"ops: {ops}  not covered: {not_covered}")

    perf_rows = {}
    families = {
        "typecast": [o for o in ops if o == "Typecast"],
        "unary": [o for o in ops if o != "Typecast"],
    }
    for family, fam_ops in families.items():
        if not fam_ops:
            continue
        runs = perf.sweep(
            base,
            head,
            args.arch,
            family,
            fam_ops,
            work / "perf",
            log,
            iterations=args.iterations,
            jobs=args.jobs,
        )
        verdicts = perf.compare(runs, THRESHOLDS[args.arch], THRESHOLDS["min_cycles"])
        perf_rows[family] = perf.rows(runs, verdicts)

    acc_ops = [o for o in ops if o != "Typecast"]
    acc = []
    if acc_ops:
        accuracy.measure(
            base, args.arch, acc_ops, work / "accuracy" / "base", log, jobs=args.jobs
        )
        accuracy.measure(
            head, args.arch, acc_ops, work / "accuracy" / "head", log, jobs=args.jobs
        )
        acc = accuracy.compare(work / "accuracy" / "base", work / "accuracy" / "head")
    if "Typecast" in ops:
        notes.append(
            "Typecast accuracy is not in this version of the report; its perf is."
        )

    cmd = " ".join(["python3", "tt_metal/tt-llk/sfpu_report/cli.py", *sys.argv[1:]])
    summary = {
        "arch": args.arch,
        "host": socket.gethostname(),
        "host_board": _board(),
        "run_url": args.run_url,
        "mode": args.mode,
        "tool_sha": plan.tool_sha,
        "base_sha": plan.base_sha,
        "head_sha": plan.head_sha,
        "head_moved_to": args.head_moved_to,
        "merge_base_age_days": _merge_base_age(plan),
        "applied": plan.applied,
        "not_applied": plan.not_applied,
        "detection": found,
        "ops": {"measured": ops, "not_covered": not_covered, "why": why},
        "iterations": args.iterations,
        "thresholds": THRESHOLDS,
        "perf": perf_rows,
        "accuracy": acc,
        "notes": notes,
        "commands": [cmd],
        "seconds": round(time.time() - t0),
    }
    out = work / f"summary-{args.arch}.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    (work / f"report-{args.arch}.md").write_text(report.render([summary]))
    print(f"wrote {out} in {summary['seconds']} s")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=str(runner.TOOL_LLK.parents[1]))
    ap.add_argument("--arch", required=True, choices=["wormhole", "blackhole"])
    ap.add_argument("--head", required=True, help="PR head: sha or ref")
    ap.add_argument("--base", help="baseline ref (default: merge-base with --main-ref)")
    ap.add_argument("--main-ref", default="origin/main")
    ap.add_argument("--work", default="/tmp/llk-sfpu-report")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--mode", choices=["merge-base", "rebase"], default="merge-base")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect", help="list the ops whose machine code changed")
    run = sub.add_parser(
        "run", help="detect, measure perf and accuracy, write summary + report"
    )
    run.add_argument(
        "--ops", help="comma-separated MathOperation names; skips detection"
    )
    run.add_argument("--iterations", type=int, default=3)
    run.add_argument("--run-url")
    run.add_argument(
        "--head-moved-to", help="the PR's current head, if it moved after the command"
    )
    args = ap.parse_args(argv)
    {"detect": cmd_detect, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
