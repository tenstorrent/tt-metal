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
    print(f"tool {plan.tool_sha[:12]}  base {plan.base_sha[:12]}  head {plan.head_sha[:12]}")
    print(f"device files taken from the PR: {len(plan.applied)}")
    for p in plan.applied:
        print(f"  {p}")
    base, head = _sides(plan, work, args.mode)
    found = detect.changed_ops(base, head, args.arch, work / "detect.log", jobs=args.jobs)
    print(json.dumps(found, indent=2))
    (work / "detect.json").write_text(json.dumps(found, indent=2))


def _board():
    try:
        out = subprocess.run(["tt-smi", "-ls"], capture_output=True, text=True, timeout=60).stdout
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
    """detect.py reports SfpuType names (``exponential``); the harness selects
    MathOperation names (``Exp``). They map through the enum value the harness writes
    into build.h; an SfpuType no unary MathOperation uses is listed as not covered."""
    sys.path.insert(0, str(runner.PYTHON_TESTS))
    from helpers.llk_params import SFPU_UNARY_OPERATIONS

    names = {op.cpp_enum_value: op.name for op in SFPU_UNARY_OPERATIONS}
    mapped, unmapped = [], []
    for t in sfpu_types:
        (mapped if t in names else unmapped).append(names.get(t, t))
    return mapped, unmapped


def _prioritize(ops, changed_files):
    """Ops whose own kernel file the PR edits first: ``ckernel_sfpu_recip.h`` puts
    ``Reciprocal`` ahead of the ops that merely call it. Stable otherwise."""
    stems = {
        Path(p).stem.replace("ckernel_sfpu_", "").replace("_", "").lower()
        for p in changed_files
        if Path(p).name.startswith("ckernel_sfpu_")
    }

    def direct(op):
        name = op.lower()
        return any(stem and (name.startswith(stem) or stem.startswith(name)) for stem in stems)

    return sorted(ops, key=lambda op: not direct(op))


def _binary_enum_to_op():
    """``ckernel::BinaryOp`` names (detect.py) -> MathOperation names (``--op``)."""
    sys.path.insert(0, str(runner.PYTHON_TESTS))
    from helpers.llk_params import SFPU_BINARY_OPERATIONS

    return {op.cpp_enum_value: op.name for op in SFPU_BINARY_OPERATIONS}


def _requested_ops(text):
    """Comma-separated names in any case (``tanh``) as MathOperation names (``Tanh``).

    Returns ``(ops, unknown)``. ``Typecast`` selects the typecast family.
    """
    names = _math_op_names()
    names["typecast"] = "Typecast"
    ops, unknown = [], []
    for raw in (o.strip() for o in text.split(",")):
        if not raw:
            continue
        if raw.lower() in names:
            ops.append(names[raw.lower()])
        else:
            unknown.append(raw)
    return list(dict.fromkeys(ops)), unknown


def _family_of(op):
    if op == "Typecast":
        return "typecast"
    return "binary" if op.startswith("Sfpu") else "unary"


def _detected_ops(found, changed_files):
    """The ops to measure from detect.changed_ops(): ``(ops, not_covered, one_sided)``.

    An op whose variants built on one side only is measured too: on the side where it
    builds, and the report flags what is missing on the other.
    """
    unary, binary = found.get("unary", {}), found.get("binary", {})
    ops, not_covered = _sfpu_type_to_op(unary.get("changed", []) + unary.get("unmatched", []))
    one_sided, _ = _sfpu_type_to_op(unary.get("unmatched", []))
    binary_names = _binary_enum_to_op()
    for enum in binary.get("changed", []) + binary.get("unmatched", []):
        (ops if enum in binary_names else not_covered).append(binary_names.get(enum, enum))
    one_sided += [binary_names[e] for e in binary.get("unmatched", []) if e in binary_names]
    ops = _prioritize(list(dict.fromkeys(ops)), changed_files)
    typecast = found.get("typecast", {})
    if typecast.get("changed") or typecast.get("unmatched"):
        ops.append("Typecast")
        if typecast.get("unmatched"):
            one_sided.append("Typecast")
    return ops, not_covered, one_sided


def _harness_compiles(side, arch, log, jobs):
    """Compile one ordinary op on ``side``: a failure means harness skew, not the PR."""
    built = detect.compile_all(side, arch, "unary", log, jobs, only_ops=("neg",))
    return built in (0, runner.NO_TESTS_COLLECTED)


class NotMeasured(Exception):
    """No two sides can be built; ``reason`` says why and what the author can do."""

    def __init__(self, reason, files=()):
        super().__init__(reason)
        self.reason = reason
        self.files = list(files)


def _choose_sides(plan, work, mode, arch, log, jobs, notes):
    """The two sides to compare, and the mode they were built in.

    In merge-base mode the sides are old trees built with today's harness. When the
    merge-base's device code no longer builds with it (a toolchain or harness change
    on main since), every variant would fail on both sides and nothing would be
    measured, so the run falls back to rebase mode and says so.

    Rebase mode needs the PR's kernel diff to apply to main. When main has changed
    the same lines since, nothing can be measured until the PR is rebased: raises
    NotMeasured.
    """
    fell_back = False
    if mode == "merge-base":
        base, head = _sides(plan, work, mode)
        if _harness_compiles(base, arch, log, jobs):
            return mode, base, head
        mode, fell_back = "rebase", True
    try:
        base, head = _sides(plan, work, mode)
    except overlay.PatchConflict as conflict:
        cause = "The merge-base does not build with today's test harness, and this" if fell_back else "This"
        raise NotMeasured(
            f"{cause} PR's kernel changes do not apply cleanly to current main "
            f"(`{plan.tool_sha[:10]}`), which has changed the same lines since. "
            "Rebase the PR onto main, then comment `/llk-sfpu-test` again.",
            conflict.files,
        ) from conflict
    if fell_back:
        notes.append(
            "The merge-base does not build with today's test harness, so this report "
            "compares current main with main plus this PR's kernel changes."
        )
    return mode, base, head


def cmd_run(args):
    t0 = time.time()
    formats = [f.strip() for f in (args.formats or "").split(",") if f.strip()]
    if args.simulator:
        args.no_perf = True
        runner.SIMULATOR = True
        # ttsim aborts on Int32 and fp32 binary SFPU inputs (its unpacker model).
        formats = formats or ["Float16_b", "Float16"]
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    log = work / "run.log"
    log.write_text("")
    plan = overlay.make_plan(args.repo, args.head, args.base, args.main_ref)
    print(f"tool {plan.tool_sha[:12]}  base {plan.base_sha[:12]}  head {plan.head_sha[:12]}  mode {args.mode}")
    notes = []
    try:
        args.mode, base, head = _choose_sides(plan, work, args.mode, args.arch, log, args.jobs, notes)
    except NotMeasured as why_not:
        print(f"not measured: {why_not.reason}")
        _write_summary(
            args,
            plan,
            work,
            t0,
            detection=None,
            ops={"measured": [], "not_covered": [], "why": "not measured"},
            perf={},
            accuracy=[],
            notes=notes,
            not_measured=why_not.reason,
            conflicts=why_not.files,
        )
        if args.check:
            sys.exit(2)
        return
    if args.ops:
        ops, unknown = _requested_ops(args.ops)
        if unknown:
            notes.append("Not an op name, skipped: " + ", ".join(report.code(o) for o in unknown) + ".")
        why = "requested in the command"
        not_covered = []
        found = None
    else:
        found = detect.changed_ops(base, head, args.arch, log, jobs=args.jobs)
        ops, not_covered, one_sided = _detected_ops(found, plan.applied)
        if one_sided:
            notes.append(
                "Built on one side only, so measured where it builds: "
                + ", ".join(f"`{o}`" for o in one_sided)
                + " (the PR added or removed variants, or one side does not compile them; see run.log)."
            )
        why = "auto-detected: their compiled SFPU code differs between the two sides"
        if len(ops) > MAX_OPS:
            notes.append(
                f"{len(ops)} ops changed; measured the first {MAX_OPS}. Measure others with "
                f"`/llk-sfpu-test {' '.join(o.lower() for o in ops[MAX_OPS:MAX_OPS + 4])} ...`."
            )
            not_covered += ops[MAX_OPS:]
            ops = ops[:MAX_OPS]
    sfpu_files = [p for p in plan.applied if "sfpu" in p.lower() and "/tests/" not in p]
    arch_dir = {"wormhole": "wormhole_b0", "blackhole": "blackhole"}[args.arch]
    other_arch_only = sfpu_files and all(
        any(d in p for d in ("wormhole_b0", "blackhole", "quasar")) and arch_dir not in p for p in sfpu_files
    )
    if not ops and other_arch_only:
        notes.append(
            f"This PR changes SFPU kernels of other architectures only, not {args.arch.title()}'s: "
            + ", ".join(report.code(p) for p in sfpu_files[:4])
            + ". Nothing to measure here."
        )
    elif not ops and sfpu_files:
        notes.append(
            "This PR changes SFPU kernel files, but the machine code of no op this report covers "
            "(elementwise unary and binary SFPU, typecast) changed. The changed kernels may be "
            "ternary or structural SFPU ops (reduce, topk, ...), which a later version will cover: "
            + ", ".join(report.code(Path(p).name) for p in sfpu_files[:8])
            + "."
        )
    print(f"ops: {ops}  not covered: {not_covered}")
    harness = overlay.harness_changes(plan, ops)

    perf_rows = {}
    if args.no_perf:
        notes.append("Perf was not measured in this run (`--no-perf`).")
    else:
        families = {}
        for op in ops:
            families.setdefault(_family_of(op), []).append(op)
        for family, fam_ops in families.items():
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
                formats=formats,
            )
            if runs is None:
                notes.append(f"No perf test covers {', '.join(f'`{o}`' for o in fam_ops)}: accuracy only.")
                continue
            verdicts = perf.compare(runs, THRESHOLDS[args.arch], THRESHOLDS["min_cycles"])
            perf_rows[family] = perf.rows(runs, verdicts)
            if not any(perf_rows[family].values()):
                notes.append(
                    f"⚠️ The perf runs of {', '.join(f'`{o}`' for o in fam_ops)} left no results on "
                    "either side; see run.log."
                )

    acc_ops = [o for o in ops if o != "Typecast"]
    acc = []
    if acc_ops:
        exit_codes = [
            accuracy.measure(
                side,
                args.arch,
                acc_ops,
                work / "accuracy" / side.name,
                log,
                jobs=args.jobs,
                formats=formats,
            )
            for side in (base, head)
        ]
        acc = accuracy.compare(work / "accuracy" / "base", work / "accuracy" / "head")
        # Missing on the head side is a regression the report flags; on the base side,
        # the PR made the variant build.
        head_only = [r for r in acc if any(m.startswith("base") for m in r.get("missing", []))]
        if head_only:
            notes.append(
                "Accuracy measured on the head side only (the variant did not build or run at the "
                "base; see run.log): "
                + ", ".join(f"`{r['key'][0]} {r['key'][1]} dest_acc={r['key'][4]}`" for r in head_only[:8])
            )
        measured = {r["key"][0] for r in acc}
        missing = [o for o in acc_ops if o not in measured]
        if missing and any(code not in (0, runner.NO_TESTS_COLLECTED) for code in exit_codes):
            notes.append(
                "⚠️ The accuracy run failed for "
                + ", ".join(f"`{o}`" for o in missing)
                + ", so this report has no accuracy for it; see run.log."
            )
        elif missing:
            notes.append(
                "No accuracy driver covers "
                + ", ".join(f"`{o}`" for o in missing)
                + " (not an elementwise function of its inputs, or the harness cannot feed it)."
            )
        for cov in sorted({(r["key"][0], r["coverage"]) for r in acc if r.get("coverage")}):
            notes.append(f"`{cov[0]}` accuracy: {cov[1]}.")
    if "Typecast" in ops:
        notes.append("Typecast accuracy is not in this version of the report; its perf is.")
    if args.simulator:
        notes.append("Run on ttsim, the functional simulator: accuracy only, no perf.")

    summary = _write_summary(
        args,
        plan,
        work,
        t0,
        detection=found,
        ops={"measured": ops, "not_covered": not_covered, "why": why},
        perf=perf_rows,
        accuracy=acc,
        harness_changed=harness,
        notes=notes,
    )
    if args.check:
        found = report.findings([summary])
        for f in found:
            print(f"{'NOT RUN' if f['kind'] == 'not run' else 'REGRESSION'} {f['text']}")
        if found:
            sys.exit(1)
        print("no regressions")


def _write_summary(args, plan, work, t0, **results):
    """Write this run's summary-<arch>.json and its report-<arch>.md; return the summary."""
    cmd = " ".join(["python3", ".github/llk_sfpu_report/cli.py", *sys.argv[1:]])
    summary = {
        "arch": args.arch,
        "pr_number": args.pr,
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
        "iterations": args.iterations,
        "thresholds": THRESHOLDS,
        **results,
        "commands": [cmd],
        "seconds": round(time.time() - t0),
    }
    out = work / f"summary-{args.arch}.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    (work / f"report-{args.arch}.md").write_text(report.render([summary]))
    print(f"wrote {out} in {summary['seconds']} s")
    return summary


def cmd_rerender(args):
    """Recompute the comparison from a finished run's raw data; measure nothing."""
    work = Path(args.work).resolve()
    path = work / f"summary-{args.arch}.json"
    summary = json.loads(path.read_text())
    for family in summary["perf"]:
        runs = {
            sched: {
                side: sorted((work / "perf" / family / sched / side).glob("run_*/*.csv")) for side in ("base", "head")
            }
            for sched in perf.SCHEDULES
            if (work / "perf" / family / sched).is_dir()
        }
        verdicts = perf.compare(runs, THRESHOLDS[args.arch], THRESHOLDS["min_cycles"])
        summary["perf"][family] = perf.rows(runs, verdicts)
    if (work / "accuracy" / "head").is_dir():
        summary["accuracy"] = accuracy.compare(work / "accuracy" / "base", work / "accuracy" / "head")
    path.write_text(json.dumps(summary, indent=1, default=str))
    (work / f"report-{args.arch}.md").write_text(report.render([summary]))
    print(f"rewrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(runner.TOOL_ROOT))
    ap.add_argument("--arch", required=True, choices=["wormhole", "blackhole"])
    ap.add_argument("--head", required=True, help="PR head: sha or ref")
    ap.add_argument("--base", help="baseline ref (default: merge-base with --main-ref)")
    ap.add_argument("--main-ref", default="origin/main")
    ap.add_argument("--work", default="/tmp/llk-sfpu-report")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--mode", choices=["merge-base", "rebase"], default="merge-base")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect", help="list the ops whose machine code changed")
    run = sub.add_parser("run", help="detect, measure perf and accuracy, write summary + report")
    run.add_argument("--ops", help="comma-separated MathOperation names; skips detection")
    run.add_argument("--iterations", type=int, default=3)
    run.add_argument("--no-perf", action="store_true", help="accuracy only")
    run.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the report has a ⚠️ (a regression, or an op that did not run on the "
        "PR side), 2 when nothing could be measured",
    )
    run.add_argument("--pr", help="PR number, for the reproduce commands in the report")
    run.add_argument(
        "--formats",
        help="only these input formats, e.g. Float16_b,Float32 (perf and accuracy)",
    )
    run.add_argument(
        "--simulator",
        action="store_true",
        help="run on ttsim ($TT_METAL_SIMULATOR): implies --no-perf; for developing the tool",
    )
    run.add_argument("--run-url")
    run.add_argument("--head-moved-to", help="the PR's current head, if it moved after the command")
    sub.add_parser("rerender", help="recompute summary + report from a finished run's data")
    args = ap.parse_args(argv)
    {"detect": cmd_detect, "run": cmd_run, "rerender": cmd_rerender}[args.cmd](args)


if __name__ == "__main__":
    main()
