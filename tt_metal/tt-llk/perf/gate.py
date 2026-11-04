# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The LLK perf gate: compare one arch under gate_policy.yaml and write its verdict.

    gate.py compare --gate merge --arch wormhole --current 'cur/**/*.csv' --baseline 'base/**/*.csv'
    gate.py verdict verdict-wormhole.json      (exit 1 when the verdict fails the job)
    gate.py baseline --gate nightly --arch wormhole  (read the policy's baseline from the warehouse)
    gate.py policy --gate merge --arch wormhole  (the compare arguments, for stage 2 in a leg)
    gate.py arches --gate merge --stage2         (the arches that run stage 2)

verdict-<arch>.json is the only thing the later steps read: the PR comment, Slack and the job result.
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regression_compare as rc  # noqa: E402

POLICY = Path(__file__).with_name("gate_policy.yaml")
SCHEMA = 1


def policy(gate, arch, path=POLICY):
    """The rules one gate applies to one arch."""
    data = yaml.safe_load(Path(path).read_text())
    g = data["gates"][gate]
    a = g["arches"][arch]
    return {
        "gate": gate,
        "title": g["title"],
        "speed_of_light": g["speed_of_light"],
        "baseline": g["baseline"],
        "threshold": a["threshold"],
        "modules": a.get("modules", {}),
        "run_types": a["run_types"],
        "min_cycles": data["min_cycles"],
        "block": a.get("block", False),
        "stage2": a.get("stage2", False),
    }


def _csvs(pattern):
    return sorted(
        p for p in glob.glob(pattern, recursive=True) if not p.endswith(".post.csv")
    )


def _speed_of_light(paths):
    """The mode the first CSV that names one was measured in, or None."""
    for path in paths:
        with open(path, newline="") as fh:
            row = next(csv.DictReader(fh), None)
        if row and "speed_of_light" in row:
            return str(row["speed_of_light"]).strip().lower() == "true"
    return None


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def decide(p, *, incomplete="", baseline=None, result=None, error=""):
    """``(status, reason)``: pass, regression, incomplete, no-baseline or no-compare."""
    if incomplete:
        return "incomplete", f"{incomplete} The gate compared nothing."
    if baseline is None:
        return (
            "no-compare",
            "The baseline read did not finish, so the gate has no verdict.",
        )
    if not baseline.get("have"):
        return "no-baseline", "No baseline was found, so the gate compared nothing."
    if error:
        return "no-compare", error
    if result["regressions"]:
        return "regression", ""
    return "pass", ""


def fails(verdict):
    """Whether the job fails: on anything that is not a result, and on a blocking regression."""
    status = verdict["status"]
    if status == "pass":
        return False
    if status == "regression":
        return verdict["policy"]["block"]
    return True


def compare(a):
    p = policy(a.gate, a.arch, a.policy)
    meta = _read_json(a.baseline_meta)
    baseline = None
    if meta is not None:
        have = Path(a.have_baseline).read_text().strip() == "true"
        baseline = {**meta, "have": have}

    result, error = None, ""
    current, base = _csvs(a.current), _csvs(a.baseline)
    if not a.incomplete and baseline and baseline["have"]:
        sol = _speed_of_light(current)
        if not current:
            error = f"No {a.arch} perf CSVs for the current side, so the gate measured nothing."
        elif sol is not None and sol != p["speed_of_light"]:
            error = (
                f"The current side is speed_of_light={sol}, the {a.gate} gate compares "
                f"speed_of_light={p['speed_of_light']}; the two share no points."
            )
        else:
            result = rc.compare_runs(
                current,
                base,
                threshold=p["threshold"],
                min_cycles=p["min_cycles"],
                run_types=",".join(p["run_types"]),
                module_thresholds=p["modules"],
            )
            if not result["records"]:
                error = "No point was compared: the two sides share no points."

    status, reason = decide(
        p, incomplete=a.incomplete, baseline=baseline, result=result, error=error
    )
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if result and result["records"]:
        b = p["baseline"]
        label = (
            f"pinned baseline, run {b['run']}"
            if isinstance(b, dict)
            else b.replace("_", " ")
        )
        if baseline.get("stale"):
            label += ", stale"
        report = rc.render_report(
            result,
            threshold=p["threshold"],
            min_cycles=p["min_cycles"],
            test="LLK perf",
            baseline_sha=baseline.get("commit_sha") or "unknown",
            current_sha=a.current_sha,
            baseline_iters=len(base),
            current_iters=len(current),
            baseline_label=label,
            current_label=a.current_label,
            module_thresholds=p["modules"],
            arch=a.arch,
        )
        (out / "perf_gate_report.md").write_text(report + "\n")
        rc._write_points_csv(result["records"], out / "perf_gate_report.points.csv")
        rc._write_points_csv(
            result["regressions"], out / "perf_gate_report.regressions.csv"
        )
    verdict = {
        "schema": SCHEMA,
        "gate": a.gate,
        "arch": a.arch,
        "status": status,
        "reason": reason,
        "policy": p,
        "baseline": {k: v for k, v in (baseline or {}).items() if k != "have"},
        "current": {"commit_sha": a.current_sha, "label": a.current_label},
        "counts": {
            "points": len(result["records"]) if result else 0,
            "regressions": len(result["regressions"]) if result else 0,
            "improvements": len(result["improvements"]) if result else 0,
            "new": len(result["new_points"]) if result else 0,
        },
    }
    verdict["fails"] = fails(verdict)
    path = out / f"verdict-{a.arch}.json"
    path.write_text(json.dumps(verdict, indent=1) + "\n")
    print(f"{a.arch}: {status} {verdict['counts']} {reason}".rstrip())
    return 0


def verdict_cmd(a):
    """Print the verdict, add the report to the step summary, and exit 1 when the job fails."""
    v = _read_json(a.path)
    if v is None:
        print(
            f"::error::{a.path} is missing: the compare did not finish, so the gate has no verdict."
        )
        return 1
    report = Path(a.path).with_name("perf_gate_report.md")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and report.exists():
        with open(summary, "a") as fh:
            fh.write(report.read_text())
    status, reason = v["status"], v["reason"]
    level = "error" if v["fails"] else "warning"
    if status == "pass":
        print(f"✅ {v['arch']}: no regressions ({v['counts']['points']} points)")
    elif status == "regression":
        print(
            f"::{level}::{v['arch']}: {v['counts']['regressions']} point(s) regressed."
        )
        if not v["fails"]:
            print("Report-only: the job does not fail.")
    else:
        print(f"::{level}::{v['arch']}: {status}. {reason}")
        print(
            "This is not a performance result: a measuring leg, the baseline or the compare failed."
        )
    return 1 if v["fails"] else 0


def policy_cmd(a):
    p = policy(a.gate, a.arch, a.policy)
    args = ["--threshold", str(p["threshold"]), "--min-cycles", str(p["min_cycles"])]
    if p["modules"]:
        modules = ",".join(f"{m}={t}" for m, t in p["modules"].items())
        args += ["--module-threshold", modules]
    print(" ".join(args))
    return 0


def arches_cmd(a):
    data = yaml.safe_load(Path(a.policy).read_text())
    arches = data["gates"][a.gate]["arches"]
    print(" ".join(k for k, v in arches.items() if v.get("stage2") or not a.stage2))
    return 0


def baseline_cmd(a):
    """Read this gate's baseline from the warehouse into ``--out``."""
    import baseline_from_warehouse as bw

    p = policy(a.gate, a.arch, a.policy)
    b = p["baseline"]
    pipeline, run = (b["pipeline"], b["run"]) if isinstance(b, dict) else (b, "")
    argv = [
        "--arch",
        a.arch,
        "--pipeline",
        pipeline,
        "--speed-of-light",
        str(p["speed_of_light"]).lower(),
        "--exclude-workflow-run",
        a.exclude_run,
        "--wait-minutes",
        a.wait_minutes,
        "--out",
        a.out,
        "--metadata",
        a.metadata,
    ]
    if run:
        argv += ["--run", str(run)]
    if a.commits is not None:
        argv += ["--commits", a.commits]
    return bw.main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default=str(POLICY))
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="compare one arch and write verdict-<arch>.json")
    c.add_argument("--gate", required=True)
    c.add_argument("--arch", required=True)
    c.add_argument("--current", required=True, help="glob of the current side's CSVs")
    c.add_argument("--baseline", required=True, help="glob of the baseline's CSVs")
    c.add_argument("--baseline-meta", default="baseline_metadata.json")
    c.add_argument("--have-baseline", default="have_baseline.txt")
    c.add_argument("--current-sha", default="?")
    c.add_argument("--current-label", default="")
    c.add_argument(
        "--incomplete", default="", help="why the measurement is incomplete, if it is"
    )
    c.add_argument("--out-dir", default=".")
    c.set_defaults(func=compare)

    v = sub.add_parser("verdict", help="exit 1 when the verdict fails the job")
    v.add_argument("path")
    v.set_defaults(func=verdict_cmd)

    b = sub.add_parser("baseline", help="read the gate's baseline from the warehouse")
    b.add_argument("--gate", required=True)
    b.add_argument("--arch", required=True)
    b.add_argument(
        "--commits", default=None, help="newest first; the run at the first is wanted"
    )
    b.add_argument(
        "--exclude-run",
        default="",
        help="this workflow run, whose upload is no baseline",
    )
    b.add_argument("--wait-minutes", default="0")
    b.add_argument("--out", default="baseline_perf/baseline.csv")
    b.add_argument("--metadata", default="baseline_metadata.json")
    b.set_defaults(func=baseline_cmd)

    r = sub.add_parser("arches", help="the arches of one gate")
    r.add_argument("--gate", required=True)
    r.add_argument("--stage2", action="store_true", help="only those with stage2: true")
    r.set_defaults(func=arches_cmd)

    p = sub.add_parser("policy", help="the compare arguments of one gate and arch")
    p.add_argument("--gate", required=True)
    p.add_argument("--arch", required=True)
    p.set_defaults(func=policy_cmd)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
