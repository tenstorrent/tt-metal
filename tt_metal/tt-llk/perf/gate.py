# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The LLK perf gate: compare one arch under gate_policy.yaml and write its verdict.

    gate.py compare --gate merge --arch wormhole --current current_perf --baseline baseline_perf
    gate.py complete --dir perf-data --arch wormhole  (exit 1 unless every leg ran its whole plan)
    gate.py verdict verdict-wormhole.json      (exit 1 when the verdict fails the job)
    gate.py baseline --gate nightly --arch wormhole  (read the policy's baseline from the warehouse)
    gate.py policy --gate merge --arch wormhole  (the compare arguments, for stage 2 in a leg)
    gate.py arches --gate merge --stage2         (the arches that run stage 2)
    gate.py render --gate merge --reports DIR [--comment PR] [--slack]  (one PR comment, one Slack post)

verdict-<arch>.json is the only thing the later steps read: the PR comment, Slack and the job result.
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement_check as pc  # noqa: E402
import regression_compare as rc  # noqa: E402

POLICY = Path(__file__).with_name("gate_policy.yaml")
WAIVERS = Path(__file__).with_name("waivers.yaml")
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
        "baseline_coverage": data["baseline_coverage"],
        "block": a.get("block", False),
        "stage2": a.get("stage2", False),
    }


def _csvs(directory):
    return sorted(
        p
        for p in glob.glob(f"{directory}/**/*.csv", recursive=True)
        if not p.endswith(".post.csv")
    )


def waivers(arch, path=WAIVERS):
    """The waivers.yaml entries for ``arch``; a malformed entry is an error, never skipped."""
    entries = yaml.safe_load(Path(path).read_text()).get("waivers") or []
    for i, w in enumerate(entries, 1):
        lacking = [k for k in ("test", "max_delta", "reason", "issue") if not w.get(k)]
        if lacking:
            raise ValueError(f"waivers.yaml entry {i} lacks {', '.join(lacking)}.")
    return [w for w in entries if w.get("arch", arch) == arch]


def _waived_by(point, entries):
    """The first entry that accepts this regressed point, or None."""
    config = {k: rc._short(v) for k, v in point["config"]}
    for w in entries:
        if (
            w["test"] == point[rc.MODULE_COL]
            and w.get("run_type", point["run_type"]) == point["run_type"]
            and w.get("marker", point["marker"]) == point["marker"]
            and all(
                str(config.get(k)) == str(v) for k, v in (w.get("config") or {}).items()
            )
            and point["delta"] <= w["max_delta"]
        ):
            return w
    return None


def apply_waivers(result, entries):
    """Move the regressions a waiver accepts to ``result["waived"]``."""
    kept, waived = [], []
    for point in result["regressions"]:
        w = _waived_by(point, entries)
        if w:
            waived.append({**point, "waiver": w})
        else:
            kept.append(point)
    result["regressions"], result["waived"] = kept, waived
    return result


def _waived_table(waived):
    lines = [
        "## Waived regressions (waivers.yaml)",
        "",
        "| test | marker | run type | Δ | allowed | issue | reason |",
        "|---|---|---|--:|--:|---|---|",
    ]
    for p in sorted(waived, key=lambda p: -p["delta"]):
        w = p["waiver"]
        lines.append(
            f"| {p[rc.MODULE_COL]} | {p['marker']} | {p['run_type']} | {p['delta'] * 100:+.1f}% | "
            f"{w['max_delta'] * 100:.0f}% | #{w['issue']} | {w['reason']} |"
        )
    return "\n".join(lines)


def complete(directory, arch):
    """ "" when every leg of ``arch`` under ``directory`` ran its whole plan; else why not.

    Each leg writes plan.json (its node ids, its group of N) and ran.txt (what passed or skipped).
    """
    plans = []
    for path in sorted(glob.glob(f"{directory}/**/plan.json", recursive=True)):
        plan = _read_json(path)
        if plan and plan.get("arch") == arch:
            plans.append((Path(path), plan))
    if not plans:
        return f"No {arch} leg left a plan (plan.json), so nothing shows what was measured."
    splits = {plan["groups"] for _, plan in plans}
    if len(splits) != 1:
        return (
            f"The {arch} legs were cut {' and '.join(map(str, sorted(splits)))} ways."
        )
    n = splits.pop()
    problems = [
        f"leg {g}/{n} left no data"
        for g in sorted(set(range(1, n + 1)) - {plan["group"] for _, plan in plans})
    ]
    for path, plan in plans:
        ran = path.with_name("ran.txt")
        ran = set(ran.read_text().splitlines()) if ran.exists() else set()
        left = [node for node in plan["nodes"] if node not in ran]
        if left:
            problems.append(
                f"leg {plan['group']}/{n} ran {len(plan['nodes']) - len(left)} of "
                f"{len(plan['nodes'])} tests (first not run: {left[0]})"
            )
    if problems:
        return f"A {arch} measuring leg did not complete: {'; '.join(problems)}."
    return ""


def legs(directory, arch):
    """Each leg of ``arch``: its group, its time against its limit (leg.json), and whether it ran its plan."""
    out = []
    for path in sorted(glob.glob(f"{directory}/**/leg.json", recursive=True)):
        leg = _read_json(path) or {}
        if leg.get("arch") != arch:
            continue
        plan = _read_json(Path(path).with_name("plan.json")) or {}
        ran = Path(path).with_name("ran.txt")
        ran = set(ran.read_text().splitlines()) if ran.exists() else set()
        out.append(
            {
                "group": leg.get("group"),
                "groups": plan.get("groups", "?"),
                "minutes": leg.get("minutes"),
                "timeout_minutes": leg.get("timeout_minutes"),
                "complete": bool(plan) and set(plan["nodes"]) <= ran,
            }
        )
    return sorted(out, key=lambda leg: leg["group"] or 0)


def _short_baseline(p, result):
    """Why a baseline holds too little of the current run to judge it, or ""."""
    modules = lambda rows: {r[rc.MODULE_COL] for r in rows}  # noqa: E731
    current = modules(result["records"]) | modules(result["new_points"])
    shared = modules(result["records"])
    if len(shared) < p["baseline_coverage"] * len(current):
        return (
            f"The baseline holds points of {len(shared)} of the {len(current)} test modules measured now: "
            "a partly uploaded or loaded run, or tests whose parameters changed since the baseline "
            "(move the pin in gate_policy.yaml)."
        )
    new = sum(r[rc.MODULE_COL] in shared for r in result["new_points"])
    if new > (1 - p["baseline_coverage"]) * (len(result["records"]) + new):
        return (
            f"{new} points of modules the baseline has are missing from it: a partly uploaded or "
            "loaded run, or tests whose parameters changed since the baseline (move the pin in gate_policy.yaml)."
        )
    return ""


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
        # With stage 2 (merge gate), a regression blocks only when stage 2 confirms it.
        stage2 = verdict.get("stage2") or {}
        return verdict["policy"]["block"] and stage2.get("blocks", True)
    return True


def _run_commit(run_id):
    """The commit a workflow run measured, or "" when the API does not answer."""
    out = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{run_id}",
        ]
        + ["--jq", ".head_sha"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def _baseline(a, base):
    """What the baseline is, or None when its read did not finish. ``have``: it holds points."""
    if a.baseline_run:
        meta = {"run_id": a.baseline_run, "commit_sha": _run_commit(a.baseline_run)}
    elif a.baseline_commit:
        meta = {"commit_sha": a.baseline_commit}
    else:
        meta = _read_json(a.baseline_meta)
    return None if meta is None else {**meta, "have": bool(base)}


def _legs(current, arch):
    """The measuring legs of ``arch`` that wrote data: one perf_data/runs/<run>-<arch>-<job> directory each."""
    return {
        part
        for path in current
        for part in Path(path).parts
        if re.fullmatch(rf"\d+-{arch}-\d+", part)
    }


def _stage2(a, current, regressions_csv, gate_base):
    """The stage-2 section of the report, from the legs' stage2.json. Report-only; a fault is a note."""
    try:
        paths = sorted(glob.glob(a.stage2, recursive=True))
        results, notes = pc.gate(
            [json.loads(Path(path).read_text()) for path in paths],
            len(_legs(current, a.arch)),
            (
                pc.read_candidates(regressions_csv)
                if Path(regressions_csv).exists()
                else []
            ),
            gate_base,
            [
                Path(b).read_text().strip()
                for b in sorted(
                    glob.glob(
                        a.stage2.replace("stage2.json", "stage2.base"), recursive=True
                    )
                )
            ],
        )
        report = pc.render(results, a.arch)
        if notes:
            report += "\n\n" + "\n".join(f"> ⚠️ {n}" for n in notes)
        counts = collections.Counter(r["outcome"] for r in results)
        stage2 = {"legs": len(paths), "notes": notes, "outcomes": dict(counts)}
        # Stage 2 clears a regression only when every leg reported and none of its rows blocks.
        stage2["blocks"] = bool(notes) or pc.blocks(results)
        return report, stage2
    except (
        Exception
    ) as e:  # noqa: BLE001 - report-only: any fault becomes a line in the report
        return f"Stage 2 could not build its report: {e!r}", {
            "error": repr(e),
            "blocks": True,
        }


def compare(a):
    p = policy(a.gate, a.arch, a.policy)
    result, error, stage2 = None, "", None
    current, base = _csvs(a.current), _csvs(a.baseline)
    baseline = _baseline(a, base)
    incomplete = complete(a.current, a.arch)
    if not incomplete and (a.baseline_commit or a.baseline_run):
        # A branch point or a run measured in CI: its legs left a plan too.
        incomplete = complete(a.baseline, a.arch).replace(
            "measuring leg", "baseline leg"
        )
    if not incomplete and baseline and baseline["have"]:
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
            elif not (a.baseline_commit or a.baseline_run):
                error = _short_baseline(p, result)
            if not error:
                try:
                    apply_waivers(result, waivers(a.arch, a.waivers))
                except (OSError, ValueError, yaml.YAMLError) as e:
                    error = f"The waivers could not be read: {e}"

    status, reason = decide(
        p, incomplete=incomplete, baseline=baseline, result=result, error=error
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
        if result.get("waived"):
            report += "\n\n" + _waived_table(result["waived"])
        missing = result["missing_points"]
        if missing:
            names = sorted({r[rc.MODULE_COL] for r in missing})
            report += (
                f"\n\n{len(missing)} baseline point(s) were not measured now, in "
                f"{', '.join(names[:10])}{' …' if len(names) > 10 else ''} (removed or renamed tests?)."
            )
        rc._write_points_csv(result["records"], out / "perf_gate_report.points.csv")
        regressions_csv = out / "perf_gate_report.regressions.csv"
        rc._write_points_csv(result["regressions"], regressions_csv)
        if a.stage2 and p["stage2"]:
            section, stage2 = _stage2(
                a, current, regressions_csv, baseline.get("commit_sha", "")
            )
            report += "\n\n" + section
        (out / "perf_gate_report.md").write_text(report + "\n")
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
            "missing": len(result["missing_points"]) if result else 0,
            "waived": len(result.get("waived", [])) if result else 0,
        },
        "stage2": stage2,
        "legs": legs(a.current, a.arch),
    }
    verdict["fails"] = fails(verdict)
    path = out / f"verdict-{a.arch}.json"
    path.write_text(json.dumps(verdict, indent=1) + "\n")
    print(f"{a.arch}: {status} {verdict['counts']} {reason}".rstrip())
    return 0


COMMENT_MARKER = "<!-- llk-perf-gate -->"


def _gh(*args, data=None):
    out = subprocess.run(
        ["gh", "api", *args], input=data, capture_output=True, text=True, check=True
    )
    return out.stdout


def _queue_pr():
    """The PR a merge-queue ref (gh-readonly-queue/main/pr-123-<sha>) carries, or {}."""
    m = re.search(
        r"gh-readonly-queue/[^/]+/pr-(\d+)-", os.environ.get("GITHUB_REF", "")
    )
    if not m:
        return {}
    repo = os.environ["GITHUB_REPOSITORY"]
    try:
        pr = json.loads(_gh(f"repos/{repo}/pulls/{m.group(1)}"))
    except (subprocess.CalledProcessError, ValueError):
        pr = {}
    return {
        "pr_number": m.group(1),
        "pr_url": f"https://github.com/{repo}/pull/{m.group(1)}",
        "pr_title": pr.get("title", ""),
        "author": (pr.get("user") or {}).get("login", ""),
    }


def _reports(directory, gate):
    """{arch: (verdict or None, its report directory)} for every arch of ``gate``."""
    found = {}
    for path in glob.glob(f"{directory}/**/verdict-*.json", recursive=True):
        v = _read_json(path)
        if v:
            found[v["arch"]] = (v, Path(path).parent)
    arches = yaml.safe_load(POLICY.read_text())["gates"][gate]["arches"]
    return {arch: found.get(arch, (None, None)) for arch in arches}


def _comment(reports, directory):
    """The PR comment: each arch's report, then its stage 2, all rendered from the files."""
    parts = [COMMENT_MARKER]
    for arch, (v, d) in reports.items():
        report = d / "perf_gate_report.md" if d else None
        if report and report.exists():
            parts.append(report.read_text().strip())
        else:
            status, reason = (
                (v["status"], v["reason"])
                if v
                else ("no verdict", "Its gate job wrote no verdict.")
            )
            parts.append(
                f"# Perf compare — LLK perf — {arch.capitalize()}\n\n**{status}**: {reason}"
            )
        for stage2 in glob.glob(f"{directory}/**/stage2-{arch}.md", recursive=True):
            parts.append(Path(stage2).read_text().strip())
    return "\n\n".join(parts) + "\n"


def _post_comment(pr, body):
    repo = os.environ["GITHUB_REPOSITORY"]
    comments = json.loads(
        _gh("--paginate", "--slurp", f"repos/{repo}/issues/{pr}/comments?per_page=100")
    )
    mine = [
        c
        for page in comments
        for c in page
        if c["user"]["type"] == "Bot" and c["body"].startswith(COMMENT_MARKER)
    ]
    payload = json.dumps({"body": body})
    if mine:
        _gh(
            "-X",
            "PATCH",
            f"repos/{repo}/issues/comments/{mine[0]['id']}",
            "--input",
            "-",
            data=payload,
        )
    else:
        _gh(
            "-X",
            "POST",
            f"repos/{repo}/issues/{pr}/comments",
            "--input",
            "-",
            data=payload,
        )


def _slack(reports, ctx, post_pass):
    """``(should_post, text)``: one post for every arch, with how a regressed point moved on the others."""
    import gate_slack_combine as combine
    import gate_slack_notify as notify

    parts, points = [], {}
    for arch, (v, d) in reports.items():
        status, arch_ctx = notify.from_verdict(v)
        arch_ctx = {
            **ctx,
            "arch": arch,
            **{k: x for k, x in arch_ctx.items() if x != ""},
        }
        rows = (
            notify.read_regressions(str(d / "perf_gate_report.regressions.csv"))
            if d and status == "regressed"
            else []
        )
        if d:
            points[arch] = str(d / "perf_gate_report.points.csv")
        parts.append(
            {
                "status": status,
                "rows": rows,
                "ctx": arch_ctx,
                "should_post": status != "clean" or post_pass,
            }
        )
    return any(p["should_post"] for p in parts), combine.build_text(parts, points)


def _post_slack(channel, text):
    import urllib.request

    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps({"channel": channel, "text": text}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
            "Content-type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.load(response)
    if not answer.get("ok"):
        raise SystemExit(f"::error::Slack rejected the gate post: {answer}")


def render_cmd(a):
    """The PR comment, the Slack post and the step summary, from the verdicts and reports only."""
    reports = _reports(a.reports, a.gate)
    body = _comment(reports, a.reports)
    Path(a.out_dir, "comment.md").write_text(body)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(body)
    if a.comment:
        _post_comment(a.comment, body)
        print(f"Commented on PR #{a.comment}.")

    run = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}"
    run += f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    ctx = {
        "run_url": run,
        "report_url": f"{run}#summary",
        "commit": os.environ.get("GITHUB_SHA", ""),
        "trigger": a.trigger,
    }
    ctx.update(_queue_pr())
    should_post, text = _slack(reports, ctx, a.post_pass)
    Path(a.out_dir, "slack.txt").write_text(text + "\n")
    print(text)
    if a.slack and should_post:
        channel = os.environ.get("SLACK_CHANNEL_ID", "")
        if not channel:
            print("SLACK_CHANNEL_ID is unset; nothing is posted.")
            return 0
        _post_slack(channel, text)
        print("Posted to Slack.")
    return 0


def complete_cmd(a):
    reason = complete(a.dir, a.arch)
    print(f"::error::{reason}" if reason else f"Every {a.arch} leg ran its whole plan.")
    return 1 if reason else 0


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
    # The nightly health check reads this line from the job log.
    slack_status = {"regression": "regressed", "pass": "clean"}.get(status, "skipped")
    print(f"verdict={slack_status} regressions={v['counts']['regressions']}")
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

    reuse = Path(a.reuse or "-") / a.arch
    if (reuse / "baseline.csv").exists():
        # The read stage 2 already made, so the gate and stage 2 compare against the same baseline.
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(reuse / "baseline.csv", a.out)
        shutil.copy(reuse / "baseline_metadata.json", a.metadata)
        return 0
    if a.match_ref:
        # Commits and trees only: enough for the merge base and the 200 commits before it.
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        subprocess.run(
            [
                "git",
                "fetch",
                "--no-tags",
                "--filter=blob:none",
                "--depth=250",
                "origin",
                head.stdout.strip(),
            ],
            check=True,
        )
        script = (
            Path(__file__).resolve().parents[3]
            / ".github/scripts/llk-perf-baseline-commits.sh"
        )
        out = subprocess.run([str(script), a.match_ref], capture_output=True, text=True)
        # An empty list is passed on, so the reader skips instead of taking any run.
        a.commits = ",".join(out.stdout.split()) if out.returncode == 0 else ""
        print(f"Expected baseline commit: {a.commits.split(',')[0] or '(none)'}")

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
    c.add_argument(
        "--current", required=True, help="directory of the current side's data"
    )
    c.add_argument("--baseline", required=True, help="directory of the baseline's data")
    c.add_argument(
        "--baseline-meta",
        default="baseline_metadata.json",
        help="what gate.py baseline wrote",
    )
    c.add_argument(
        "--baseline-commit",
        default="",
        help="the baseline is the branch point at this commit",
    )
    c.add_argument(
        "--baseline-run",
        default="",
        help="the baseline is this workflow run's perf data",
    )
    c.add_argument(
        "--stage2", default="", help="glob of the legs' stage2.json (merge gate)"
    )
    c.add_argument("--waivers", default=str(WAIVERS), help="the accepted regressions")
    c.add_argument("--current-sha", default="?")
    c.add_argument("--current-label", default="")
    c.add_argument("--out-dir", default=".")
    c.set_defaults(func=compare)

    m = sub.add_parser(
        "complete", help="exit 1 unless every leg of an arch ran its plan"
    )
    m.add_argument("--dir", required=True)
    m.add_argument("--arch", required=True)
    m.set_defaults(func=complete_cmd)

    n = sub.add_parser(
        "render", help="the PR comment and the Slack post, from the verdicts"
    )
    n.add_argument("--gate", required=True)
    n.add_argument(
        "--reports",
        required=True,
        help="directory holding every arch's verdict and report",
    )
    n.add_argument(
        "--comment", default="", help="post or update the comment on this PR"
    )
    n.add_argument("--slack", action="store_true", help="post to SLACK_CHANNEL_ID")
    n.add_argument(
        "--post-pass", action="store_true", help="also post when every arch passed"
    )
    n.add_argument("--trigger", default="nightly", help="what started a run with no PR")
    n.add_argument("--out-dir", default=".")
    n.set_defaults(func=render_cmd)

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
    b.add_argument(
        "--match-ref",
        default="",
        help="the run at the newest perf change at or before this ref",
    )
    b.add_argument(
        "--reuse",
        default="",
        help="directory with <arch>/baseline.csv read earlier in this run",
    )
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
