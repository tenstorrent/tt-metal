# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check the LLK nightlies of one day and build a Slack health message.

One run of this covers the whole day. It is scheduled after every nightly is
expected to be done, rather than driven by their completions: the nightlies
start two hours apart, so the first one to finish would ask about a run that
does not exist yet and report it missing.
"""

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys

import yaml

NIGHTLIES = (
    ("SoL", "llk-perf.yaml"),
    ("non-SoL", "llk-perf-gate.yaml"),
)
OTHER_NIGHTLIES = (
    ("bit-exact", "llk-bit-exact.yaml"),
    ("e2e", "llk-e2e.yaml"),
)
HEALTH = "llk-perf-nightly-health.yaml"
WORKFLOWS = ".github/workflows"
VERDICT = re.compile(r"verdict=(\w+) regressions=(\d+)")
BASELINE = re.compile(r"(?:^|Z )baseline \S+ \(([0-9a-f]+), (\d{4}-\d{2}-\d{2})", re.M)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LEG = re.compile(r"(llk_perf_\w+ group \d+/\d+)")
ARCHES = ("blackhole", "wormhole")
SHORT_SHA = 11
ROLLUP_JOBS = ("summary", "Report status")


def _gh(*args):
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout


def _json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _time(stamp):
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _entries(tests_yaml):
    with open(tests_yaml) as fh:
        return yaml.safe_load(fh)


def leg_limits(tests_yaml):
    return {
        e["name"]: max(s["timeout"] for s in e["skus"].values())
        for e in _entries(tests_yaml)
    }


def owners(tests_yaml):
    return sorted({e["owner_id"] for e in _entries(tests_yaml) if e.get("owner_id")})


def payloads(text, problems, dm, channel, owner_ids):
    out = [{"channel": dm, "text": text}]
    if channel:
        cc = " ".join(f"<@{o}>" for o in owner_ids)
        out.append(
            {
                "channel": channel,
                "text": f"{text}\n\ncc {cc}" if problems and cc else text,
            }
        )
    return out


def scheduled_run(repo, workflow, day):
    runs = _json_lines(
        _gh(
            "api",
            f"repos/{repo}/actions/workflows/{workflow}/runs?event=schedule&created={day}&per_page=10",
            "--jq",
            ".workflow_runs[] | {id, html_url, head_sha, status, conclusion, updated_at} | @json",
        )
    )
    return runs[0] if runs else None


def disabled_workflows(repo, workflows_dir):
    ours = {
        f"{WORKFLOWS}/{p.name}"
        for p in pathlib.Path(workflows_dir).iterdir()
        if p.name.startswith("llk-") or "quasar" in p.name
    }
    states = _json_lines(
        _gh(
            "api",
            "--paginate",
            f"repos/{repo}/actions/workflows?per_page=100",
            "--jq",
            ".workflows[] | {path, state} | @json",
        )
    )
    return sorted(
        w["path"].rsplit("/", 1)[1]
        for w in states
        if w["path"] in ours and w["state"] != "active"
    )


def run_jobs(repo, run_id):
    return _json_lines(
        _gh(
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&filter=latest",
            "--jq",
            ".jobs[] | {id, name, conclusion, started_at, completed_at} | @json",
        )
    )


def job_log(repo, job_id):
    try:
        log = _gh(
            "api",
            "--allow-escape-sequences",
            f"repos/{repo}/actions/jobs/{job_id}/logs",
        )
    except subprocess.CalledProcessError as e:
        print(f"no log for job {job_id}: {e.stderr.strip()}", file=sys.stderr)
        return ""
    return ANSI.sub("", log)


def gate_result(log, day):
    verdict = VERDICT.search(log)
    baseline = BASELINE.search(log)
    return {
        "verdict": verdict.group(1) if verdict else "no verdict",
        "regressions": int(verdict.group(2)) if verdict else 0,
        "baseline": baseline.group(1)[:SHORT_SHA] if baseline else None,
        "baseline_age": (
            (day - dt.date.fromisoformat(baseline.group(2))).days if baseline else None
        ),
    }


def _is_gate(job):
    return "Compare" in job["name"] and job["conclusion"] not in (None, "skipped")


def _leg_problem(leg):
    if leg["conclusion"] == "success":
        return None
    if leg["used"] is not None and leg["used"] >= 100:
        return "timed out"
    return leg["conclusion"] or "not finished"


def assess(name, run, jobs, logs, limits, day):
    report = {
        "name": name,
        "run": run,
        "legs": [],
        "upload": None,
        "gates": {},
        "problems": [],
    }
    if run is None:
        report["problems"].append(f"{name}: no scheduled run")
        return report
    if run["status"] != "completed":
        report["problems"].append(f"{name}: run still {run['status']}")
    other_failures = []
    for job in jobs:
        leg = LEG.search(job["name"])
        if leg:
            minutes = None
            if job["started_at"] and job["completed_at"]:
                minutes = (
                    _time(job["completed_at"]) - _time(job["started_at"])
                ).total_seconds() / 60
            limit = limits.get(leg.group(1))
            used = (
                round(100 * minutes / limit) if minutes is not None and limit else None
            )
            report["legs"].append(
                {"leg": leg.group(1), "conclusion": job["conclusion"], "used": used}
            )
        elif "Upload perf data to the warehouse" in job["name"]:
            report["upload"] = job["conclusion"]
            if job["conclusion"] != "success":
                report["problems"].append(
                    f"{name}: upload {job['conclusion'] or 'not finished'}"
                )
        elif _is_gate(job):
            arch = next((a for a in ARCHES if a in job["name"]), "?")
            gate = gate_result(logs.get(job["id"], ""), day)
            report["gates"][arch] = gate
            if gate["verdict"] != "clean":
                count = (
                    f", {gate['regressions']:,} points" if gate["regressions"] else ""
                )
                report["problems"].append(
                    f"{name} {arch}: gate {gate['verdict']}{count}"
                )
        elif job["conclusion"] not in (None, "success", "skipped") and not any(
            rollup in job["name"] for rollup in ROLLUP_JOBS
        ):
            other_failures.append(f"{name}: {job['name']} {job['conclusion']}")
    report["legs"].sort(key=lambda l: l["leg"])
    report["problems"][:0] = [
        f"{name}: {l['leg']} {_leg_problem(l)}"
        for l in report["legs"]
        if _leg_problem(l)
    ]
    report["problems"] += other_failures
    for arch in ARCHES:
        if run["status"] == "completed" and arch not in report["gates"]:
            report["problems"].append(f"{name} {arch}: no gate verdict")
    if run["conclusion"] not in (None, "success") and not report["problems"]:
        report["problems"].append(f"{name}: run {run['conclusion']}")
    return report


def assess_plain(name, run, jobs):
    report = {"name": name, "run": run, "legs": [], "upload": None, "gates": {}}
    if run is None:
        report["problems"] = [f"{name}: no scheduled run"]
        return report
    problems = [
        f"{name}: {j['name']} {j['conclusion'] or 'not finished'}"
        for j in jobs
        if j["conclusion"] not in ("success", "skipped")
        and not any(rollup in j["name"] for rollup in ROLLUP_JOBS)
    ]
    if run["status"] != "completed":
        problems.append(f"{name}: run still {run['status']}")
    elif run["conclusion"] != "success" and not problems:
        problems.append(f"{name}: run {run['conclusion']}")
    report["problems"] = problems
    report["plain"] = True
    return report


def read_extras(paths):
    """Each report is {"name", "problems", "lines"}."""
    out = []
    for path in paths:
        try:
            with open(path) as fh:
                out.append(json.load(fh))
        except (OSError, ValueError) as e:
            out.append(
                {"name": path, "problems": [f"{path}: no report ({e})"], "lines": []}
            )
    return out


def _legs_line(legs):
    parts = []
    for arch, short in (("blackhole", "BH"), ("wormhole", "WH")):
        mine = [l for l in legs if f"_{arch} " in l["leg"]]
        if mine:
            cells = " ".join(
                f"{l['leg'].split()[-1]} {l['used'] if l['used'] is not None else '?'}%"
                + ("" if l["conclusion"] == "success" else " ✗")
                for l in mine
            )
            parts.append(f"{short}: {cells}")
    return " | ".join(parts) or "none"


def build_text(day, reports, extras=()):
    problems = [p for r in [*reports, *extras] for p in r["problems"]]
    head = ":white_check_mark:" if not problems else ":red_circle:"
    lines = [
        f"{head} *LLK nightly health, {day}*"
        + (f": {len(problems)} problem(s)" if problems else ": healthy")
    ]
    lines += [f"• {p}" for p in problems]
    for r in reports:
        run = r["run"]
        lines.append("")
        if run is None:
            lines.append(f"*{r['name']}*: no scheduled run")
            continue
        lines.append(
            f"*{r['name']}* <{run['html_url']}|run> · `{run['head_sha'][:SHORT_SHA]}` · {run['conclusion'] or run['status']}"
        )
        if r.get("plain"):
            continue
        lines.append(f"legs, % of limit used: {_legs_line(r['legs'])}")
        lines.append(f"upload: {r['upload'] or 'not run'}")
        for arch, g in sorted(r["gates"].items()):
            age = (
                f"{g['baseline_age']} days old"
                if g["baseline_age"] is not None
                else "age unknown"
            )
            count = f", {g['regressions']:,} points" if g["regressions"] else ""
            lines.append(
                f"gate {arch}: {g['verdict']}{count} · baseline `{g['baseline'] or '?'}`, {age}"
            )
    for e in extras:
        if e["lines"]:
            lines += ["", f"*{e['name']}*", *e["lines"]]
    return "\n".join(lines)


def watchdog(repo, day, workflows_dir, dm, out):
    problems = [
        f"{w} is disabled"
        for w in disabled_workflows(repo, workflows_dir)
        if w == HEALTH
    ]
    run = scheduled_run(repo, HEALTH, day)
    if run is None:
        problems.append("no health check ran today")
    elif run["conclusion"] != "success":
        problems.append(
            f"the health check ended {run['conclusion'] or run['status']}: {run['html_url']}"
        )
    if problems:
        with open(out, "w") as fh:
            json.dump(
                [
                    {
                        "channel": dm,
                        "text": f":rotating_light: *LLK nightly watchdog, {day}*\n"
                        + "\n".join(f"• {p}" for p in problems),
                    }
                ],
                fh,
            )
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dm", required=True, help="Slack user id that always gets it")
    ap.add_argument("--channel", default="", help="channel that also gets it")
    ap.add_argument("--tests-yaml", default="tests/pipeline_reorg/llk_perf_tests.yaml")
    ap.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    ap.add_argument("--out", default="health_payloads.json")
    ap.add_argument("--workflows", default=WORKFLOWS)
    ap.add_argument("--extra", nargs="*", default=[], help="reports of other checks")
    ap.add_argument(
        "--watchdog", action="store_true", help="check only the health check"
    )
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.date)
    if a.watchdog:
        for p in watchdog(a.repo, a.date, a.workflows, a.dm, a.out):
            print(p)
        return 0
    limits = leg_limits(a.tests_yaml)
    reports = []
    for name, workflow in NIGHTLIES:
        run = scheduled_run(a.repo, workflow, a.date)
        jobs = run_jobs(a.repo, run["id"]) if run else []
        logs = {j["id"]: job_log(a.repo, j["id"]) for j in jobs if _is_gate(j)}
        reports.append(assess(name, run, jobs, logs, limits, day))
    for name, workflow in OTHER_NIGHTLIES:
        run = scheduled_run(a.repo, workflow, a.date)
        reports.append(
            assess_plain(name, run, run_jobs(a.repo, run["id"]) if run else [])
        )
    disabled = disabled_workflows(a.repo, a.workflows)
    extras = read_extras(a.extra)
    if disabled:
        extras.insert(
            0,
            {
                "name": "Disabled workflows",
                "problems": [
                    f"{len(disabled)} LLK workflow(s) disabled: {', '.join(disabled)}"
                ],
                "lines": [],
            },
        )
    text = build_text(a.date, reports, extras)
    problems = any(r["problems"] for r in [*reports, *extras])
    with open(a.out, "w") as fh:
        json.dump(payloads(text, problems, a.dm, a.channel, owners(a.tests_yaml)), fh)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
