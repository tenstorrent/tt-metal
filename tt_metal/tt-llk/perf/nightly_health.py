# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check the LLK perf nightlies of one day and build a Slack health message."""

import argparse
import datetime as dt
import json
import re
import subprocess
import sys

import yaml

NIGHTLIES = (
    ("SoL", "llk-perf.yaml"),
    ("non-SoL", "llk-perf-gate.yaml"),
)
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


def build_text(day, reports):
    problems = [p for r in reports for p in r["problems"]]
    head = ":white_check_mark:" if not problems else ":red_circle:"
    lines = [
        f"{head} *LLK perf nightly health, {day}*"
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
    return "\n".join(lines)


def last_to_finish(runs, trigger_run):
    if any(r is None or r["status"] != "completed" for r in runs):
        return False
    return max(runs, key=lambda r: r["updated_at"])["id"] == trigger_run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dm", required=True, help="Slack user id that always gets it")
    ap.add_argument("--channel", default="", help="channel that also gets it")
    ap.add_argument("--tests-yaml", default="tests/pipeline_reorg/llk_perf_tests.yaml")
    ap.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    ap.add_argument(
        "--trigger-run",
        type=int,
        help="send only once every nightly of the day is done and this run finished last",
    )
    ap.add_argument("--out", default="health_payloads.json")
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.date)
    runs = [scheduled_run(a.repo, workflow, a.date) for _, workflow in NIGHTLIES]
    if a.trigger_run and not last_to_finish(
        [r for r in runs if r is not None], a.trigger_run
    ):
        print("not the last nightly of the day to finish; that one sends the message")
        return 0
    limits = leg_limits(a.tests_yaml)
    reports = []
    for (name, _), run in zip(NIGHTLIES, runs):
        jobs = run_jobs(a.repo, run["id"]) if run else []
        logs = {j["id"]: job_log(a.repo, j["id"]) for j in jobs if _is_gate(j)}
        reports.append(assess(name, run, jobs, logs, limits, day))
    text = build_text(a.date, reports)
    problems = any(r["problems"] for r in reports)
    with open(a.out, "w") as fh:
        json.dump(payloads(text, problems, a.dm, a.channel, owners(a.tests_yaml)), fh)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
