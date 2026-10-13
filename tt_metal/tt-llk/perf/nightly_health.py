# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check the LLK perf nightlies of one day and build a Slack health message."""

import argparse
import datetime as dt
import json
import re
import subprocess

import yaml

NIGHTLIES = (
    ("SoL", ".github/workflows/llk-perf.yaml"),
    ("non-SoL", ".github/workflows/llk-perf-gate.yaml"),
)
VERDICT = re.compile(r"verdict=(\w+) regressions=(\d+)")
BASELINE = re.compile(r"(?:^|Z )baseline \S+ \(([0-9a-f]+), (\d{4}-\d{2}-\d{2})", re.M)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LEG = re.compile(r"(llk_perf_\w+ group \d+/\d+)")
ARCHES = ("blackhole", "wormhole")


def _gh(*args):
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout


def _json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _time(stamp):
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def leg_limits(tests_yaml):
    with open(tests_yaml) as fh:
        entries = yaml.safe_load(fh)
    return {e["name"]: max(s["timeout"] for s in e["skus"].values()) for e in entries}


def scheduled_run(repo, path, day):
    wf = _gh(
        "api",
        "--paginate",
        f"repos/{repo}/actions/workflows?per_page=100",
        "--jq",
        f'.workflows[] | select(.path == "{path}") | .id',
    ).split()[0]
    runs = _json_lines(
        _gh(
            "api",
            f"repos/{repo}/actions/workflows/{wf}/runs?event=schedule&created={day}&per_page=10",
            "--jq",
            ".workflow_runs[] | {id, html_url, head_sha, status, conclusion} | @json",
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
    return ANSI.sub(
        "",
        _gh(
            "api",
            "--allow-escape-sequences",
            f"repos/{repo}/actions/jobs/{job_id}/logs",
        ),
    )


def gate_result(log, day):
    verdict = VERDICT.search(log)
    baseline = BASELINE.search(log)
    return {
        "verdict": verdict.group(1) if verdict else "no verdict",
        "regressions": int(verdict.group(2)) if verdict else 0,
        "baseline": baseline.group(1)[:11] if baseline else None,
        "baseline_age": (
            (day - dt.date.fromisoformat(baseline.group(2))).days if baseline else None
        ),
    }


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
            if job["conclusion"] != "success":
                report["problems"].append(
                    f"{name}: {leg.group(1)} {job['conclusion'] or 'not finished'}"
                )
        elif "Upload perf data to the warehouse" in job["name"]:
            report["upload"] = job["conclusion"]
            if job["conclusion"] != "success":
                report["problems"].append(
                    f"{name}: upload {job['conclusion'] or 'not finished'}"
                )
        elif "Compare" in job["name"] and job["conclusion"] != "skipped":
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
    for arch in ARCHES:
        if run["status"] == "completed" and arch not in report["gates"]:
            report["problems"].append(f"{name} {arch}: no gate verdict")
    return report


def _legs_line(legs):
    parts = []
    for arch, short in (("blackhole", "BH"), ("wormhole", "WH")):
        mine = sorted(
            (l for l in legs if f"_{arch} " in l["leg"]), key=lambda l: l["leg"]
        )
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
            f"*{r['name']}* <{run['html_url']}|run> · `{run['head_sha'][:11]}` · {run['conclusion'] or run['status']}"
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--tests-yaml", default="tests/pipeline_reorg/llk_perf_tests.yaml")
    ap.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    ap.add_argument("--out", default="health_payload.json")
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.date)
    limits = leg_limits(a.tests_yaml)
    reports = []
    for name, path in NIGHTLIES:
        run = scheduled_run(a.repo, path, a.date)
        jobs = run_jobs(a.repo, run["id"]) if run else []
        logs = {
            j["id"]: job_log(a.repo, j["id"])
            for j in jobs
            if "Compare" in j["name"] and j["conclusion"] not in (None, "skipped")
        }
        reports.append(assess(name, run, jobs, logs, limits, day))
    text = build_text(a.date, reports)
    with open(a.out, "w") as fh:
        json.dump({"channel": a.channel, "text": text}, fh)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
