# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check the LLK perf nightlies of one day and send a Slack health message.

It reads two things per nightly: the run's conclusion, and the verdict files its gates uploaded
(gate.py compare: status, regressions, baseline, and each leg's time). It runs on its own
schedule, after both nightlies, so it also reports a nightly that never ran.
"""

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import tempfile
import urllib.request

# The run-name of each scheduled run of llk-perf.yaml.
NIGHTLIES = ("LLK perf nightly, speed of light", "LLK perf nightly")
ARCHES = ("blackhole", "wormhole")
SHORT_SHA = 11


def _gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def scheduled_runs(repo, day):
    """{run-name: the newest scheduled run of llk-perf.yaml with that name on ``day``}."""
    runs = json.loads(
        _gh(
            "api",
            f"repos/{repo}/actions/workflows/llk-perf.yaml/runs?event=schedule&created={day}&per_page=20",
            "--jq",
            "[.workflow_runs[] | {id, display_title, html_url, head_sha, status, conclusion}]",
        )
    )
    found = {}
    for run in runs:
        found.setdefault(run["display_title"], run)
    return found


def verdicts(repo, run_id):
    """{arch: verdict} from the run's gate artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _gh(
                "run",
                "download",
                str(run_id),
                "-R",
                repo,
                "-p",
                "llk-perf-gate-report-*",
                "-D",
                tmp,
            )
        except subprocess.CalledProcessError:
            return {}
        out = {}
        for path in glob.glob(f"{tmp}/**/verdict-*.json", recursive=True):
            with open(path) as fh:
                v = json.load(fh)
            out[v["arch"]] = v
        return out


def assess(name, run, found):
    """The problems of one nightly, and what the message shows of it."""
    report = {"name": name, "run": run, "verdicts": found, "problems": []}
    if run is None:
        report["problems"].append(f"{name}: no scheduled run")
        return report
    if run["status"] != "completed":
        report["problems"].append(f"{name}: run still {run['status']}")
    for arch in ARCHES:
        v = found.get(arch)
        if v is None:
            if run["status"] == "completed":
                report["problems"].append(f"{name} {arch}: no gate verdict")
            continue
        if v["status"] == "regression":
            regressions = v["counts"]["regressions"]
            report["problems"].append(f"{name} {arch}: gate regressed, {regressions:,} points")
        elif v["status"] != "pass":
            report["problems"].append(f"{name} {arch}: gate {v['status']}. {v['reason']}")
    if run["conclusion"] not in (None, "success") and not report["problems"]:
        report["problems"].append(f"{name}: run {run['conclusion']}")
    return report


def _legs_line(v):
    cells = []
    for leg in v.get("legs", []):
        minutes, limit = leg.get("minutes"), leg.get("timeout_minutes")
        used = f"{round(100 * minutes / limit)}%" if minutes is not None and limit else "?%"
        cells.append(f"{leg['group']}/{leg['groups']} {used}" + ("" if leg["complete"] else " ✗"))
    return " ".join(cells) or "none"


def _baseline_line(v, day):
    base = v.get("baseline") or {}
    sha = (base.get("commit_sha") or "?")[:SHORT_SHA]
    stamp = str(base.get("run_ts") or "")[:10]
    age = f"{(day - dt.date.fromisoformat(stamp)).days} days old" if stamp else "age unknown"
    return f"baseline `{sha}`, {age}"


def build_text(day, reports):
    problems = [p for r in reports for p in r["problems"]]
    head = ":white_check_mark:" if not problems else ":red_circle:"
    lines = [
        f"{head} *LLK perf nightly health, {day}*" + (f": {len(problems)} problem(s)" if problems else ": healthy")
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
        for arch, v in sorted(r["verdicts"].items()):
            regressions = v["counts"]["regressions"]
            count = f", {regressions:,} points" if regressions else ""
            lines.append(f"{arch}: {v['status']}{count} · {_baseline_line(v, day)}")
            lines.append(f"    legs, % of limit used: {_legs_line(v)}")
    return "\n".join(lines)


def post(channel, text):
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
        raise SystemExit(f"::error::Slack rejected the health message: {answer}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    ap.add_argument("--post", nargs="*", default=[], help="Slack channel or user ids to send it to")
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.date)
    runs = scheduled_runs(a.repo, a.date)
    reports = []
    for name in NIGHTLIES:
        run = runs.get(name)
        reports.append(assess(name, run, verdicts(a.repo, run["id"]) if run else {}))
    text = build_text(day, reports)
    print(text)
    for channel in filter(None, a.post):
        post(channel, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
