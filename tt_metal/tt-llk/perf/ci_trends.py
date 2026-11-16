# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CI cost and reliability of the LLK jobs, for the nightly health message.

Leg minutes against their caps, LLK job failures in the merge queue by cause,
and the age of the test durations that split the legs. The merge queue runs
hundreds of times a day, so each day is read once and kept in --history.
"""

import argparse
import concurrent.futures
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
import time

from nightly_health import LEG, _gh, _json_lines, _time, job_log, leg_limits

NAME = "CI cost and reliability"
MERGE_GATE = "merge-gate.yaml"
NIGHTLIES = (
    ("SoL nightly", "llk-perf.yaml"),
    ("non-SoL nightly", "llk-perf-gate.yaml"),
)
# The sources of .github/scripts/llk-fetch-test-durations.sh: workflow, branch, event.
DURATIONS = (
    ("smoke", "pr-gate.yaml", "main", "push"),
    ("SoL", "llk-perf.yaml", "main", "schedule"),
    ("non-SoL", "llk-perf-gate.yaml", "main", "schedule"),
    ("merge gate", MERGE_GATE, "", "merge_group"),
)
DURATION_RUNS = 60
WINDOW = 7
HEADROOM = 0.8
GROWTH = 0.2
MIN_SAMPLES = 3
FAILURE_RATE = 0.1
LLK_JOB = re.compile(r"(?i)llk|quasar")
QUEUE_CANCEL = re.compile(r"higher priority waiting request")
TIMEOUT = re.compile(r"exceeded the maximum execution time")
CAUSES = (
    ("API rate limit", re.compile(r"(?i)API rate limit exceeded|secondary rate limit")),
    (
        "network",
        re.compile(
            r"ECONNRESET|ETIMEDOUT|socket hang up|Could not resolve host|Connection reset|Connection timed out|502 Bad Gateway"
        ),
    ),
    (
        "credentials",
        re.compile(
            r"Permission denied \(publickey\)|error in libcrypto|PRIVATE_KEY is empty|Bad credentials"
        ),
    ),
    (
        "runner lost",
        re.compile(
            r"lost communication with the server|received a shutdown signal|runner has encountered an error"
        ),
    ),
)
SHORT = {"blackhole": "BH", "wormhole": "WH"}


def _api(*args):
    for attempt in range(4):
        try:
            return _gh("api", *args)
        except subprocess.CalledProcessError as e:
            if attempt == 3:
                raise
            print(f"retry {args[:2]}: {e.stderr.strip()}", file=sys.stderr)
            time.sleep(5 * 2**attempt)


def _minutes(job):
    if not (job["started_at"] and job["completed_at"]):
        return None
    return (_time(job["completed_at"]) - _time(job["started_at"])).total_seconds() / 60


def runs(repo, workflow, query):
    return _json_lines(
        _api(
            "--paginate",
            f"repos/{repo}/actions/workflows/{workflow}/runs?{query}&per_page=100",
            "--jq",
            ".workflow_runs[] | {id, conclusion, created_at} | @json",
        )
    )


def jobs(repo, run_id):
    return _json_lines(
        _api(
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&filter=latest",
            "--jq",
            '.jobs[] | {id, name, conclusion, started_at, completed_at, steps: [.steps[]? | select(.conclusion != "success" and .conclusion != "skipped") | .name]} | @json',
        )
    )


def annotations(repo, job_id):
    try:
        return _api(
            f"repos/{repo}/check-runs/{job_id}/annotations",
            "--jq",
            '.[] | select(.annotation_level == "failure") | .message',
        )
    except subprocess.CalledProcessError:
        return ""


def cause(job, text):
    """Why a job did not succeed, from its annotations or log and its first unfinished step."""
    step = job["steps"][0] if job["steps"] else ""
    if TIMEOUT.search(text):
        return "checkout hang" if "checkout" in step.lower() else "timeout"
    if QUEUE_CANCEL.search(text):
        return "queue cancel"
    for name, pattern in CAUSES:
        if pattern.search(text):
            return name
    if job["conclusion"] == "startup_failure":
        return "startup_failure"
    if re.search(r"(?i)test", step):
        return "test failure"
    return "unknown"


def _jobs_or_none(repo, run_id):
    try:
        return jobs(repo, run_id)
    except subprocess.CalledProcessError:
        return None


def _is_timeout(minutes, cap):
    return cap is not None and minutes is not None and minutes >= 0.95 * cap


def read_day(repo, day, caps, max_logs, workers=8):
    """One day of merge queue runs: LLK job outcomes by cause, and leg minutes."""
    listed = runs(repo, MERGE_GATE, f"event=merge_group&created={day}")
    done = [
        r
        for r in listed
        if r["conclusion"] in ("success", "failure", "cancelled", "timed_out")
    ]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        all_jobs = list(pool.map(lambda r: _jobs_or_none(repo, r["id"]), done))
    out = {
        "runs": len(listed),
        "startup_failure": sum(r["conclusion"] == "startup_failure" for r in listed),
        "jobs": {},
        "legs": {},
        "unread": sum(j is None for j in all_jobs),
    }
    from_log = {}
    for run, run_jobs in zip(done, all_jobs):
        for job in run_jobs or []:
            if not LLK_JOB.search(job["name"]) or job["conclusion"] in (
                None,
                "skipped",
            ):
                continue
            name = re.sub(r" \[.*\]$", "", job["name"])
            leg = LEG.search(name)
            minutes = _minutes(job)
            timed_out = _is_timeout(minutes, caps.get(leg.group(1)) if leg else None)
            why = None
            if (
                job["conclusion"] == "cancelled"
                and run["conclusion"] == "cancelled"
                and not timed_out
            ):
                why = "queue cancel"
            elif job["conclusion"] != "success":
                text = annotations(repo, job["id"])
                why = "timeout" if timed_out else cause(job, text)
                # The same job failing the same way has the same cause: one log is enough.
                key = (name, tuple(job["steps"]), text)
                if why == "unknown" and key in from_log:
                    why = from_log[key]
                elif why == "unknown" and len(from_log) < max_logs:
                    why = from_log[key] = cause(job, job_log(repo, job["id"]))
            stats = out["jobs"].setdefault(
                name,
                {"runs": 0, "failures": 0, "cancelled": 0, "causes": {}, "run": None},
            )
            if why == "queue cancel":
                stats["cancelled"] += 1
                continue
            stats["runs"] += 1
            if why:
                stats["failures"] += 1
                stats["causes"][why] = stats["causes"].get(why, 0) + 1
                stats["run"] = run["id"]
            if leg and minutes is not None:
                out["legs"].setdefault(leg.group(1), []).append(round(minutes, 1))
    return out


def nightly_legs(repo, workflow, since, caps):
    """Leg minutes of the scheduled runs since ``since``, by day."""
    by_day = {}
    for run in runs(repo, workflow, f"event=schedule&created=>={since}"):
        if run["conclusion"] not in ("success", "failure", "cancelled", "timed_out"):
            continue
        day = by_day.setdefault(run["created_at"][:10], {})
        for job in jobs(repo, run["id"]):
            leg = LEG.search(job["name"])
            minutes = _minutes(job)
            if not leg or minutes is None or job["conclusion"] in (None, "skipped"):
                continue
            if job["conclusion"] == "cancelled" and not _is_timeout(
                minutes, caps.get(leg.group(1))
            ):
                continue
            day.setdefault(leg.group(1), []).append(round(minutes, 1))
    return by_day


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def _window(legs_by_day, end, days):
    out = {}
    for n in range(days):
        for leg, values in legs_by_day.get(
            (end - dt.timedelta(days=n)).isoformat(), {}
        ).items():
            out.setdefault(leg, []).extend(values)
    return out


def leg_report(label, legs_by_day, day, caps):
    """p50/p90/max per leg over the last 7 days; p90 near the cap or 20 % up is a problem."""
    last = day - dt.timedelta(days=1)
    now = _window(legs_by_day, last, WINDOW)
    before = _window(legs_by_day, last - dt.timedelta(days=WINDOW), WINDOW)
    problems, lines = [], []
    for arch, short in SHORT.items():
        legs = sorted(l for l in now if f"_{arch} " in l)
        if not legs:
            continue
        cells = []
        for leg in legs:
            v = now[leg]
            p90 = _pct(v, 0.9)
            cap = caps.get(leg)
            if cap and p90 > HEADROOM * cap:
                problems.append(
                    f"{label} {leg}: p90 {p90:.1f} min = {100 * p90 / cap:.0f} % of the {cap} min cap"
                )
            old = before.get(leg, [])
            if (
                len(v) >= MIN_SAMPLES
                and len(old) >= MIN_SAMPLES
                and p90 > (1 + GROWTH) * _pct(old, 0.9)
            ):
                problems.append(
                    f"{label} {leg}: p90 {_pct(old, 0.9):.1f} → {p90:.1f} min against the {WINDOW} days before"
                )
            cells.append(f"{leg.split()[-1]} {_pct(v, 0.5):.1f}/{p90:.1f}/{max(v):.1f}")
        cap = caps.get(legs[0])
        lines.append(
            f"{label} {short} (cap {cap} min, {max(len(now[l]) for l in legs)} runs): "
            + " · ".join(cells)
        )
    if not now:
        lines.append(f"{label}: no leg ran in the last {WINDOW} days")
    return problems, lines


def cap_line(label, legs_by_day, day, caps):
    """How much of its cap the slowest leg of each arch used over the last 14 days."""
    legs = _window(legs_by_day, day - dt.timedelta(days=1), 2 * WINDOW)
    cells = []
    for arch, short in SHORT.items():
        mine = {l: max(v) for l, v in legs.items() if f"_{arch} " in l and caps.get(l)}
        if mine:
            leg = max(mine, key=mine.get)
            cells.append(
                f"{short} {mine[leg]:.0f} of {caps[leg]} min ({100 * mine[leg] / caps[leg]:.0f} %)"
            )
    return f"{label} slowest leg in {2 * WINDOW} days: " + (", ".join(cells) or "none")


def queue_report(day_data, day):
    """LLK jobs of one day of the merge queue: runs, failures and their causes."""
    if day_data is None:
        return [f"merge queue {day}: not read"], []
    problems, lines = [], []
    if (
        day_data["runs"]
        and day_data["startup_failure"] > FAILURE_RATE * day_data["runs"]
    ):
        problems.append(
            f"merge queue {day}: {day_data['startup_failure']} of {day_data['runs']} runs startup_failure"
        )
    stats = day_data["jobs"].values()
    lines.append(
        f"merge queue {day}: {day_data['runs']} runs, {day_data['startup_failure']} startup_failure; "
        f"LLK jobs {sum(s['runs'] for s in stats)} ran, {sum(s['failures'] for s in stats)} failed, "
        f"{sum(s['cancelled'] for s in stats)} cancelled by the queue"
        + (f"; {day_data['unread']} runs not read" if day_data.get("unread") else "")
    )
    failed = sorted(
        ((n, s) for n, s in day_data["jobs"].items() if s["failures"]),
        key=lambda x: -x[1]["failures"],
    )
    for name, s in failed:
        causes = ", ".join(
            f"{c} {n}" for c, n in sorted(s["causes"].items(), key=lambda x: -x[1])
        )
        text = f"{name}: {s['failures']}/{s['runs']} failed ({causes})"
        if s["failures"] >= 2 and s["failures"] > FAILURE_RATE * s["runs"]:
            problems.append(f"merge queue {day}: {text}")
        lines.append(f"• {text}, e.g. run {s['run']}")
    return problems, lines[:6]


def durations_age(repo, workflow, branch, event, now):
    """Age in days of the durations that llk-fetch-test-durations.sh would pick, or None."""
    query = f"event={event}&status=success" + (f"&branch={branch}" if branch else "")
    listed = _json_lines(
        _api(
            f"repos/{repo}/actions/workflows/{workflow}/runs?{query}&per_page={DURATION_RUNS}",
            "--jq",
            ".workflow_runs[] | {id} | @json",
        )
    )
    for run in listed:
        found = _json_lines(
            _api(
                f"repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100",
                "--jq",
                '.artifacts[] | select(.name | startswith("test-durations-")) | select(.expired | not) | {created_at} | @json',
            )
        )
        if found:
            newest = max(_time(a["created_at"]) for a in found)
            return (now - newest).total_seconds() / 86400, run["id"]
    return None, None


def durations_report(ages, stale_days):
    problems, cells = [], []
    for label, (age, run) in ages.items():
        if age is None:
            problems.append(
                f"durations {label}: none in the last {DURATION_RUNS} green runs"
            )
            cells.append(f"{label} none")
            continue
        if age > stale_days:
            problems.append(
                f"durations {label}: newest is {age:.1f} days old (run {run})"
            )
        cells.append(f"{label} {age:.1f} d")
    return problems, ["test durations age: " + " · ".join(cells)]


def update_history(repo, history, day, caps, fetch_days, max_logs):
    """Read the missing days of the last two windows, newest first; drop older days."""
    days = [(day - dt.timedelta(days=n)).isoformat() for n in range(1, 2 * WINDOW + 1)]
    kept = {d: history[d] for d in days if d in history}
    for d in [d for d in days if d not in kept][:fetch_days]:
        kept[d] = read_day(repo, d, caps, max_logs)
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    ap.add_argument("--tests-yaml", default="tests/pipeline_reorg/llk_perf_tests.yaml")
    ap.add_argument(
        "--gate-tests-yaml",
        default="tests/pipeline_reorg/llk_perf_merge_gate_tests.yaml",
    )
    ap.add_argument("--history", default="ci_trends_history.json")
    ap.add_argument(
        "--fetch-days", type=int, default=3, help="merge queue days to read per run"
    )
    ap.add_argument("--max-logs", type=int, default=10)
    ap.add_argument("--stale-days", type=float, default=7)
    ap.add_argument("--out", default="nightly_checks/ci_trends.json")
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.date)
    gate_caps = leg_limits(a.gate_tests_yaml)
    nightly_caps = leg_limits(a.tests_yaml)
    try:
        with open(a.history) as fh:
            history = json.load(fh)
    except (OSError, ValueError):
        history = {}
    history = update_history(a.repo, history, day, gate_caps, a.fetch_days, a.max_logs)
    with open(a.history, "w") as fh:
        json.dump(history, fh)

    problems, lines = leg_report(
        "merge gate", {d: v["legs"] for d, v in history.items()}, day, gate_caps
    )
    since = (day - dt.timedelta(days=2 * WINDOW)).isoformat()
    for label, workflow in NIGHTLIES:
        legs = nightly_legs(a.repo, workflow, since, nightly_caps)
        p, l = leg_report(label, legs, day, nightly_caps)
        problems += p
        lines += l + [cap_line(label, legs, day, nightly_caps)]
    yesterday = (day - dt.timedelta(days=1)).isoformat()
    p, l = queue_report(history.get(yesterday), yesterday)
    problems += p
    lines += l
    now = dt.datetime.now(dt.timezone.utc)
    ages = {label: durations_age(a.repo, *source, now) for label, *source in DURATIONS}
    p, l = durations_report(ages, a.stale_days)
    problems += p
    lines += l
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"name": NAME, "problems": problems, "lines": lines}, fh)
    print("\n".join([*(f"PROBLEM {p}" for p in problems), *lines]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
