#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI digest: report the current state of watched workflows.

Stateless by design — each run reports the latest completed run of each watched
workflow and the jobs that failed, split into real (🔴) vs infra (🟣) issues.
No history, no incident tracking: "since when" is deliberately out of scope.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone


def _status(j: dict) -> str | None:
    return (j.get("_job") or {}).get("status_code")


def derive_outcome(latest_run: dict, job_summaries: list[dict]) -> tuple[str, list[dict]]:
    """Classify a run as GREEN / REAL_FAIL / INFRA.

    Returns (outcome, non_green_jobs). A non-green job is infra only when its
    status_code is explicitly PURPLE; anything else non-green (including an
    unknown/missing status) is a real failure, not hidden as infra. Without any
    job summaries we cannot prove infra, so we report REAL_FAIL.
    """
    if latest_run.get("conclusion") == "success":
        return "GREEN", []
    if not job_summaries:
        return "REAL_FAIL", []
    non_green = [j for j in job_summaries if _status(j) != "GREEN"]
    if any(_status(j) != "PURPLE" for j in non_green):
        return "REAL_FAIL", non_green
    if non_green:
        return "INFRA", non_green
    return "REAL_FAIL", []


def _cron_field(field: str, value: int, lo: int, hi: int) -> bool:
    """True if `value` matches cron `field` (`*`, `a`, `a-b`, lists, `*/n`);
    `lo`/`hi` are the wildcard expansion bounds."""
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            step = int(s)
        if part in ("*", ""):
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def due(schedule: str | None, now: datetime) -> bool:
    """Is a subscription with this 5-field cron due at `now`?

    The minute field is ignored — the job fires hourly and may land any minute
    in the slot's hour — so it must be 0 or * (a stray minute would silently
    mislead). No schedule = always due.
    """
    if not schedule:
        return True
    fields = schedule.split()
    if len(fields) != 5:
        raise ValueError(f"cron needs 5 fields: {schedule!r}")
    minute, hour, dom, mon, dow = fields
    if minute not in ("0", "*"):
        raise ValueError(f"minute must be 0 or * (the job fires hourly): {schedule!r}")
    if not _cron_field(hour, now.hour, 0, 23) or not _cron_field(mon, now.month, 1, 12):
        return False
    cron_dow = (now.weekday() + 1) % 7  # cron: 0=Sun..6=Sat (7 also Sun)
    dow_ok = _cron_field(dow, cron_dow, 0, 7) or (cron_dow == 0 and _cron_field(dow, 7, 0, 7))
    dom_ok = _cron_field(dom, now.day, 1, 31)
    if dom.strip() != "*" and dow.strip() != "*":  # vixie-cron: either matches
        return dom_ok or dow_ok
    return dom_ok and dow_ok


def _gh_json(args: list[str]) -> object:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out) if out.strip() else None


def latest_run(repo: str, workflow: str, branch: str) -> dict | None:
    """Latest completed scheduled run of a workflow (by file name, e.g. foo.yaml).

    Restricted to ``--event schedule``: developer-triggered runs (workflow_dispatch,
    push) often filter the matrix to a subset of legs, so their job set is partial
    and would understate the digest. Scheduled runs always exercise the full matrix.
    """
    runs = _gh_json(
        [
            "run",
            "list",
            "-R",
            repo,
            "--workflow",
            workflow,
            "--branch",
            branch,
            "--event",
            "schedule",
            "--status",
            "completed",
            "--limit",
            "1",
            "--json",
            "databaseId,conclusion,createdAt,headSha,url,workflowName",
        ]
    )
    return runs[0] if runs else None


def fetch_job_summaries(repo: str, run_id: int) -> list[dict]:
    """Download every ai_job_summary_* artifact of a run and parse the JSONs.

    Returns [] when none exist (older/uninstrumented runs) — the caller treats
    that as undetermined-real via derive_outcome.
    """
    # --jq streams names per page; avoids json.loads choking on multi-page
    # concatenated objects when a run has >100 artifacts.
    out = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
            "--jq",
            ".artifacts[].name",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    names = [n for n in out.splitlines() if n.startswith("ai_job_summary_")]
    if not names:
        return []
    summaries: list[dict] = []
    with tempfile.TemporaryDirectory() as d:
        cmd = ["gh", "run", "download", str(run_id), "-R", repo, "-D", d]
        for n in names:
            cmd += ["-n", n]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        for path in _glob.glob(os.path.join(d, "**", "ai_job_summary_*.json"), recursive=True):
            with open(path, encoding="utf-8") as f:
                summaries.append(json.load(f))
    return summaries


def ai_summary_step_url(repo: str, job: dict) -> str:
    """Deep-link to the job's '🤖 AI job summary' step; the 'Post …' teardown
    step shares the name and must be excluded. Falls back to the job page."""
    url = job.get("url", "")
    if "/job/" not in url:
        return url
    job_id = url.rsplit("/job/", 1)[1].split("#")[0].split("/")[0]
    jq = '[.steps[] | select((.name | test("ai job summary"; "i")) and (.name | startswith("Post") | not))][0].number // empty'
    try:
        n = subprocess.run(
            ["gh", "api", f"repos/{repo}/actions/jobs/{job_id}", "--jq", jq],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return url
    return f"{url}#step:{n}:1" if n else url


def _sev_emoji(code: str | None) -> str:
    # Two signals only: infra (PURPLE) vs real (every other non-green code).
    return "🟣" if code == "PURPLE" else "🔴"


def _job_cell(j: dict) -> tuple[str, str, str]:
    """(linked job name, category cell, test-count cell)."""
    job = j.get("_job") or {}
    link = job.get("summary_url") or job.get("url")
    nm = job.get("name", "?")
    name = f"[{nm}]({link})" if link else nm
    # Non-breaking hyphen so "tt-metal:*" isn't wrapped across lines in the
    # narrow Category column (markdown gives no column-width control).
    cat = (j.get("category") or "").replace("-", "‑")
    cat_cell = f"`{cat}`" if cat else "—"
    n = len(j.get("failed_tests") or [])
    return name, cat_cell, (str(n) if n else "—")


def _error_cell(j: dict) -> str:
    # When the upstream LLM summary was unparseable, root_cause holds the parse
    # marker and error_message holds the raw broken blob — neither is usable.
    if (j.get("root_cause") or "").startswith("Failed to parse LLM response"):
        return "_(AI summary unavailable)_"
    em = (j.get("error_message") or "").strip().replace("\n", " ").replace("|", "\\|")
    return em or "—"


def all_green(results: list[dict]) -> bool:
    """True when every watched workflow's latest run is fully green."""
    return all(r["outcome"] == "GREEN" for r in results)


def _fmt_ts(iso: str) -> str:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")


def _health_bar(counts: dict, width: int = 20) -> str | None:
    """Bar of passing / total jobs; None when there are no jobs to score."""
    total = counts.get("broken", 0) + counts.get("infra", 0) + counts.get("passing", 0)
    if not total:
        return None
    pct = round(100 * counts.get("passing", 0) / total)
    filled = round(pct / 100 * width)
    return f"Health: `{'█' * filled}{'░' * (width - filled)}` {pct}%"


def _link(r: dict) -> str:
    name = r.get("label") or r["workflow"]
    return f"[{name}]({r['latest_url']})" if r.get("latest_url") else name


def _section(r: dict) -> list[str]:
    jobs = (r.get("real_jobs") or []) + (r.get("infra_jobs") or [])
    when = f" · {_fmt_ts(r['latest_ts'])} UTC" if r.get("latest_ts") else ""
    out = [f"### {_link(r)}"]
    if not jobs:  # failed run with no AI job summaries (uninstrumented or expired artifacts)
        out += [f"🔴 run failed — no AI job summaries available{when}", ""]
        return out
    c = r.get("counts") or {}
    bar = _health_bar(c)
    if bar:
        out.append(bar)
    out.append(f"🔴 {c.get('broken', 0)} · 🟣 {c.get('infra', 0)} · 🟢 {c.get('passing', 0)}{when}")
    rows = []
    for j in jobs:
        jl, cat_cell, tests = _job_cell(j)
        rows.append(f"| {_sev_emoji(_status(j))} | {jl} | {cat_cell} | {tests} | {_error_cell(j)} |")
    # Blank lines around the table are required for GFM to render it inside <details>.
    out += [
        "",
        "<details><summary>Failed jobs</summary>",
        "",
        "| | Job | Category | Tests | Error |",
        "|--|--|--|--|--|",
        *rows,
        "",
        "</details>",
        "",
    ]
    return out


def render_markdown(name: str, results: list[dict]) -> str:
    broken = [r for r in results if r["outcome"] == "REAL_FAIL"]
    infra = [r for r in results if r["outcome"] == "INFRA"]
    healthy = [r for r in results if r["outcome"] == "GREEN"]
    nodata = [r for r in results if r["outcome"] in ("UNKNOWN", "ERROR")]

    lines = [f"## CI Digest: {name}", "", "Legend: 🔴 broken · 🟣 infra · 🟢 success", ""]
    for r in broken + infra:
        lines += _section(r)
    if healthy:
        lines += ["**🟢 Passing:** " + ", ".join(_link(r) for r in healthy), ""]
    if nodata:
        lines += [
            "**⚠️ No data:** " + ", ".join(f"{_link(r)} ({r.get('note') or r['outcome'].lower()})" for r in nodata),
            "",
        ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CI digest engine — one digest per invocation")
    p.add_argument("--self-test", action="store_true", help="run embedded unit tests and exit")
    p.add_argument("--name", help="name of this digest")
    p.add_argument("--workflows", nargs="+", default=[], help="workflow file names to check")
    p.add_argument("--schedule", help="5-field cron; skipped unless due this hour (default: always)")
    p.add_argument("--force", action="store_true", help="ignore --schedule and run now (manual dispatch)")
    p.add_argument("--branch", default="main")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "tenstorrent/tt-metal"))
    p.add_argument("--out-dir", default=".")
    return p


def check_workflow(repo: str, branch: str, workflow: str) -> dict:
    base = {"workflow": workflow, "label": workflow, "latest_url": "", "real_jobs": [], "infra_jobs": [], "counts": {}}
    try:
        run = latest_run(repo, workflow, branch)
        if run is None:  # never ran / renamed / typo'd — not the same as "passing"
            return {**base, "outcome": "UNKNOWN", "note": "no completed run found"}
        summaries = fetch_job_summaries(repo, run["databaseId"]) if run["conclusion"] != "success" else []
        outcome, non_green = derive_outcome(run, summaries)
        real_jobs = [j for j in non_green if _status(j) != "PURPLE"]
        infra_jobs = [j for j in non_green if _status(j) == "PURPLE"]
        passing = sum(1 for j in summaries if _status(j) == "GREEN")
        for j in real_jobs + infra_jobs:
            jb = j.get("_job")
            if jb and jb.get("url"):
                jb["summary_url"] = ai_summary_step_url(repo, jb)
        return {
            **base,
            "label": run.get("workflowName") or workflow,
            "outcome": outcome,
            "latest_url": run["url"],
            "latest_ts": run["createdAt"],
            "real_jobs": real_jobs,
            "infra_jobs": infra_jobs,
            "counts": {"broken": len(real_jobs), "infra": len(infra_jobs), "passing": passing},
        }
    except subprocess.CalledProcessError as exc:
        # One flaky gh call must not discard the other workflows' results.
        err = ((exc.stderr or "").strip().splitlines() or ["gh command failed"])[-1]
        return {**base, "outcome": "ERROR", "note": err[:200]}
    except (json.JSONDecodeError, OSError) as exc:
        return {**base, "outcome": "ERROR", "note": str(exc)[:200]}


def main(argv: list[str]) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1

    if not args.name:
        raise SystemExit("no --name provided")

    now = datetime.now(timezone.utc)
    if not args.force and not due(args.schedule, now):  # not this digest's slot
        return 0

    results = [check_workflow(args.repo, args.branch, wf) for wf in args.workflows]
    md = render_markdown(args.name, results)
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")  # colon-free so the file name is portable/shareable
    stem = os.path.join(args.out_dir, f"ci-digest-{args.name}-{stamp}")
    with open(f"{stem}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "all_green": all_green(results), "results": results}, f, indent=2)
    print(md)
    return 0


# --- embedded tests (run via --self-test) ---------------------------------


class TestDeriveOutcome(unittest.TestCase):
    def _job(self, code):
        return {"_job": {"status_code": code, "name": "j", "url": "u"}}

    def test_green_when_conclusion_success(self):
        self.assertEqual(derive_outcome({"conclusion": "success"}, []), ("GREEN", []))

    def test_real_fail_when_any_non_purple(self):
        out, jobs = derive_outcome({"conclusion": "failure"}, [self._job("PURPLE"), self._job("ORANGE")])
        self.assertEqual(out, "REAL_FAIL")
        self.assertEqual(len(jobs), 2)

    def test_infra_when_all_non_green_are_purple(self):
        out, jobs = derive_outcome({"conclusion": "failure"}, [self._job("PURPLE"), self._job("GREEN")])
        self.assertEqual(out, "INFRA")
        self.assertEqual(len(jobs), 1)

    def test_no_summaries_is_conservative_real(self):
        self.assertEqual(derive_outcome({"conclusion": "failure"}, [])[0], "REAL_FAIL")

    def test_unknown_status_is_real_not_infra(self):
        self.assertEqual(derive_outcome({"conclusion": "failure"}, [self._job(None)])[0], "REAL_FAIL")

    def test_malformed_summaries_do_not_crash(self):
        self.assertEqual(derive_outcome({"conclusion": "failure"}, [{}, {"_job": {}}])[0], "REAL_FAIL")


class TestDue(unittest.TestCase):
    MON_8 = datetime(2024, 1, 1, 8, 30)  # 2024-01-01 is a Monday; :30 proves minute-agnostic
    SUN_8 = datetime(2024, 1, 7, 8, 0)

    def test_no_schedule_always_due(self):
        self.assertTrue(due(None, self.MON_8))

    def test_hour_and_weekday(self):
        self.assertTrue(due("0 8 * * 1-5", self.MON_8))
        self.assertFalse(due("0 9 * * 1-5", self.MON_8))
        self.assertFalse(due("0 8 * * 6,0", self.MON_8))

    def test_sunday_zero_and_seven(self):
        self.assertTrue(due("0 8 * * 0", self.SUN_8))
        self.assertTrue(due("0 8 * * 7", self.SUN_8))

    def test_step_hours(self):
        self.assertTrue(due("0 */2 * * *", datetime(2024, 1, 1, 8, 0)))
        self.assertFalse(due("0 */2 * * *", datetime(2024, 1, 1, 9, 0)))

    def test_hour_list_and_range(self):
        self.assertTrue(due("0 8,16 * * *", datetime(2024, 1, 1, 16, 0)))
        self.assertTrue(due("0 8-10 * * *", datetime(2024, 1, 1, 9, 0)))
        self.assertFalse(due("0 8,16 * * *", datetime(2024, 1, 1, 12, 0)))

    def test_month_gate(self):
        self.assertFalse(due("0 8 * 2 *", datetime(2024, 1, 1, 8, 0)))
        self.assertTrue(due("0 8 * 2 *", datetime(2024, 2, 1, 8, 0)))

    def test_day_of_month(self):
        self.assertTrue(due("0 8 15 * *", datetime(2024, 1, 15, 8, 0)))
        self.assertFalse(due("0 8 15 * *", datetime(2024, 1, 14, 8, 0)))

    def test_nonzero_minute_rejected(self):
        with self.assertRaises(ValueError):
            due("30 8 * * *", self.MON_8)

    def test_bad_field_count_rejected(self):
        with self.assertRaises(ValueError):
            due("8 * * *", self.MON_8)


class TestRender(unittest.TestCase):
    def _broken(self):
        return {
            "workflow": "WF-A",
            "label": "WF-A",
            "outcome": "REAL_FAIL",
            "latest_url": "http://run/2",
            "counts": {"broken": 1, "infra": 1, "passing": 3},
            "real_jobs": [
                {
                    "_job": {"name": "job-x", "url": "http://job/x", "status_code": "RED"},
                    "category": "tt-metal:compile",
                    "error_message": "boom",
                    "failed_tests": [],
                }
            ],
            "infra_jobs": [
                {
                    "_job": {"name": "infra-y", "url": "http://job/y", "status_code": "PURPLE"},
                    "category": "infra:ci",
                    "error_message": "runner died",
                    "failed_tests": [],
                }
            ],
        }

    def test_sections_and_links(self):
        md = render_markdown(
            "models", [self._broken(), {"workflow": "WF-D", "outcome": "GREEN", "latest_url": "http://run/3"}]
        )
        self.assertIn("[WF-A](http://run/2)", md)
        self.assertIn("[job-x](http://job/x)", md)
        self.assertIn("🟣", md)
        self.assertIn("boom", md)
        self.assertIn("Failed jobs", md)
        self.assertIn("Health:", md)  # 3 passing / 5 total
        self.assertIn("60%", md)
        self.assertIn("WF-D", md)

    def test_broken_without_job_summaries(self):
        md = render_markdown(
            "m",
            [
                {
                    "workflow": "WF-N",
                    "outcome": "REAL_FAIL",
                    "latest_url": "u",
                    "real_jobs": [],
                    "infra_jobs": [],
                    "counts": {},
                }
            ],
        )
        self.assertIn("no AI job summaries available", md)
        self.assertNotIn("🟢 0", md)  # no misleading all-zero counts
        self.assertNotIn("Failed jobs", md)  # no empty placeholder table

    def test_infra_only(self):
        md = render_markdown(
            "m",
            [
                {
                    "workflow": "WF-I",
                    "outcome": "INFRA",
                    "latest_url": "u",
                    "counts": {"broken": 0, "infra": 1, "passing": 0},
                    "infra_jobs": [{"_job": {"name": "i", "status_code": "PURPLE"}, "error_message": "x"}],
                }
            ],
        )
        self.assertIn("WF-I", md)

    def test_empty_and_all_green(self):
        self.assertIn("CI Digest: m", render_markdown("m", []))
        self.assertIn("Passing", render_markdown("m", [{"workflow": "G", "outcome": "GREEN", "latest_url": "u"}]))

    def test_no_data_section_guards_empty_url(self):
        md = render_markdown(
            "m", [{"workflow": "WF-U", "outcome": "UNKNOWN", "latest_url": "", "note": "no completed run found"}]
        )
        self.assertIn("No data", md)
        self.assertIn("WF-U", md)
        self.assertNotIn("WF-U]()", md)  # no broken empty markdown link

    def test_all_green(self):
        self.assertTrue(all_green([{"outcome": "GREEN"}, {"outcome": "GREEN"}]))
        self.assertFalse(all_green([{"outcome": "INFRA"}, {"outcome": "GREEN"}]))
        self.assertFalse(all_green([{"outcome": "REAL_FAIL"}]))
        self.assertFalse(all_green([{"outcome": "UNKNOWN"}]))


class TestErrorCell(unittest.TestCase):
    def test_parse_failure(self):
        self.assertIn("unavailable", _error_cell({"root_cause": "Failed to parse LLM response: ..."}))

    def test_pipe_and_newline_escaped(self):
        self.assertEqual(_error_cell({"error_message": "a|b\nc"}), "a\\|b c")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
