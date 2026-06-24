#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI digest: report the current state of watched workflows.

Thin aggregator. For each watched workflow it finds the latest completed
scheduled run and reads that run's machine-readable ``ai_run_summary_<run_id>``
artifact — a factual JSON the ai_summary/run action already produces (succeeded
/ failed / infra_failure jobs). The digest does no classification of its own; it
collects those per-run summaries and renders them at one point so a team can
react. Stateless by design — no history, no incident tracking.
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


def fetch_run_summary(repo: str, run_id: int) -> dict | None:
    """Download a run's ``ai_run_summary_<run_id>`` artifact and parse its JSON.

    The ai_summary/run action uploads ``ai_run_summary_<run_id>.json`` (the
    factual, deterministic run report: succeeded / failed / infra_failure) inside
    that artifact. Returns None when the artifact is absent — the workflow doesn't
    run ai_summary/run, or the run predates JSON output — so the caller can fall
    back to the run's conclusion.
    """
    name = f"ai_run_summary_{run_id}"
    with tempfile.TemporaryDirectory() as d:
        try:
            subprocess.run(
                ["gh", "run", "download", str(run_id), "-R", repo, "-n", name, "-D", d],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            return None  # no such artifact on this run
        matches = _glob.glob(os.path.join(d, "**", f"{name}.json"), recursive=True)
        if not matches:
            return None  # artifact present but .md-only (run predates JSON output)
        with open(matches[0], encoding="utf-8") as f:
            return json.load(f)


def summarize_run(data: dict) -> tuple[str, list[dict], list[dict], int]:
    """Map a run-summary JSON into (outcome, failed_rows, infra_rows, passing).

    A run is REAL_FAIL if it has any real failure, else INFRA if it has any infra
    failure, else GREEN. failed/infra rows are passed through verbatim — they
    already carry job_name, job_url, status, category, error_message, root_cause.
    """
    failed = data.get("failed") or []
    infra = data.get("infra_failure") or []
    passing = len(data.get("succeeded") or [])
    outcome = "REAL_FAIL" if failed else "INFRA" if infra else "GREEN"
    return outcome, failed, infra, passing


def _sev_emoji(row: dict) -> str:
    # 🟣 infra; ⌛️ a failure whose log was truncated/killed (log_complete is
    # False — i.e. it timed out); 🔴 every other non-green status.
    if row.get("status") == "INFRA_FAILURE":
        return "🟣"
    if row.get("log_complete") is False:
        return "⌛️"
    return "🔴"


def _job_link(row: dict) -> str:
    nm = row.get("job_name") or "?"
    url = row.get("job_url") or ""
    return f"[{nm}]({url})" if url else nm


def _cat_cell(row: dict) -> str:
    # Non-breaking hyphen so "infra:no-artifact" / "tt-metal:*" isn't wrapped in
    # the narrow Category column (markdown gives no column-width control).
    cat = (row.get("category") or "").replace("-", "‑")
    return f"`{cat}`" if cat else "—"


def _error_cell(row: dict) -> str:
    msg = (row.get("error_message") or row.get("root_cause") or "").strip().replace("\n", " ").replace("|", "\\|")
    return (msg[:200] + "…") if len(msg) > 200 else (msg or "—")


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
    """One section per workflow — same shape for green and broken: name, health
    bar, and the 🔴/🟣/🟢 + date line. Broken/infra runs add a collapsible
    failed-jobs table; a green run stops at the counts line (it's enough to see
    it's green)."""
    when = f" · {_fmt_ts(r['latest_ts'])} UTC" if r.get("latest_ts") else ""
    out = [f"### {_link(r)}"]
    c = r.get("counts") or {}
    total = c.get("broken", 0) + c.get("infra", 0) + c.get("passing", 0)
    if total:
        bar = _health_bar(c)
        if bar:
            out.append(bar)
        out.append(f"🔴 {c.get('broken', 0)} · 🟣 {c.get('infra', 0)} · 🟢 {c.get('passing', 0)}{when}")
    elif r.get("outcome") == "GREEN":
        # Green via the run-conclusion fallback (no per-job counts available).
        out.append(f"🟢 green{when}")
    else:
        out.append(f"🔴 run not green — no per-job detail in the run summary{when}")
    jobs = (r.get("real_jobs") or []) + (r.get("infra_jobs") or [])
    if jobs:
        rows = [
            f"| {_sev_emoji(j)} | {_job_link(j)} | {j.get('status') or '—'} | {_cat_cell(j)} | {_error_cell(j)} |"
            for j in jobs
        ]
        # Blank lines around the table are required for GFM to render it inside <details>.
        out += [
            "",
            "<details><summary>Failed jobs</summary>",
            "",
            "| | Job | Status | Category | Error |",
            "|--|--|--|--|--|",
            *rows,
            "",
            "</details>",
        ]
    out.append("")
    return out


def render_markdown(name: str, results: list[dict]) -> str:
    broken = [r for r in results if r["outcome"] == "REAL_FAIL"]
    infra = [r for r in results if r["outcome"] == "INFRA"]
    healthy = [r for r in results if r["outcome"] == "GREEN"]
    nodata = [r for r in results if r["outcome"] in ("UNKNOWN", "ERROR")]

    lines = [f"## CI Digest: {name}", "", "Legend: 🔴 broken · 🟣 infra · 🟢 success", ""]
    for r in broken + infra + healthy:  # failures first, then green — same section shape
        lines += _section(r)
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
        label = run.get("workflowName") or workflow
        meta = {"label": label, "latest_url": run["url"], "latest_ts": run["createdAt"]}
        data = fetch_run_summary(repo, run["databaseId"])
        if data is None:
            # No machine-readable summary. Fall back to the run conclusion: a green
            # run is healthy; a non-green one we can't detail (workflow doesn't run
            # ai_summary/run, or the run predates JSON output).
            if run.get("conclusion") == "success":
                return {**base, **meta, "outcome": "GREEN"}
            return {**base, **meta, "outcome": "UNKNOWN", "note": "no ai_run_summary artifact"}
        outcome, failed, infra, passing = summarize_run(data)
        return {
            **base,
            **meta,
            "outcome": outcome,
            "real_jobs": failed,
            "infra_jobs": infra,
            "counts": {"broken": len(failed), "infra": len(infra), "passing": passing},
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


class TestSummarizeRun(unittest.TestCase):
    def test_real_fail_when_any_failure(self):
        out, failed, infra, passing = summarize_run(
            {"failed": [{"job_name": "a"}], "infra_failure": [{"job_name": "b"}], "succeeded": [{}, {}]}
        )
        self.assertEqual((out, len(failed), len(infra), passing), ("REAL_FAIL", 1, 1, 2))

    def test_infra_when_only_infra(self):
        out, _, infra, passing = summarize_run({"infra_failure": [{"job_name": "b"}], "succeeded": [{}]})
        self.assertEqual((out, len(infra), passing), ("INFRA", 1, 1))

    def test_green_when_only_success(self):
        self.assertEqual(summarize_run({"succeeded": [{}, {}, {}]}), ("GREEN", [], [], 3))

    def test_empty_is_green(self):
        self.assertEqual(summarize_run({}), ("GREEN", [], [], 0))


class TestRender(unittest.TestCase):
    def _broken(self):
        return {
            "workflow": "WF-A",
            "label": "WF-A",
            "outcome": "REAL_FAIL",
            "latest_url": "http://run/2",
            "latest_ts": "2026-06-14T06:34:32Z",
            "counts": {"broken": 1, "infra": 1, "passing": 3},
            "real_jobs": [
                {
                    "job_name": "job-x",
                    "job_url": "http://job/x",
                    "status": "TESTS_FAILED",
                    "category": "tt-metal:compile",
                    "error_message": "boom",
                }
            ],
            "infra_jobs": [
                {
                    "job_name": "infra-y",
                    "job_url": "http://job/y",
                    "status": "INFRA_FAILURE",
                    "category": "infra:ci",
                    "error_message": "runner died",
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
        self.assertIn("TESTS_FAILED", md)  # precise status surfaced
        self.assertIn("boom", md)
        self.assertIn("Failed jobs", md)
        self.assertIn("Health:", md)  # 3 passing / 5 total
        self.assertIn("60%", md)
        self.assertIn("WF-D", md)

    def test_section_without_rows(self):
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
        self.assertIn("no per-job detail", md)
        self.assertNotIn("Failed jobs", md)

    def test_infra_only(self):
        md = render_markdown(
            "m",
            [
                {
                    "workflow": "WF-I",
                    "outcome": "INFRA",
                    "latest_url": "u",
                    "counts": {"broken": 0, "infra": 1, "passing": 0},
                    "infra_jobs": [{"job_name": "i", "status": "INFRA_FAILURE", "error_message": "x"}],
                }
            ],
        )
        self.assertIn("WF-I", md)
        self.assertIn("🟣", md)

    def test_empty_and_green_fallback(self):
        self.assertIn("CI Digest: m", render_markdown("m", []))
        # GREEN with no per-job counts (run-conclusion fallback) → "🟢 green".
        md = render_markdown("m", [{"workflow": "G", "outcome": "GREEN", "latest_url": "u"}])
        self.assertIn("🟢 green", md)
        self.assertIn("G", md)

    def test_green_section_keeps_full_format(self):
        # A passing run renders name + health + semaphore/date, no jobs table.
        md = render_markdown(
            "m",
            [
                {
                    "workflow": "WF-G",
                    "outcome": "GREEN",
                    "latest_url": "http://run/9",
                    "latest_ts": "2026-06-14T06:34:32Z",
                    "counts": {"broken": 0, "infra": 0, "passing": 5},
                }
            ],
        )
        self.assertIn("[WF-G](http://run/9)", md)
        self.assertIn("Health:", md)
        self.assertIn("100%", md)
        self.assertIn("🔴 0 · 🟣 0 · 🟢 5", md)
        self.assertNotIn("Failed jobs", md)

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
    def test_error_message_preferred(self):
        self.assertEqual(_error_cell({"error_message": "boom", "root_cause": "rc"}), "boom")

    def test_fallback_to_root_cause(self):
        self.assertEqual(_error_cell({"root_cause": "rc only"}), "rc only")

    def test_pipe_and_newline_escaped(self):
        self.assertEqual(_error_cell({"error_message": "a|b\nc"}), "a\\|b c")

    def test_empty(self):
        self.assertEqual(_error_cell({}), "—")

    def test_truncation(self):
        self.assertTrue(_error_cell({"error_message": "x" * 250}).endswith("…"))


class TestSevEmoji(unittest.TestCase):
    def test_infra(self):
        self.assertEqual(_sev_emoji({"status": "INFRA_FAILURE"}), "🟣")

    def test_incomplete_log_is_hourglass(self):
        # log_complete is False → truncated/killed (timed out): ⌛️ instead of 🔴.
        self.assertEqual(_sev_emoji({"status": "FAILED", "log_complete": False}), "⌛️")
        self.assertEqual(_sev_emoji({"status": "TIMEOUT", "log_complete": False}), "⌛️")

    def test_complete_or_unknown_log_is_red(self):
        self.assertEqual(_sev_emoji({"status": "FAILED", "log_complete": True}), "🔴")
        self.assertEqual(_sev_emoji({"status": "CRASHED"}), "🔴")  # absent (None) → 🔴
        self.assertEqual(_sev_emoji({"status": "FAILED", "log_complete": None}), "🔴")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
