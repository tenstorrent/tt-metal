#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI digest: report the current state of watched workflows.

Thin aggregator. For each watched workflow it finds the latest completed
scheduled run and reads that run's machine-readable ``ai_run_summary[_<scope>]_r<run>_a<attempt>``
artifact — a factual JSON the ai_summary/run action already produces (succeeded
/ failed / infra_failure jobs). The digest does no classification of its own; it
collects those per-run summaries and renders them at one point so a team can
react. Stateless by design — no history, no incident tracking.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import zipfile
from dataclasses import dataclass
from types import SimpleNamespace
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


def _summary_name_re(run_id: int) -> "re.Pattern[str]":
    """Matches this run's report artifacts, capturing the attempt.

    ``ai_run_summary[_<scope>]_r<run>_a<n>``. The scope is present when a
    workflow is invoked more than once per run and publishes one report per
    invocation. Anchored at both ends, so a longer run id (``r421`` for run 42),
    an unrelated prefix and a suffixed copy (``_a3_backup``) are all rejected.

    The scope segment is ``[^_]+`` rather than a character class: the producer
    (``qualified_stem``/``slugify_scope`` in
    tenstorrent/tt-github-actions ``.github/actions/ai_summary/tool/common/artifact_names.py``)
    slugifies every non-alphanumeric character to ``-``, so a slug never
    contains ``_`` but may hold any alphanumeric, Unicode included.
    """
    return re.compile(rf"^ai_run_summary_(?:[^_]+_)?r{run_id}_a(\d+)$")


def _artifacts_jq() -> str:
    """jq projection listing every artifact as ``name\\tcreated_at\\tid``.

    Projection only — the name match happens in Python, so one engine decides
    it and the tests exercise the same code CI runs.
    """
    return '.artifacts[] | "\\(.name)\\t\\(.created_at)\\t\\(.id)"'


def _pick_latest_report(listing: list[str], run_id: int) -> tuple[str, str] | None:
    """``(name, id)`` of the highest-attempt report, or None if none match.

    The attempt in the name ranks first; created_at then the higher id settle
    ties. Ties are real: an unscoped workflow invoked more than once per run
    publishes one report per invocation under a single name, and only one of
    them can be read — so that case is reported rather than resolved quietly.

    Lines that don't parse are skipped, not raised: one bad line must not cost
    the caller every other workflow's result.
    """
    name_re = _summary_name_re(run_id)
    ranked = []
    for line in listing:
        if not line.strip():
            continue
        # Artifact names may contain tabs — GitHub rejects only " : < > | * ?
        # \r \n \ / — so just the two machine-generated trailing fields split off.
        parts = line.rsplit("\t", 2)
        if len(parts) != 3 or not (parts[2].isascii() and parts[2].isdigit()):
            print(f"Skipping malformed artifact listing line: {line!r}", file=sys.stderr)
            continue
        name, created_at, art_id = parts
        m = name_re.match(name)
        if m:
            ranked.append(((int(m.group(1)), created_at, int(art_id)), name, art_id))
    if not ranked:
        return None
    best = max(ranked, key=lambda r: r[0])
    tied = [r for r in ranked if r[0][0] == best[0][0]]
    if len(tied) > 1:
        print(
            f"Run {run_id} has {len(tied)} reports at attempt {best[0][0]}; "
            f"reading {best[1]} and ignoring the rest — the producing workflow "
            f"is invoked more than once per run without a distinct scope",
            file=sys.stderr,
        )
    return best[1], best[2]


def fetch_run_summary(repo: str, run_id: int) -> dict | None:
    """Download the highest attempt's ``ai_run_summary[_<scope>]_r<run>_a<n>`` JSON.

    Each attempt uploads its own report, so every attempt stays downloadable and
    the highest wins. Returns None when the run has no such artifact — the
    workflow doesn't run ai_summary/run, or it produced only markdown — so the
    caller can fall back to the run's conclusion.
    """
    listing = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
            "--jq",
            _artifacts_jq(),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    latest = _pick_latest_report(listing, run_id)
    if latest is None:
        return None
    art_name, art_id = latest
    with tempfile.TemporaryDirectory() as d:
        zip_path = os.path.join(d, "artifact.zip")
        with open(zip_path, "wb") as fh:
            subprocess.run(["gh", "api", f"repos/{repo}/actions/artifacts/{art_id}/zip"], stdout=fh, check=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(d)
        # Escaped: GitHub permits [ and ] in an artifact name, and glob would
        # read them as a character class and resolve to a different file.
        matches = _glob.glob(os.path.join(d, "**", _glob.escape(art_name) + ".json"), recursive=True)
        if not matches:
            return None  # artifact present but markdown-only
        with open(matches[0], encoding="utf-8") as fh:
            return json.load(fh)


# Conclusions where the run genuinely broke, as opposed to merely not finishing
# (cancelled/skipped/neutral/…), which we can't score either way.
_BROKEN_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}


@dataclass(frozen=True)
class RunReport:
    """One watched run's result: the failure rows plus the run's own conclusion.

    ``outcome`` is derived, never stored, so it cannot drift from the rows it
    summarizes. With no rows to show, the conclusion decides: a broken one
    (matrix generation died before any leg, checkout failed) is REAL_FAIL — an
    empty summary must never read as GREEN — while a merely-unfinished one
    (cancelled, skipped) is UNKNOWN rather than a false red.

    failed/infra rows are passed through verbatim — they already carry job_name,
    job_url, status, category, error_message, root_cause.
    """

    conclusion: str
    failed: list[dict]
    infra: list[dict]
    passing: int

    @property
    def outcome(self) -> str:
        if self.failed:
            return "REAL_FAIL"
        if self.infra:
            return "INFRA"
        if self.conclusion == "success":
            return "GREEN"
        return "REAL_FAIL" if self.conclusion in _BROKEN_CONCLUSIONS else "UNKNOWN"


def summarize_run(data: dict, conclusion: str) -> RunReport:
    """Build a RunReport from a run-summary JSON and the run's GH conclusion."""
    return RunReport(
        conclusion=conclusion,
        failed=data.get("failed") or [],
        infra=data.get("infra_failure") or [],
        passing=len(data.get("succeeded") or []),
    )


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


def _health_bar(counts: dict, width: int = 20) -> str:
    """Bar of passing / total jobs. No jobs scores 0% — a failed run with no
    per-job detail is 0% healthy, and the bar keeps every section uniform."""
    total = counts.get("broken", 0) + counts.get("infra", 0) + counts.get("passing", 0)
    pct = round(100 * counts.get("passing", 0) / total) if total else 0
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
    attempt = r.get("run_attempt")
    hdr = _link(r) + (f" (attempt {attempt})" if attempt and attempt > 1 else "")
    out = [f"### {hdr}"]
    c = r.get("counts") or {}
    broken, infra, passing = c.get("broken", 0), c.get("infra", 0), c.get("passing", 0)
    if broken or infra or (r.get("outcome") == "GREEN" and passing):
        out.append(_health_bar(c))
        out.append(f"🔴 {broken} · 🟣 {infra} · 🟢 {passing}{when}")
    elif r.get("outcome") == "GREEN":
        # Green via the run-conclusion fallback (no per-job counts available).
        out.append(f"🟢 green{when}")
    elif passing:
        # Broken run whose summarized legs all passed — the failure is outside
        # them, so neither a 100% nor a 0% bar is meaningful. State it plainly.
        out.append(f"🔴 run failed; {passing} summarized leg(s) passed, none failed{when}")
    else:
        out.append(_health_bar(c))  # 0% — uniform with scored sections
        out.append(f"🔴 no per-job summary{when}")
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
        conclusion = run.get("conclusion") or ""
        data = fetch_run_summary(repo, run["databaseId"])
        if data is None:
            # No machine-readable summary (uninstrumented workflow, or the run
            # predates JSON output); classify from the conclusion alone, the same
            # rule a present-but-empty summary gets.
            outcome = RunReport(conclusion, [], [], 0).outcome
            result = {**base, **meta, "outcome": outcome}
            if outcome == "UNKNOWN":
                result["note"] = "no ai_run_summary artifact"
            return result
        report = summarize_run(data, conclusion)
        return {
            **base,
            **meta,
            "outcome": report.outcome,
            "run_attempt": data.get("run_attempt"),
            "real_jobs": report.failed,
            "infra_jobs": report.infra,
            "counts": {"broken": len(report.failed), "infra": len(report.infra), "passing": report.passing},
        }
    except subprocess.CalledProcessError as exc:
        # One flaky gh call must not discard the other workflows' results.
        err = ((exc.stderr or "").strip().splitlines() or ["gh command failed"])[-1]
        return {**base, "outcome": "ERROR", "note": err[:200]}
    except (json.JSONDecodeError, ValueError, OSError, zipfile.BadZipFile) as exc:
        # A corrupt/truncated artifact, or any parse error, marks only this
        # workflow ERROR and never aborts the others' reports.
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
        r = summarize_run(
            {"failed": [{"job_name": "a"}], "infra_failure": [{"job_name": "b"}], "succeeded": [{}, {}]},
            "failure",
        )
        self.assertEqual((r.outcome, len(r.failed), len(r.infra), r.passing), ("REAL_FAIL", 1, 1, 2))

    def test_infra_when_only_infra(self):
        r = summarize_run({"infra_failure": [{"job_name": "b"}], "succeeded": [{}]}, "failure")
        self.assertEqual((r.outcome, len(r.infra), r.passing), ("INFRA", 1, 1))

    def test_green_when_success_and_only_success(self):
        r = summarize_run({"succeeded": [{}, {}, {}]}, "success")
        self.assertEqual((r.outcome, r.failed, r.infra, r.passing), ("GREEN", [], [], 3))

    def test_empty_summary_but_failed_conclusion_is_real_fail(self):
        # A run that failed before producing any leg (matrix generation died,
        # checkout failed) has an empty summary — must not read as green.
        r = summarize_run({}, "failure")
        self.assertEqual((r.outcome, r.passing), ("REAL_FAIL", 0))

    def test_empty_summary_and_success_is_green(self):
        self.assertEqual(summarize_run({}, "success").outcome, "GREEN")

    def test_cancelled_or_skipped_with_no_rows_is_unknown(self):
        # A run that didn't finish isn't broken — don't render it as a false red.
        self.assertEqual(summarize_run({}, "cancelled").outcome, "UNKNOWN")
        self.assertEqual(summarize_run({}, "skipped").outcome, "UNKNOWN")


class TestPickLatestReport(unittest.TestCase):
    NAME = "ai_run_summary_r42"

    def _id(self, lines):
        got = _pick_latest_report(lines, 42)
        return got[1] if got else None

    def test_higher_attempt_wins_over_newer_created_at(self):
        # The attempt is authoritative; created_at is incidental.
        lines = [
            f"{self.NAME}_a2\t2026-07-23T01:00:00Z\t100",
            f"{self.NAME}_a1\t2026-07-23T09:00:00Z\t900",
        ]
        self.assertEqual(self._id(lines), "100")

    def test_created_at_then_id_settle_same_attempt_ties(self):
        newer_lower_id = [
            f"{self.NAME}_a1\t2026-07-23T01:00:00Z\t900",
            f"{self.NAME}_a1\t2026-07-23T09:00:00Z\t100",
        ]
        self.assertEqual(self._id(newer_lower_id), "100")
        same_time = [
            f"{self.NAME}_a1\t2026-07-23T01:00:00Z\t100",
            f"{self.NAME}_a1\t2026-07-23T01:00:00Z\t200",
        ]
        self.assertEqual(self._id(same_time), "200")

    def test_a_tie_is_reported_because_the_other_reports_are_lost(self):
        # An unscoped workflow invoked more than once per run publishes one
        # report per invocation under a single name; only one can be read.
        lines = [
            f"{self.NAME}_a1\t2026-07-23T01:00:00Z\t100",
            f"{self.NAME}_a1\t2026-07-23T02:00:00Z\t200",
            f"{self.NAME}_a1\t2026-07-23T03:00:00Z\t300",
        ]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertEqual(self._id(lines), "300")
        self.assertIn("3 reports at attempt 1", buf.getvalue())

    def test_returns_the_winning_name_for_addressing_the_report(self):
        lines = [f"{self.NAME}_a2\t2026-07-23T01:00:00Z\t100"]
        self.assertEqual(_pick_latest_report(lines, 42), (f"{self.NAME}_a2", "100"))

    def test_malformed_lines_are_skipped_not_raised(self):
        # A ValueError here escapes check_workflow's guards and leaves the
        # digest silent instead of red.
        lines = [
            "not-a-tabbed-line",
            "",
            f"{self.NAME}_a1\t2026-07-23T01:00:00Z\tnope",
            f"{self.NAME}_a1\t2026-07-23T01:00:00Z\t100",
        ]
        self.assertEqual(self._id(lines), "100")

    def test_a_non_ascii_digit_id_is_skipped_not_raised(self):
        # "²".isdigit() is True while int("²") raises, and a ValueError here
        # would abort every remaining workflow.
        lines = [f"{self.NAME}_a1\t2026-07-23T01:00:00Z\t²", f"{self.NAME}_a1\t2026-07-23T01:00:00Z\t100"]
        self.assertEqual(self._id(lines), "100")

    def test_a_tab_bearing_name_is_accepted(self):
        # GitHub rejects " : < > | * ? in artifact names but not tab, so the
        # split takes only the two machine-generated trailing fields.
        lines = [f"ai_run_summary_r42\tstray_a1\t2026-07-23T01:00:00Z\t100"]
        self.assertIsNone(self._id(lines))

    def test_nothing_matching_is_none(self):
        lines = ["garbage", "", f"ai_run_summary_r7_a1\t2026-07-23T01:00:00Z\t100"]
        self.assertIsNone(_pick_latest_report(lines, 42))


class TestFetchRunSummary(unittest.TestCase):
    """The download seam: which artifact is asked for, and which file is read."""

    def _fake_gh(self, listing, members):
        def run(argv, **kw):
            if "--jq" in argv:
                return SimpleNamespace(stdout="\n".join(listing), stderr="", returncode=0)
            self.requested = argv[-1]
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for name, body in members.items():
                    z.writestr(name, body)
            kw["stdout"].write(buf.getvalue())
            return SimpleNamespace(stdout=b"", stderr="", returncode=0)

        return run

    def test_reads_the_highest_attempt_report_by_exact_name(self):
        # A scope containing [ ] would make an unescaped glob resolve to the
        # sibling file instead. GitHub permits both characters.
        name = "ai_run_summary_[ab]_r42_a2"
        listing = [
            f"ai_run_summary_x_r42_a1\t2026-07-23T09:00:00Z\t900",
            f"{name}\t2026-07-23T01:00:00Z\t100",
        ]
        members = {
            f"{name}.json": json.dumps({"run_id": "42", "failed": [], "infra_failure": [], "succeeded": []}),
            "ai_run_summary_a_r42_a2.json": json.dumps({"run_id": "WRONG"}),
        }
        with unittest.mock.patch("subprocess.run", side_effect=self._fake_gh(listing, members)):
            data = fetch_run_summary("o/r", 42)
        self.assertEqual(data["run_id"], "42")
        self.assertIn("/100/", self.requested)

    def test_no_matching_artifact_is_none(self):
        listing = ["ai_run_summary_r7_a1\t2026-07-23T01:00:00Z\t100"]
        with unittest.mock.patch("subprocess.run", side_effect=self._fake_gh(listing, {})):
            self.assertIsNone(fetch_run_summary("o/r", 42))


class TestSummaryNameRe(unittest.TestCase):
    def _matches(self, name):
        return _summary_name_re(42).match(name) is not None

    def test_accepts_plain_and_scoped(self):
        for name in ("ai_run_summary_r42_a3", "ai_run_summary_ubuntu-24-04_r42_a1"):
            self.assertTrue(self._matches(name), name)

    def test_accepts_a_unicode_scope(self):
        # The producer keeps any alphanumeric, so the segment cannot be an
        # ASCII character class.
        self.assertTrue(self._matches("ai_run_summary_ubuntú_r42_a1"))

    def test_captures_the_attempt(self):
        self.assertEqual(_summary_name_re(42).match("ai_run_summary_s_r42_a13").group(1), "13")

    def test_rejects_near_misses(self):
        # r421 for run 42, another run, another artifact kind, a stray prefix,
        # an unversioned name, a suffixed copy, and a missing attempt.
        for name in (
            "ai_run_summary_r421_a1",
            "ai_run_summary_r7_a1",
            "ai_job_summary_r42_a1_j99",
            "xai_run_summary_r42_a1",
            "ai_run_summary_42",
            "ai_run_summary_r42_a3_backup",
            "ai_run_summary_r42",
        ):
            self.assertFalse(self._matches(name), name)


class TestArtifactsJq(unittest.TestCase):
    def test_projects_the_three_fields_resolution_needs(self):
        # Dropping a field makes every line fail the parse guard, so every
        # workflow silently degrades to its run conclusion.
        self.assertEqual(_artifacts_jq(), '.artifacts[] | "\\(.name)\\t\\(.created_at)\\t\\(.id)"')


class TestRender(unittest.TestCase):
    def _broken(self):
        return {
            "workflow": "WF-A",
            "label": "WF-A",
            "outcome": "REAL_FAIL",
            "latest_url": "http://run/2",
            "latest_ts": "2026-06-14T06:34:32Z",
            "run_attempt": 2,
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
        self.assertIn("(attempt 2)", md)  # re-run exposed in the header
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
        self.assertIn("no per-job summary", md)
        self.assertIn("0%", md)  # health bar present and at zero, uniform with scored sections
        self.assertNotIn("Failed jobs", md)

    def test_broken_run_with_only_passing_legs_is_not_100_percent(self):
        # Run concluded failure but every summarized leg passed (failure outside
        # them): must not render as a healthy 100% bar.
        md = render_markdown(
            "m",
            [
                {
                    "workflow": "WF-P",
                    "outcome": "REAL_FAIL",
                    "latest_url": "u",
                    "real_jobs": [],
                    "infra_jobs": [],
                    "counts": {"broken": 0, "infra": 0, "passing": 5},
                }
            ],
        )
        self.assertIn("run failed", md)
        self.assertIn("5 summarized leg(s) passed", md)
        self.assertNotIn("100%", md)

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
