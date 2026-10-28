from datetime import datetime, timezone
import subprocess

import pytest

from runner_failure_common import JobScanResult, job_from_dict, scan_job
from runner_failure_report import (
    LiveGithubPopulationMetrics,
    build_runner_markdown_report,
    recent_job_from_runner_api_row,
    runner_report_results,
)


def test_runner_report_preserves_triggering_disconnect_without_logs(monkeypatch) -> None:
    job = job_from_dict({"job_id": "1", "status": "completed", "conclusion": "failure"})
    triggering_result = JobScanResult(
        job=job,
        log_status="not available: runner lost communication with GitHub (HTTP 404)",
        log_checked=False,
        log_unavailable=True,
        signature_labels=("Runner disconnected",),
        fabric_missing_links="",
    )

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("The triggering failure already has confirmed metadata")

    monkeypatch.setattr("runner_failure_report.scan_jobs", unexpected_scan)

    results = runner_report_results(runner_jobs=[job], known_results=[triggering_result], gh_timeout=120, log_workers=8)

    assert results == [triggering_result]


def test_runner_report_shows_leftover_process_in_table_and_summary() -> None:
    job = job_from_dict({"job_id": "1", "status": "completed", "conclusion": "failure"})
    result = JobScanResult(
        job=job,
        log_status="fetched",
        log_checked=True,
        signature_labels=("Leftover process",),
        fabric_missing_links="",
    )
    timestamp = datetime(2026, 10, 28, tzinfo=timezone.utc)

    report = build_runner_markdown_report(
        generated_at=timestamp,
        since=timestamp,
        hours=24,
        runner_name="runner",
        job_source="test",
        runner_jobs=[job],
        scan_results=[result],
        triggering_jobs_added=0,
        live_github_metrics=LiveGithubPopulationMetrics(statuses=()),
    )

    assert "- Failure summary: `1x Leftover process`" in report
    assert "| Leftover process |" in report


@pytest.mark.parametrize(
    ("status", "success", "expected"),
    [
        ("FAILED", False, True),
        ("failure", False, True),
        ("timed_out", False, True),
        ("timed out", False, True),
        ("Completed", False, True),
        ("", False, True),
        ("success", True, False),
        ("cancelled", False, False),
        ("canceled", False, False),
        ("skipped", False, False),
        ("in_progress", False, False),
    ],
)
def test_runner_api_status_allows_leftover_process_only_for_failed_jobs(monkeypatch, status, success, expected) -> None:
    def download(command, **kwargs):
        kwargs["stdout"].write(
            b"Waiting for KMD lock 20 on /dev/tenstorrent/16, which is currently held by another handle\n"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("runner_failure_common.subprocess.run", download)
    monkeypatch.setattr("runner_failure_common.gh_api_json", lambda *_args, **_kwargs: {})
    job = recent_job_from_runner_api_row(
        {
            "GITHUB_JOB_LINK": "https://github.com/tenstorrent/tt-metal/actions/runs/1/job/2",
            "JOB_STATUS": status,
            "JOB_SUCCESS": success,
        },
        fallback_runner_name="runner",
        fallback_owner_repo="tenstorrent/tt-metal",
    )

    result = scan_job(job, timeout=120)

    assert job.status == ("in_progress" if status == "in_progress" else "completed")
    assert ("Leftover process" in result.signature_labels) is expected
