from datetime import datetime, timezone

from runner_failure_common import JobScanResult, job_from_dict
from runner_failure_report import LiveGithubPopulationMetrics, build_runner_markdown_report, runner_report_results


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
