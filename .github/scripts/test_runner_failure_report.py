from runner_failure_common import JobScanResult, job_from_dict
from runner_failure_report import runner_report_results


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
