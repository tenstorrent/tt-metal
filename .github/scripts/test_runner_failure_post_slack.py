from runner_failure_common import JobScanResult, RecentJob
from runner_failure_post_slack import (
    failure_summary_for_slack_cell,
    format_scan_health_alert,
    scan_health_from_report,
    should_post_health_alert,
)


def test_failure_summary_shows_metadata_signature_when_log_not_checked() -> None:
    result = JobScanResult(
        job=RecentJob(
            owner_repo="tenstorrent/tt-metal",
            workflow="all-model-tests",
            workflow_id="all-model-tests.yaml",
            run_id="1",
            run_attempt="1",
            run_url="https://example.test/run",
            job_id="2",
            name="job",
            runner_name="runner",
            status="completed",
            conclusion="failure",
            html_url="https://example.test/job",
            started_at="",
            completed_at="",
            setup_runner_conclusion="failure",
        ),
        log_status="gh api timed out",
        log_checked=False,
        signature_labels=("Set up runner failure",),
        fabric_missing_links="",
    )

    assert failure_summary_for_slack_cell(result) == "Set up runner failure"


def test_scan_health_alerts_above_ten_percent() -> None:
    report = {
        "counts": {
            "jobs_to_scan": 20,
            "log_download_attempts": 20,
            "log_download_successes": 17,
            "log_download_failures": 3,
        },
        "scan_results": [
            {"log_checked": False, "log_status": "gh api timed out"},
            {"log_checked": False, "log_status": "gh api timed out"},
            {"log_checked": False, "log_status": "gh api failed: HTTP 500"},
        ],
    }

    health = scan_health_from_report(report)

    assert health.failure_rate == 0.15
    assert should_post_health_alert(health, 0.10)
    message = format_scan_health_alert(health, 0.10, "https://example.test/run")
    assert "3 of 20 job logs (15.0%)" in message
    assert "2x gh api timed out" in message
    assert "<https://example.test/run|View workflow run>" in message


def test_scan_health_does_not_alert_at_threshold() -> None:
    report = {
        "counts": {
            "log_download_attempts": 20,
            "log_download_successes": 18,
            "log_download_failures": 2,
        },
        "scan_results": [],
    }

    health = scan_health_from_report(report)

    assert health.failure_rate == 0.10
    assert not should_post_health_alert(health, 0.10)
