import pytest

from runner_failure_common import JobScanResult, RecentJob
from runner_failure_post_slack import (
    failure_summary_for_slack_cell,
    format_scan_health_alert,
    scan_health_from_report,
    should_post_health_alert,
)


@pytest.mark.parametrize("signature", ["Set up runner failure", "Runner disconnected"])
def test_failure_summary_shows_metadata_signature_when_log_not_checked(signature) -> None:
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
        signature_labels=(signature,),
        fabric_missing_links="",
    )

    assert failure_summary_for_slack_cell(result) == signature


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


def test_unavailable_logs_do_not_raise_health_alerts() -> None:
    report = {
        "counts": {
            "jobs_to_scan": 4,
            "log_download_attempts": 0,
            "log_download_successes": 0,
            "log_download_failures": 0,
            "log_download_unavailable": 4,
        },
        "scan_results": [
            {
                "log_checked": False,
                "log_unavailable": True,
                "log_status": "not available: runner lost communication with GitHub (HTTP 404)",
            }
            for _ in range(4)
        ],
    }

    health = scan_health_from_report(report)

    assert health.attempts == 0
    assert health.unavailable == 4
    assert not health.failure_statuses
    assert not should_post_health_alert(health, 0.10)


def test_real_download_errors_still_alert_with_unavailable_logs() -> None:
    report = {
        "counts": {"jobs_to_scan": 5},
        "scan_results": [
            {"log_checked": True},
            {"log_checked": False, "log_status": "gh api timed out"},
            {"log_checked": False, "log_unavailable": True},
            {"log_checked": False, "log_unavailable": True},
        ],
    }

    health = scan_health_from_report(report)

    assert health.attempts == 3
    assert health.failures == 2
    assert should_post_health_alert(health, 0.10)
    assert health.failure_statuses == {"gh api timed out": 1, "job scan did not return a result": 1}
