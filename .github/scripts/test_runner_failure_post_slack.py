from runner_failure_common import JobScanResult, RecentJob
from runner_failure_post_slack import failure_summary_for_slack_cell


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
