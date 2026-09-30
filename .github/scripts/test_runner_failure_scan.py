from runner_failure_common import JobScanResult, RecentJob
from runner_failure_scan import log_download_counts


def make_job(job_id: str) -> RecentJob:
    return RecentJob(
        owner_repo="tenstorrent/tt-metal",
        workflow="vllm-model-tests",
        workflow_id="vllm-model-tests.yaml",
        run_id="1",
        run_attempt="1",
        run_url="https://example.test/run",
        job_id=job_id,
        name=f"job-{job_id}",
        runner_name="runner",
        status="completed",
        conclusion="failure",
        html_url=f"https://example.test/job/{job_id}",
        started_at="",
        completed_at="",
        setup_runner_conclusion="",
    )


def make_result(job: RecentJob, *, log_checked: bool) -> JobScanResult:
    return JobScanResult(
        job=job,
        log_status="fetched" if log_checked else "gh api failed",
        log_checked=log_checked,
        signature_labels=(),
        fabric_missing_links="",
    )


def test_log_download_counts_handles_no_attempts() -> None:
    assert log_download_counts([], []) == (0, 0, 0)


def test_log_download_counts_includes_failed_and_missing_results() -> None:
    jobs = [make_job("1"), make_job("2"), make_job("3")]
    results = [make_result(jobs[0], log_checked=True), make_result(jobs[1], log_checked=False)]

    assert log_download_counts(jobs, results) == (3, 1, 2)
