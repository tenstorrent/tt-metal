import argparse
import json
from dataclasses import replace

from runner_failure_common import JobScanResult, RecentJob
from runner_failure_scan import is_failed_job, log_download_counts, main


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
    assert log_download_counts([], []) == (0, 0, 0, 0)


def test_log_download_counts_includes_failed_and_missing_results() -> None:
    jobs = [make_job("1"), make_job("2"), make_job("3")]
    results = [make_result(jobs[0], log_checked=True), make_result(jobs[1], log_checked=False)]

    assert log_download_counts(jobs, results) == (3, 1, 2, 0)


def test_unavailable_logs_are_excluded_from_health() -> None:
    jobs = [make_job("1"), make_job("2"), make_job("3")]
    results = [
        make_result(jobs[0], log_checked=True),
        replace(make_result(jobs[1], log_checked=False), log_unavailable=True),
        make_result(jobs[2], log_checked=False),
    ]

    assert log_download_counts(jobs, results) == (2, 1, 1, 1)


def test_incomplete_jobs_are_not_selected_as_failures() -> None:
    job = make_job("1")

    assert is_failed_job(job)
    assert not is_failed_job(replace(job, status="in_progress"))
    assert not is_failed_job(replace(job, status="queued"))


def test_confirmed_disconnect_is_reported_once_but_unknown_download_failure_is_retried(monkeypatch, tmp_path) -> None:
    disconnected, unknown = make_job("1"), make_job("2")
    disconnected_result = replace(
        make_result(disconnected, log_checked=False),
        log_unavailable=True,
        signature_labels=("Runner disconnected",),
    )
    unknown_result = make_result(unknown, log_checked=False)
    args = argparse.Namespace(
        hours=24,
        gh_timeout=120,
        log_workers=8,
        config=tmp_path / "config.yaml",
        state_in=None,
        state_out=tmp_path / "state.json",
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        force_fresh=False,
    )
    scanned_jobs = []

    def scan(jobs, **_kwargs):
        scanned_jobs.append(jobs)
        return [result for result in (disconnected_result, unknown_result) if result.job in jobs]

    monkeypatch.setattr("runner_failure_scan.parse_args", lambda: args)
    monkeypatch.setattr("runner_failure_scan.ensure_gh_available", lambda: None)
    monkeypatch.setattr("runner_failure_scan.load_workflows", lambda _path: [])
    monkeypatch.setattr("runner_failure_scan.list_recent_jobs", lambda *_args, **_kwargs: [disconnected, unknown])
    monkeypatch.setattr("runner_failure_scan.scan_jobs", scan)

    assert main() == 0
    report = json.loads(args.report_json.read_text())
    assert report["counts"]["runner_failure_jobs"] == 1
    assert report["runner_failures"]["runner"][0]["signatures"] == ["Runner disconnected"]
    assert report["runner_failures"]["runner"][0]["log_checked"] is False

    args.state_in = args.state_out
    assert main() == 0
    assert scanned_jobs == [[disconnected, unknown], [unknown]]
    report = json.loads(args.report_json.read_text())
    assert report["counts"]["runner_failure_jobs"] == 0
