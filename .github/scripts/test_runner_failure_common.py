import subprocess

from runner_failure_common import (
    RecentJob,
    fetch_github_job_log,
    matching_job_metadata_signature_labels,
    matching_signature_labels,
    recent_job_from_api,
)


def test_setup_runner_failure_signature_matches_step_conclusion() -> None:
    job = recent_job_from_api(
        owner_repo="tenstorrent/tt-metal",
        workflow_name="all-model-tests",
        workflow_id="all-model-tests.yaml",
        run={"id": 31525883863, "run_attempt": 1, "html_url": "https://example.test/run"},
        job={
            "id": 93905365734,
            "name": "t3-e2e-tests / MNIST MLP e2e tests [wh_n150]",
            "conclusion": "failure",
            "steps": [
                {"name": "Set up job", "conclusion": "success"},
                {"name": "Set up runner", "conclusion": "failure"},
            ],
        },
    )

    labels = matching_job_metadata_signature_labels(job)
    assert job.setup_runner_conclusion == "failure", job
    assert "Set up runner failure" in labels, labels


def test_setup_runner_failure_signature_ignores_successful_step() -> None:
    job = recent_job_from_api(
        owner_repo="tenstorrent/tt-metal",
        workflow_name="all-model-tests",
        workflow_id="all-model-tests.yaml",
        run={"id": 31525883863, "run_attempt": 1, "html_url": "https://example.test/run"},
        job={
            "id": 93905365734,
            "name": "t3-e2e-tests / MNIST MLP e2e tests [wh_n150]",
            "conclusion": "failure",
            "steps": [
                {"name": "Set up job", "conclusion": "success"},
                {"name": "Set up runner", "conclusion": "success"},
            ],
        },
    )

    labels = matching_job_metadata_signature_labels(job)
    assert labels == [], labels


def test_eth_heartbeat_timeout_signature() -> None:
    log_text = (
        "RuntimeError: TT standard-DP device discovery failed: RuntimeError: "
        "Timed out waiting for \x1b[36;1mETH heartbeat\x1b[0m on device ASIC ID: 87033183734870352, "
        "ETH core e9-0 (NOC0) to advance. Stuck at 0xabcdae0a"
    )

    assert "ETH heartbeat timeout" in matching_signature_labels(log_text)


def test_job_log_fetch_allows_escape_sequences(monkeypatch) -> None:
    captured_command: list[str] = []

    def fake_run(command, **_kwargs):
        captured_command.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="\x1b[36;1mlog\x1b[0m", stderr="")

    monkeypatch.setattr("runner_failure_common.subprocess.run", fake_run)
    job = RecentJob(
        owner_repo="tenstorrent/tt-metal",
        workflow="vllm-model-tests",
        workflow_id="vllm-model-tests.yaml",
        run_id="35290238153",
        run_attempt="1",
        run_url="https://example.test/run",
        job_id="105436322615",
        name="vllm-tests / test",
        runner_name="runner",
        status="completed",
        conclusion="failure",
        html_url="https://example.test/job",
        started_at="",
        completed_at="",
        setup_runner_conclusion="",
    )

    result = fetch_github_job_log(job, timeout=120)

    assert captured_command == [
        "gh",
        "api",
        "--allow-escape-sequences",
        "repos/tenstorrent/tt-metal/actions/jobs/105436322615/logs",
    ]
    assert result.log_text == "\x1b[36;1mlog\x1b[0m"
