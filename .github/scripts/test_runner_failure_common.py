import subprocess

import pytest

from runner_failure_common import (
    RecentJob,
    fetch_github_job_log,
    job_from_dict,
    matching_job_metadata_signature_labels,
    matching_signature_labels,
    recent_job_from_api,
    result_to_dict,
    scan_job,
    scan_result_from_dict,
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


@pytest.mark.parametrize("requested_shape", ["[4, 8]", "[8, 4]"])
def test_unrotatable_system_mesh_signature(requested_shape) -> None:
    log_text = (
        "TT_THROW @ /work/tt_metal/distributed/system_mesh.cpp:224: "
        f"Requested mesh is too big and is not rotatable: MeshShape({requested_shape}) "
        "and SystemMesh MeshShape([32, 1]), offset MeshCoordinate([0, 0])"
    )

    assert matching_signature_labels(log_text) == ["Wrong mesh shape"]


def test_mgd_topology_signature() -> None:
    log_text = "Graph specified in MGD could not fit in the discovered physical topology"

    assert matching_signature_labels(log_text) == ["Fabric link down (MGD topology)"]


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


def test_active_job_logs_are_deferred_without_downloading(monkeypatch) -> None:
    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("Active job logs should not be downloaded")

    monkeypatch.setattr("runner_failure_common.subprocess.run", unexpected_download)
    job = job_from_dict({"job_id": "1", "status": "in_progress"})

    result = scan_job(job, timeout=120)

    assert not result.log_checked
    assert result.log_unavailable
    assert "in_progress" in result.log_status
    assert scan_result_from_dict(result_to_dict(result)) == result


@pytest.mark.parametrize(
    ("annotation", "unavailable"),
    [
        ("The self-hosted runner lost communication with the server.", True),
        ("Process completed with exit code 1.", False),
    ],
)
def test_log_404_is_unavailable_only_with_runner_disconnect_evidence(monkeypatch, annotation, unavailable) -> None:
    def failed_download(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="gh: HTTP 404")

    def metadata(endpoint, **_kwargs):
        if endpoint == "repos/tenstorrent/tt-metal/actions/jobs/1":
            return {
                "status": "completed",
                "check_run_url": "https://api.github.com/repos/tenstorrent/tt-metal/check-runs/2",
            }
        assert endpoint == "repos/tenstorrent/tt-metal/check-runs/2/annotations?per_page=100"
        return [[{"message": annotation}]]

    monkeypatch.setattr("runner_failure_common.subprocess.run", failed_download)
    monkeypatch.setattr("runner_failure_common.gh_api_json", metadata)
    job = job_from_dict({"owner_repo": "tenstorrent/tt-metal", "job_id": "1", "status": "completed"})

    result = scan_job(job, timeout=120)

    assert not result.log_checked
    assert result.log_unavailable is unavailable
    assert "HTTP 404" in result.log_status
    assert result.signature_labels == (("Runner disconnected",) if unavailable else ())
    assert scan_result_from_dict(result_to_dict(result)) == result


def test_log_404_counts_as_failure_when_metadata_lookup_fails(monkeypatch) -> None:
    def failed_download(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="gh: HTTP 404")

    def failed_metadata(*_args, **_kwargs):
        raise RuntimeError("gh api failed: HTTP 403")

    monkeypatch.setattr("runner_failure_common.subprocess.run", failed_download)
    monkeypatch.setattr("runner_failure_common.gh_api_json", failed_metadata)
    job = job_from_dict({"owner_repo": "tenstorrent/tt-metal", "job_id": "1", "status": "completed"})

    result = fetch_github_job_log(job, timeout=120)

    assert not result.unavailable
    assert result.status == "gh api failed: gh: HTTP 404"
