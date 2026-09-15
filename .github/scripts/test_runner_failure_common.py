from runner_failure_common import matching_job_metadata_signature_labels, recent_job_from_api


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
