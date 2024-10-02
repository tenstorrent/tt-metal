import pytest
import pathlib

from infra.data_collection.cicd import create_cicd_json_for_data_analysis
from infra.data_collection.models import InfraErrorV1


def test_dummy():
    pass


def test_create_pipeline_json_with_passing_post_commit(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = "tests/_data/data_collection/cicd/all_post_commit_passing_10662355710/workflow.json"
    github_jobs_json_filename = (
        "tests/_data/data_collection/cicd/all_post_commit_passing_10662355710/workflow_jobs.json"
    )

    workflow_outputs_dir = pathlib.Path(
        "tests/_data/data_collection/cicd/all_post_commit_passing_10662355710/"
    ).resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    assert pipeline.github_pipeline_id == 10662355710
    assert len(pipeline.jobs) == 99, "There should be 99 jobs according to github jobs API"

    for job in pipeline.jobs:
        assert job.failure_signature is None


def test_create_pipeline_json_to_detect_generic_set_up_error_v1(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = (
        "tests/_data/data_collection/cicd/all_post_commit_runner_died_12626_10996802864/workflow.json"
    )
    github_jobs_json_filename = (
        "tests/_data/data_collection/cicd/all_post_commit_runner_died_12626_10996802864/workflow_jobs.json"
    )

    workflow_outputs_dir = pathlib.Path(
        "tests/_data/data_collection/cicd/all_post_commit_runner_died_12626_10996802864/"
    ).resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    assert pipeline.github_pipeline_id == 10996802864

    for job in pipeline.jobs:
        if job.github_job_id == 30531878948:
            assert job.failure_signature == str(InfraErrorV1.GENERIC_SET_UP_FAILURE)
        else:
            assert job.failure_signature is None
