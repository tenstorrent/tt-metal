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


def get_non_success_jobs_(pipeline):
    is_failing_job = lambda job_: not job_.job_success
    matching_jobs = list(filter(is_failing_job, pipeline.jobs))

    return matching_jobs


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

    failing_jobs = get_non_success_jobs_(pipeline)

    assert len(failing_jobs) == 1

    for job in pipeline.jobs:
        if job.github_job_id == 30531878948:
            assert job.failure_signature == str(InfraErrorV1.GENERIC_SET_UP_FAILURE)
        else:
            assert job.failure_signature is None


def test_create_pipeline_json_to_detect_generic_set_up_error_v1_among_other_failures(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = (
        "tests/_data/data_collection/cicd/all_post_commit_runner_died_12626_11110261767/workflow.json"
    )
    github_jobs_json_filename = (
        "tests/_data/data_collection/cicd/all_post_commit_runner_died_12626_11110261767/workflow_jobs.json"
    )

    workflow_outputs_dir = pathlib.Path(
        "tests/_data/data_collection/cicd/all_post_commit_runner_died_12626_11110261767/"
    ).resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    failing_jobs = get_non_success_jobs_(pipeline)

    # some are skipped
    assert len(failing_jobs) == 4

    assert pipeline.github_pipeline_id == 11110261767

    for job in pipeline.jobs:
        if job.github_job_id == 30868260202:
            assert job.failure_signature == str(InfraErrorV1.GENERIC_SET_UP_FAILURE)
        else:
            assert job.failure_signature is None


def test_create_pipeline_json_for_run_github_timed_out_job(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = (
        "tests/_data/data_collection/cicd/all_post_commit_github_timeout_11034942442/workflow.json"
    )
    github_jobs_json_filename = (
        "tests/_data/data_collection/cicd/all_post_commit_github_timeout_11034942442/workflow_jobs.json"
    )

    workflow_outputs_dir = pathlib.Path(
        "tests/_data/data_collection/cicd/all_post_commit_github_timeout_11034942442/"
    ).resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    assert pipeline.github_pipeline_id == 11034942442

    for job in pipeline.jobs:
        if job.github_job_id == 30868260202:
            assert len(job.tests) > 0
