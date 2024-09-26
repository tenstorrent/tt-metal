import pytest
import pathlib

from infra.data_collection.cicd import create_cicd_json_for_data_analysis


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
