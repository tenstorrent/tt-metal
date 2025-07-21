import pytest
import pathlib

from infra.data_collection.github import workflows
from infra.data_collection.cicd import create_cicd_json_for_data_analysis
from infra.data_collection.models import InfraErrorV1, TestErrorV1
from infra.data_collection.pydantic_models import JobStatus

INFRA_TESTS_DIR = pathlib.Path(__file__).parent.parent


def test_dummy():
    pass


def test_create_pipeline_json_with_passing_post_commit(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_passing_10662355710/workflow.json"
    )
    github_jobs_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_passing_10662355710/workflow_jobs.json"
    )

    workflow_outputs_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_passing_10662355710/"
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


def test_create_pipeline_json_to_detect_job_timeout_error_v1(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_runner_died_12626_10996802864/workflow.json"
    )
    github_jobs_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_runner_died_12626_10996802864/workflow_jobs.json"
    )

    workflow_outputs_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_runner_died_12626_10996802864/"
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
            assert job.failure_signature == str(InfraErrorV1.JOB_CUMULATIVE_TIMEOUT_FAILURE)
            assert job.failure_description is not None
            assert job.job_status == JobStatus.failure
        else:
            assert job.failure_signature is None
            assert job.failure_description is None


def test_create_pipeline_json_to_detect_runner_comm_error_v1_among_other_failures(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_runner_died_12626_11110261767/workflow.json"
    )
    github_jobs_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_runner_died_12626_11110261767/workflow_jobs.json"
    )

    workflow_outputs_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_runner_died_12626_11110261767/"
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

    # some are skipped (skipped jobs are considered success)
    assert len(failing_jobs) == 2

    assert pipeline.github_pipeline_id == 11110261767

    for job in pipeline.jobs:
        if job.github_job_id == 30868260202:
            assert job.failure_signature == str(InfraErrorV1.RUNNER_COMM_FAILURE)
            assert job.failure_description is not None
            assert job.job_status == JobStatus.failure
        else:
            assert job.failure_signature is None
            assert job.failure_description is None


def test_create_pipeline_json_for_run_github_timed_out_job(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_github_timeout_11034942442/workflow.json"
    )
    github_jobs_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_github_timeout_11034942442/workflow_jobs.json"
    )

    workflow_outputs_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_github_timeout_11034942442/"
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
            assert job.job_status == JobStatus.failure
        if job.github_job_id == 30650754720:
            assert job.tt_smi_version is not None


def test_create_pipeline_json_for_timeout_bad_testcase(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_timeout_bad_testcase_13077087562/workflow.json"
    )
    github_jobs_json_filename = str(
        INFRA_TESTS_DIR
        / "_data/data_collection/cicd/all_post_commit_timeout_bad_testcase_13077087562/workflow_jobs.json"
    )

    workflow_outputs_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_timeout_bad_testcase_13077087562/"
    ).resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    assert pipeline.github_pipeline_id == 13077087562

    for job in pipeline.jobs:
        if job.github_job_id == 36492361640:
            assert job.failure_signature == str(InfraErrorV1.JOB_HANG)
            assert len(job.tests) > 0
            assert job.job_status == JobStatus.failure


def test_create_pipeline_json_for_gtest_testcases(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_gtest_testcases_13315815702/workflow.json"
    )
    github_jobs_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_gtest_testcases_13315815702/workflow_jobs.json"
    )

    workflow_outputs_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_gtest_testcases_13315815702/"
    ).resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    assert pipeline.github_pipeline_id == 13315815702

    for job in pipeline.jobs:
        # passing gtest testcase
        if job.github_job_id == 37190230023:
            assert len(job.tests) > 0
            assert job.job_success is True
            assert job.job_status == JobStatus.success
        # failing gtest testcase
        if job.github_job_id == 37190213375:
            assert len(job.tests) > 0
            assert job.job_success is False
            # check that there are failing gtests stored in the pydantic testcase list
            assert len([x for x in job.tests if not x.success]) > 0
            assert job.job_status == JobStatus.failure
        # passing pytest testcase
        if job.github_job_id == 37190252200:
            assert len(job.tests) > 0
            assert job.job_success is True
            assert job.job_status == JobStatus.success
        # failing pytest testcase
        if job.github_job_id == 37190251054:
            assert len(job.tests) > 0
            assert job.job_success is False
            # check that there are failing pytests stored in the pydantic testcase list
            assert len([x for x in job.tests if not x.success]) > 0
            assert job.job_status == JobStatus.failure
        # job has two tests with the same full_test_name, should be deduplicated
        if job.github_job_id == 37190219113:
            assert (
                len(
                    [
                        x
                        for x in job.tests
                        if x.full_test_name
                        == "tests/tt_metal/tt_metal/device/test_device_cluster_api.cpp::N300DeviceFixture::EthValidatePhysicalCoreConversion"
                    ]
                )
                == 1
            )


def test_empty_gtest_xml(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    workflow_outputs_dir = (INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_job_37712709106/").resolve()
    assert (
        workflows.get_tests_from_test_report_path(workflow_outputs_dir / "distributed_unit_tests_wormhole_b0.xml") == []
    )


def test_create_pipeline_json_for_testcases_with_annotations(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_test_annotations_13443325356/workflow.json"
    )
    github_jobs_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_test_annotations_13443325356/workflow_jobs.json"
    )

    workflow_outputs_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_test_annotations_13443325356/"
    ).resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    assert pipeline.github_pipeline_id == 13443325356

    for job in pipeline.jobs:
        # failing gtest testcase
        if job.github_job_id == 37563095078:
            assert len(job.tests) > 0
            assert job.job_success is False
            # check that there are failing gtests stored in the pydantic testcase list
            assert len([x for x in job.tests if not x.success]) == 1
            # check that the job signature and description are present
            assert job.failure_signature == str(TestErrorV1.CPP_TEST_FAILURE)
            assert job.failure_description is not None and ".cpp" in job.failure_description
            assert job.job_status == JobStatus.failure
        # failing pytest testcase
        if job.github_job_id == 37563108566:
            assert len(job.tests) > 0
            assert job.job_success is False
            # check that there are failing pytests stored in the pydantic testcase list
            assert len([x for x in job.tests if not x.success]) == 1
            assert job.failure_signature == str(TestErrorV1.PY_TEST_FAILURE)
            assert job.failure_description is not None and ".py" in job.failure_description
            assert job.job_status == JobStatus.failure


def test_create_pipeline_json_for_ctest_case(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    github_pipeline_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/tt_train_post_commit_ctest_13858791332/workflow.json"
    )
    github_jobs_json_filename = str(
        INFRA_TESTS_DIR / "_data/data_collection/cicd/tt_train_post_commit_ctest_13858791332/workflow_jobs.json"
    )

    workflow_outputs_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/tt_train_post_commit_ctest_13858791332/"
    ).resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    assert pipeline.github_pipeline_id == 13858791332

    for job in pipeline.jobs:
        # failing ctest testcase
        if job.github_job_id == 38782158256 or job.github_job_id == 38782157821:
            assert len(job.tests) == 190
            assert job.job_success is False
            # check that there are failing cpp tests stored in the pydantic testcase list
            assert len([x for x in job.tests if not x.success]) == 2
