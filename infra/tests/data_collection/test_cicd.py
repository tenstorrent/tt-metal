import pytest
import pathlib

from infra.data_collection.github import workflows
from infra.data_collection.cicd import create_cicd_json_for_data_analysis
from infra.data_collection.models import InfraErrorV1, TestErrorV1
from infra.data_collection.github.utils import (
    get_job_failure_signature_,
    get_job_row_from_github_job,
    get_civ2_node_name_and_serial_from_annotations,
    _card_type_from_job_labels,
    _generic_runner_labels,
    _load_sku_config_skus,
)
from infra.data_collection.pydantic_models import JobStatus
from infra.data_collection.pydantic_models import Step
from loguru import logger

INFRA_TESTS_DIR = pathlib.Path(__file__).parent.parent

_pipeline_cache = {}


def _load_pipeline(workflow_run_gh_environment, data_subdir):
    if data_subdir not in _pipeline_cache:
        github_runner_environment = workflow_run_gh_environment
        workflow_outputs_dir = (INFRA_TESTS_DIR / "_data/data_collection/cicd" / data_subdir).resolve()
        assert workflow_outputs_dir.is_dir()
        assert workflow_outputs_dir.exists()
        _pipeline_cache[data_subdir] = create_cicd_json_for_data_analysis(
            workflow_outputs_dir,
            github_runner_environment,
            str(workflow_outputs_dir / "workflow.json"),
            str(workflow_outputs_dir / "workflow_jobs.json"),
        )
    return _pipeline_cache[data_subdir]


def _find_job(pipeline, github_job_id):
    for job in pipeline.jobs:
        if job.github_job_id == github_job_id:
            return job
    assert False, "Job {} not found in pipeline".format(github_job_id)


def test_dummy():
    pass


def test_create_pipeline_json_with_passing_post_commit(workflow_run_gh_environment):
    pipeline = _load_pipeline(workflow_run_gh_environment, "all_post_commit_passing_10662355710")

    assert pipeline.github_pipeline_id == 10662355710
    assert len(pipeline.jobs) == 99, "There should be 99 jobs according to github jobs API"

    for job in pipeline.jobs:
        assert job.failure_signature is None


def get_non_success_jobs_(pipeline):
    is_failing_job = lambda job_: not job_.job_success
    matching_jobs = list(filter(is_failing_job, pipeline.jobs))

    return matching_jobs


def test_create_pipeline_json_to_detect_job_timeout_error_v1(workflow_run_gh_environment):
    pipeline = _load_pipeline(workflow_run_gh_environment, "all_post_commit_runner_died_12626_10996802864")

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
    pipeline = _load_pipeline(workflow_run_gh_environment, "all_post_commit_runner_died_12626_11110261767")

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


@pytest.mark.parametrize(
    "github_job_id,check_tests,check_tt_smi",
    [
        (30650764191, True, False),
        (30650754720, False, True),
    ],
)
def test_create_pipeline_json_for_run_github_timed_out_job(
    workflow_run_gh_environment, github_job_id, check_tests, check_tt_smi
):
    pipeline = _load_pipeline(workflow_run_gh_environment, "all_post_commit_github_timeout_11034942442")

    assert pipeline.github_pipeline_id == 11034942442

    job = _find_job(pipeline, github_job_id)
    if check_tests:
        assert len(job.tests) > 0
        assert job.job_status == JobStatus.failure
    if check_tt_smi:
        assert job.tt_smi_version is not None


def test_create_pipeline_json_for_timeout_bad_testcase(workflow_run_gh_environment):
    pipeline = _load_pipeline(workflow_run_gh_environment, "all_post_commit_timeout_bad_testcase_13077087562")

    assert pipeline.github_pipeline_id == 13077087562

    for job in pipeline.jobs:
        if job.github_job_id == 36492361640:
            assert job.failure_signature == str(InfraErrorV1.JOB_HANG)
            assert len(job.tests) > 0
            assert job.job_status == JobStatus.failure


@pytest.mark.parametrize(
    "github_job_id,job_success,check_failing_tests,dedupe_test_name",
    [
        (37190230023, True, False, None),
        (37190213375, False, True, None),
        (37190252200, True, False, None),
        (37190251054, False, True, None),
        (
            37190219113,
            None,
            False,
            "tests/tt_metal/tt_metal/device/test_device_cluster_api.cpp::N300MeshDeviceFixture::EthValidatePhysicalCoreConversion",
        ),
    ],
)
def test_create_pipeline_json_for_gtest_testcases(
    workflow_run_gh_environment, github_job_id, job_success, check_failing_tests, dedupe_test_name
):
    pipeline = _load_pipeline(workflow_run_gh_environment, "all_post_commit_gtest_testcases_13315815702")

    assert pipeline.github_pipeline_id == 13315815702

    job = _find_job(pipeline, github_job_id)

    if dedupe_test_name is not None:
        # job has two tests with the same full_test_name, should be deduplicated
        assert len([x for x in job.tests if x.full_test_name == dedupe_test_name]) == 1
        return

    assert len(job.tests) > 0
    assert job.job_success is job_success
    assert job.job_status == JobStatus.success if job_success else JobStatus.failure
    if check_failing_tests:
        assert len([x for x in job.tests if not x.success]) > 0


def test_empty_gtest_xml(workflow_run_gh_environment):
    github_runner_environment = workflow_run_gh_environment
    workflow_outputs_dir = (INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_job_37712709106/").resolve()
    assert (
        workflows.get_tests_from_test_report_path(workflow_outputs_dir / "distributed_unit_tests_wormhole_b0.xml") == []
    )


@pytest.mark.parametrize(
    "github_job_id,failure_signature,failure_description_contains",
    [
        (37563095078, str(TestErrorV1.CPP_TEST_FAILURE), ".cpp"),
        (37563108566, str(TestErrorV1.PY_TEST_FAILURE), ".py"),
    ],
)
def test_create_pipeline_json_for_testcases_with_annotations(
    workflow_run_gh_environment, github_job_id, failure_signature, failure_description_contains
):
    pipeline = _load_pipeline(workflow_run_gh_environment, "all_post_commit_test_annotations_13443325356")

    assert pipeline.github_pipeline_id == 13443325356

    job = _find_job(pipeline, github_job_id)
    assert len(job.tests) > 0
    assert job.job_success is False
    # check that there are failing gtests/pytests stored in the pydantic testcase list
    assert len([x for x in job.tests if not x.success]) == 1
    # check that the job signature and description are present
    assert job.failure_signature == failure_signature
    assert job.failure_description is not None and failure_description_contains in job.failure_description
    assert job.job_status == JobStatus.failure


@pytest.mark.parametrize("github_job_id", [38782158256, 38782157821])
def test_create_pipeline_json_for_ctest_case(workflow_run_gh_environment, github_job_id):
    pipeline = _load_pipeline(workflow_run_gh_environment, "tt_train_post_commit_ctest_13858791332")

    assert pipeline.github_pipeline_id == 13858791332

    job = _find_job(pipeline, github_job_id)
    # failing ctest testcase
    assert len(job.tests) == 190
    assert job.job_success is False
    # check that there are failing cpp tests stored in the pydantic testcase list
    assert len([x for x in job.tests if not x.success]) == 2


def test_pipeline_job_contains_valid_steps():
    workflow_data_dir = (
        INFRA_TESTS_DIR / "_data/data_collection/cicd/all_post_commit_gtest_testcases_13315815702/"
    ).resolve()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir=workflow_data_dir,
        github_runner_environment={"github_event_name": "push"},
        github_pipeline_json_filename=str(workflow_data_dir / "workflow.json"),
        github_jobs_json_filename=str(workflow_data_dir / "workflow_jobs.json"),
    )

    assert len(pipeline.jobs) > 0, "Pipeline contains no jobs."

    # Select a known job that includes steps
    target_job_name = "ttnn-unit-tests (grayskull, E150) / ttnn group 2 grayskull E150"
    target_job = next(job for job in pipeline.jobs if job.name == target_job_name)

    step_names = [step.name for step in target_job.steps]
    assert len(step_names) >= 1, f"Expected at least 1 step, got {len(step_names)}"

    # Confirm the step names include general-purpose CI keywords
    expected_keywords = {"set up", "run", "post", "test", "checkout", "build", "artifact"}
    matching_steps = [name for name in step_names if any(keyword in name.lower() for keyword in expected_keywords)]
    assert (
        matching_steps
    ), f"Expected at least one step containing a keyword from {expected_keywords}, got: {step_names}"

    # Validate step schema fields
    valid_statuses = {None, "queued", "in_progress", "completed"}
    valid_conclusions = {None, "success", "skipped", "failure", "cancelled", "timed_out", "neutral", "stale"}

    assert all(step.status in valid_statuses for step in target_job.steps)
    assert all(step.conclusion in valid_conclusions for step in target_job.steps)

    logger.debug(f"\n✅ Job `{target_job_name}` has {len(target_job.steps)} steps:")
    for step in target_job.steps:
        logger.debug(
            f" - {step.name}: status={step.status}, conclusion={step.conclusion}, "
            f"started_at={step.started_at}, completed_at={step.completed_at}"
        )


def _make_mock_job(step_name="Checkout", step_conclusion="failure"):
    """Helper: minimal GitHub job dict for get_job_failure_signature_ unit tests."""
    return {
        "id": 1,
        "run_id": 1,
        "steps": [
            {
                "name": step_name,
                "status": "completed",
                "conclusion": step_conclusion,
                "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:01:00Z",
            }
        ],
    }


@pytest.mark.parametrize(
    "failure_description,step_name,step_conclusion",
    [
        # Annotation-level matches (specific error text surfaced in GitHub annotation)
        (
            "fatal: clone of 'https://github.com/tenstorrent/tt-umd.git' into submodule path failed",
            "Run something",
            "success",
        ),
        ("Failed to clone 'tt_metal/third_party/umd'. Retry scheduled", "Run something", "success"),
        ("could not read Username for 'https://github.com': terminal prompts disabled", "Run something", "success"),
        ("terminal prompts disabled", "Run something", "success"),
        (
            "Fetched in submodule path 'tt_metal/third_party/tracy', but it did not contain abc123",
            "Run something",
            "success",
        ),
        # Step-name match: generic git exit code annotation but checkout step failed
        ("The process '/usr/bin/git' failed with exit code 1", "Checkout", "failure"),
        ("The process '/usr/bin/git' failed with exit code 1", "\u2b07\ufe0f Checkout", "failure"),
    ],
)
def test_checkout_failure_classified_correctly(failure_description, step_name, step_conclusion):
    """git submodule clone failures and checkout step failures → CHECKOUT_FAILURE, not GENERIC_FAILURE."""
    mock_job = _make_mock_job(step_name=step_name, step_conclusion=step_conclusion)
    result = get_job_failure_signature_(mock_job, failure_description, workflow_outputs_dir=None)
    assert result == str(InfraErrorV1.CHECKOUT_FAILURE), (
        f"Expected CHECKOUT_FAILURE for description={failure_description!r}, "
        f"step={step_name!r}/{step_conclusion!r}, got {result!r}"
    )


def test_non_checkout_git_failure_stays_generic():
    """A git failure outside a checkout step should remain GENERIC_FAILURE."""
    mock_job = _make_mock_job(step_name="Run build script", step_conclusion="failure")
    result = get_job_failure_signature_(
        mock_job, "The process '/usr/bin/git' failed with exit code 1", workflow_outputs_dir=None
    )
    assert result == str(InfraErrorV1.GENERIC_FAILURE)


@pytest.fixture(autouse=True)
def clear_sku_config_cache():
    from infra.data_collection.github.utils import _generic_runner_labels, _root_sku_for, _sku_config_sku_names

    _load_sku_config_skus.cache_clear()
    _sku_config_sku_names.cache_clear()
    _generic_runner_labels.cache_clear()
    _root_sku_for.cache_clear()
    yield
    _load_sku_config_skus.cache_clear()
    _sku_config_sku_names.cache_clear()
    _generic_runner_labels.cache_clear()
    _root_sku_for.cache_clear()


def test_generic_runner_labels_derived_from_sim_skus():
    expected_labels: set[str] = set()
    for sku_name, sku_entry in _load_sku_config_skus().items():
        if sku_name.startswith("sim_"):
            expected_labels.update(sku_entry.get("runs_on") or [])

    assert _generic_runner_labels() == frozenset(expected_labels)


@pytest.mark.parametrize(
    "labels,expected_card_type",
    [
        (["N300", "cloud-virtual-machine", "in-service"], "wh_n300"),
        (["N150", "cloud-virtual-machine", "in-service"], "wh_n150"),
        (
            ["P300-viommu", "arch-blackhole", "in-service", "pipeline-yyz2-lfc"],
            "bh_p300",
        ),
        (["P300-viommu", "in-service", "pipeline-yyz2-lfc"], "bh_p300"),
        (["tt-ubuntu-2204-N300-viommu-stable"], "wh_n300"),
        (
            ["P150", "arch-blackhole", "in-service", "pipeline-functional"],
            "bh_p150",
        ),
        (["P100", "cloud-virtual-machine", "in-service"], "bh_p100"),
        (
            ["config-t3000", "arch-wormhole_b0", "in-service", "pipeline-functional"],
            "wh_llmbox",
        ),
        (
            ["arch-wormhole_b0", "topology-6u", "in-service", "pipeline-perf"],
            "wh_galaxy",
        ),
        # tm-fabric-style runs_on: strict match fails, label fallback applies
        (["P300-viommu", "arch-blackhole", "in-service"], "bh_p300"),
        # model perf-style runs_on: missing arch-blackhole for bh_p150_perf
        (["P150", "pipeline-perf", "bare-metal", "in-service"], "bh_p150"),
        # legacy partial wh_n300 labels
        (["N300", "in-service"], "wh_n300"),
        (["build", "in-service"], None),
        (["ubuntu-latest"], "ubuntu-latest"),
        (["tt-ubuntu-2204-large-stable"], "tt-ubuntu-2204-large-stable"),
    ],
)
def test_card_type_from_job_labels(labels, expected_card_type):
    assert _card_type_from_job_labels(labels) == expected_card_type


def test_get_civ2_node_name_and_serial_from_annotations():
    # Card runner: both node name and (possibly composite) serial present
    annotations = [
        {"title": "k8s-node-name", "message": "CIV2 runner foo is running on Kubernetes node: aus-glx-03"},
        {"title": "tt-card-serial", "message": "CIV2 runner foo has serial number(s): TT-BH-00111:TT-BH-00222"},
        {"title": "", "message": "some unrelated infra annotation"},
    ]
    assert get_civ2_node_name_and_serial_from_annotations(annotations) == ("aus-glx-03", "TT-BH-00111:TT-BH-00222")

    # CPU-only runner: a tt-card-serial notice is still emitted, but with a
    # "Not a Tenstorrent card runner" message that carries no serial
    cpu_annotations = [
        {
            "title": "tt-card-serial",
            "message": "Not a Tenstorrent card runner (runner name: tt-ubuntu-2204-large-stable-w77km-runner-djpn9)",
        },
        {
            "title": "k8s-node-name",
            "message": "CIV2 runner tt-ubuntu-2204-large-stable-w77km-runner-djpn9 is running on Kubernetes node: f10-cpu-01",
        },
    ]
    assert get_civ2_node_name_and_serial_from_annotations(cpu_annotations) == ("f10-cpu-01", None)

    # No annotations at all
    assert get_civ2_node_name_and_serial_from_annotations(None) == (None, None)


def _make_completed_civ2_job(job_id, runner_name, labels):
    return {
        "id": job_id,
        "run_id": 999,
        "runner_name": runner_name,
        "labels": labels,
        "status": "completed",
        "conclusion": "success",
        "created_at": "2026-07-10T00:00:00Z",
        "started_at": "2026-07-10T00:01:00Z",
        "completed_at": "2026-07-10T00:05:00Z",
        "name": "some test job",
        "html_url": "https://github.com/tenstorrent/tt-metal/actions/runs/999/job/1",
        "run_attempt": 1,
        "steps": [],
    }


def test_civ2_host_name_replaced_with_node_and_serial():
    job = _make_completed_civ2_job(
        1,
        "tt-ubuntu-2204-n300-viommu-stable-abcde-runner-fghij",
        ["tt-ubuntu-2204-n300-viommu-stable"],
    )
    annotations = {
        1: [
            {
                "title": "k8s-node-name",
                "message": "CIV2 runner foo is running on Kubernetes node: aus-glx-03",
                "annotation_level": "notice",
                "path": ".github",
            },
            {
                "title": "tt-card-serial",
                "message": "CIV2 runner foo has serial number(s): TT-BH-02345",
                "annotation_level": "notice",
                "path": ".github",
            },
        ]
    }
    row = get_job_row_from_github_job(job, annotations, INFRA_TESTS_DIR)
    assert row["host_name"] == "aus-glx-03_TT-BH-02345"


def test_civ2_host_name_falls_back_to_truncation_without_annotations():
    job = _make_completed_civ2_job(
        2,
        "tt-ubuntu-2204-n300-viommu-stable-abcde-runner-fghij",
        ["tt-ubuntu-2204-n300-viommu-stable"],
    )
    # No node/serial annotations -> keep current ephemeral-suffix truncation behavior
    row = get_job_row_from_github_job(job, {}, INFRA_TESTS_DIR)
    assert row["host_name"] == "tt-ubuntu-2204-n300-viommu-stable-abcde-runner"


def test_create_pipeline_json_assigns_sku_card_type_to_n300_job(workflow_run_gh_environment):
    pipeline = _load_pipeline(workflow_run_gh_environment, "all_post_commit_passing_10662355710")

    wh_n300_labels = {"N300", "cloud-virtual-machine", "in-service"}
    full_wh_n300_jobs = [
        job for job in pipeline.jobs if job.job_label and wh_n300_labels.issubset(set(job.job_label.split(",")))
    ]
    assert full_wh_n300_jobs
    assert all(job.card_type == "wh_n300" for job in full_wh_n300_jobs)

    # Legacy post-commit jobs may only carry a subset of wh_n300 runs_on labels.
    partial_n300_jobs = [
        job
        for job in pipeline.jobs
        if job.job_label
        and "N300" in job.job_label.split(",")
        and "cloud-virtual-machine" not in job.job_label.split(",")
    ]
    assert partial_n300_jobs
    assert all(job.card_type == "wh_n300" for job in partial_n300_jobs)
