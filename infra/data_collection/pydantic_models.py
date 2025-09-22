# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Definition of the pydantic models used for data production.
"""

from datetime import datetime
from typing import List, Optional, Union, Tuple

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class Test(BaseModel):
    """
    Table containing information about the execution of CI/CD tests, each one associated
    with a specific CI/CD job execution.

    Only some CI/CD jobs execute tests, which are executed sequentially.
    """

    test_start_ts: datetime = Field(description="Timestamp with timezone when the test execution started.")
    test_end_ts: datetime = Field(description="Timestamp with timezone when the test execution ended.")
    test_case_name: str = Field(description="Name of the pytest function.")
    filepath: str = Field(description="Test file path and name.")
    category: str = Field(description="Name of the test category.")
    group: Optional[str] = Field(None, description="Name of the test group.")
    owner: Optional[str] = Field(None, description="Developer of the test.")
    error_message: Optional[str] = Field(None, description="Succinct error string, such as exception type.")
    success: bool = Field(description="Test execution success.")
    skipped: bool = Field(description="Some tests in a job can be skipped.")
    full_test_name: str = Field(description="Test name plus config.")
    config: Optional[dict] = Field(None, description="Test configuration key/value " "pairs.")
    tags: Optional[dict] = Field(None, description="Tags associated with the test, as key/value pairs.")


class Step(BaseModel):
    """
    Contains information about the execution of CI/CD steps, each one associated with a
    specific CI/CD job execution.
    """

    name: Optional[str] = Field(description="Name of the step.")
    status: Optional[str] = Field(description="Status of the step.")
    conclusion: Optional[str] = Field(description="Conclusion of the step.")
    number: int = Field(description="Step number.")
    started_at: Optional[datetime] = Field(description="Timestamp with timezone when the step execution started.")
    completed_at: Optional[datetime] = Field(description="Timestamp with timezone when the step execution ended.")


class JobStatus(str, Enum):
    success = "success"
    failure = "failure"
    skipped = "skipped"
    cancelled = "cancelled"
    neutral = "neutral"
    unknown = "unknown"
    timed_out = "timed_out"
    action_required = "action_required"


class Job(BaseModel):
    """
    Contains information about the execution of CI/CD jobs, each one associated with a
    specific CI/CD pipeline.

    Each job may execute multiple tests, which are executed sequentially on a unique
    host.
    """

    github_job_id: Optional[int] = Field(
        None,
        description="Identifier for the Github Actions CI job, for pipelines " "orchestrated and executed by Github.",
    )
    github_job_link: Optional[str] = Field(
        None,
        description="Link to the Github Actions CI job, for pipelines orchestrated and " "executed by Github.",
    )
    name: str = Field(description="Name of the job.")
    job_submission_ts: datetime = Field(description="Timestamp with timezone when the job was submitted.")
    job_start_ts: datetime = Field(description="Timestamp with timezone when the job execution started.")
    job_end_ts: datetime = Field(description="Timestamp with timezone when the job execution ended.")
    job_success: bool = Field(
        description="Job execution success, independently from the test success "
        "criteria. Failure mechanisms that are only descriptive of the "
        "job itself."
    )
    job_status: Optional[JobStatus] = Field(
        None,
        description="Job execution status, possible statuses include success, failure, "
        "skipped, cancelled, neutral, etc.",
    )
    docker_image: Optional[str] = Field(None, description="Name of the Docker image used for the CI job.")
    is_build_job: bool = Field(description="Flag identifying if the job is a software build.")
    job_matrix_config: Optional[dict] = Field(
        None, description="This attribute is included for future feature enhancement."
    )
    host_name: Optional[str] = Field(description="Unique host name.")
    card_type: Optional[str] = Field(description="Card type and version.")
    os: Optional[str] = Field(description="Operating system of the host.")
    location: Optional[str] = Field(description="Where the host is located.")
    failure_signature: Optional[str] = Field(None, description="Failure signature.")
    failure_description: Optional[str] = Field(None, description="Failure description.")
    tests: List[Test] = []
    steps: Optional[List[Step]] = Field(None, description="Steps of the job.")
    job_label: Optional[str] = Field(None, description="GitHub CI runner label for the job.")
    tt_smi_version: Optional[str] = Field(
        None, description="Version of the tt-smi tool in order to check consistency across CI fleets."
    )

    # Model validator to check the unique combination constraint
    @model_validator(mode="before")
    def check_unique_tests(cls, values):
        tests = values.get("tests", [])
        seen_combinations = set()

        for test in tests:
            # for each job, the test constraint is full_test_name, test_start_ts
            test_combination = (test.full_test_name, test.test_start_ts)
            if test_combination in seen_combinations:
                raise ValueError(f"Duplicate test combination found: {test_combination}")
            seen_combinations.add(test_combination)
        return values


class PipelineStatus(str, Enum):
    success = "success"
    failure = "failure"
    skipped = "skipped"
    cancelled = "cancelled"
    neutral = "neutral"
    timed_out = "timed_out"
    action_required = "action_required"
    completed = "completed"
    stale = "stale"


class Pipeline(BaseModel):
    """
    Contains information about the execution of CI/CD pipelines, which consist of the
    sequential execution of one or more jobs.

    Each pipeline is associated with a specific code repository and a specific commit.;
    """

    github_pipeline_id: Optional[int] = Field(
        None,
        description="Identifier for the Github Actions CI pipeline, for pipelines "
        "orchestrated and executed by Github.",
    )
    github_pipeline_link: Optional[str] = Field(
        None,
        description="Link to the Github Actions CI pipeline, for pipelines " "orchestrated and executed by Github.",
    )
    pipeline_submission_ts: datetime = Field(
        description="Timestamp with timezone when the pipeline was submitted for " "execution.",
    )
    pipeline_start_ts: datetime = Field(description="Timestamp with timezone when the pipeline execution started.")
    pipeline_end_ts: datetime = Field(description="Timestamp with timezone when the pipeline execution ended.")
    pipeline_status: Optional[PipelineStatus] = Field(
        None,
        description="Pipeline execution status, possible statuses include success, failure, skipped, cancelled, neutral, etc.",
    )
    name: str = Field(description="Name of the pipeline.")
    project: Optional[str] = Field(None, description="Name of the software project.")
    trigger: Optional[str] = Field(None, description="Type of trigger that initiated the pipeline.")
    vcs_platform: Optional[str] = Field(
        None,
        description="Version control software used for the code tested in the pipeline.",
    )
    repository_url: str = Field(description="URL of the code repository.")
    git_branch_name: Optional[str] = Field(description="Name of the Git branch tested by the pipeline.")
    git_commit_hash: str = Field(description="Git commit that triggered the execution of the pipeline.")
    git_author: str = Field(description="Author of the Git commit.")
    orchestrator: Optional[str] = Field(None, description="CI/CD pipeline orchestration platform.")
    jobs: List[Job] = []

    # Model validator to check the unique combination constraint
    @model_validator(mode="before")
    def check_unique_jobs(cls, values):
        jobs = values.get("jobs", [])
        seen_combinations = set()

        for job in jobs:
            # for each pipeline, the job constraint is (name, job_submission_ts, job_start_ts, job_end_ts)
            job_combination = (job.name, job.job_submission_ts, job.job_start_ts, job.job_end_ts)
            if job_combination in seen_combinations:
                raise ValueError(f"Duplicate job combination found: {job_combination}")
            seen_combinations.add(job_combination)
        return values


class BenchmarkMeasurement(BaseModel):
    """
    Contains measurements for each benchmark run, iteration and step.

    A run can have multiple iterations, each iteration can have multiple steps and each
    step can execute multiple measurements.
    """

    step_start_ts: datetime = Field(description="Timestamp with time zone when the step started.")
    step_end_ts: datetime = Field(description="Timestamp with time zone when the step ended.")
    iteration: int = Field(
        description="A benchmark run can comprise a loop that repeats with the same "
        "parameters the same sequence of steps and measurements for each. "
        "This integer is the repetition number."
    )
    step_name: str = Field(description="Name of the benchmark step within the run.")
    step_warm_up_num_iterations: Optional[int] = Field(
        None, description="Number of iterations for device warm-up at each step."
    )
    name: str = Field(
        description="Name of the measurement performed, e.g. tokens_per_sec_per_user, "
        "tokens_per_sec, images_per_sec, pearson_correlation, "
        "top1/top5 ratios."
    )
    value: float = Field(description="Measured value.")
    target: Optional[float] = Field(None, description="Target value.")
    device_power: Optional[float] = Field(
        None,
        description="Average power consumption in Watts during the benchmark step.",
    )
    device_temperature: Optional[float] = Field(
        None, description="Average temperature of the device during the benchmark."
    )


class PartialBenchmarkRun(BaseModel):
    """
    Contains information about each execution of an AI model benchmark, called benchmark
    run, composed of steps each of which performs a set of measurements. The sequence of
    steps in a run can be iterated in a loop.

    Note that this model differs from the CompleteBenchamarkRun model in adding
    optionality for more attributes which might not be known during a data production
    phase, such as device_hostname.
    """

    run_start_ts: datetime = Field(description="Timestamp with time zone when the benchmark run started.")
    run_end_ts: datetime = Field(description="Timestamp with time zone when the benchmark run ended.")
    run_type: str = Field(description="Description of the benchmark run, e.g. a100_fp16_experiments.")
    git_repo_name: Optional[str] = Field(
        None,
        description="Name of the Git repository containing the code that executes the " "benchmark.",
    )
    git_commit_hash: Optional[str] = Field(
        None,
        description="Git commit hash of the code used to run the benchmark (software " "version info).",
    )
    git_commit_ts: Optional[datetime] = Field(None, description="Timestamp with timezone of the git commit.")
    git_branch_name: Optional[str] = Field(
        None, description="Name of the Git branch associated with the benchmark run."
    )
    github_pipeline_id: Optional[int] = Field(
        None,
        description="Unique identifier for the pipeline record from GitHub Actions.",
    )
    github_pipeline_link: Optional[str] = Field(
        None,
        description="Link to the GitHub job run associated with the benchmark run.",
    )
    github_job_id: Optional[int] = Field(None, description="Unique GitHub Actions CI job ID.")
    user_name: Optional[str] = Field(None, description="Name of the person that executed the benchmark run.")
    docker_image: Optional[str] = Field(
        None,
        description="Name or ID of the Docker image used for benchmarking (software "
        "version info), e.g., trt-llm-v080.",
    )
    device_hostname: Optional[str] = Field(
        None, description="Host name of the device on which the benchmark is performed."
    )
    device_ip: Optional[str] = Field(None, description="Host IP address.")
    device_info: Optional[dict] = Field(
        None,
        description="Device information as JSON, such as manufacturer, card_type, "
        "dram_size, num_cores, price, bus_interface, optimal_clock_speed.",
    )
    ml_model_name: str = Field(description="Name of the benchmarked neural network model.")
    ml_model_type: Optional[str] = Field(
        None,
        description="Model type, such as text generation, classification, question " "answering, etc.",
    )
    num_layers: Optional[int] = Field(None, description="Number of layers of the model.")
    batch_size: Optional[int] = Field(None, description="Batch size.")
    config_params: Optional[dict] = Field(None, description="Additional training/inference parameters.")
    precision: Optional[str] = Field(
        None,
        description="Numerical precision, such as bfp8, fp16, or a mix such as " "fp16_act_bfp8_weights, etc.",
    )
    dataset_name: Optional[str] = Field(None, description="Name of the dataset used for the benchmark.")
    profiler_name: Optional[str] = Field(None, description="Profiler to time the benchmark.")
    input_sequence_length: Optional[int] = Field(
        None,
        description="Length of the sequence used as input to the model, applicable to " "sequence models.",
    )
    output_sequence_length: Optional[int] = Field(
        None,
        description="Length of the sequence used as output by the model, applicable to " "sequence models.",
    )
    image_dimension: Optional[str] = Field(
        None,
        description="Dimension of the image, e.g. 224x224x3, applicable to computer v" "ision models.",
    )
    perf_analysis: Optional[bool] = Field(
        None,
        description="If the model was run in perf analysis mode. This is " "kernel/operation execution mode.",
    )
    training: Optional[bool] = Field(None, description="ML model benchmarks for training or inference.")
    measurements: List[BenchmarkMeasurement] = Field(description="List of benchmark measurements.")


class CompleteBenchmarkRun(BaseModel):
    """
    Contains information about each execution of an AI model benchmark, called benchmark
    run, composed of steps each of which performs a set of measurements.

    The sequence of steps in a run can be iterated in a loop.
    """

    run_start_ts: datetime = Field(description="Timestamp with time zone when the benchmark run started.")
    run_end_ts: datetime = Field(description="Timestamp with time zone when the benchmark run ended.")
    run_type: str = Field(description="Description of the benchmark run, e.g. a100_fp16_experiments.")
    git_repo_name: Optional[str] = Field(
        None,
        description="Name of the Git repository containing the code that executes " "the benchmark.",
    )
    git_commit_hash: Optional[str] = Field(
        None,
        description="Git commit hash of the code used to run the benchmark (software " "version info).",
    )
    git_commit_ts: Optional[datetime] = Field(None, description="Timestamp with timezone of the git commit.")
    git_branch_name: Optional[str] = Field(
        None, description="Name of the Git branch associated with the benchmark run."
    )
    github_pipeline_id: Optional[int] = Field(
        None,
        description="Unique identifier for the pipeline record from GitHub Actions.",
    )
    github_pipeline_link: Optional[str] = Field(
        None,
        description="Link to the GitHub job run associated with the benchmark run.",
    )
    github_job_id: Optional[int] = Field(None, description="Unique GitHub Actions CI job ID.")
    user_name: Optional[str] = Field(None, description="Name of the person that executed the benchmark run.")
    docker_image: Optional[str] = Field(
        None,
        description="Name or ID of the Docker image used for benchmarking (software "
        "version info), e.g., trt-llm-v080.",
    )
    device_hostname: str = Field(description="Host name of the device on which the benchmark is performed.")
    device_ip: Optional[str] = Field(None, description="Host IP address.")
    device_info: Optional[dict] = Field(
        None,
        description="Device information as JSON, such as manufacturer, card_type, "
        "dram_size, num_cores, price, bus_interface, optimal_clock_speed.",
    )
    ml_model_name: str = Field(description="Name of the benchmarked neural network model.")
    ml_model_type: Optional[str] = Field(
        None,
        description="Model type, such as text generation, classification, question " "answering, etc.",
    )
    num_layers: Optional[int] = Field(None, description="Number of layers of the model.")
    batch_size: Optional[int] = Field(None, description="Batch size.")
    config_params: Optional[dict] = Field(None, description="Additional training/inference parameters.")
    precision: Optional[str] = Field(
        None,
        description="Numerical precision, such as bfp8, fp16, or a mix such as " "fp16_act_bfp8_weights, etc.",
    )
    dataset_name: Optional[str] = Field(None, description="Name of the dataset used for the benchmark.")
    profiler_name: Optional[str] = Field(None, description="Profiler to time the benchmark.")
    input_sequence_length: Optional[int] = Field(
        None,
        description="Length of the sequence used as input to the model, applicable " "to sequence models.",
    )
    output_sequence_length: Optional[int] = Field(
        None,
        description="Length of the sequence used as output by the model, applicable " "to sequence models.",
    )
    image_dimension: Optional[str] = Field(
        None,
        description="Dimension of the image, e.g. 224x224x3, applicable to computer " "vision models.",
    )
    perf_analysis: Optional[bool] = Field(
        None,
        description="If the model was run in perf analysis mode. This is " "kernel/operation execution mode.",
    )
    training: Optional[bool] = Field(None, description="ML model benchmarks for training or inference.")
    measurements: List[BenchmarkMeasurement] = Field(description="List of benchmark measurements.")


class TensorDesc(BaseModel):
    """
    Contains descriptions of tensors used as inputs or outputs of the operation in a ML
    kernel operation test.
    """

    shape: List[int] = Field(description="Shape of the tensor.")
    data_type: str = Field(description="Data type of the tensor, e.g. Float32, " "BFloat16, etc.")
    buffer_type: str = Field(description="Memory space of the tensor, e.g. Dram, L1, " "System.")
    layout: str = Field(description="Layout of the tensor, e.g. Interleaved, " "SingleBank, HeightSharded.")
    grid_shape: List[int] = Field(
        description="The grid shape describes a 2D region of cores which are used to "
        "store the tensor in memory. E.g. You have a tensor with shape "
        "128x128, you might decide to put this on a 2x2 grid of cores, "
        "meaning each core has a 64x64 slice."
    )


class TestStatus(Enum):
    """
    Status of the test execution.
    """

    # This includes the statuses from both the TTNN (Steven) and Forge (Collin) side.
    compile_failed = "compile_failed"
    run_failed = "run_failed"
    golden_failed = "golden_failed"
    success = "success"
    passed = "pass"
    fail_assert_exception = "fail_assert_exception"
    fail_l1_out_of_mem = "fail_l1_out_of_mem"
    fail_watcher = "fail_watcher"
    fail_crash_hang = "fail_crash_hang"
    fail_unsupported_device_perf = "fail_unsupported_device_perf"
    skipped = "skipped"
    error = "error"


class RunStatus(Enum):
    """
    Status of the run execution.
    """

    passed = "passed"
    fail = "fail"
    did_not_finish = "did_not_finish"
    exception = "exception"


class PerfMetric(BaseModel):
    """
    Metric name and its value.
    """

    metric_name: str = Field(description="Metric name.")
    metric_value: float = Field(description="Metric value.")

    class Config:
        frozen = True


class OpParam(BaseModel):
    """
    Test parameter (i.e. test vector) and its value.
    """

    param_name: str = Field(description="Test parameter name.")

    # Test parameter values can be a single str/int/float value or a list of values,
    # e.g. dtype="int8", shape=[1, 2, 3].
    param_value_numeric: Optional[float] = Field(default=None, description="Test parameter value in float.")
    param_value_text: Optional[str] = Field(default=None, description="Test parameter value in text.")
    # JSON value holder for complex params: can be a JSON object (dict) or JSON array (list).
    param_value_json: Optional[Union[dict, list]] = Field(
        default=None, description="Test parameter value as JSON (object or array)."
    )

    class Config:
        frozen = True


class OpTest(BaseModel):
    """
    Contains information about ML kernel operation tests, such as test execution,
    results, configuration.
    """

    # Made this optional since TTNN (Steven) or Forge (Collin) side may have tests that
    # are not executed by CI runners.
    github_job_id: Optional[int] = Field(
        description="Identifier for the Github Actions CI job, which ran the test.",
    )
    full_test_name: str = Field(description="Test name plus config.")
    test_start_ts: datetime = Field(description="Timestamp with timezone when the test execution started.")
    test_end_ts: datetime = Field(description="Timestamp with timezone when the test execution ended.")
    # test_case_name will be suite_name for the TTNN (Steven) side.
    test_case_name: Optional[str] = Field(description="Name of the pytest function.")
    filepath: str = Field(description="Test file path and name.")
    success: bool = Field(description="Test execution success.")
    skipped: bool = Field(description="Some tests in a job can be skipped.")
    error_message: Optional[str] = Field(None, description="Succinct error string, such as exception type.")
    error_hash: Optional[str] = Field(None, description="Hash of the error message for traceability.")
    config: Optional[dict] = Field(default=None, description="Test configuration, as key/value pairs.")
    frontend: str = Field(description="ML frontend or framework used to run the test.")
    model_name: str = Field(description="Name of the ML model in which this operation is used.")
    op_kind: str = Field(description="Kind of operation, e.g. Eltwise.")
    op_name: str = Field(description="Name of the operation, e.g. ttnn.conv2d")
    framework_op_name: str = Field(description="Name of the operation within the framework, e.g. torch.conv2d")
    # Made these optional since TTNN (Steven) side doesn't have these fields.
    inputs: Optional[List[TensorDesc]] = Field(description="List of input tensors.")
    outputs: Optional[List[TensorDesc]] = Field(description="List of output tensors.")
    op_params: Optional[dict] = Field(
        default=None,
        description="Parametrization criteria for the operation, based on its kind, "
        "as key/value pairs, e.g. stride, padding, etc.",
    )

    # Fields added for Forge (Collin) side.
    git_sha: Optional[str] = Field(description="Git commit SHA of the op test.")
    status: Optional[TestStatus] = Field(
        description="Status of the op test, e.g. success, " "run_failed, skipped, error."
    )
    card_type: Optional[str] = Field(description="Type of hardware card used for testing, e.g. N150, N300.")
    backend: Optional[str] = Field(description="Backend used for the op test.")

    # Fields added for TTNN (Steven) side.
    data_source: Optional[str] = Field(
        None,
        description="Source of the data for the op test, indicating the data producer.",
    )
    input_hash: Optional[str] = Field(None, description="Hash of the input vector for deduplication and traceability.")
    message: Optional[str] = Field(None, description="Optional informational message about the execution outcome.")
    exception: Optional[str] = Field(None, description="Exception text when a failure occurred, if any.")
    # Performance metrics (optional)
    metrics: Optional[set[PerfMetric]] = Field(
        None,
        description="Set of performance metrics for the test, including both end-to-end performance "
        "metric and device-level performance measurements, when device profiling is enabled.",
    )
    # Note that the op_params_set field is very similar to the op_params field above.
    # op_params will be kept as a JSONB column in the ml_kernel_op_test table,
    # while op_params_set will be popped out and normalized into a vertical table,
    # i.e. op_param table.
    # We have both implementations to avoid the impact on the data production/consumption
    # of the primary data producer (Forge: Vladimir, James).
    op_params_set: Optional[List[OpParam]] = Field(
        description="Original test vector contents captured for traceability, normalized for JSON compatibility.",
    )


# This model is only adopted by TTNN (Steven) side.
class OpRun(BaseModel):
    """
    High-level metadata describing a sweep run session or unit
    test run.
    """

    initiated_by: str = Field(description="User or CI pipeline that initiated the run.")
    host: str = Field(description="Hostname of the machine executing the run.")
    card_type: str = Field(description="Target device/architecture identifier, if available.")
    run_type: str = Field(description="Type of op test run (e.g., sweeps, unit_test).")
    run_contents: str = Field(
        description="Human-readable description of run contents (e.g., module/suite selection).",
    )

    git_author: str = Field(description="Git author configured in the environment.")
    git_branch_name: str = Field(description="Current git branch name.")
    git_sha: str = Field(description="Short git commit hash for the run.")
    github_pipeline_id: Optional[int] = Field(
        None,
        description="Identifier for the GitHub Actions pipeline run (GITHUB_RUN_ID) or analogous CI pipeline id.",
    )

    run_start_ts: datetime = Field(description="Timestamp with timezone when the sweeps run started.")
    run_end_ts: datetime = Field(description="Timestamp with timezone when the sweeps run ended.")
    status: RunStatus = Field(description="Overall run status aggregated from testcases.")
    tests: List[OpTest] = Field(description="List of tests executed in the run.")
