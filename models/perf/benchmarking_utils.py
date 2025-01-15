# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
import pytz
from loguru import logger
from typing import List, Optional
from pydantic import BaseModel, Field


class BenchmarkProfiler:
    def __init__(self):
        self.start_times = dict()
        self.end_times = dict()

    def __call__(self, step_name: str, iteration: int = 0):
        # Return a context manager for this step
        return self.StepContext(self, step_name, iteration)

    class StepContext:
        def __init__(self, profiler, step_name: str, iteration: int):
            self.profiler = profiler
            self.step_name = step_name
            self.iteration = iteration

        def __enter__(self):
            self.profiler.start(self.step_name, self.iteration)
            return self.profiler

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.profiler.end(self.step_name, self.iteration)
            return False

    def start(self, step_name: str, iteration: int = 0):
        self.start_times[(iteration, step_name)] = datetime.now(tz=pytz.UTC)

    def end(self, step_name: str, iteration: int = 0):
        self.end_times[(iteration, step_name)] = datetime.now(tz=pytz.UTC)

    def get_duration(self, step_name: str, iteration: int = 0):
        start_time = self.start_times[(iteration, step_name)]
        end_time = self.end_times[(iteration, step_name)]
        return (end_time - start_time).total_seconds()

    def get_str_start(self, step_name: str, iteration: int = 0):
        return self._get_str_ts(self.start_times[(iteration, step_name)])

    def get_str_end(self, step_name: str, iteration: int = 0):
        return self._get_str_ts(self.end_times[(iteration, step_name)])

    def contains_step(self, step_name: str, iteration: int = 0):
        return (iteration, step_name) in self.start_times and (iteration, step_name) in self.end_times

    def _get_str_ts(self, timestamp):
        return timestamp.strftime("%Y-%m-%dT%H:%M:%S%z")


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


class BenchmarkData:
    def __init__(self):
        self.measure_data: List[BenchmarkMeasurement] = []
        self.output_folder = "generated/benchmark_data/"

    def add_measurement(
        self,
        profiler: BenchmarkProfiler,
        iteration: int,
        step_name: str,
        name: str,
        value: float,
        step_warm_up_num_iterations: int = None,
        target: float = None,
        device_power: float = None,
        device_temperature: float = None,
    ):
        """
        Measurement data contains records and attributes for each measurement performed at each iteration and each step of the benchmark run.
        The triad of fields (iteration, step_name, name) must be unique.
        """
        assert None not in [profiler, iteration, step_name, name, value], "Missing required fields"
        assert profiler.contains_step(
            step_name, iteration
        ), f"Completed step '{step_name}' for iteration {iteration} not found in profiler"
        self.measure_data.append(
            BenchmarkMeasurement(
                step_start_ts=profiler.get_str_start(step_name, iteration),
                step_end_ts=profiler.get_str_end(step_name, iteration),
                iteration=iteration,
                step_name=step_name,
                name=name,
                value=value,
                step_warm_up_num_iterations=step_warm_up_num_iterations,
                target=target,
                device_power=device_power,
                device_temperature=device_temperature,
            )
        )

    def save_partial_run_json(
        self,
        profiler: BenchmarkProfiler,  # must contain a "run" step for the entire run
        run_type: str,
        ml_model_name: str,
        ml_model_type: str = None,
        num_layers: int = None,
        batch_size: int = None,
        config_params: dict = None,
        precision: str = None,
        dataset_name: str = None,
        profiler_name: str = None,
        input_sequence_length: int = None,
        output_sequence_length: int = None,
        image_dimension: int = None,
        perf_analysis: bool = None,
        training: bool = None,
    ):
        assert None not in [profiler, run_type, ml_model_name], "Missing required fields"
        assert profiler.contains_step("run"), "Run step not found in profiler"

        run_start_ts = profiler.get_str_start("run")
        run_end_ts = profiler.get_str_end("run")
        partial_benchmark_run = PartialBenchmarkRun(
            run_start_ts=run_start_ts,
            run_end_ts=run_end_ts,
            run_type=run_type,
            ml_model_name=ml_model_name,
            ml_model_type=ml_model_type,
            num_layers=num_layers,
            batch_size=batch_size,
            config_params=config_params,
            precision=precision,
            dataset_name=dataset_name,
            profiler_name=profiler_name,
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            image_dimension=image_dimension,
            perf_analysis=perf_analysis,
            training=training,
            measurements=self.measure_data,
        )

        json_data = partial_benchmark_run.model_dump_json()

        filename = os.path.join(self.output_folder, f"partial_run_{run_start_ts}.json")
        parent_dir = os.path.dirname(filename)
        if parent_dir != "" and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(filename, "w") as f:
            f.write(json_data)
        logger.info(f"Run and measurement data saved to {filename}")
