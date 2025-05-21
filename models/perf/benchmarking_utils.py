# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from typing import List

import pytz
from loguru import logger

# Decouple dependency of model tests on infra folder unless running in CI
IS_CI_ENV = os.getenv("CI") == "true"
if IS_CI_ENV:
    from infra.data_collection.pydantic_models import BenchmarkMeasurement, PartialBenchmarkRun
else:
    logger.warning("Skipping import of pydantic_models for benchmarking since not running in CI environment")


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

    def _get_step_durations(self, step_name: str, start_iteration: int = 0):
        """Helper method to collect all durations for a given step across iterations."""
        durations = []
        iteration = start_iteration
        while self.contains_step(step_name, iteration):
            durations.append(self.get_duration(step_name, iteration))
            iteration += 1
        return durations

    def get_duration_average(self, step_name: str, start_iteration: int = 0):
        durations = self._get_step_durations(step_name, start_iteration)
        return sum(durations) / len(durations)

    def get_duration_sum(self, step_name: str, start_iteration: int = 0):
        durations = self._get_step_durations(step_name, start_iteration)
        return sum(durations)


class BenchmarkData:
    def __init__(self):
        if IS_CI_ENV:
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
        if IS_CI_ENV:  # no-op if not running in CI environment
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
        if IS_CI_ENV:  # no-op if not running in CI environment
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
        else:
            logger.info("Skipping saving benchmark data JSON since not running in CI environment")
