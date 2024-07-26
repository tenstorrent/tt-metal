# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import csv
import json
from datetime import datetime
import pytz
from loguru import logger
from typing import List


class BenchmarkProfiler:
    def __init__(self):
        self.start_times = dict()
        self.end_times = dict()

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


def save_data_csv(data: List[dict], filename: str):
    assert len(data) > 0, "No data to save"

    parent_dir = os.path.dirname(filename)
    if parent_dir != "" and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)


class BenchmarkData:
    def __init__(self):
        self.measure_data = []
        self.output_folder = "generated/benchmark_data/"

    def add_measurement(
        self,
        profiler: BenchmarkProfiler,
        iteration: int,
        step_name: str,
        measurement_name: str,
        value: float,
        step_warm_up_num_iterations: int = None,
        target: float = None,
        device_power: float = None,
        device_temperature: float = None,
    ):
        assert None not in [profiler, iteration, step_name, measurement_name, value], "Missing required fields"
        assert profiler.contains_step(
            step_name, iteration
        ), f"Completed step '{step_name}' for iteration {iteration} not found in profiler"
        self.measure_data.append(
            {
                "step_start_ts": profiler.get_str_start(step_name, iteration),
                "step_end_ts": profiler.get_str_end(step_name, iteration),
                "iteration": iteration,
                "step_name": step_name,
                "measurement_name": measurement_name,
                "value": value,
                "step_warm_up_num_iterations": step_warm_up_num_iterations,
                "target": target,
                "device_power": device_power,
                "device_temperature": device_temperature,
            }
        )

    def prep_csvs(
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
    ):
        assert None not in [profiler, run_type, ml_model_name], "Missing required fields"
        assert profiler.contains_step("run"), "Run step not found in profiler"
        run_start_ts = profiler.get_str_start("run")
        run_end_ts = profiler.get_str_end("run")

        def prep_run_csv():
            """
            Run data contains a single record with attributes for a single benchmark run.
            """

            run_data = [
                {
                    "run_start_ts": run_start_ts,
                    "run_end_ts": run_end_ts,
                    "run_type": run_type,
                    "ml_model_name": ml_model_name,
                    "ml_model_type": ml_model_type,
                    "num_layers": num_layers,
                    "batch_size": batch_size,
                    "config_params": json.dumps(config_params),
                    "precision": precision,
                    "dataset_name": dataset_name,
                    "profiler_name": profiler_name,
                    "input_sequence_length": input_sequence_length,
                    "output_sequence_length": output_sequence_length,
                    "image_dimension": image_dimension,
                }
            ]

            filename = os.path.join(self.output_folder, f"run_{run_start_ts}.csv")
            save_data_csv(run_data, filename)
            logger.info(f"Run data saved to {filename}")

        def prep_measurement_csv():
            """
            Measurement data contains records and attributes for each measurement performed at each iteration and each step of the benchmark run.
            The triad of fields (iteration, step_name, name) must be unique.
            """

            filename = os.path.join(self.output_folder, f"measurement_{run_start_ts}.csv")
            save_data_csv(self.measure_data, filename)
            logger.info(f"Measurement data saved to {filename}")

        prep_run_csv()
        prep_measurement_csv()
