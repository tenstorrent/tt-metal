# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Convert the isolated evaluator's result to tt-metal's CI benchmark format."""

import json
import sys
from datetime import datetime
from pathlib import Path

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


def report(summary):
    profiler = BenchmarkProfiler()
    for phase in ("run", "inference", "inference_prefill", "inference_decode"):
        start = summary["run_start" if phase == "run" else "measurement_start"]
        profiler.start_times[(0, phase)] = datetime.fromisoformat(start)
        profiler.end_times[(0, phase)] = datetime.fromisoformat(summary["measurement_end"])
    benchmark = BenchmarkData()
    for phase, name, value in (
        ("inference", "ifeval_accuracy", 100 * summary["mean_accuracy"]),
        ("inference_prefill", "time_to_token", summary["mean_ttft_ms"] / 1000),
        ("inference_decode", "tokens/s/user", summary["mean_decode_tokens_per_s"]),
        ("inference_decode", "tokens/s", summary["aggregate_output_tokens_per_s"]),
    ):
        benchmark.add_measurement(profiler, 0, phase, name, value)
    benchmark.save_partial_run_json(
        profiler,
        run_type="demo",
        ml_model_name="llama3.1-8b-qb2",
        ml_model_type="LLM",
        device_name="P150x4",
        num_layers=32,
        batch_size=1,
        dataset_name="Meta IFEval, first 28",
        config_params={"input_sha256": summary["input_sha256"], "requests": summary["requests"]},
    )


if __name__ == "__main__":
    report(json.loads(Path(sys.argv[1]).read_text()))
