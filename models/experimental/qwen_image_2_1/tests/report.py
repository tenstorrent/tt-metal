# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Report image agreement and completed-request throughput to the standard CI collector."""

import json
import sys
from datetime import datetime
from pathlib import Path

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


def report(directory):
    records = [json.loads((directory / f"served-{case}.json").read_text()) for case in ("text", "edit")]
    profiler = BenchmarkProfiler()
    for phase in ("run", "inference"):
        profiler.start_times[(0, phase)] = datetime.fromisoformat(records[0]["run_start"])
        profiler.end_times[(0, phase)] = datetime.fromisoformat(records[-1]["run_end"])
    benchmark = BenchmarkData()
    for record in records:
        case = record["case"]
        benchmark.add_measurement(profiler, 0, "inference", f"{case}_image_pcc", record["image_pcc"])
    completed = sum(row["measured_requests"] for row in records)
    seconds = sum(timing["wall_s"] for row in records for timing in row["timings"])
    benchmark.add_measurement(profiler, 0, "inference", "fps", completed / seconds)
    benchmark.save_partial_run_json(
        profiler,
        run_type="demo",
        ml_model_name="qwen-image-2.1-p150",
        ml_model_type="diffusion",
        device_name="P150",
        num_layers=32,
        batch_size=1,
        dataset_name="One fixed text prompt; one two-image edit regression",
        config_params={
            "steps": 40,
            "seed": 42,
            "warmup_requests_per_case": 1,
            "measured_requests_per_case": 3,
            "text_reference": "independent CUDA bf16",
            "edit_reference": "original P150 regression",
            "source_revision": records[0]["source_revision"],
            "checkpoint_revision": records[0]["checkpoint_revision"],
        },
    )


if __name__ == "__main__":
    report(Path(sys.argv[1]))
