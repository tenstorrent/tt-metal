# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Publish measured serving results in the shared CI benchmark format."""

import json
import sys
from datetime import datetime
from pathlib import Path

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


def report(summary):
    runs = []
    if "gpqa_result" in summary:
        runs.append((summary["gpqa_result"], "GPQA Diamond, first 10/198", True))
    runs.extend(
        (row, f"Fixed-length text serving, server capacity {summary['server_capacity']}", False)
        for row in summary.get("performance_results", [])
    )
    for row, dataset, accuracy in runs:
        profiler = BenchmarkProfiler()
        for phase in ("run", "inference", "inference_prefill", "inference_decode"):
            profiler.start_times[(0, phase)] = datetime.fromisoformat(row["measurement_start"])
            profiler.end_times[(0, phase)] = datetime.fromisoformat(row["measurement_end"])
        benchmark = BenchmarkData()
        metrics = [
            (
                "inference_prefill",
                "time_to_token",
                row["mean_ttft_ms"] / 1000 if row["mean_ttft_ms"] is not None else None,
            ),
            ("inference_decode", "tokens/s/user", row["mean_decode_tokens_per_s"]),
            ("inference_decode", "tokens/s", row["aggregate_output_tokens_per_s"]),
        ]
        if accuracy:
            metrics.append(("inference", "gpqa_accuracy", 100 * row["accuracy"]))
        for phase, name, value in metrics:
            if value is not None:
                benchmark.add_measurement(profiler, 0, phase, name, value)
        benchmark.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name="gemma4-31b-qb2",
            ml_model_type="LLM",
            device_name="P300x2",
            num_layers=60,
            batch_size=row["concurrency"],
            dataset_name=dataset,
            input_sequence_length=row.get("input_tokens"),
            output_sequence_length=row.get("output_tokens"),
            precision="BF4 projections; BF8 KV; BF16 residuals",
            config_params={
                "checkpoint_revision": summary["checkpoint_revision"],
                "server_capacity": summary["server_capacity"],
                "prompt_sha256": row.get("prompt_sha256"),
                "input_sha256": summary["input_sha256"],
                "requests": row["requests"],
                "scope": summary["scope"] if accuracy else "warmed, greedy, ignored EOS",
            },
        )


if __name__ == "__main__":
    report(json.loads(Path(sys.argv[1]).read_text()))
