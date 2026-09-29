# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Write one measured task run to the standard benchmark artifact format."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


def report(summary, *, device_name):
    if device_name not in ("P150x4", "QB2"):
        raise ValueError("device_name must come from the qualified physical allocation: P150x4 or QB2")
    if summary["completed_samples"] != summary["requested_samples"]:
        raise ValueError("An incomplete task run cannot supply the CI accuracy gate")
    profiler = BenchmarkProfiler()
    for phase in ("run", "inference", "inference_prefill", "inference_decode"):
        profiler.start_times[(0, phase)] = datetime.fromisoformat(summary["measurement_start"])
        profiler.end_times[(0, phase)] = datetime.fromisoformat(summary["measurement_end"])
    benchmark = BenchmarkData()
    for phase, name, value in (
        ("inference", "gsm8k_accuracy", 100 * summary["accuracy"]),
        ("inference_prefill", "time_to_token", summary["mean_ttft_s"]),
        ("inference_decode", "tokens/s/user", summary["mean_stream_decode_tokens_per_s"]),
        ("inference_decode", "tokens/s", summary["aggregate_output_tokens_per_s"]),
    ):
        if value is not None:
            benchmark.add_measurement(profiler, 0, phase, name, value)
    benchmark.save_partial_run_json(
        profiler,
        run_type="demo",
        ml_model_name="qwen38-flash-next",
        ml_model_type="LLM",
        device_name=device_name,
        num_layers=48,
        batch_size=1,
        dataset_name=f"GSM8K: {summary['scope']}",
        precision="BF4 routed experts; BF8 dense weights; BF16 activations",
        config_params={
            key: summary[key]
            for key in (
                "checkpoint_revision",
                "dataset_revision",
                "dataset_sha256",
                "scope",
                "completed_samples",
                "requested_samples",
                "max_output_tokens",
                "ignore_eos",
                "timing_note",
                "accuracy_floor",
                "passed",
            )
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--device-name",
        required=True,
        choices=("P150x4", "QB2"),
        help="Physical hardware from allocation/qualification evidence; a logical 1x4 mesh does not identify the SKU",
    )
    args = parser.parse_args(argv)
    report(json.loads(args.summary.read_text()), device_name=args.device_name)


if __name__ == "__main__":
    main()
