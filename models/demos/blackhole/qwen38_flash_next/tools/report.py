# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Write one measured task run to the standard benchmark artifact format."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

DEVICE_NAMES = ("P150x4", "P300x2")  # four p150, or the two p300 of a QuietBox 2 (the in-tree QB2 models' name)


def report(summary, *, device_name, mtp_drafts=None):
    if device_name not in DEVICE_NAMES:
        raise ValueError(f"device_name must name the chips the run used, one of {DEVICE_NAMES}, got {device_name!r}")
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
        }
        | {"mtp_drafts": mtp_drafts},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--device-name",
        required=True,
        choices=DEVICE_NAMES,
        help="the chips the run used; a 1x4 mesh does not tell four p150 from a QuietBox 2",
    )
    parser.add_argument("--mtp-drafts", type=int, choices=(3, 4, 5), help="the server's --mtp, when it drafted")
    args = parser.parse_args(argv)
    report(json.loads(args.summary.read_text()), device_name=args.device_name, mtp_drafts=args.mtp_drafts)


if __name__ == "__main__":
    main()
