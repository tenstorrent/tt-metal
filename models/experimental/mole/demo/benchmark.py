# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MoLE TTNN inference benchmark using the shared checkpoint endpoint."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import ttnn

from models.experimental.mole.demo.run import (
    CheckpointEndpointOptions,
    CheckpointInferenceEndpoint,
    add_dataset_arguments,
    add_model_arguments,
    close_ttnn_device,
    model_config_from_args,
    open_ttnn_device,
    set_random_seed,
    unpack_batch,
    upload_mole_inputs,
)
from models.experimental.mole.reference.config import MoLEConfig, replace_num_experts


MILLISECONDS_PER_SECOND = 1000.0
TEST_SPLIT = "test"
CHECKPOINT_BASE_DIR = "/demo_checkpoints"


@dataclass(frozen=True)
class BenchmarkOptions:
    checkpoint_path: str | None = None
    checkpoint_debug_keys: int = 0
    batch_size: int = 8
    warmup_iterations: int = 2
    measure_iterations: int = 20
    seed: int = 0
    dataset_dir: str | None = None
    dataset_file: str | None = None


def _resolve_checkpoint_path(checkpoint_file: str) -> str:
    base_dir = os.path.abspath(CHECKPOINT_BASE_DIR)
    checkpoint_path = os.path.abspath(os.path.join(base_dir, checkpoint_file))
    if not checkpoint_path.startswith(base_dir + os.sep):
        raise ValueError("checkpoint path escapes checkpoint_dir")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _validate_benchmark_options(options: BenchmarkOptions) -> None:
    if not options.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not options.dataset_dir:
        raise ValueError("dataset_dir is required")
    if options.measure_iterations <= 0:
        raise ValueError("measure_iterations must be > 0")


def _measure_loop(
    infer_once: Callable[[], object],
    *,
    batch_size: int,
    iterations: int,
    device: Any,
) -> dict[str, float]:
    start = time.perf_counter()
    for _ in range(iterations):
        infer_once()
    ttnn.synchronize_device(device)
    elapsed_s = time.perf_counter() - start

    return {
        "total_calls": float(iterations),
        "latency_ms": (elapsed_s / iterations) * MILLISECONDS_PER_SECOND,
        "sequences_per_second": (batch_size * iterations) / elapsed_s,
    }


def _forward_prediction_only(model, tt_input, tt_marks):
    if hasattr(model, "forward_prediction_no_trace"):
        return model.forward_prediction_no_trace(tt_input, tt_marks)
    return model.forward_prediction(tt_input, tt_marks)


def _benchmark_single_model(
    *,
    model: Any,
    device: Any,
    torch_input,
    torch_input_mark,
    batch_size: int,
    warmup_iterations: int,
    measure_iterations: int,
) -> dict[str, float]:
    tt_input, tt_marks = upload_mole_inputs(
        model=model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
    )

    def infer_once() -> object:
        return _forward_prediction_only(model, tt_input, tt_marks)

    for _ in range(warmup_iterations):
        infer_once()
    ttnn.synchronize_device(device)

    return _measure_loop(
        infer_once=infer_once,
        batch_size=batch_size,
        iterations=measure_iterations,
        device=device,
    )


def run_benchmark(
    device: Any,
    config: MoLEConfig,
    *,
    options: BenchmarkOptions | None = None,
) -> dict[str, float]:
    benchmark_options = options or BenchmarkOptions()
    _validate_benchmark_options(benchmark_options)

    set_random_seed(benchmark_options.seed)

    endpoint = CheckpointInferenceEndpoint(
        device=device,
        options=CheckpointEndpointOptions(
            checkpoint_path=benchmark_options.checkpoint_path,
            checkpoint_debug_keys=benchmark_options.checkpoint_debug_keys,
        ),
    )
    loaders, config = endpoint.resolve_dataset(
        config,
        dataset_dir=benchmark_options.dataset_dir,
        dataset_file=benchmark_options.dataset_file,
        eval_batch_size=benchmark_options.batch_size,
    )

    batch = next(iter(loaders[TEST_SPLIT]))
    torch_input, _, torch_input_mark, _ = unpack_batch(batch)
    if torch_input_mark is None:
        raise ValueError("benchmark requires x_mark time features")

    batch_size = int(torch_input.shape[0])

    primary_model = endpoint.build_mole_ttnn(config)
    primary_metrics = _benchmark_single_model(
        model=primary_model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
        batch_size=batch_size,
        warmup_iterations=benchmark_options.warmup_iterations,
        measure_iterations=benchmark_options.measure_iterations,
    )

    baseline_latency_ms = 0.0
    baseline_sequences_per_second = 0.0
    expert_overhead_x = 0.0

    if config.num_experts > 1:
        baseline_config = replace_num_experts(config, num_experts=1)
        baseline_model = endpoint.build_mole_ttnn(baseline_config)
        baseline_metrics = _benchmark_single_model(
            model=baseline_model,
            device=device,
            torch_input=torch_input,
            torch_input_mark=torch_input_mark,
            batch_size=batch_size,
            warmup_iterations=benchmark_options.warmup_iterations,
            measure_iterations=benchmark_options.measure_iterations,
        )
        baseline_latency_ms = baseline_metrics["latency_ms"]
        baseline_sequences_per_second = baseline_metrics["sequences_per_second"]
        if baseline_latency_ms > 0.0:
            expert_overhead_x = primary_metrics["latency_ms"] / baseline_latency_ms

    return {
        "batch_size": float(batch_size),
        "total_calls": primary_metrics["total_calls"],
        "ttnn_latency_ms": primary_metrics["latency_ms"],
        "ttnn_sequences_per_second": primary_metrics["sequences_per_second"],
        "baseline_ttnn_latency_ms": baseline_latency_ms,
        "baseline_ttnn_sequences_per_second": baseline_sequences_per_second,
        "expert_overhead_x": expert_overhead_x,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure MoLE TTNN inference latency and throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dataset_arguments(parser)
    add_model_arguments(parser)
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint.pth",
        help="Checkpoint file path relative to /demo_checkpoints",
    )
    parser.add_argument(
        "--checkpoint-debug-keys",
        type=int,
        default=0,
        help="If > 0, print a checkpoint key/shape sample before load for mismatch debugging",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Warmup iterations before measurement. Use 10+ to avoid cold-start artifacts.",
    )
    parser.add_argument("--measure-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = model_config_from_args(args)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_file)

    device = open_ttnn_device()
    try:
        options = BenchmarkOptions(
            checkpoint_path=checkpoint_path,
            checkpoint_debug_keys=args.checkpoint_debug_keys,
            batch_size=args.batch_size,
            warmup_iterations=args.warmup_iterations,
            measure_iterations=args.measure_iterations,
            seed=args.seed,
            dataset_dir=args.dataset_dir,
            dataset_file=args.dataset_file,
        )
        metrics = run_benchmark(device, config, options=options)
    finally:
        close_ttnn_device(device)

    print("MoLE inference benchmark")
    print(f"- batch_size: {metrics['batch_size']:.0f}")
    print(f"- total_calls: {metrics['total_calls']:.0f}")
    print(f"- ttnn_latency_ms: {metrics['ttnn_latency_ms']:.3f}")
    print(f"- ttnn_sequences_per_second: {metrics['ttnn_sequences_per_second']:.3f}")
    print(f"- baseline_ttnn_latency_ms: {metrics['baseline_ttnn_latency_ms']:.3f}")
    print(f"- baseline_ttnn_sequences_per_second: {metrics['baseline_ttnn_sequences_per_second']:.3f}")
    if metrics["expert_overhead_x"] > 0.0:
        overhead_note = (
            ""
            if metrics["expert_overhead_x"] >= 1.0
            else " (values <1.0 indicate measurement variance, not real speedup)"
        )
        print(f"- expert_overhead_x: {metrics['expert_overhead_x']:.3f}{overhead_note}")


if __name__ == "__main__":
    main()
