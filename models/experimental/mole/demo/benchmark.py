# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MoLE TTNN inference benchmark using the shared checkpoint endpoint."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import ttnn

from models.experimental.mole.demo.run import (
    CheckpointEndpointOptions,
    CheckpointInferenceEndpoint,
    add_dataset_arguments,
    add_model_arguments,
    close_ttnn_device,
    config_from_checkpoint_resolution,
    open_ttnn_device,
    resolve_mole_checkpoint,
    set_random_seed,
    unpack_batch,
    upload_mole_inputs,
)
from models.experimental.mole.reference.config import MoLEConfig


MILLISECONDS_PER_SECOND = 1000.0
TEST_SPLIT = "test"


@dataclass(frozen=True)
class BenchmarkOptions:
    checkpoint_path: str = ""
    dataset_csv_path: str = ""
    assets_root: str = ""
    dataset: str = ""
    batch_size: int = 8
    warmup_iterations: int = 2
    measure_iterations: int = 20
    seed: int = 0


def _validate_benchmark_options(options: BenchmarkOptions) -> None:
    if not options.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if options.measure_iterations <= 0:
        raise ValueError("measure_iterations must be > 0")


def _measure_loop(infer_once, *, batch_size: int, iterations: int, device: Any) -> dict[str, float]:
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
        return model.forward_prediction(tt_input, tt_marks)

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
) -> dict[str, object]:
    benchmark_options = options or BenchmarkOptions()
    _validate_benchmark_options(benchmark_options)

    set_random_seed(benchmark_options.seed)

    endpoint = CheckpointInferenceEndpoint(
        device=device,
        options=CheckpointEndpointOptions(
            checkpoint_path=benchmark_options.checkpoint_path,
            dataset_csv_path=benchmark_options.dataset_csv_path,
            assets_root=benchmark_options.assets_root,
            dataset=benchmark_options.dataset,
        ),
    )
    loaders, config = endpoint.resolve_dataset(
        config,
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

    return {
        "batch_size": float(batch_size),
        "total_calls": primary_metrics["total_calls"],
        "checkpoint_path": endpoint.options.checkpoint_path,
        "dataset_csv_path": endpoint.options.dataset_csv_path,
        "num_experts": float(config.num_experts),
        "ttnn_latency_ms": primary_metrics["latency_ms"],
        "ttnn_sequences_per_second": primary_metrics["sequences_per_second"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure MoLE TTNN inference latency and throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dataset_arguments(parser)
    add_model_arguments(parser)
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

    resolution = resolve_mole_checkpoint(
        dataset=args.dataset,
        base_model_type=args.base_model_type,
        num_experts=args.num_experts,
        assets_root=args.dataset_dir,
    )
    config = config_from_checkpoint_resolution(
        resolution,
        base_model_type=args.base_model_type,
        num_experts=args.num_experts,
    )

    device = open_ttnn_device()
    try:
        options = BenchmarkOptions(
            checkpoint_path=resolution.checkpoint_path,
            dataset_csv_path=resolution.dataset_csv_path,
            assets_root=args.dataset_dir,
            dataset=args.dataset,
            batch_size=args.batch_size,
            warmup_iterations=args.warmup_iterations,
            measure_iterations=args.measure_iterations,
            seed=args.seed,
        )
        metrics = run_benchmark(device, config, options=options)
    finally:
        close_ttnn_device(device)

    print("MoLE inference benchmark")
    print(f"- checkpoint_path: {metrics['checkpoint_path']}")
    print(f"- dataset_csv_path: {metrics['dataset_csv_path']}")
    print(f"- num_experts: {metrics['num_experts']:.0f}")
    print(f"- batch_size: {metrics['batch_size']:.0f}")
    print(f"- total_calls: {metrics['total_calls']:.0f}")
    print(f"- ttnn_latency_ms: {metrics['ttnn_latency_ms']:.3f}")
    print(f"- ttnn_sequences_per_second: {metrics['ttnn_sequences_per_second']:.3f}")


if __name__ == "__main__":
    main()
