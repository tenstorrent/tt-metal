# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MoLE benchmark: CPU reference timing vs TTNN inference timing.

**trace_replay (default)** — One-time trace capture, then repeated ``execute_trace`` replay 

**e2e** — Each iteration: ``upload_mole_inputs`` → forward → ``ttnn.to_torch`` readback
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import partial

import torch
import ttnn

try:
    from tracy import signpost as _tracy_signpost
except ImportError:

    def _tracy_signpost(*_args: object, **_kwargs: object) -> None:
        """Tracy profiler hook; no-op when the optional ``tracy`` package is absent."""


from models.experimental.mole.demo.core import (
    add_dataset_arguments,
    build_reference_mole,
    build_ttnn_mole,
    capture_trace,
    execute_trace,
    open_ttnn_device,
    release_trace,
    resolve_eval_input,
    set_random_seed,
    upload_mole_inputs,
)
from models.experimental.mole.reference.config import MoLEConfig, replace_num_experts


MILLISECONDS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class BenchmarkOptions:
    batch_size: int = 8
    warmup_iterations: int = 2
    measure_iterations: int = 6
    inner_iterations: int = 12
    seed: int = 0
    tt_only: bool = False
    include_expert_overhead: bool = True
    profile_single_replay: bool = False
    e2e: bool = False
    dataset_path: str | None = None


def _forward_prediction_only(model, tt_input, tt_marks):
    if hasattr(model, "forward_prediction_no_trace"):
        return model.forward_prediction_no_trace(tt_input, tt_marks)
    return model.forward_prediction(tt_input, tt_marks)


def _measure_infer_loop(
    infer_once,
    *,
    batch_size: int,
    iterations: int,
    inner_iterations: int,
    sync_device=None,
) -> dict[str, float]:
    total_calls = iterations * inner_iterations
    start = time.perf_counter()
    for _ in range(iterations):
        for _ in range(inner_iterations):
            infer_once()
    if sync_device is not None:
        ttnn.synchronize_device(sync_device)
    elapsed_s = time.perf_counter() - start
    return {
        "total_calls": float(total_calls),
        "latency_ms": (elapsed_s / total_calls) * MILLISECONDS_PER_SECOND,
        "sequences_per_second": (batch_size * total_calls) / elapsed_s,
    }


def _readback_ttnn_output(output) -> None:
    if isinstance(output, tuple):
        for tensor in output:
            ttnn.to_torch(tensor)
    else:
        ttnn.to_torch(output)


def _e2e_infer_once_step(model, device, torch_input: torch.Tensor, torch_input_mark: torch.Tensor) -> None:
    tt_input, tt_marks = upload_mole_inputs(
        model=model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
    )
    output = _forward_prediction_only(model, tt_input, tt_marks)
    _readback_ttnn_output(output)


def _run_e2e_metrics(
    *,
    device,
    tt_model,
    tt_compare_model,
    torch_input: torch.Tensor,
    torch_input_mark: torch.Tensor,
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    measured_inner_iterations: int,
    include_expert_overhead: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    primary_once = partial(_e2e_infer_once_step, tt_model, device, torch_input, torch_input_mark)
    for _ in range(warmup_iterations):
        primary_once()

    ttnn_metrics = _measure_infer_loop(
        primary_once,
        batch_size=batch_size,
        iterations=measured_iterations,
        inner_iterations=measured_inner_iterations,
    )

    if (not include_expert_overhead) or tt_compare_model is None:
        return ttnn_metrics, {"latency_ms": 0.0, "sequences_per_second": 0.0}

    compare_once = partial(_e2e_infer_once_step, tt_compare_model, device, torch_input, torch_input_mark)
    for _ in range(warmup_iterations):
        compare_once()

    compare_metrics = _measure_infer_loop(
        compare_once,
        batch_size=batch_size,
        iterations=measured_iterations,
        inner_iterations=measured_inner_iterations,
    )
    return ttnn_metrics, compare_metrics


def _resolve_replay_counts(
    *,
    measure_iterations: int,
    inner_iterations: int,
    profile_single_replay: bool,
) -> tuple[int, int]:
    if profile_single_replay:
        return 1, 1
    return measure_iterations, inner_iterations


@torch.no_grad()
def _cpu_infer_step(reference_model, torch_input: torch.Tensor, torch_input_mark: torch.Tensor) -> None:
    reference_model(torch_input, torch_input_mark)


def _cpu_metrics(
    *,
    reference_model,
    torch_input: torch.Tensor,
    torch_input_mark: torch.Tensor,
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    measured_inner_iterations: int,
    tt_only: bool,
) -> dict[str, float]:
    if tt_only:
        return {
            "total_calls": float(measured_iterations * measured_inner_iterations),
            "latency_ms": 0.0,
            "sequences_per_second": 0.0,
        }

    cpu_once = partial(_cpu_infer_step, reference_model, torch_input, torch_input_mark)
    for _ in range(warmup_iterations):
        cpu_once()

    return _measure_infer_loop(
        cpu_once,
        batch_size=batch_size,
        iterations=measured_iterations,
        inner_iterations=measured_inner_iterations,
    )


def _replay_trace_step(trace_state) -> None:
    execute_trace(trace_state, blocking=False)


def _run_single_trace_metrics(
    *,
    device,
    trace_state,
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    measured_inner_iterations: int,
) -> dict[str, float]:
    replay_once = partial(_replay_trace_step, trace_state)
    for _ in range(warmup_iterations):
        replay_once()
    ttnn.synchronize_device(device)

    _tracy_signpost("start")
    try:
        return _measure_infer_loop(
            replay_once,
            batch_size=batch_size,
            iterations=measured_iterations,
            inner_iterations=measured_inner_iterations,
            sync_device=device,
        )
    finally:
        _tracy_signpost("stop")


def _trace_replay_metrics_pair(
    *,
    device,
    tt_model,
    tt_compare_model,
    tt_input,
    tt_marks,
    tt_compare_input,
    tt_compare_marks,
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    measured_inner_iterations: int,
    include_expert_overhead: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run primary (and optional compare) trace capture + replay timing."""
    compare_metrics: dict[str, float] = {"latency_ms": 0.0, "sequences_per_second": 0.0}
    primary_trace_state = None
    try:
        primary_trace_state = capture_trace(
            model=tt_model,
            device=device,
            tt_input=tt_input,
            tt_marks=tt_marks,
            prediction_only=True,
        )
        ttnn_metrics = _run_single_trace_metrics(
            device=device,
            trace_state=primary_trace_state,
            batch_size=batch_size,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            inner_iterations=measured_inner_iterations,
        )
    finally:
        if primary_trace_state is not None:
            release_trace(primary_trace_state)

    if include_expert_overhead and tt_compare_model is not None:
        compare_trace_state = None
        try:
            compare_trace_state = capture_trace(
                model=tt_compare_model,
                device=device,
                tt_input=tt_compare_input,
                tt_marks=tt_compare_marks,
                prediction_only=True,
            )
            compare_metrics = _run_single_trace_metrics(
                device=device,
                trace_state=compare_trace_state,
                batch_size=batch_size,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
                inner_iterations=measured_inner_iterations,
            )
        finally:
            if compare_trace_state is not None:
                release_trace(compare_trace_state)

    return ttnn_metrics, compare_metrics


def run_benchmark(
    device,
    config: MoLEConfig,
    *,
    dataset_name: str,
    options: BenchmarkOptions | None = None,
) -> dict[str, float]:
    benchmark_options = options or BenchmarkOptions()

    set_random_seed(benchmark_options.seed)

    torch_input, torch_input_mark, config = resolve_eval_input(
        config,
        batch_size=benchmark_options.batch_size,
        dataset_name=dataset_name,
        dataset_path=benchmark_options.dataset_path,
    )
    actual_batch_size = torch_input.shape[0]

    reference_model = build_reference_mole(config).eval()
    tt_model = build_ttnn_mole(device, config, reference_model)
    tt_compare_model = None
    if benchmark_options.include_expert_overhead:
        compare_config = replace_num_experts(config, num_experts=1)
        compare_reference_model = build_reference_mole(compare_config).eval()
        tt_compare_model = build_ttnn_mole(device, compare_config, compare_reference_model)
    tt_input = None
    tt_marks = None
    tt_compare_input = None
    tt_compare_marks = None
    if not benchmark_options.e2e:
        tt_input, tt_marks = upload_mole_inputs(
            model=tt_model,
            device=device,
            torch_input=torch_input,
            torch_input_mark=torch_input_mark,
        )
        if tt_compare_model is not None:
            tt_compare_input, tt_compare_marks = upload_mole_inputs(
                model=tt_compare_model,
                device=device,
                torch_input=torch_input,
                torch_input_mark=torch_input_mark,
            )

    measured_iterations, measured_inner_iterations = _resolve_replay_counts(
        measure_iterations=benchmark_options.measure_iterations,
        inner_iterations=benchmark_options.inner_iterations,
        profile_single_replay=benchmark_options.profile_single_replay,
    )

    include_expert_overhead = benchmark_options.include_expert_overhead

    cpu_metrics = _cpu_metrics(
        reference_model=reference_model,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
        batch_size=actual_batch_size,
        warmup_iterations=benchmark_options.warmup_iterations,
        measured_iterations=measured_iterations,
        measured_inner_iterations=measured_inner_iterations,
        tt_only=benchmark_options.tt_only,
    )

    if benchmark_options.e2e:
        ttnn_metrics, compare_metrics = _run_e2e_metrics(
            device=device,
            tt_model=tt_model,
            tt_compare_model=tt_compare_model,
            torch_input=torch_input,
            torch_input_mark=torch_input_mark,
            batch_size=actual_batch_size,
            warmup_iterations=benchmark_options.warmup_iterations,
            measured_iterations=measured_iterations,
            measured_inner_iterations=measured_inner_iterations,
            include_expert_overhead=include_expert_overhead,
        )
    else:
        ttnn_metrics, compare_metrics = _trace_replay_metrics_pair(
            device=device,
            tt_model=tt_model,
            tt_compare_model=tt_compare_model,
            tt_input=tt_input,
            tt_marks=tt_marks,
            tt_compare_input=tt_compare_input,
            tt_compare_marks=tt_compare_marks,
            batch_size=actual_batch_size,
            warmup_iterations=benchmark_options.warmup_iterations,
            measured_iterations=measured_iterations,
            measured_inner_iterations=measured_inner_iterations,
            include_expert_overhead=include_expert_overhead,
        )

    if benchmark_options.tt_only:
        speedup_x = 0.0
    else:
        speedup_x = ttnn_metrics["sequences_per_second"] / cpu_metrics["sequences_per_second"]

    if compare_metrics["latency_ms"] > 0.0:
        expert_overhead_x = ttnn_metrics["latency_ms"] / compare_metrics["latency_ms"]
    else:
        expert_overhead_x = 0.0

    return {
        "batch_size": float(actual_batch_size),
        "total_calls": cpu_metrics["total_calls"],
        "cpu_latency_ms": cpu_metrics["latency_ms"],
        "cpu_sequences_per_second": cpu_metrics["sequences_per_second"],
        "ttnn_latency_ms": ttnn_metrics["latency_ms"],
        "ttnn_sequences_per_second": ttnn_metrics["sequences_per_second"],
        "speedup_x": speedup_x,
        "expert_overhead_x": expert_overhead_x,
        "tt_only": float(1 if benchmark_options.tt_only else 0),
        "profile_single_replay": float(1 if benchmark_options.profile_single_replay else 0),
        "e2e": float(1 if benchmark_options.e2e else 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure MoLE inference timing on TTNN, with optional CPU reference timing for context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dataset_arguments(
        parser,
        dataset_help="Dataset name, e.g. weather, etth1, etth2, ettm1, ettm2",
        dataset_path_help="Optional CSV path for dataset evaluation",
    )
    parser.add_argument("--base-model-type", choices=("dlinear", "rlinear", "rmlp"), default="dlinear")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="Time-feature layout for marks: hourly-style (suffix 'h') uses 4 features; otherwise 5",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measure-iterations", type=int, default=6)
    parser.add_argument("--inner-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tt-only",
        action="store_true",
        help="Skip CPU reference timing and report TTNN-only measurements",
    )
    parser.add_argument(
        "--skip-expert-overhead",
        action="store_true",
        help="Skip the TTNN num_experts=1 comparison used to compute expert_overhead_x",
    )
    parser.add_argument(
        "--profile-single-replay",
        action="store_true",
        help="Measure exactly one trace replay call for cleaner profiler captures",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Measure end-to-end TTNN inference including upload, execution, and host readback",
    )
    args = parser.parse_args()

    config = MoLEConfig(
        base_model_type=args.base_model_type,
        num_experts=args.num_experts,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        freq=args.freq,
    )
    device = open_ttnn_device()
    try:
        options = BenchmarkOptions(
            batch_size=args.batch_size,
            warmup_iterations=args.warmup_iterations,
            measure_iterations=args.measure_iterations,
            inner_iterations=args.inner_iterations,
            seed=args.seed,
            tt_only=args.tt_only,
            include_expert_overhead=not args.skip_expert_overhead,
            profile_single_replay=args.profile_single_replay,
            e2e=args.e2e,
            dataset_path=args.dataset_path,
        )
        metrics = run_benchmark(
            device,
            config,
            dataset_name=args.dataset_name,
            options=options,
        )
    finally:
        ttnn.close_device(device)

    mode = "e2e" if metrics["e2e"] > 0 else "trace_replay"
    print("MoLE inference benchmark")
    if metrics["tt_only"] > 0:
        print("- mode: tt_only")
    if metrics["profile_single_replay"] > 0:
        print("- replay_mode: single")
    print(f"- measurement_mode: {mode}")
    print(f"- batch_size: {metrics['batch_size']:.0f}")
    print(f"- total_calls: {metrics['total_calls']:.0f}")
    print(f"- cpu_latency_ms: {metrics['cpu_latency_ms']:.3f}")
    print(f"- cpu_sequences_per_second: {metrics['cpu_sequences_per_second']:.3f}")
    print(f"- ttnn_latency_ms: {metrics['ttnn_latency_ms']:.3f}")
    print(f"- ttnn_sequences_per_second: {metrics['ttnn_sequences_per_second']:.3f}")
    print(f"- speedup_x: {metrics['speedup_x']:.3f}")
    print(f"- expert_overhead_x: {metrics['expert_overhead_x']:.3f}")


if __name__ == "__main__":
    main()
