# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded interleaved D/N × reader-barrier combined-trace experiment.

Uses unchanged Vtransposed_fullchip.build: full compensated BF16, K8/V4,
native exp, Q256/K512 and two KV slots. No clock/power overrides or telemetry
processes. Timing includes real preprocessing; it is not attention-only time.
"""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("paired_vaxis_base", HERE / "Vtransposed_fullchip.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def candidates(barriers):
    assert 1 <= len(barriers) <= 3 and len(set(barriers)) == len(barriers)
    assert all(value in (2, 8, 16) for value in barriers)
    return [(axis, barrier) for barrier in barriers for axis in ("D", "N")]


def round_order(count, ordinal):
    """Rotate each pair's start; second round is the exact reverse of first."""
    assert 1 <= count <= 6
    order = [(ordinal // 2 + offset) % count for offset in range(count)]
    return list(reversed(order)) if ordinal % 2 else order


def memory_estimate(length, heads, choices, trace_bytes):
    """Tensor bytes for independent unchanged-builder instances, plus reserve.

    Per tile: three original BF16 tensors; Q BF16/K B8/V B4; BF16 output;
    N-axis adds a BF16 transposed V tensor. Program/allocator overhead is
    budgeted separately, not silently treated as zero.
    """
    tiles = heads * length * 128 // 1024
    tensors = {
        f"{axis}-b{barrier}": tiles * (3 * 2048 + 2048 + 1088 + 576 + 2048 + (2048 if axis == "N" else 0))
        for axis, barrier in choices
    }
    reserve = 256 * 1024**2
    return dict(
        candidate_count=len(choices),
        tensor_bytes_by_candidate=tensors,
        tensor_bytes=sum(tensors.values()),
        trace_region_bytes=trace_bytes,
        program_allocator_reserve_bytes=reserve,
        total_budget_bytes=sum(tensors.values()) + trace_bytes + reserve,
        warning="Conservative logical tensor budget; actual allocator view is also checked",
    )


def memory_view(device):
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    names = (
        "num_banks",
        "total_bytes_per_bank",
        "total_bytes_allocated_per_bank",
        "total_bytes_free_per_bank",
        "largest_contiguous_bytes_free_per_bank",
    )
    result = {name: int(getattr(view, name)) for name in names}
    result["free_bytes"] = result["num_banks"] * result["total_bytes_free_per_bank"]
    return result


def check_immutable(candidate, inputs):
    assert all(
        BASE.bf16_bitwise_equal(ttnn.to_torch(t).bfloat16(), x) for t, x in zip(candidate["originals"], inputs)
    ), "Original device input changed"


def check_output(candidate):
    actual = ttnn.to_torch(candidate["out"]).bfloat16()
    assert bool(torch.isfinite(actual).all()), "Nonfinite trace output"
    digest = BASE.tensor_hash(actual)
    assert digest == candidate["output_sha256"], "Combined replay changed output bits"
    return digest


def release_captures(device, traces, active_capture, emit):
    """Best-effort cleanup of every retained trace without masking a failure."""
    errors = []
    if active_capture is not None:
        try:
            ttnn.end_trace_capture(device, active_capture, cq_id=0)
        except Exception as error:
            errors.append(dict(action="end_failed_capture", error=repr(error)))
    for trace in reversed(traces):
        try:
            ttnn.release_trace(device, trace)
        except Exception as error:
            errors.append(dict(action="release_trace", trace=str(trace), error=repr(error)))
    emit(dict(kind="trace_cleanup", trace_count=len(traces), errors=errors))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument("--barriers", type=int, nargs="+", default=[2, 8, 16])
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", choices=("normal", "channel_v", "outliers"), default="normal")
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--iters", type=int, default=7, help="Measured complete round-robin rounds")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed complete warmup rounds")
    parser.add_argument("--trace-repeats", type=int, default=1)
    parser.add_argument("--trace-mib", type=int, default=64)
    parser.add_argument(
        "--max-retained-gib",
        type=float,
        default=24,
        help="Hard tensor/trace/reserve budget before any device allocation",
    )
    args = parser.parse_args()
    assert 512 <= args.length <= 262144 and args.length % 512 == 0
    assert args.heads > 0 and args.cores >= args.heads and args.cores % args.heads == 0
    assert 1 <= args.iters <= 31 and 0 <= args.warmup <= 10 and 1 <= args.trace_repeats <= 4
    assert 16 <= args.trace_mib <= 256 and args.sample_rows > 0 and 0 < args.max_retained_gib <= 32
    assert Path(args.label).name == args.label
    choices = candidates(args.barriers)
    budget = memory_estimate(args.length, args.heads, choices, args.trace_mib * 1024**2)
    assert (
        budget["total_budget_bytes"] <= args.max_retained_gib * 1024**3
    ), "Candidate memory budget exceeded; reduce --barriers or length, not the kernel geometry"
    paths = sorted(
        set(
            BASE.source_files("fast_bf16")
            + [
                Path(__file__).resolve(),
                ROOT / "ttnn/ttnn/device.py",
                ROOT / "ttnn/cpp/ttnn-nanobind/device.cpp",
                ROOT / "tt_metal/api/tt-metalium/memory_reporter.hpp",
                ROOT / "tt_metal/detail/reports/memory_reporter.cpp",
                ROOT / "tt_metal/hw/inc/api/dataflow/noc.h",
                ROOT / "tt_metal/hw/inc/api/dataflow/dataflow_api.h",
                ROOT / "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h",
            ]
        )
    )
    pinned = BASE.hashes(paths)
    torch.set_num_threads(4)
    inputs = BASE.REPRO.make_inputs(
        args.heads,
        args.length,
        args.length,
        128,
        args.seed,
        "normal" if args.distribution == "channel_v" else args.distribution,
    )
    if args.distribution == "channel_v":
        inputs[2][..., ::16] *= 32
    input_hashes = [BASE.tensor_hash(x) for x in inputs]
    count = args.length if args.length <= 1024 else min(args.sample_rows, args.length)
    rows = torch.linspace(0, args.length - 1, count).long().unique()
    reference = BASE.REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
    retained, traces, active_capture, device = [], [], None, None
    with (HERE / (args.label + ".jsonl")).open("x") as stream:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                args=vars(args),
                source_sha256=pinned,
                memory_budget=budget,
                input_sha256=input_hashes,
                sampled_query_rows=rows.tolist(),
                timing_scope="Combined real device Q/K/V quantization, optional V transpose and attention; host uploads/oracles excluded; host wall-clock blocking trace latency per invocation",
                clock_scope="No clock/power overrides or continuous telemetry; interleaving limits but does not eliminate thermal/frequency drift",
                order_policy="Pairs of rounds share a rotated order; second round is exact reverse; next pair rotates its start; all candidates once per round",
            )
        )
        try:
            device = ttnn.open_device(device_id=0, trace_region_size=budget["trace_region_bytes"])
            initial_memory = memory_view(device)
            # Trace reservation may already be reflected in free DRAM. The
            # independent 10% guard leaves headroom rather than double-counting it.
            assert (
                budget["tensor_bytes"] + budget["program_allocator_reserve_bytes"] < 0.90 * initial_memory["free_bytes"]
            ), "Insufficient free DRAM for all retained candidates; reduce --barriers"
            emit(dict(kind="initial_memory", **initial_memory))
            axis_hashes = {}
            for axis, barrier in choices:
                config = SimpleNamespace(
                    destination="fast_bf16",
                    denom_only=False,
                    kv_formats="b8_b4",
                    length=args.length,
                    heads=args.heads,
                    cores=args.cores,
                    check_preprocess=True,
                    grid7_exp=False,
                    v_transposed=axis == "N",
                    read_barrier_tiles=barrier,
                )
                built = BASE.build(device, config, inputs)
                originals, tensors, out, attention, preprocess, combined, kernel = built
                combined()
                actual = ttnn.to_torch(out).bfloat16()
                assert bool(torch.isfinite(actual).all())
                digest = BASE.tensor_hash(actual)
                assert axis not in axis_hashes or axis_hashes[axis] == digest, "Barrier changed arithmetic bits"
                axis_hashes[axis] = digest
                quiet = torch.arange(128) % 16 != 0
                result = dict(
                    id=f"{axis}-b{barrier}",
                    axis=axis,
                    barrier=barrier,
                    output_sha256=digest,
                    originals=originals,
                    tensors=tensors,
                    out=out,
                    attention=attention,
                    preprocess=preprocess,
                    combined=combined,
                    built=built,
                    samples_ms=[],
                    kernel=kernel,
                )
                retained.append(result)
                check_immutable(result, inputs)
                assert BASE.hashes(paths) == pinned, "Sources changed during qualification"
                emit(
                    dict(
                        kind="qualified",
                        id=result["id"],
                        axis=axis,
                        barrier=barrier,
                        kernel=kernel,
                        output_sha256=digest,
                        accuracy=BASE.REPRO.metrics(actual[..., rows, :], reference),
                        quiet_value_channel_accuracy=(
                            BASE.REPRO.metrics(actual[..., rows, :][..., quiet], reference[..., quiet])
                            if args.distribution == "channel_v"
                            else None
                        ),
                        all_output_finite=True,
                        exact_preprocessing=True,
                        original_inputs_immutable=True,
                        barrier_output_bits_equal=True,
                        memory=memory_view(device),
                    )
                )
                del actual

            # All candidates are qualified BEFORE any benchmark traces exist.
            for candidate in retained:
                trace = ttnn.begin_trace_capture(device, cq_id=0)
                traces.append(trace)
                active_capture = trace
                for _ in range(args.trace_repeats):
                    candidate["combined"]()
                ttnn.end_trace_capture(device, trace, cq_id=0)
                active_capture = None
                candidate["trace"] = trace
                for _ in range(2):
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    check_output(candidate)
            emit(dict(kind="trace_qualification", each_candidate_replay_bitwise_equal=True, traces=len(traces)))

            origin = time.perf_counter()
            for ordinal in range(args.warmup + args.iters):
                order = round_order(len(retained), ordinal)
                measured = ordinal >= args.warmup
                emit(
                    dict(
                        kind="round_order",
                        ordinal=ordinal,
                        phase="measure" if measured else "warmup",
                        candidates=[retained[i]["id"] for i in order],
                    )
                )
                round_samples = []
                for position, index in enumerate(order):
                    candidate = retained[index]
                    start = time.perf_counter()
                    ttnn.execute_trace(device, candidate["trace"], cq_id=0, blocking=True)
                    elapsed = 1000 * (time.perf_counter() - start) / args.trace_repeats
                    if measured:
                        candidate["samples_ms"].append(elapsed)
                    round_samples.append(
                        dict(
                            id=candidate["id"],
                            position=position,
                            start_since_timing_origin_s=start - origin,
                            combined_ms=elapsed,
                        )
                    )
                # No disk logging/telemetry between candidates in a round.
                emit(dict(kind="round_samples", ordinal=ordinal, measured=measured, samples=round_samples))

            for candidate in retained:
                digest = check_output(candidate)
                assert digest == axis_hashes[candidate["axis"]]
                check_immutable(candidate, inputs)
            assert [BASE.tensor_hash(x) for x in inputs] == input_hashes
            assert BASE.hashes(paths) == pinned, "Sources changed during timing"
            emit(
                dict(
                    kind="summary",
                    sources_unchanged=True,
                    each_candidate_replay_bitwise_equal=True,
                    barrier_output_bits_equal=True,
                    original_inputs_immutable=True,
                    candidates=[
                        dict(
                            id=c["id"],
                            axis=c["axis"],
                            barrier=c["barrier"],
                            samples_ms=c["samples_ms"],
                            median_combined_ms=statistics.median(c["samples_ms"]),
                            output_sha256=c["output_sha256"],
                        )
                        for c in retained
                    ],
                    final_memory=memory_view(device),
                )
            )
        except BaseException as error:
            emit(dict(kind="failure", error=repr(error), qualified_candidates=len(retained)))
            raise
        finally:
            if device is not None:
                try:
                    cleanup_errors = release_captures(device, traces, active_capture, emit)
                finally:
                    ttnn.close_device(device)
        assert not cleanup_errors, "Trace cleanup had errors; inspect ledger"
        emit(
            dict(
                kind="complete",
                sources_unchanged=BASE.hashes(paths) == pinned,
                retained_candidate_count=len(retained),
                all_traces_released=True,
            )
        )


if __name__ == "__main__":
    main()
