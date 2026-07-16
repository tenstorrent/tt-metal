# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure trace-allocation tracker overhead when all tracker toggles are off.

For a release comparison, run this script repeatedly on the tracker commit and
on its pre-tracker base commit, alternating the order on the same machine. The
script intentionally targets the three hot paths touched by the tracker:

* cached TTNN operation dispatch;
* device buffer allocation/deallocation;
* trace replay.

Each result is written as JSON so samples from multiple fresh processes can be
combined for an equivalence analysis.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

TRACKER_ENV_VARS = (
    "TT_METAL_TRACE_ALLOC_TRACKING",
    "TT_METAL_TRACE_ALLOC_TRACEBACKS",
    "TT_METAL_TRACE_ALLOC_REFERRER_DEPTH",
)


def measure_batch(fn, *, iterations, samples, synchronize=None):
    for _ in range(3):
        for _ in range(iterations):
            fn()
        if synchronize is not None:
            synchronize()

    timings_ns = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            if synchronize is not None:
                synchronize()
            start_ns = time.perf_counter_ns()
            for _ in range(iterations):
                fn()
            if synchronize is not None:
                synchronize()
            timings_ns.append((time.perf_counter_ns() - start_ns) / iterations)
    finally:
        if gc_enabled:
            gc.enable()

    return {
        "iterations_per_sample": iterations,
        "samples_ns_per_iteration": timings_ns,
        "median_ns_per_iteration": statistics.median(timings_ns),
        "mean_ns_per_iteration": statistics.fmean(timings_ns),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--dispatch-iterations", type=int, default=2_000)
    parser.add_argument("--allocation-iterations", type=int, default=5_000)
    parser.add_argument("--trace-iterations", type=int, default=5_000)
    return parser.parse_args()


def main():
    args = parse_args()
    enabled = {name: os.environ.get(name) for name in TRACKER_ENV_VARS if os.environ.get(name) is not None}
    if any(enabled.get(name) == "1" for name in TRACKER_ENV_VARS):
        raise RuntimeError(f"Tracker benchmark requires all tracker toggles disabled; got {enabled}")

    # Import only after validating the environment because tracker settings are
    # deliberately captured during module/shared-library initialization.
    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=1_000_000)
    trace_id = None
    tensors = []
    try:
        shape = ttnn.Shape([1, 1, 32, 32])

        def allocate():
            tensor = ttnn.allocate_tensor_on_device(
                shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(tensor, force=True)

        input_a = ttnn.allocate_tensor_on_device(
            shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        input_b = ttnn.allocate_tensor_on_device(
            shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        output = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        tensors.extend((input_a, input_b, output))

        def dispatch():
            ttnn.add(input_a, input_b, output_tensor=output)

        # Compile the operation before either dispatch or trace timing.
        dispatch()
        ttnn.synchronize_device(device)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        dispatch()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        def replay_trace():
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

        synchronize = lambda: ttnn.synchronize_device(device)
        results = {
            "operation_dispatch": measure_batch(
                dispatch,
                iterations=args.dispatch_iterations,
                samples=args.samples,
                synchronize=synchronize,
            ),
            "buffer_allocate_deallocate": measure_batch(
                allocate,
                iterations=args.allocation_iterations,
                samples=args.samples,
            ),
            "trace_replay": measure_batch(
                replay_trace,
                iterations=args.trace_iterations,
                samples=args.samples,
                synchronize=synchronize,
            ),
        }

        direct_binding = getattr(ttnn, "_ttnn_execute_trace", ttnn.execute_trace) is ttnn.execute_trace
        payload = {
            "label": args.label,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
            ).stdout.strip(),
            "tracker_environment": enabled,
            "execute_trace_is_direct_binding": direct_binding,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        for tensor in tensors:
            if tensor.is_allocated():
                ttnn.deallocate(tensor, force=True)
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
