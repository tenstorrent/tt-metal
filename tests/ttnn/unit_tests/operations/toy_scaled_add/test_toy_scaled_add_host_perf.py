# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-time measurements for toy_scaled_add: the Python generic_op version against the C++ device
operation.

Both versions dispatch the same kernels with the same per-launch payload, so device time is the same
(test_toy_scaled_add_device_perf.py) and every difference measured here is host code.

Dispatch time, cold (program-cache miss) and hot (hit):
  * the program cache stays enabled; a cold sample clears it first (compiled kernels stay warm on
    disk), a hot sample primes it with one untimed call. Each sample's cache counter is checked: a
    cold call must add one entry, a hot call none;
  * a sample times one public call from a synchronized device until it returns;
  * blocks alternate which route goes first, so neither always runs in the other's wake;
  * route `generic_op_prebuilt` dispatches the generic version's descriptor built once, ahead of
    time, into one preallocated output. It is not an op anyone calls; it splits the generic
    version's cost into building the descriptor in Python and the generic_op path itself. Its
    counterpart is `native_preallocated`, the C++ op writing into that same output: the other two
    routes allocate an output on every call, and on a sharded output that allocation is most of
    a hit's cost.

Back to back: calls issued one after another, as a model loop does, with one synchronize at the
end. `enqueue` is the host's cost per call; `end_to_end` is the time per call until the device is
done, which is the host's cost when the route is host-bound and the device's when it is
device-bound. These are the steadier numbers: a single call's dispatch time also depends on how
long the host sat idle before it (the same for every route), which spreads it over two modes.

Host timing varies between processes, so run the file several times, each round in a fresh process
(TOY_SCALED_ADD_PERF_ROUND=<n>); results go to generated/toy_scaled_add/host_perf_round_<n>.json.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/toy_scaled_add/test_toy_scaled_add_host_perf.py
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.operations.toy_scaled_add import toy_scaled_add_generic
from ttnn.operations.toy_scaled_add.toy_scaled_add_program_descriptor import (
    create_height_sharded_descriptor,
    create_interleaved_descriptor,
)

BLOCKS = 30
HOT_PER_BLOCK = 5
THROUGHPUT_CALLS = 20
THROUGHPUT_ROUNDS = 20
RESULTS = Path(f"generated/toy_scaled_add/host_perf_round_{os.environ.get('TOY_SCALED_ADD_PERF_ROUND', '0')}.json")


def _tensor(device, shape, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _height_sharded(device, rows, width, num_cores):
    grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    shard_rows = -(-rows // num_cores)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_rows * 32, width], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _timed(device, call):
    """One sample: (dispatch_ns, cache entries added). The output is released outside the timing."""
    ttnn.synchronize_device(device)
    before = device.num_program_cache_entries()
    t0 = time.perf_counter_ns()
    out = call()
    t1 = time.perf_counter_ns()
    ttnn.synchronize_device(device)
    del out
    return t1 - t0, device.num_program_cache_entries() - before


def _reset_cache(device):
    ttnn.synchronize_device(device)
    device.clear_program_cache()
    device.enable_program_cache()


def _measure(device, calls):
    """Cold and hot samples for every route in `calls`, alternating route order across blocks."""
    samples = {route: {"cold": [], "hot": []} for route in calls}
    for call in calls.values():  # compile the kernels once; no sample includes a JIT build
        _reset_cache(device)
        call()
    order = list(calls)
    for block in range(BLOCKS):
        for route in order if block % 2 == 0 else reversed(order):
            call = calls[route]
            _reset_cache(device)
            dispatch, added = _timed(device, call)
            assert added == 1, f"{route}: a cold call must add one program-cache entry, added {added}"
            samples[route]["cold"].append(dispatch)
            hot = []
            for _ in range(HOT_PER_BLOCK):
                dispatch, added = _timed(device, call)
                assert added == 0, f"{route}: a hot call must reuse the cached program, added {added}"
                hot.append(dispatch)
            samples[route]["hot"].append(statistics.median(hot))
    return samples


def _throughput(device, calls):
    """Per-call host enqueue and end-to-end time of THROUGHPUT_CALLS calls issued back to back."""
    out = {}
    for route, call in calls.items():
        enqueue, total = [], []
        for _ in range(THROUGHPUT_ROUNDS):
            ttnn.synchronize_device(device)
            outputs = []
            t0 = time.perf_counter_ns()
            for _ in range(THROUGHPUT_CALLS):
                outputs.append(call())
            t1 = time.perf_counter_ns()
            ttnn.synchronize_device(device)
            t2 = time.perf_counter_ns()
            enqueue.append((t1 - t0) / THROUGHPUT_CALLS)
            total.append((t2 - t0) / THROUGHPUT_CALLS)
            del outputs
        out[route] = {
            "enqueue_us": statistics.median(enqueue) / 1e3,
            "end_to_end_us": statistics.median(total) / 1e3,
        }
    return out


def _summary(samples):
    return {
        route: {
            state: {
                "dispatch_us": statistics.median(values) / 1e3,
                "dispatch_p90_us": statistics.quantiles(values, n=10)[-1] / 1e3,
                "blocks": len(values),
            }
            for state, values in states.items()
        }
        for route, states in samples.items()
    }


def _record(case_id, summary, throughput=None):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[case_id] = {"dispatch": summary, **({"throughput": throughput} if throughput else {})}
    RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\n{case_id}")
    print(f"  {'route':20} {'state':5} {'dispatch us':>12} {'p90 us':>8}")
    for route, states in summary.items():
        for state, s in states.items():
            print(f"  {route:20} {state:5} {s['dispatch_us']:12.1f} {s['dispatch_p90_us']:8.1f}")
    for route, t in (throughput or {}).items():
        print(
            f"  {route:20} back-to-back: enqueue {t['enqueue_us']:.1f} us/call, end to end {t['end_to_end_us']:.1f} us/call"
        )


CASES = {
    # case id: (shape, gamma, placement, sharded core count)
    "one_tile": ([1, 1, 32, 32], False, "interleaved", None),
    "grid_64_cores": ([1, 1, 32 * 256, 256], False, "interleaved", None),
    "grid_64_cores_gamma": ([1, 1, 32 * 256, 256], True, "interleaved", None),
    "height_sharded_64_cores_gamma": ([1, 1, 32 * 64, 256], True, "height_sharded", 64),
}


@pytest.mark.parametrize("case_id", list(CASES))
def test_host_time_generic_vs_native(device, case_id):
    shape, with_gamma, placement, num_cores = CASES[case_id]
    rows = shape[-2] // 32
    memory_config = (
        _height_sharded(device, rows, shape[-1], num_cores)
        if placement == "height_sharded"
        else ttnn.DRAM_MEMORY_CONFIG
    )
    a = _tensor(device, shape, memory_config)
    b = _tensor(device, shape, memory_config)
    gamma = _tensor(device, [1, 1, 1, shape[-1]]) if with_gamma else None
    prebuilt_out = ttnn.allocate_tensor_on_device(a.shape, a.dtype, ttnn.TILE_LAYOUT, device, memory_config)
    create = create_height_sharded_descriptor if placement == "height_sharded" else create_interleaved_descriptor
    prebuilt = create(a, b, gamma, prebuilt_out, 0.5, None)
    io_tensors = [a, b] + ([gamma] if with_gamma else []) + [prebuilt_out]
    calls = {
        "generic": lambda: toy_scaled_add_generic(a, b, alpha=0.5, gamma=gamma),
        "generic_op_prebuilt": lambda: ttnn.generic_op(io_tensors, prebuilt),
        "native": lambda: ttnn.toy_scaled_add(a, b, alpha=0.5, gamma=gamma),
        "native_preallocated": lambda: ttnn.toy_scaled_add(a, b, alpha=0.5, gamma=gamma, output_tensor=prebuilt_out),
    }

    summary = _summary(_measure(device, calls))
    throughput = _throughput(device, {route: calls[route] for route in ("generic", "native")})
    _record(case_id, summary, throughput)

    native, generic = summary["native"], summary["generic"]
    assert native["hot"]["dispatch_us"] < native["cold"]["dispatch_us"], "a cache hit must cost less than a miss"
    assert native["hot"]["dispatch_us"] < generic["hot"]["dispatch_us"], "the C++ hit path must beat the Python one"
    assert throughput["native"]["end_to_end_us"] <= throughput["generic"]["end_to_end_us"]


def test_hit_cost_does_not_grow_with_core_count(device):
    """One core against the whole grid: the cache-hit path writes a fixed number of common runtime
    args, so its host time stays flat however many cores the work spreads over."""
    grid = device.compute_with_storage_grid_size()
    full_grid = grid.x * grid.y
    width = 256
    calls, inputs = {}, []  # inputs keeps every case's tensors alive for the whole measurement
    for label, rows in (("1_core", 1), (f"{full_grid}_cores", full_grid)):
        a = _tensor(device, [1, 1, 32 * rows, width])
        b = _tensor(device, [1, 1, 32 * rows, width])
        inputs.append((a, b))
        calls[label] = lambda a=a, b=b: ttnn.toy_scaled_add(a, b, alpha=0.5)

    summary = _summary(_measure(device, calls))
    _record("native_hit_vs_core_count", summary)

    one, full = summary["1_core"]["hot"]["dispatch_us"], summary[f"{full_grid}_cores"]["hot"]["dispatch_us"]
    assert full < 2.0 * one, f"cache-hit dispatch grew from {one:.1f} us on 1 core to {full:.1f} us on {full_grid}"
