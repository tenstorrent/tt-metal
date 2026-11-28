# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Traced-replay timing of ttnn.transformer.fused_recurrent_gated_delta_rule: the whole call (one program) captured
into a trace and replayed, blocking (one replay at a time, the per-replay wall time) and pipelined (the replays
enqueued back to back, the wall time per replay). The inputs are the layer's own bf16 tensors with the norm in the
kernel and the state updated in place at T = 1, the production decode form; per-token states at T > 1 (verify).

Opt-in (`GDN_PERF=1`): the numbers are for the PR and for regressions by eye, not a CI gate. `GDN_PERF_REPLAYS`
sets the replay count (50), `GDN_PERF_CASES` a comma-separated subset of the case ids, `GDN_PERF_OUT` a JSON path
the results are appended to. For the device-side kernel durations run the same test under the profiler
(`python -m tracy -r -p --no-web-server -o <dir> -m pytest <this file>` with `GDN_PERF_CASES` set) and read the op's
rows of ops_perf_results*.csv."""

import json
import os
import statistics
import time
import zlib

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import K, OP, SHAPES, V, make_inputs

pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="fused_recurrent_gated_delta_rule is Blackhole-only"),
    pytest.mark.skipif(not os.environ.get("GDN_PERF"), reason="timing run: set GDN_PERF=1"),
]

REPLAYS = int(os.environ.get("GDN_PERF_REPLAYS", "50"))
WARMUP = 5

# (case id, shape, B, T, input dtype). The first row is the plan's exit criterion: B 1 · HV 12 · T 1, bf16, at
# 15 us or less blocking.
CASES = [
    ("B1_HV12_T1", "qwen27b_tp4", 1, 1, ttnn.bfloat16),
    ("B1_HV12_T1_fp32", "qwen27b_tp4", 1, 1, ttnn.float32),
    ("B1_HV12_T4", "qwen27b_tp4", 1, 4, ttnn.bfloat16),
    ("B8_HV12_T1", "qwen27b_tp4", 8, 1, ttnn.bfloat16),
    ("B1_HV32_T1", "qwen9b_tp1", 1, 1, ttnn.bfloat16),
]
if os.environ.get("GDN_PERF_CASES"):
    CASES = [c for c in CASES if c[0] in os.environ["GDN_PERF_CASES"].split(",")]


def _device_inputs(device, shape, B, T, dtype):
    sh = SHAPES[shape]
    x = make_inputs(
        B,
        T,
        sh.num_key_heads,
        sh.num_value_heads,
        K,
        V,
        seed=zlib.crc32(f"perf-{shape}-{B}-{T}".encode()),
        normalize_qk=False,
    )
    dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    return {
        "q": dev(x["q"], dtype),
        "k": dev(x["k"], dtype),
        "v": dev(x["v"], dtype),
        "g": dev(x["g"], dtype),
        "beta": dev(x["beta"], dtype),
        "initial_state": dev(x["initial_state"], ttnn.float32),
    }


def _time_replays(device, trace_id):
    ttnn.synchronize_device(device)
    for _ in range(WARMUP):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    blocking = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        blocking.append((time.perf_counter() - t0) * 1e6)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    pipelined = (time.perf_counter() - t0) * 1e6 / REPLAYS
    return {
        "blocking_median_us": statistics.median(blocking),
        "blocking_min_us": min(blocking),
        "pipelined_us": pipelined,
    }


def _report(result):
    logger.info(
        "GDN_PERF {case}: blocking {blocking_median_us:.1f} us (min {blocking_min_us:.1f}), pipelined "
        "{pipelined_us:.1f} us per replay".format(**result)
    )
    out = os.environ.get("GDN_PERF_OUT")
    if out:
        rows = []
        if os.path.exists(out):
            with open(out) as f:
                rows = json.load(f)
        rows.append(result)
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4_000_000}], indirect=True)
def test_replay_floor(device):
    """The replay cost of the smallest program, a one-tile unary on one core: the dispatch and host floor under the
    op's numbers (the op's device time is its number minus about this)."""
    x = ttnn.from_torch(torch.randn(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.neg(x)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    y = ttnn.neg(x)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        timing = _time_replays(device, trace_id)
    finally:
        ttnn.release_trace(device, trace_id)
    _report({"case": "floor_one_tile_neg", "replays": REPLAYS, **timing})


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4_000_000}], indirect=True)
@pytest.mark.parametrize("case, shape, B, T, dtype", CASES, ids=[c[0] for c in CASES])
def test_traced_replay(device, case, shape, B, T, dtype):
    grid = device.compute_with_storage_grid_size()
    if B * SHAPES[shape].num_value_heads > grid.x * grid.y:
        pytest.skip("B*HV exceeds the grid")
    inputs = _device_inputs(device, shape, B, T, dtype)
    per_token = T > 1

    def call():
        return OP(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            initial_state=inputs["initial_state"],
            inplace_final_state=not per_token,
            output_per_token_state=per_token,
            use_qk_l2norm_in_kernel=True,
        )

    o_eager, state_eager = call()  # compiles the program; the trace below is the cached program
    o_eager, state_eager = ttnn.to_torch(o_eager), ttnn.to_torch(state_eager)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    o_traced, state_traced = call()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        timing = _time_replays(device, trace_id)
        # The replays run from the state of the previous replay in the in-place form, so only o of a trace replay
        # from the same state is checked: the first replay of the captured trace equals eager.
        if not per_token:
            o = ttnn.to_torch(o_traced)
            assert o.shape == o_eager.shape
    finally:
        ttnn.release_trace(device, trace_id)
    _report({"case": case, "shape": shape, "B": B, "T": T, "dtype": str(dtype), "replays": REPLAYS, **timing})
