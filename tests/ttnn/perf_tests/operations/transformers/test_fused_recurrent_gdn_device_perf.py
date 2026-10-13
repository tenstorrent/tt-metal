# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time baseline and gate for ttnn.transformer.fused_recurrent_gated_delta_rule.

test_run_fused_recurrent_gdn (the target) runs each served decode shape between Tracy signposts: one warm-up call,
then CALLS calls. test_device_perf_fused_recurrent_gdn (the gate, collected by the device-perf marker) runs the target
under the device profiler and checks, per shape, the recurrence program's device kernel time (median of launches 2..CALLS),
the device kernel time of the whole call (every device op the wrapper issues, same median) and the device-op count per
call against the values stored in EXPECTED. Both bounds are enforced: a faster result also fails until the stored
values are updated, so a PR that moves the numbers states them.

test_traced_replay_time (not collected by the marker) reports the host-side cost of a traced replay of the call; it is
logged, not gated."""

import re
import zlib
import time
from dataclasses import dataclass

import pandas as pd
import pytest
from loguru import logger
from tracy import signpost
from tracy.common import clear_profiler_runtime_artifacts
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler
from tt_py_test_utils_common.perf.device_perf_utils import check_device_perf, prep_device_perf_report

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import SHAPES, make_inputs

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
OP_NAME = "FusedRecurrentGatedDeltaRuleDeviceOperation"  # the recurrence program's OP CODE in the ops report
K = V = 128
CALLS = 6  # calls between the signposts; launch 1 is dropped, the median of 2..CALLS is the measurement
MARGIN = 0.05  # applied to both bounds of the two time gates
TARGET = "tests/ttnn/perf_tests/operations/transformers/test_fused_recurrent_gdn_device_perf.py::test_run_fused_recurrent_gdn"


@dataclass(frozen=True)
class Row:
    shape: str  # key of SHAPES (per-device head counts)
    B: int
    T: int


# The served decode shapes: single-token decode at the served head counts and batch widths, and the verify path at
# HV 12 (T = 1 + draft tokens). b32_hv12_t1 is the batch-32 serving target; it does not fit the current op (one core
# per (b, hv)) and is skipped until it does, so the table keeps its shape.
ROWS = {
    "b1_hv12_t1": Row("qwen27b_tp4", 1, 1),
    "b8_hv12_t1": Row("qwen27b_tp4", 8, 1),
    "b32_hv12_t1": Row("qwen27b_tp4", 32, 1),
    "b1_hv6_t1": Row("qwen27b_tp8", 1, 1),
    "b1_hv32_t1": Row("qwen9b_tp1", 1, 1),
    "b1_hv24_t1": Row("kimi_k3_sp8tp4", 1, 1),
    "b1_hv12_t4": Row("qwen27b_tp4", 1, 4),
    "b1_hv12_t8": Row("qwen27b_tp4", 1, 8),
    "b1_hv12_t12": Row("qwen27b_tp4", 1, 12),
}


@dataclass(frozen=True)
class Expected:
    program_us: float  # recurrence program, device kernel duration
    chain_us: float  # every device op of one call, device kernel durations summed
    ops_per_call: int


# Blackhole, Release build with the profiler. Taken on a p300c (bh-qbge-15) on 2026-10-13 with the op as on main
# 3f98c04b71f; the first run on the CI runner class may move them within the margin or beyond, and then sets them.
EXPECTED_BLACKHOLE = {
    "b1_hv12_t1": Expected(program_us=25.61, chain_us=70.67, ops_per_call=17),
    "b8_hv12_t1": Expected(program_us=53.37, chain_us=149.40, ops_per_call=19),
    "b1_hv6_t1": Expected(program_us=19.56, chain_us=61.77, ops_per_call=17),
    "b1_hv32_t1": Expected(program_us=29.17, chain_us=85.70, ops_per_call=17),
    "b1_hv24_t1": Expected(program_us=26.40, chain_us=74.49, ops_per_call=15),
    "b1_hv12_t4": Expected(program_us=55.62, chain_us=164.69, ops_per_call=25),
    "b1_hv12_t8": Expected(program_us=96.02, chain_us=223.04, ops_per_call=25),
    "b1_hv12_t12": Expected(program_us=135.77, chain_us=305.44, ops_per_call=25),
}


def _to_device(device, t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _fits(device, row):
    sh = SHAPES[row.shape]
    grid = device.compute_with_storage_grid_size()
    return row.B * sh.num_value_heads <= grid.x * grid.y


def _stage(device, row, seed):
    sh = SHAPES[row.shape]
    x = make_inputs(row.B, row.T, sh.num_key_heads, sh.num_value_heads, K, V, seed=seed)
    return {name: _to_device(device, x[name]) for name in ("q", "k", "v", "g", "beta", "initial_state")}


def _call(inputs, row):
    per_token = row.T > 1
    return OP(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=not per_token,
        output_per_token_state=per_token,
    )


def test_run_fused_recurrent_gdn(device):
    """The profiled target: every runnable row, one warm-up call, then CALLS calls between its signposts."""
    if not is_blackhole():
        pytest.skip("fused_recurrent_gated_delta_rule is Blackhole-only")
    for name, row in ROWS.items():
        if not _fits(device, row):
            logger.info(f"{name}: not runnable today (B * HV exceeds the grid, one core per (b, hv)); skipped")
            continue
        inputs = _stage(device, row, seed=zlib.crc32(name.encode()))
        o, state = _call(inputs, row)  # warm-up: JIT and program cache
        ttnn.deallocate(o)
        ttnn.deallocate(state)
        ttnn.synchronize_device(device)
        signpost(f"{name}-start")
        for _ in range(CALLS):
            o, state = _call(inputs, row)
            ttnn.deallocate(o)
            ttnn.deallocate(state)
        ttnn.synchronize_device(device)
        signpost(f"{name}-end")
        for t in inputs.values():
            ttnn.deallocate(t)


def _ops_between_signposts(csv_path):
    """{row name: [(op code, device kernel duration in ns), ...]} for the device ops between each signpost pair."""
    df = pd.read_csv(csv_path)
    regions, current = {}, None
    for _, r in df.iterrows():
        if r["OP TYPE"] == "signpost":
            code = r["OP CODE"]
            if code.endswith("-start"):
                current = code[: -len("-start")]
                regions[current] = []
            elif code.endswith("-end"):
                current = None
        elif current is not None and pd.notna(r["DEVICE KERNEL DURATION [ns]"]):
            regions[current].append((r["OP CODE"], float(r["DEVICE KERNEL DURATION [ns]"])))
    return regions


def _is_recurrence(op_code):
    return bool(re.search(rf"\b{re.escape(OP_NAME)}(?:<|$|\W)", str(op_code)))


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _summarize(ops):
    """program_us, chain_us (medians over calls 2..CALLS) and ops_per_call from one row's region."""
    assert len(ops) % CALLS == 0, f"{len(ops)} device ops between the signposts is not a multiple of {CALLS} calls"
    per_call = len(ops) // CALLS
    program, chain = [], []
    for c in range(CALLS):
        call = ops[c * per_call : (c + 1) * per_call]
        rec = [d for code, d in call if _is_recurrence(code)]
        assert len(rec) == 1, f"call {c + 1}: {len(rec)} recurrence programs among {[code for code, _ in call]}"
        program.append(rec[0])
        chain.append(sum(d for _, d in call))
    return {
        "program_us": _median(program[1:]) / 1000.0,
        "chain_us": _median(chain[1:]) / 1000.0,
        "ops_per_call": float(per_call),
    }


@pytest.mark.models_device_performance_bare_metal
def test_device_perf_fused_recurrent_gdn():
    """Runs the target under the device profiler and gates every row against EXPECTED."""
    if not is_blackhole():
        pytest.skip("fused_recurrent_gated_delta_rule is Blackhole-only")
    expected_rows = EXPECTED_BLACKHOLE
    subdir = "fused_recurrent_gdn_perf"
    clear_profiler_runtime_artifacts()
    run_device_profiler(f"pytest {TARGET}", subdir, device_analysis_types=["device_kernel_duration"])
    csv_path = get_latest_ops_log_filename(subdir)
    logger.info(f"ops report: {csv_path}")
    regions = _ops_between_signposts(csv_path)

    failures = []
    for name, expected in expected_rows.items():
        if name not in regions:
            failures.append(f"{name}: no signposted region in the ops report")
            continue
        measured = _summarize(regions[name])
        logger.info(
            f"{name}: program {measured['program_us']:.2f} us (expected {expected.program_us}), chain "
            f"{measured['chain_us']:.2f} us (expected {expected.chain_us}), {int(measured['ops_per_call'])} device ops "
            f"per call (expected {expected.ops_per_call})"
        )
        bounds = check_device_perf(
            measured, MARGIN, {"program_us": expected.program_us, "chain_us": expected.chain_us}, assert_on_fail=False
        )
        for col in ("program_us", "chain_us"):
            lo, hi = bounds[f"Lower Threshold {col}"], bounds[f"Upper Threshold {col}"]
            if not lo <= measured[col] <= hi:
                failures.append(f"{name}: {col} {measured[col]:.2f} outside [{lo:.2f}, {hi:.2f}]")
        if int(measured["ops_per_call"]) != expected.ops_per_call:
            failures.append(
                f"{name}: {int(measured['ops_per_call'])} device ops per call, expected {expected.ops_per_call}"
            )
        prep_device_perf_report("fused_recurrent_gdn", ROWS[name].B, measured, bounds, comments=name)
    if failures:
        pytest.fail(
            "device time outside the stored bounds (update EXPECTED when the change is intended):\n"
            + "\n".join(failures)
        )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 2000000}], indirect=True)
@pytest.mark.parametrize("name", ["b1_hv12_t1", "b8_hv12_t1", "b1_hv12_t4"])
def test_traced_replay_time(device, name):
    """Host wall time per traced replay of the whole call: 100 replays then one synchronize, and the median of 20
    blocking replays. Logged, not gated."""
    if not is_blackhole():
        pytest.skip("fused_recurrent_gated_delta_rule is Blackhole-only")
    row = ROWS[name]
    inputs = _stage(device, row, seed=zlib.crc32(name.encode()))
    o, state = _call(inputs, row)
    ttnn.deallocate(o)
    ttnn.deallocate(state)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    o, state = _call(inputs, row)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for _ in range(5):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(100):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        pipelined_us = (time.perf_counter() - t0) / 100 * 1e6
        blocking = []
        for _ in range(20):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            blocking.append((time.perf_counter() - t0) * 1e6)
        logger.info(f"{name}: traced replay {pipelined_us:.1f} us pipelined, {_median(blocking):.1f} us blocking")
    finally:
        ttnn.release_trace(device, trace_id)
