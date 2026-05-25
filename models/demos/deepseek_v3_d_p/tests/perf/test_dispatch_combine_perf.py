# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device performance tests for DeepSeek V3 MoE dispatch and combine operations.

Measures device kernel duration for dispatch and combine separately on 8-chip
linear and ring topologies (2 links, 7K payload). Each perf case spawns a
worker pytest at a parametrize id whose config ends in `perf_no_pcc`; those
workers skip PCC validation and run the op under test with the same
production-format inputs it would see in the end-to-end MoE pipeline.
`run_model_device_perf_test_with_merge` merges the per-device rows
in the Tracy CSV into per-op totals; `op_filter` isolates the combine op
when the worker also emits a preceding dispatch.
"""

import math
from collections import defaultdict

import pandas as pd
import pytest
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


def merge_device_rows(df):
    """
    Merge multi-device operation rows into single rows.

    For collective operations (AllGather, ReduceScatter, AllReduce, Matmul_RS):
      Uses AVERAGE duration across devices (synchronized operations)

    For non-collective operations:
      Uses MAX duration across devices (critical path bottleneck)

    Args:
        df: pandas DataFrame with profiler data. The DEVICE KERNEL DURATION
            column may arrive as object dtype if Tracy wrote non-numeric
            placeholders; it is coerced to float here so downstream NaN
            handling works regardless.

    Returns:
        DataFrame with merged rows.

    Raises:
        pytest.fail: if any device's op sequence diverges from the others,
            since the merged totals would otherwise silently compare
            apples-to-oranges.
    """
    duration_col = "DEVICE KERNEL DURATION [ns]"
    df = df.copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")

    block_by_device = defaultdict(list)

    for _, row in df.iterrows():
        op_name = row["OP CODE"]
        op_type = row["OP TYPE"]
        if op_type == "tt_dnn_device":
            device_id = int(row["DEVICE ID"])
            block_by_device[device_id].append((op_name, row.to_dict()))

    device_ids = sorted(block_by_device.keys())
    merged_blocks = []

    while device_ids and max(len(block_by_device[device_id]) for device_id in device_ids) > 0:
        blocks = []
        op_name = None

        for device_id in device_ids:
            if len(block_by_device[device_id]) > 0:
                current_op_name, current_block = block_by_device[device_id].pop(0)
                if op_name is None:
                    op_name = current_op_name
                elif op_name != current_op_name:
                    pytest.fail(
                        f"Mismatched ops across devices at merge index: "
                        f"device {device_id} has {current_op_name!r}, expected {op_name!r}"
                    )
                blocks.append((device_id, current_block))
            else:
                pytest.fail(f"Device {device_id} is missing an op during merge (truncated trace?)")

        if not blocks:
            continue

        is_collective = (
            "AllGather" in op_name or "ReduceScatter" in op_name or "AllReduce" in op_name or "Matmul_RS" in op_name
        )

        if is_collective:
            device_kernel_durations = [
                d[duration_col] for _, d in blocks if duration_col in d and not math.isnan(d[duration_col])
            ]
            average_duration = (
                sum(device_kernel_durations) / len(device_kernel_durations) if device_kernel_durations else float("nan")
            )
            base_block = blocks[0][1].copy()
            base_block[duration_col] = average_duration
            merged_blocks.append(base_block)
        else:
            max_duration_block = max(blocks, key=lambda x: x[1].get(duration_col, 0))
            merged_blocks.append(max_duration_block[1])

    return pd.DataFrame(merged_blocks)


def run_model_device_perf_test_with_merge(
    command: str,
    expected_device_perf_ns_per_iteration: float,
    subdir: str,
    model_name: str,
    num_iterations: int = 1,
    batch_size: int = 1,
    margin: float = 0.015,
    comments: str = "",
    op_filter: str = "",
    extra_env: dict | None = None,
):
    """
    Run device performance test with multi-device row merging.

    Extends `run_model_device_perf_test` with multi-chip row merging:
    CCL ops are averaged across devices, non-CCL ops use the
    slowest device's duration (critical path).

    `op_filter`, if set, restricts the measurement to rows whose OP CODE
    contains the given substring — useful when the worker pytest runs more
    than one op and only one of them is under test.

    `extra_env`, if set, is applied to `os.environ` for the duration of the
    subprocess invocation. Use this for vars the worker reads directly
    (e.g. TT_DS_CAPTURED_LAYER) — prefixing them into `command` doesn't work
    because tracy's `-m` flag mis-parses leading KEY=VAL tokens as module
    names.

    Only `num_iterations=1` is supported today. `run_device_perf` can run
    multiple iterations and average them, but this helper rereads only the
    latest ops CSV and rewrites the averaged metric from that single file —
    mixing that with a multi-iteration average would produce inconsistent
    numbers. Extending to N>1 requires merging across per-iteration CSVs.
    """
    if num_iterations != 1:
        pytest.fail(
            f"run_model_device_perf_test_with_merge currently supports num_iterations=1 only "
            f"(got {num_iterations}); per-iteration CSV merging is not implemented."
        )

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"

    import os

    saved_env = {k: os.environ.get(k) for k in (extra_env or {})}
    try:
        if extra_env:
            os.environ.update(extra_env)
        post_processed_results = run_device_perf(
            command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
        )
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    total_rows = len(df)
    device_rows_before_filter = len(df[df["OP TYPE"] == "tt_dnn_device"])

    logger.info(f"CSV total rows: {total_rows}")
    logger.info(f"Device operation rows: {device_rows_before_filter}")

    df = df[df["OP TYPE"] == "tt_dnn_device"]

    if op_filter:
        df = df[df["OP CODE"].str.contains(op_filter, na=False)]
        if df.empty:
            pytest.fail(f"op_filter={op_filter!r} matched no rows in {filename}")
        logger.info(f"Rows after op_filter={op_filter!r}: {len(df)}")

    logger.info(f"Device rows before merge: {len(df)}")
    df_merged = merge_device_rows(df)
    logger.info(f"Device rows after merge: {len(df_merged)}")

    if not df_merged.empty:
        merged_kernel_durations = df_merged["DEVICE KERNEL DURATION [ns]"].dropna().tolist()
        if merged_kernel_durations:
            merged_sum_ns = sum(merged_kernel_durations)
            logger.info(f"Merged operations count: {len(merged_kernel_durations)}")
            logger.info(f"Merged sum (ns): {merged_sum_ns} ({merged_sum_ns / 1000:.1f} μs)")
            logger.info(
                f"Original post_processed_results[{inference_time_key}]: "
                f"{post_processed_results.get(inference_time_key, 'N/A')}"
            )
            post_processed_results[inference_time_key] = merged_sum_ns

    expected_perf_cols = {inference_time_key: expected_device_perf_ns_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=margin, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )

    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=comments,
    )


def run_model_device_perf_test_per_op(
    command: str,
    expected_per_op: dict,
    subdir: str,
    model_name: str,
    margin: float = 0.015,
    comments: str = "",
    extra_env: dict | None = None,
):
    """Run one worker subprocess and assert performance for multiple ops independently.

    Use when a single worker invocation produces multiple device ops in the Tracy CSV
    (e.g. dispatch + combine in one forward pass) and each needs its own baseline so
    a regression points to the responsible kernel rather than the combined total.

    Args:
        expected_per_op: dict mapping OP CODE substring → expected duration in ns.
            For each entry, the merged-device-rows DataFrame is filtered by substring,
            durations summed, and asserted against expected ± margin. Every entry must
            match at least one row in the CSV; missing matches `pytest.fail`.
        margin: tolerance applied uniformly to all entries in `expected_per_op`.
    """
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"

    import os

    saved_env = {k: os.environ.get(k) for k in (extra_env or {})}
    try:
        if extra_env:
            os.environ.update(extra_env)
        run_device_perf(command, subdir=subdir, num_iterations=1, cols=cols, batch_size=1)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)
    df = df[df["OP TYPE"] == "tt_dnn_device"]
    df_merged = merge_device_rows(df)
    logger.info(f"[per-op] CSV={filename}  device rows={len(df)}  merged ops={len(df_merged)}")

    measured_per_op = {}
    failures = []
    for op_substring, expected_ns in expected_per_op.items():
        rows = df_merged[df_merged["OP CODE"].str.contains(op_substring, na=False)]
        if rows.empty:
            pytest.fail(
                f"No merged rows match op_substring={op_substring!r} in {filename}; "
                f"available op codes: {sorted(df_merged['OP CODE'].unique())}"
            )
        measured_ns = float(rows["DEVICE KERNEL DURATION [ns]"].sum())
        measured_per_op[op_substring] = measured_ns
        lo = (1 - margin) * expected_ns
        hi = (1 + margin) * expected_ns
        passing = lo <= measured_ns <= hi
        logger.info(
            f"[per-op] {op_substring}: measured={measured_ns:,.0f} ns  "
            f"expected={expected_ns:,.0f} ns  bounds=[{lo:,.0f}, {hi:,.0f}]  "
            f"{'PASS' if passing else 'FAIL'}"
        )
        if not passing:
            failures.append((op_substring, measured_ns, expected_ns, lo, hi))

    total_measured = sum(measured_per_op.values())
    post_processed_results = {inference_time_key: total_measured}
    for op_substring, measured_ns in measured_per_op.items():
        post_processed_results[f"{op_substring} DEVICE KERNEL DURATION [ns]"] = measured_ns

    prep_device_perf_report(
        model_name=model_name,
        batch_size=1,
        post_processed_results=post_processed_results,
        expected_results={},
        comments=comments,
    )

    if failures:
        msg = "Per-op perf checks failed:\n  " + "\n  ".join(
            f"{op}: measured={m:,.0f} ns  expected={e:,.0f} ns (bounds [{lo:,.0f}, {hi:,.0f}], margin ±{margin*100:.0f}%)"
            for op, m, e, lo, hi in failures
        )
        pytest.fail(msg)


def _perf_param(
    op,
    worker_file,
    worker_test,
    topo,
    nlinks,
    expected_ns,
    op_filter,
    margin=0.1,
    layout="tile",
    dtype_filter="",
    captured_layer: int | None = None,
    captured_col: int | None = None,
    worker_filter_extras: str | None = None,
):
    """Build one pytest.param tuple for the perf tests.

    When `captured_layer`+`captured_col` are set, the test sets TT_DS_CAPTURED_LAYER /
    TT_DS_CAPTURED_COL on `os.environ` before invoking the tracy/pytest subprocess
    (env vars are not prefixed into the command string — tracy's `-m` flag would
    mis-parse them as a module name). The worker then loads real captured Galaxy
    gate indices for that (layer, col) and the parametrize filter selects the
    `perf_real_indices` worker config instead of `perf_no_pcc`.

    `worker_filter_extras`, if set, replaces the default `"random and {layout}"`
    extras appended to the worker pytest `-k` filter. Use this when targeting a
    worker test whose parametrize matrix doesn't include the random/predictable
    or tile/row_major axes (e.g. `test_ttnn_dispatch_combine`).
    """
    worker_id = f"{topo}-8-{nlinks}link"
    model_name = f"deepseek_v3_{op}_{topo}_8_{nlinks}link"
    if layout != "tile":
        model_name += f"_{layout}"
    use_captured = captured_layer is not None and captured_col is not None
    parametrize_id = "perf_real_indices" if use_captured else "perf_no_pcc"
    if worker_filter_extras is None:
        worker_filter_extras = f"random and {layout}"
    k_filter = f"{parametrize_id} and {worker_id}"
    if worker_filter_extras:
        k_filter += f" and {worker_filter_extras}"
    if dtype_filter:
        k_filter += f" and {dtype_filter}"
    if use_captured:
        model_name += f"_real_l{captured_layer:02d}_col{captured_col}"
        extra_env = {"TT_DS_CAPTURED_LAYER": str(captured_layer), "TT_DS_CAPTURED_COL": str(captured_col)}
    else:
        extra_env = {}
    command = f"pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/{worker_file}::{worker_test} " f"-k '{k_filter}'"
    return (
        command,
        expected_ns,
        f"deepseek_v3_{op}",
        model_name,
        1,  # num_iterations
        1,  # batch_size
        margin,
        f"{topo}-8-{nlinks}link",
        op_filter,
        extra_env,
    )


# Baselines measured on BH LoudBox (bh-rb-01), seq_len=3200, emb=7168, experts=64, top-k=2.
# Dispatch worker emits only DispatchDeviceOperation -> op_filter is empty.
# Combine worker runs real dispatch + combine -> op_filter="CombineDeviceOperation".

# CI set (BH LoudBox pipeline): keep small.
_DISPATCH_PERF_PARAMS = [
    _perf_param(
        "dispatch",
        "test_prefill_dispatch.py",
        "test_ttnn_dispatch",
        "linear",
        2,
        4_108_262,
        "",
        dtype_filter="bf16_out",
    ),
    _perf_param(
        "dispatch", "test_prefill_dispatch.py", "test_ttnn_dispatch", "ring", 2, 3_683_084, "", dtype_filter="bf16_out"
    ),
]
_COMBINE_PERF_PARAMS = [
    _perf_param(
        "combine", "test_prefill_combine.py", "test_ttnn_combine", "linear", 2, 3_538_087, "CombineDeviceOperation"
    ),
    _perf_param(
        "combine", "test_prefill_combine.py", "test_ttnn_combine", "ring", 2, 2_290_921, "CombineDeviceOperation"
    ),
]


# Real captured Galaxy gate indices, replayed col-by-col on LB 8x1 — single
# worker spawns dispatch+combine end-to-end on device (mirroring the
# nmilicevic/ds-glx-lb-measure replay), with **per-op** perf assertions so a
# regression points to dispatch or combine specifically.
#
# Each (layer, col) entry spawns `test_ttnn_dispatch_combine`, which runs
# TtDispatchModule → production layout transform (squeeze → TILE+bfp8 → unsqueeze)
# → TtCombineModule(init_zeros=True) in one forward pass. Tracy captures
# DispatchDeviceOperation, the layout op(s), and CombineDeviceOperation in one CSV;
# the perf wrapper merges per-device rows, filters by OP CODE substring per
# entry in `expected_per_op`, and asserts each independently. The 256-experts
# indexing space, top-k=8, experts_per_chip=8 explicit override are needed
# because the captures are Galaxy-global IDs in [0, 256); the loader
# (`load_captured_routing`) remaps them to [0, 64) ∪ {255} so the LB single-col
# combine kernel (first_expert_id=0) interprets them correctly, then slices the
# gate outputs to [0:1] for LB's 1 dispatch group.
#
# Pick set: the 4 hottest (layer, col) pairs from the real longbook_qa_eng_25600
# CI fixture (LONGBOOK_QA_ENG_25600/expert_routing.safetensors) — one per col-
# index for kernel-routing diversity — plus L27 col 0 (the SAME layer as the
# absolute hottest pick, but its coldest col at 16.4%) as a same-layer hot/cold
# cross-reference baseline.
_REAL_INDICES_PICKS: list[tuple[int, int]] = [
    # (layer, col)
    (38, 0),  # hot col 0 (41.2%)
    (28, 1),  # hot col 1 (39.5%)
    (27, 2),  # hot col 2 (43.2%) — hottest in the corpus
    (41, 3),  # hot col 3 (38.5%)
    (27, 0),  # cold baseline — same layer as the hottest pick, but its coldest col (16.4%)
]
_REAL_INDICES_TOPOS = [("linear", 2), ("ring", 2)]

# Per-(topo, nlinks, layer, col) baselines in nanoseconds. Fill in from a one-time
# LB-400G measurement run; the default ±10% margin from `_perf_param_per_op`
# applies. Dispatch and combine are developed separately, so each is asserted
# against its own baseline — a regression localizes to the responsible kernel.
_DISPATCH_REAL_INDICES_EXPECTED_NS: dict[tuple[str, int, int, int], int] = {
    # (topo, nlinks, layer, col): expected_ns. Measured on LB-400G against
    # LONGBOOK_QA_ENG_25600/expert_routing.safetensors; ±10% margin.
    ("linear", 2, 38, 0): 7_327_917,
    ("linear", 2, 28, 1): 11_265_747,
    ("linear", 2, 27, 2): 12_233_719,  # hottest pick (43.2% in-col)
    ("linear", 2, 41, 3): 10_306_467,
    ("linear", 2, 27, 0): 3_580_883,  # cold baseline (same layer as hot col2 pick)
    ("ring", 2, 38, 0): 6_494_076,
    ("ring", 2, 28, 1): 6_538_480,
    ("ring", 2, 27, 2): 7_524_959,  # hottest pick
    ("ring", 2, 41, 3): 5_802_859,
    ("ring", 2, 27, 0): 3_492_424,  # cold baseline
}
_COMBINE_REAL_INDICES_EXPECTED_NS: dict[tuple[str, int, int, int], int] = {
    ("linear", 2, 38, 0): 9_872_124,
    ("linear", 2, 28, 1): 16_409_416,
    ("linear", 2, 27, 2): 17_535_757,  # hottest pick
    ("linear", 2, 41, 3): 15_824_209,
    ("linear", 2, 27, 0): 4_227_779,  # cold baseline
    ("ring", 2, 38, 0): 8_382_228,
    ("ring", 2, 28, 1): 15_641_010,
    ("ring", 2, 27, 2): 17_103_829,  # hottest pick
    ("ring", 2, 41, 3): 14_305_275,
    ("ring", 2, 27, 0): 3_713_016,  # cold baseline
}


def _perf_param_per_op(
    op,
    worker_file,
    worker_test,
    topo,
    nlinks,
    expected_per_op: dict,
    margin: float = 0.1,
    captured_layer: int | None = None,
    captured_col: int | None = None,
    worker_filter_extras: str | None = "",
):
    """Build one pytest.param tuple for a per-op perf test.

    Identical wiring to `_perf_param` (env-var injection for captured layer/col,
    `worker_filter_extras` to drop the random/tile/dtype filter tokens when the
    worker test doesn't have those parametrize axes), but the result tuple
    carries an `expected_per_op` dict (op_code_substring → expected_ns) instead
    of a single `(expected_ns, op_filter)` pair. Used by
    `run_model_device_perf_test_per_op`.
    """
    worker_id = f"{topo}-8-{nlinks}link"
    model_name = f"deepseek_v3_{op}_{topo}_8_{nlinks}link"
    use_captured = captured_layer is not None and captured_col is not None
    parametrize_id = "perf_real_indices" if use_captured else "perf_no_pcc"
    k_filter = f"{parametrize_id} and {worker_id}"
    if worker_filter_extras:
        k_filter += f" and {worker_filter_extras}"
    if use_captured:
        model_name += f"_real_l{captured_layer:02d}_col{captured_col}"
        extra_env = {"TT_DS_CAPTURED_LAYER": str(captured_layer), "TT_DS_CAPTURED_COL": str(captured_col)}
    else:
        extra_env = {}
    command = f"pytest models/demos/deepseek_v3_d_p/tests/perf/{worker_file}::{worker_test} " f"-k '{k_filter}'"
    return (
        command,
        expected_per_op,
        f"deepseek_v3_{op}",
        model_name,
        margin,
        f"{topo}-8-{nlinks}link",
        extra_env,
    )


_DISPATCH_COMBINE_PERF_REAL_INDICES_PARAMS = [
    _perf_param_per_op(
        "dispatch_combine",
        "test_prefill_dispatch_combine.py",
        "test_ttnn_dispatch_combine",
        topo,
        nlinks,
        expected_per_op={
            "DispatchDeviceOperation": _DISPATCH_REAL_INDICES_EXPECTED_NS[(topo, nlinks, layer, col)],
            "CombineDeviceOperation": _COMBINE_REAL_INDICES_EXPECTED_NS[(topo, nlinks, layer, col)],
        },
        captured_layer=layer,
        captured_col=col,
    )
    for topo, nlinks in _REAL_INDICES_TOPOS
    for layer, col in _REAL_INDICES_PICKS
]
_PARAMS_HEADER_PER_OP = "command, expected_per_op, subdir, model_name, margin, comments, extra_env"

# Full matrix (heavy local/manual run): all 8-chip topo x num_links combos.
# Payload is auto-selected by get_max_payload_size() (7k on WH, 14k on BH).
# Baselines below are from BH (14k payload).
_DISPATCH_PERF_PARAMS_FULL = [
    _perf_param(
        "dispatch",
        "test_prefill_dispatch.py",
        "test_ttnn_dispatch",
        topo,
        nlinks,
        expected,
        "",
        dtype_filter="bf16_out",
    )
    for topo, nlinks, expected in [
        ("linear", 1, 6_564_151),
        ("linear", 2, 3_907_070),
        ("ring", 1, 5_392_448),
        ("ring", 2, 3_690_830),
    ]
]
_COMBINE_PERF_PARAMS_FULL = [
    _perf_param(
        "combine", "test_prefill_combine.py", "test_ttnn_combine", topo, nlinks, expected, "CombineDeviceOperation"
    )
    for topo, nlinks, expected in [
        ("linear", 1, 4_893_014),
        ("linear", 2, 3_552_410),
        ("ring", 1, 2_837_530),
        ("ring", 2, 2_298_073),
    ]
] + [
    _perf_param(
        "combine",
        "test_prefill_combine.py",
        "test_ttnn_combine",
        "linear",
        1,
        6_055_667,
        "CombineDeviceOperation",
        layout="row_major",
    ),
]


def _ids_for(params):
    # model_name (4th tuple element) like deepseek_v3_dispatch_linear_8_2link_7k -> linear-8-2link-7k
    ids = []
    for p in params:
        mn = p[3]
        # Strip the compound prefix first so dispatch_combine doesn't fall through to "combine_…".
        mn = (
            mn.removeprefix("deepseek_v3_dispatch_combine_")
            .removeprefix("deepseek_v3_dispatch_")
            .removeprefix("deepseek_v3_combine_")
        )
        ids.append(mn.replace("_", "-"))
    return ids


_PARAMS_HEADER = (
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, "
    "num_iterations, batch_size, margin, comments, op_filter, extra_env"
)


# --- CI (BH LoudBox pipeline) -------------------------------------------------


@pytest.mark.parametrize(_PARAMS_HEADER, _DISPATCH_PERF_PARAMS, ids=_ids_for(_DISPATCH_PERF_PARAMS))
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_dispatch(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
    op_filter,
    extra_env,
):
    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
        op_filter=op_filter,
        extra_env=extra_env,
    )


@pytest.mark.parametrize(_PARAMS_HEADER, _COMBINE_PERF_PARAMS, ids=_ids_for(_COMBINE_PERF_PARAMS))
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_combine(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
    op_filter,
    extra_env,
):
    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
        op_filter=op_filter,
        extra_env=extra_env,
    )


# --- CI real-indices replay (BH LoudBox pipeline) ----------------------------
# One worker spawn per (layer, col, topo) runs TtDispatchModule → production layout
# transform → TtCombineModule(init_zeros=True) end-to-end on device, mirroring the
# nmilicevic/ds-glx-lb-measure replay flow. Tracy captures both ops in one CSV; the
# perf wrapper sums them via merge_device_rows (empty op_filter). Real captured
# Galaxy gate indices come from $LONGBOOK_QA_ENG_25600/expert_routing.safetensors.


@pytest.mark.parametrize(
    _PARAMS_HEADER_PER_OP,
    _DISPATCH_COMBINE_PERF_REAL_INDICES_PARAMS,
    ids=_ids_for(_DISPATCH_COMBINE_PERF_REAL_INDICES_PARAMS),
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_dispatch_combine_real_indices(
    command,
    expected_per_op,
    subdir,
    model_name,
    margin,
    comments,
    extra_env,
):
    run_model_device_perf_test_per_op(
        command=command,
        expected_per_op=expected_per_op,
        subdir=subdir,
        model_name=model_name,
        margin=margin,
        comments=comments,
        extra_env=extra_env,
    )


# --- Full matrix (manual/local) ----------------------------------------------
# Not marked models_device_performance_bare_metal so they do not run in CI by default.


@pytest.mark.parametrize(_PARAMS_HEADER, _DISPATCH_PERF_PARAMS_FULL, ids=_ids_for(_DISPATCH_PERF_PARAMS_FULL))
def test_device_perf_dispatch_full(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
    op_filter,
    extra_env,
):
    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
        op_filter=op_filter,
        extra_env=extra_env,
    )


@pytest.mark.parametrize(_PARAMS_HEADER, _COMBINE_PERF_PARAMS_FULL, ids=_ids_for(_COMBINE_PERF_PARAMS_FULL))
def test_device_perf_combine_full(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
    op_filter,
    extra_env,
):
    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
        op_filter=op_filter,
        extra_env=extra_env,
    )
