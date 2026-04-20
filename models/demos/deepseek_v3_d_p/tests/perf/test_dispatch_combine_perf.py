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

TODO: `run_model_device_perf_test_with_merge` and `merge_device_rows` are
duplicated here pending a follow-up PR that moves them into
`models/perf/device_perf_utils.py`.
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
):
    """
    Run device performance test with multi-device row merging.

    Extends `run_model_device_perf_test` with multi-chip row merging:
    CCL ops are averaged across devices, non-CCL ops use the
    slowest device's duration (critical path).

    `op_filter`, if set, restricts the measurement to rows whose OP CODE
    contains the given substring — useful when the worker pytest runs more
    than one op and only one of them is under test.

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

    post_processed_results = run_device_perf(
        command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
    )

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


def _perf_param(op, worker_file, worker_test, topo, nlinks, payload, expected_ns, op_filter, margin=0.03):
    """Build one pytest.param tuple for the perf tests."""
    worker_id = f"perf-{topo}-8-{nlinks}link-{payload}"
    return (
        f"pytest models/demos/deepseek_v3_d_p/tests/pcc/{worker_file}::{worker_test} "
        f"-k 'perf_no_pcc and {worker_id} and random'",
        expected_ns,
        f"deepseek_v3_{op}",
        f"deepseek_v3_{op}_{topo}_8_{nlinks}link_{payload}",
        1,  # num_iterations
        1,  # batch_size
        margin,
        f"{topo}-8-{nlinks}link-{payload.upper()}",
        op_filter,
    )


# Baselines measured on BH LoudBox (bh-rb-01), seq_len=3200, emb=7168, experts=64, top-k=2.
# Dispatch worker emits only DispatchDeviceOperation -> op_filter is empty.
# Combine worker runs real dispatch + combine -> op_filter="CombineDeviceOperation".

# CI set (BH LoudBox pipeline): keep small.
_DISPATCH_PERF_PARAMS = [
    _perf_param("dispatch", "test_prefill_dispatch.py", "test_ttnn_dispatch", "linear", 2, "7k", 3_526_102, ""),
    _perf_param("dispatch", "test_prefill_dispatch.py", "test_ttnn_dispatch", "ring", 2, "7k", 2_823_079, ""),
]
_COMBINE_PERF_PARAMS = [
    _perf_param(
        "combine",
        "test_prefill_combine.py",
        "test_ttnn_combine",
        "linear",
        2,
        "7k",
        4_335_384,
        "CombineDeviceOperation",
    ),
    _perf_param(
        "combine", "test_prefill_combine.py", "test_ttnn_combine", "ring", 2, "7k", 3_113_945, "CombineDeviceOperation"
    ),
]

# Full matrix (heavy local/manual run): all 8-chip topo x num_links x payload combos.
# 14k payloads run on Blackhole only (auto-skipped on Wormhole by the worker).
# Placeholder expected=1 entries are for configs not yet baselined.
_DISPATCH_PERF_PARAMS_FULL = [
    _perf_param("dispatch", "test_prefill_dispatch.py", "test_ttnn_dispatch", topo, nlinks, payload, expected, "")
    for topo, nlinks, payload, expected in [
        ("linear", 1, "7k", 5_558_477),
        ("linear", 2, "7k", 3_533_716),
        ("linear", 1, "14k", 6_640_253),
        ("linear", 2, "14k", 3_918_331),
        ("ring", 1, "7k", 4_794_187),
        ("ring", 2, "7k", 2_815_844),
        ("ring", 1, "14k", 4_305_734),
        ("ring", 2, "14k", 2_619_047),
    ]
]
_COMBINE_PERF_PARAMS_FULL = [
    _perf_param(
        "combine",
        "test_prefill_combine.py",
        "test_ttnn_combine",
        topo,
        nlinks,
        payload,
        expected,
        "CombineDeviceOperation",
    )
    for topo, nlinks, payload, expected in [
        ("linear", 1, "7k", 5_995_117),
        ("linear", 2, "7k", 4_313_204),
        ("linear", 1, "14k", 5_952_111),
        ("linear", 2, "14k", 4_194_514),
        ("ring", 1, "7k", 5_136_507),
        ("ring", 2, "7k", 3_083_027),
        ("ring", 1, "14k", 4_513_822),
        ("ring", 2, "14k", 2_889_739),
    ]
]


def _ids_for(params):
    # model_name (4th tuple element) like deepseek_v3_dispatch_linear_8_2link_7k -> linear-8-2link-7k
    ids = []
    for p in params:
        mn = p[3]
        mn = mn.removeprefix("deepseek_v3_dispatch_").removeprefix("deepseek_v3_combine_")
        ids.append(mn.replace("_", "-"))
    return ids


_PARAMS_HEADER = (
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, "
    "num_iterations, batch_size, margin, comments, op_filter"
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
    )
