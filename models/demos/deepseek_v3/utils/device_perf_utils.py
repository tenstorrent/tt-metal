# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Device-performance post-processing helpers for the DeepSeek V3 decode demo.

This module consolidates the CSV filtering logic (originally in
``generated/profiler/process_decode_demo_profile.py``) and the statistics
merging logic (originally in ``generated/profiler/merge_profile_stats.py``)
so that both the standalone scripts and the pytest perf test can share the
same code.
"""

import csv
from collections import defaultdict

from loguru import logger

# ============================================================================
# CSV filtering helpers  (from process_decode_demo_profile.py)
# ============================================================================


def _parse_signposts(rows):
    """
    Parse the CSV rows to identify signpost regions.

    Returns a dict with signpost info:
    - warmup_range: (start_idx, end_idx) of warmup ops (excluding signposts)
    - trace_capture_range: (start_idx, end_idx) of trace capture ops (to delete)
    - trace_execution_ranges: list of (start_idx, end_idx) for each trace execution
    - warmup_structure: dict with embedding, dense, moe, tail ranges within warmup
    """
    signpost_info = {
        "warmup_range": None,
        "trace_capture_range": None,
        "trace_execution_ranges": [],
        "warmup_structure": {
            "embedding": None,
            "dense": None,
            "moe": None,
            "tail": None,
        },
    }

    # Find all signpost positions
    signpost_positions = []
    for idx, row in enumerate(rows):
        if row[1] == "signpost":  # OP TYPE column
            signpost_positions.append((idx, row[0]))  # (index, signpost_name)

    # Parse the signpost structure
    warmup_open = None
    warmup_close = None
    trace_capture_open = None
    trace_capture_close = None
    first_dense_layer_warmup_open = None
    first_dense_layer_warmup_close = None
    first_moe_layer_warmup_open = None
    first_moe_layer_warmup_close = None
    trace_exec_opens = []
    trace_exec_closes = []

    in_warmup = False

    for idx, name in signpost_positions:
        if name == "decode_warmup":
            if warmup_open is None:
                warmup_open = idx
                in_warmup = True
            else:
                warmup_close = idx
                in_warmup = False
        elif name == "decode_trace_capture":
            if trace_capture_open is None:
                trace_capture_open = idx
            else:
                trace_capture_close = idx
        elif name == "decode_execute_trace":
            if len(trace_exec_opens) == len(trace_exec_closes):
                trace_exec_opens.append(idx)
            else:
                trace_exec_closes.append(idx)
        elif name == "first_dense_layer" and in_warmup:
            if first_dense_layer_warmup_open is None:
                first_dense_layer_warmup_open = idx
            else:
                first_dense_layer_warmup_close = idx
        elif name == "first_moe_layer" and in_warmup:
            if first_moe_layer_warmup_open is None:
                first_moe_layer_warmup_open = idx
            else:
                first_moe_layer_warmup_close = idx

    # Set ranges (excluding signpost rows themselves)
    if warmup_open is not None and warmup_close is not None:
        signpost_info["warmup_range"] = (warmup_open + 1, warmup_close - 1)

        if first_dense_layer_warmup_open is not None:
            signpost_info["warmup_structure"]["embedding"] = (warmup_open + 1, first_dense_layer_warmup_open - 1)

        if first_dense_layer_warmup_open is not None and first_dense_layer_warmup_close is not None:
            signpost_info["warmup_structure"]["dense"] = (
                first_dense_layer_warmup_open + 1,
                first_dense_layer_warmup_close - 1,
            )

        if first_moe_layer_warmup_open is not None and first_moe_layer_warmup_close is not None:
            signpost_info["warmup_structure"]["moe"] = (
                first_moe_layer_warmup_open + 1,
                first_moe_layer_warmup_close - 1,
            )

        if first_moe_layer_warmup_close is not None:
            signpost_info["warmup_structure"]["tail"] = (first_moe_layer_warmup_close + 1, warmup_close - 1)

    if trace_capture_open is not None and trace_capture_close is not None:
        signpost_info["trace_capture_range"] = (trace_capture_open, trace_capture_close)

    for i in range(len(trace_exec_closes)):
        signpost_info["trace_execution_ranges"].append((trace_exec_opens[i] + 1, trace_exec_closes[i] - 1))

    return signpost_info


def _get_op_level_for_warmup(idx, warmup_structure):
    """Determine OP LEVEL for a warmup op based on its index."""
    if warmup_structure["embedding"] and warmup_structure["embedding"][0] <= idx <= warmup_structure["embedding"][1]:
        return "Embedding"
    elif warmup_structure["dense"] and warmup_structure["dense"][0] <= idx <= warmup_structure["dense"][1]:
        return "Dense Decoder"
    elif warmup_structure["moe"] and warmup_structure["moe"][0] <= idx <= warmup_structure["moe"][1]:
        return "MoE Decoder"
    elif warmup_structure["tail"] and warmup_structure["tail"][0] <= idx <= warmup_structure["tail"][1]:
        return "Tail"
    return "Unknown"


def _get_op_level_for_trace_exec(local_idx, warmup_structure):
    """
    Determine OP LEVEL for a trace execution op based on its local index
    within the trace.  Uses the same op counts as warmup to determine
    boundaries.
    """
    embedding_count = (
        warmup_structure["embedding"][1] - warmup_structure["embedding"][0] + 1 if warmup_structure["embedding"] else 0
    )
    dense_count = warmup_structure["dense"][1] - warmup_structure["dense"][0] + 1 if warmup_structure["dense"] else 0
    moe_count = warmup_structure["moe"][1] - warmup_structure["moe"][0] + 1 if warmup_structure["moe"] else 0

    if local_idx < embedding_count:
        return "Embedding"
    elif local_idx < embedding_count + dense_count:
        return "Dense Decoder"
    elif local_idx < embedding_count + dense_count + moe_count:
        return "MoE Decoder"
    else:
        return "Tail"


def _is_ccl_op(attributes):
    """Check if the op is a CCL op (has 'cluster_axis' in ATTRIBUTES)."""
    return "cluster_axis" in attributes


def _convert_ns_to_us(ns_value):
    """Convert nanoseconds to microseconds."""
    if ns_value == "" or ns_value is None:
        return ""
    try:
        return float(ns_value) / 1_000
    except (ValueError, TypeError):
        return ""


def filter_profile_csv(input_path, output_path):
    """
    Filter the raw profiler CSV and write a cleaned-up version.

    The output CSV contains:
    OP CODE, DEVICE ID, RUN TYPE, OP TYPE, OP LEVEL,
    OP TO OP LATENCY [us], DEVICE KERNEL DURATION [us]

    Trace-capture rows are removed; ns values are converted to us.
    """
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    op_code_idx = header.index("OP CODE")
    op_type_idx = header.index("OP TYPE")
    device_id_idx = header.index("DEVICE ID")
    attributes_idx = header.index("ATTRIBUTES")
    op_to_op_latency_idx = header.index("OP TO OP LATENCY [ns]")
    device_kernel_duration_idx = header.index("DEVICE KERNEL DURATION [ns]")

    signpost_info = _parse_signposts(rows)

    logger.info(f"Warmup range: {signpost_info['warmup_range']}")
    logger.info(
        f"Warmup structure: embedding={signpost_info['warmup_structure']['embedding']}, "
        f"dense={signpost_info['warmup_structure']['dense']}, "
        f"moe={signpost_info['warmup_structure']['moe']}, "
        f"tail={signpost_info['warmup_structure']['tail']}"
    )
    logger.info(f"Trace capture range (to delete): {signpost_info['trace_capture_range']}")
    logger.info(f"Number of trace executions: {len(signpost_info['trace_execution_ranges'])}")

    # Op counts
    ws = signpost_info["warmup_structure"]
    embedding_count = ws["embedding"][1] - ws["embedding"][0] + 1 if ws["embedding"] else 0
    dense_count = ws["dense"][1] - ws["dense"][0] + 1 if ws["dense"] else 0
    moe_count = ws["moe"][1] - ws["moe"][0] + 1 if ws["moe"] else 0
    tail_count = ws["tail"][1] - ws["tail"][0] + 1 if ws["tail"] else 0

    logger.info(
        f"Op counts per run: Embedding={embedding_count}, Dense={dense_count}, "
        f"MoE={moe_count}, Tail={tail_count}, Total={embedding_count + dense_count + moe_count + tail_count}"
    )

    output_rows = []
    output_header = [
        "OP CODE",
        "DEVICE ID",
        "RUN TYPE",
        "OP TYPE",
        "OP LEVEL",
        "OP TO OP LATENCY [us]",
        "DEVICE KERNEL DURATION [us]",
    ]

    # Indices to skip (signposts + trace capture)
    skip_indices = set()
    for idx, row in enumerate(rows):
        if row[op_type_idx] == "signpost":
            skip_indices.add(idx)
    if signpost_info["trace_capture_range"]:
        tc_start, tc_end = signpost_info["trace_capture_range"]
        for idx in range(tc_start, tc_end + 1):
            skip_indices.add(idx)

    # Warmup ops
    if signpost_info["warmup_range"]:
        warmup_start, warmup_end = signpost_info["warmup_range"]
        for idx in range(warmup_start, warmup_end + 1):
            if idx in skip_indices:
                continue
            row = rows[idx]
            output_rows.append(
                [
                    row[op_code_idx],
                    row[device_id_idx],
                    "warmup",
                    "CCL" if _is_ccl_op(row[attributes_idx]) else "Non CCL",
                    _get_op_level_for_warmup(idx, signpost_info["warmup_structure"]),
                    _convert_ns_to_us(row[op_to_op_latency_idx]),
                    _convert_ns_to_us(row[device_kernel_duration_idx]),
                ]
            )

    # Trace execution ops
    for trace_idx, (trace_start, trace_end) in enumerate(signpost_info["trace_execution_ranges"]):
        local_idx = 0
        for idx in range(trace_start, trace_end + 1):
            if idx in skip_indices:
                continue
            row = rows[idx]
            output_rows.append(
                [
                    row[op_code_idx],
                    row[device_id_idx],
                    f"trace_execution_{trace_idx}",
                    "CCL" if _is_ccl_op(row[attributes_idx]) else "Non CCL",
                    _get_op_level_for_trace_exec(local_idx, signpost_info["warmup_structure"]),
                    _convert_ns_to_us(row[op_to_op_latency_idx]),
                    _convert_ns_to_us(row[device_kernel_duration_idx]),
                ]
            )
            local_idx += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        writer.writerows(output_rows)

    logger.info(f"Filtered CSV written to: {output_path}  ({len(output_rows)} rows)")


# ============================================================================
# Statistics merging helpers  (from merge_profile_stats.py)
# ============================================================================


def _read_filtered_csv(input_path):
    """Read the filtered CSV and return rows as list of dicts."""
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _group_rows_by_run_and_device(rows):
    """Group rows by (run_type, device_id) → list of rows (in order)."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["RUN TYPE"], row["DEVICE ID"])
        groups[key].append(row)
    return groups


def _get_warmup_reference(rows):
    """
    Get the reference op sequence from warmup device 0.
    Returns list of dicts with op_code, op_level, op_type.
    """
    reference = []
    for row in rows:
        if row["RUN TYPE"] == "warmup" and row["DEVICE ID"] == "0":
            reference.append(
                {
                    "op_code": row["OP CODE"],
                    "op_level": row["OP LEVEL"],
                    "op_type": row["OP TYPE"],
                }
            )
    return reference


def _get_num_trace_executions(rows):
    """Return the count of unique ``trace_execution_N`` run types."""
    trace_indices = set()
    for row in rows:
        run_type = row["RUN TYPE"]
        if run_type.startswith("trace_execution_"):
            try:
                trace_indices.add(int(run_type.split("_")[-1]))
            except ValueError:
                pass
    return len(trace_indices)


def _calculate_stats(rows, reference, skip_traces=1):
    """
    Calculate per-op kernel duration and op-to-op latency.

    Rules
    -----
    Kernel Duration:
      - Non-CCL ops → warmup only, max across all devices.
      - CCL ops → trace executions (skip first *skip_traces*), average across
        devices then average across iterations.

    Op-to-Op Latency:
      - trace executions only (skip first *skip_traces*), device 0, average
        across iterations.
    """
    grouped = _group_rows_by_run_and_device(rows)

    device_ids = sorted(
        {row["DEVICE ID"] for row in rows if row["DEVICE ID"]},
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    num_trace_executions = _get_num_trace_executions(rows)
    num_ops = len(reference)

    logger.info(
        f"Stats: {num_ops} ops, {len(device_ids)} devices, "
        f"{num_trace_executions} trace executions (skipping first {skip_traces})"
    )

    for key, group_rows in grouped.items():
        if len(group_rows) != num_ops:
            logger.warning(f"{key} has {len(group_rows)} ops, expected {num_ops}")

    results = []
    for op_idx, op_info in enumerate(reference):
        op_code = op_info["op_code"]
        op_level = op_info["op_level"]
        op_type = op_info["op_type"]

        # --- kernel duration ---
        kernel_duration = None
        if op_type == "Non CCL":
            durations = []
            for device_id in device_ids:
                key = ("warmup", device_id)
                if key in grouped and op_idx < len(grouped[key]):
                    dur = grouped[key][op_idx]["DEVICE KERNEL DURATION [us]"]
                    if dur != "":
                        try:
                            durations.append(float(dur))
                        except ValueError:
                            pass
            if durations:
                kernel_duration = max(durations)
        else:  # CCL
            iteration_averages = []
            for trace_idx in range(skip_traces, num_trace_executions):
                run_type = f"trace_execution_{trace_idx}"
                durations = []
                for device_id in device_ids:
                    key = (run_type, device_id)
                    if key in grouped and op_idx < len(grouped[key]):
                        dur = grouped[key][op_idx]["DEVICE KERNEL DURATION [us]"]
                        if dur != "":
                            try:
                                durations.append(float(dur))
                            except ValueError:
                                pass
                if durations:
                    iteration_averages.append(sum(durations) / len(durations))
            if iteration_averages:
                kernel_duration = sum(iteration_averages) / len(iteration_averages)

        # --- op-to-op latency (device 0 only) ---
        latencies = []
        for trace_idx in range(skip_traces, num_trace_executions):
            run_type = f"trace_execution_{trace_idx}"
            key = (run_type, "0")
            if key in grouped and op_idx < len(grouped[key]):
                lat = grouped[key][op_idx]["OP TO OP LATENCY [us]"]
                if lat != "":
                    try:
                        latencies.append(float(lat))
                    except ValueError:
                        pass
        op_to_op_latency = (sum(latencies) / len(latencies)) if latencies else None

        results.append(
            {
                "OP_CODE": op_code,
                "OP_LEVEL": op_level,
                "OP_TYPE": op_type,
                "AVG KERNEL DURATION (us)": kernel_duration if kernel_duration is not None else "",
                "AVG Op-to-Op LATENCY (us)": op_to_op_latency if op_to_op_latency is not None else "",
            }
        )

    return results


def _reorder_by_level(results):
    """Reorder results: Embedding → Dense Decoder → MoE Decoder → Tail."""
    level_order = ["Embedding", "Dense Decoder", "MoE Decoder", "Tail"]
    by_level = defaultdict(list)
    for r in results:
        by_level[r["OP_LEVEL"]].append(r)
    ordered = []
    for level in level_order:
        ordered.extend(by_level[level])
    return ordered


def process_profile_stats(filtered_csv_path, merged_csv_path):
    """
    Read the *filtered* CSV, compute per-op statistics, write the merged CSV,
    and return the ordered per-op results list.

    The returned list can be passed to :func:`compute_e2e_time`.
    """
    rows = _read_filtered_csv(filtered_csv_path)
    reference = _get_warmup_reference(rows)

    level_counts = defaultdict(int)
    for op_info in reference:
        level_counts[op_info["op_level"]] += 1
    logger.info(
        "Ops per level (warmup): "
        + ", ".join(f"{lvl}={level_counts[lvl]}" for lvl in ["Embedding", "Dense Decoder", "MoE Decoder", "Tail"])
    )

    results = _calculate_stats(rows, reference)
    results = _reorder_by_level(results)

    # Write merged CSV
    output_header = ["OP_CODE", "OP_LEVEL", "OP_TYPE", "AVG KERNEL DURATION (us)", "AVG Op-to-Op LATENCY (us)"]
    with open(merged_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_header)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Merged stats CSV written to: {merged_csv_path}  ({len(results)} rows)")

    return results


# ============================================================================
# E2E time computation
# ============================================================================


def compute_e2e_time(results):
    """
    Compute 2-Layer Model E2E Time from the per-op statistics list.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`process_profile_stats`.

    Returns
    -------
    dict
        Keys: ``total_kernel_duration_us``, ``total_op_to_op_latency_us``,
        ``e2e_time_us``.
    """
    total_kernel_duration = 0.0
    total_op_to_op_latency = 0.0

    for i, r in enumerate(results):
        if r["AVG KERNEL DURATION (us)"] != "":
            total_kernel_duration += float(r["AVG KERNEL DURATION (us)"])
        # Skip the first op's latency (meaningless)
        if i > 0 and r["AVG Op-to-Op LATENCY (us)"] != "":
            total_op_to_op_latency += float(r["AVG Op-to-Op LATENCY (us)"])

    return {
        "total_kernel_duration_us": total_kernel_duration,
        "total_op_to_op_latency_us": total_op_to_op_latency,
        "e2e_time_us": total_kernel_duration + total_op_to_op_latency,
    }
