# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Complete operator and disjoint family accounting for any qualified trace graph.

Counts come from the capture, not a previous graph's operator inventory. Weight
bandwidth is a declared-byte estimate, never a physical DRAM counter. Firmware
and RISC durations include waits and do not establish compute utilization.
"""

import math
import statistics
from collections import defaultdict

from models.demos.qwen38_27b_qb2.tests.full_trace_profile import analyze, intervals_by_stage, number
from models.demos.qwen38_27b_qb2.tests.profile_export_recovery import validate_pair

BANDWIDTH = 512e9
PROJECTIONS = {
    (5120, 4608): "GDN packed projection",
    (1536, 5120): "Attention/GDN output projection",
    (5120, 9216): "MLP gate/up",
    (4352, 5120): "MLP down",
    (5120, 3584): "Full-attention packed projection",
    (5120, 16384): "Vocabulary head full chunk",
    (5120, 12928): "Vocabulary head tail chunk",
}


def family(operation):
    if operation == "MatmulDeviceOperation":
        return "matmul"
    if operation == "SdpaDecodeDeviceOperation":
        return "attention"
    if operation.startswith(("AllReduce", "AllGather", "ReduceScatter")):
        return "collectives"
    if operation in ("GenericOpDeviceOperation", "QkvCausalConv1dSiluOperation", "SigmoidGatedRmsNormOperation"):
        return "custom_gdn_conv_gated_norm"
    if operation.startswith(
        (
            "Slice",
            "Reshape",
            "Tilize",
            "Untilize",
            "Concat",
            "Reshard",
            "Pad",
            "FillPad",
            "Copy",
            "ShardedToInterleaved",
            "InterleavedToSharded",
            "Transpose",
            "NLPCreateQKVHeads",
        )
    ):
        return "layout"
    if operation.startswith("LayerNorm"):
        return "normalization"
    if operation.startswith("RotaryEmbedding"):
        return "rotary"
    if operation.startswith(("Binary", "Unary", "Typecast")):
        return "elementwise_and_typecast"
    if operation.startswith(("Topk", "Sampling", "ManualSeed")):
        return "sampling"
    if operation.startswith(("PagedFusedUpdateCache", "IndexedFill")):
        return "cache_update"
    # Unknown operations remain visible and included in every accounting total.
    return "other"


def build_report(rows, profile, baseline):
    validate_pair(profile, baseline)
    if (
        profile["precision"].get("recurrent_dtype") != "float32"
        or profile["precision"].get("kv_cache_dtype") != "bfloat8_b"
        or set(profile["precision"].get("weight_groups", {}).values()) != {"bfloat8_b"}
    ):
        raise ValueError("Bandwidth assumptions require BFP8 weights/KV and FP32 recurrence")
    rows = list(rows)
    full = analyze(rows, profile)
    if full["full_trace_reconciliation_passed"] is not True:
        raise ValueError("Full model timeline does not reconcile")
    ranks = full["ranks"]
    inventory = []
    for name in sorted({op for rank in ranks for op in rank["operations"]}):
        stats = [rank["operations"].get(name, dict(calls=0, kernel_ns=0)) for rank in ranks]
        risc = {}
        for processor in ("reader", "writer", "compute"):
            values = [r[processor + "_wait_inclusive_ns"] for r in stats if processor + "_wait_inclusive_ns" in r]
            risc[processor] = dict(
                available_rank_replays=len(values),
                median_sum_ms=statistics.median(values) / 1e6 if values else None,
                missing_rows=sum(r.get(processor + "_unavailable_rows", 0) for r in stats),
            )
        inventory.append(
            dict(
                operation=name,
                family=family(name),
                risc_wait_inclusive=risc,
                calls_range=[min(r["calls"] for r in stats), max(r["calls"] for r in stats)],
                median_calls=statistics.median(r["calls"] for r in stats),
                median_kernel_ms=statistics.median(r["kernel_ns"] for r in stats) / 1e6,
                kernel_ms_range=[min(r["kernel_ns"] for r in stats) / 1e6, max(r["kernel_ns"] for r in stats) / 1e6],
            )
        )
    inventory.sort(key=lambda row: -row["median_kernel_ms"])
    selected = defaultdict(list)
    for row in rows:
        if row.get("METAL TRACE ID") not in (str(profile["model_trace_id"]), str(profile["sample_trace_id"])):
            continue
        if row.get("METAL TRACE REPLAY SESSION ID") in (None, "", "-"):
            continue
        key = (int(row["DEVICE ID"]), int(row["METAL TRACE ID"]), int(row["METAL TRACE REPLAY SESSION ID"]))
        selected[key].append(row)
    sessions = {}
    for device in profile["device_ids"]:
        for trace in (profile["model_trace_id"], profile["sample_trace_id"]):
            sessions[device, trace] = sorted(s for d, t, s in selected if (d, t) == (device, trace))
    disjoint, matrices, metadata = [], [], defaultdict(set)
    for rank in ranks:
        device, replay = rank["device"], rank["replay"]
        raw = []
        for trace in (profile["model_trace_id"], profile["sample_trace_id"]):
            raw.extend(selected[device, trace, sessions[device, trace][replay]])
        if len(raw) != rank["device_op_rows"]:
            raise ValueError("Operator inventory lost executed rows")
        intervals = [
            (
                number(row, "DEVICE FW START CYCLE", integer=True),
                number(row, "DEVICE FW END CYCLE", integer=True),
                family(row["OP CODE"]),
            )
            for row in raw
        ]
        cycles = intervals_by_stage(intervals)
        times = {name: value / rank["cycles_per_ns"] / 1e6 for name, value in cycles.items()}
        if not math.isclose(sum(times.values()), rank["span_ns"] / 1e6, rel_tol=1e-12):
            raise ValueError("Disjoint operator timeline lost elapsed time")
        disjoint.append(
            dict(
                device=device,
                replay=replay,
                firmware_span_ms=rank["span_ns"] / 1e6,
                families_ms=times,
                device_op_rows=len(raw),
            )
        )
        per_shape = defaultdict(lambda: dict(times_ns=[], bytes_per_call=None))
        all_matmuls = []
        for row in raw:
            if row["OP CODE"] != "MatmulDeviceOperation":
                continue
            if row.get("INPUT_1_DATATYPE") != "BFLOAT8_B" or "DRAM" not in row.get("INPUT_1_MEMORY", ""):
                raise ValueError("Weight byte model requires BFP8 DRAM matrices")
            shape = tuple(int(row[f"INPUT_1_{c}_PAD[LOGICAL]"].split("[")[0]) for c in "WZYX")
            if shape[:2] != (1, 1) or any(v <= 0 or v % 32 for v in shape[-2:]):
                raise ValueError("Unexpected weight tile shape")
            shape = shape[-2:]
            target = per_shape[shape]
            target["times_ns"].append(number(row, "DEVICE KERNEL DURATION [ns]"))
            target["bytes_per_call"] = math.prod(shape) // 1024 * 1088
            all_matmuls.append(target["times_ns"][-1])
            metadata[shape].add(row.get("ATTRIBUTES", ""))
        expected = rank["operations"].get("MatmulDeviceOperation", dict(calls=0, kernel_ns=0))
        if len(all_matmuls) != expected["calls"] or not math.isclose(
            sum(all_matmuls), expected["kernel_ns"], abs_tol=1e-4
        ):
            raise ValueError("Weight inventory does not cover all matrix multiplication")
        matrices.append(per_shape)
    shapes = set().union(*(row.keys() for row in matrices))
    projections = []
    total_bytes = 0
    for shape in sorted(shapes):
        cells = [row.get(shape, dict(times_ns=[], bytes_per_call=0)) for row in matrices]
        counts = {len(row["times_ns"]) for row in cells}
        if len(counts) != 1 or 0 in counts:
            raise ValueError("Projection calls differ across ranks or replays")
        count = counts.pop()
        encoded_bytes = cells[0]["bytes_per_call"] * count
        total_bytes += encoded_bytes
        elapsed_ms = statistics.median(sum(row["times_ns"]) for row in cells) / 1e6
        if elapsed_ms <= 0:
            raise ValueError("Non-positive matmul timing")
        projections.append(
            dict(
                projection=PROJECTIONS.get(shape, "Other weight projection"),
                stored_kn=list(shape),
                calls_per_step=count,
                median_kernel_ms=elapsed_ms,
                encoded_weight_bytes=encoded_bytes,
                encoded_weight_gbs=encoded_bytes / elapsed_ms / 1e6,
                fraction_of_assumed_peak=encoded_bytes / (elapsed_ms / 1000) / BANDWIDTH,
                weight_only_floor_ms=encoded_bytes / BANDWIDTH * 1000,
                program_attributes=sorted(metadata[shape]),
            )
        )
    model_ms = statistics.median(row["host_step_s"] for row in baseline["replays"]) * 1000
    profiled_ms = statistics.median(row["host_step_s"] for row in profile["replays"]) * 1000
    longest = [
        max((row for row in disjoint if row["replay"] == replay), key=lambda row: row["firmware_span_ms"])
        for replay in range(3)
    ]
    return dict(
        scope=profile["scope"],
        state="completed",
        full_trace_reconciliation_passed=True,
        policy=profile["precision"],
        batch=profile["batch"],
        input_tokens=profile["input_tokens"],
        distinct_operation_types=len(inventory),
        device_op_rows_per_rank=[r["device_op_rows"] for r in ranks],
        inventory=inventory,
        projections=projections,
        disjoint_rank_timelines=disjoint,
        longest_rank_timelines=longest,
        comparisons=full["comparisons"],
        encoded_weight_bytes_per_chip=total_bytes,
        unprofiled_step_ms=model_ms,
        profiled_step_ms=profiled_ms,
        profiler_overhead_fraction=profiled_ms / model_ms - 1,
        counts_are_program_counts=False,
        physical_dram_counters=False,
        compute_roofline_calibrated=False,
        accounting="Each rank's firmware span is partitioned by op family with overlap and gaps explicit. "
        "Use each replay's longest-rank timeline; do not sum rank times or family medians. "
        "Firmware includes waits. Profile overhead is measured for the full step, not per operation.",
        bandwidth_assumptions="One read of each declared padded BFP8 weight tile (1088 B/32x32); "
        "512 GB/s/chip assumed. Extra transactions, activation traffic and compute excluded.",
    )
