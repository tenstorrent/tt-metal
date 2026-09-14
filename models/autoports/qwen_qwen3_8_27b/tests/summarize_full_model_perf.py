# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize saved TP4 full-model evidence without importing a device runtime.

The profile must contain one real linear layer, one real full-attention layer,
and the surrounding token-out path. Layer counts and stored projection bytes
come from this autoport's saved HF config and Stage5 accounting. All times are
microseconds unless the field says otherwise. No four-device times are summed.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shlex
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_ROOT.parents[2]
BANKS = 8
TP = 4
TILE_BYTES = {"BFLOAT4_B": 576, "BFLOAT8_B": 1088, "BFLOAT16": 2048}
PHASES = ("prefill", "model", "sample", "token_out")
KINDS = ("linear_attention", "full_attention")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Sources:
    def __init__(self):
        self.hashes = {}

    def register(self, path):
        path = path.resolve()
        label = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        self.hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def json(self, path):
        return json.loads(self.register(path).read_text())

    def csv(self, path):
        path = self.register(path)
        with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
            return list(csv.DictReader(stream))


def optional_path(path):
    for candidate in (path, Path(str(path) + ".gz")):
        if candidate.exists():
            return candidate
    return None


def number(value):
    return float(value) if value and value not in ("-", "N/A") else 0.0


def op_name(row):
    return row.get("OP Code", row.get("OP CODE", "")).split()[0]


def metrics(rows):
    kernel = sum(number(row["Device Time"]) for row in rows)
    gap = sum(number(row["Op-to-Op Gap"]) for row in rows)
    return {
        "ops": len(rows),
        "kernel_us": kernel,
        "incoming_gap_us": gap,
        "sum_us": kernel + gap,
        "first_id": int(rows[0]["ID"]) if rows else None,
        "last_id": int(rows[-1]["ID"]) if rows else None,
    }


def span(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def dimension(row, tensor, axis, *, logical=True):
    match = re.fullmatch(r"(\d+)\[(\d+)\]", row.get(f"{tensor}_{axis}_PAD[LOGICAL]", ""))
    return int(match.group(2 if logical else 1)) if match else None


def shape(row, tensor):
    return [dimension(row, tensor, axis) for axis in "WZYX"]


def attribute_int(attributes, key):
    match = re.search(rf"\b{key}=(\d+)", attributes)
    return int(match.group(1)) if match else None


def raw_window(raw, phase):
    start_name = f"PERF_{phase.upper()}"
    starts = [i for i, row in enumerate(raw) if row.get("OP TYPE") == "signpost" and op_name(row) == start_name]
    ends = [i for i, row in enumerate(raw) if row.get("OP TYPE") == "signpost" and op_name(row) == start_name + "_END"]
    if not starts and not ends:
        return [], None
    require(len(starts) == len(ends) == 1 and starts[0] < ends[0], f"Ambiguous {phase} signposts")
    start, end = starts[0], ends[0]
    host_us = (int(raw[end]["HOST START TS"]) - int(raw[start]["HOST START TS"])) / 1000
    require(host_us > 0, f"Invalid {phase} host signpost duration")
    rows = [row for row in raw[start + 1 : end] if row.get("OP TYPE") == "tt_dnn_device"]
    return rows, host_us


def align_raw(rows, raw):
    require(len(rows) == len(raw), "Token-out raw/table row counts differ")
    for row, original in zip(rows, raw):
        require(op_name(row) == op_name(original), "Token-out raw/table operation order differs")
        require(
            int(number(row["Global Call Count"])) == int(number(original["GLOBAL CALL COUNT"])),
            "Token-out raw/table global call counts differ",
        )


def partition(rows, raw, hidden_size):
    ops = [op_name(row) for row in rows]
    norms = [i for i, op in enumerate(ops) if op == "LayerNormDeviceOperation"]
    hidden_norms = [
        i
        for i in norms
        if (dimension(raw[i], "INPUT_0", "X") == hidden_size if raw else int(number(rows[i]["Cores"])) == 40)
    ]
    require(
        len(norms) == 7 and len(hidden_norms) == 5, f"Expected 7 norms / 5 hidden norms, got {norms}/{hidden_norms}"
    )
    linear_start, full_start, terminal_start = hidden_norms[::2]
    require(any("GdnScan" in op for op in ops[linear_start:full_start]), "First representative layer is not GDN")
    require(sum("SdpaDecode" in op for op in ops[full_start:terminal_start]) == 1, "Second layer is not full attention")
    require(not any("SdpaDecode" in op for op in ops[linear_start:full_start]), "Mixed layer partition")
    candidates = [
        i for i in range(terminal_start, len(ops)) if "Topk" in ops[i] or "TopK" in ops[i] or "ArgMax" in ops[i]
    ]
    require(candidates, "No recognizable sampling route after terminal norm")
    sample_start = candidates[0]
    boundary_source = "first TopK/ArgMax route operation"
    if raw:
        trace_ids = [row.get("METAL TRACE ID", "") for row in raw]
        transitions = [i for i in range(1, len(raw)) if trace_ids[i] != trace_ids[i - 1]]
        if all(value not in ("", "-", "N/A") for value in trace_ids) and len(transitions) == 1:
            require(
                terminal_start < transitions[0] <= sample_start, "Trace boundary conflicts with terminal/sample route"
            )
            sample_start = transitions[0]
            boundary_source = "contiguous model/sampling trace-ID transition"
    require(any(op.startswith("Matmul") for op in ops[terminal_start:sample_start]), "Terminal segment has no LM head")
    history = [i for i in range(sample_start, len(ops)) if ops[i] == "IndexedFillDeviceOperation"]
    require(len(history) <= 1, "Ambiguous history append")
    history_start = history[0] if history else len(ops)
    if history:
        require(
            ops[history_start:] == ["IndexedFillDeviceOperation", "CopyDeviceOperation", "PlusOneDeviceOperation"],
            "History suffix changed; audit its operations before assigning costs",
        )
    cuts = [0, linear_start, full_start, terminal_start, sample_start, history_start, len(rows)]
    names = ["entry_and_rope", *KINDS, "terminal_model", "sampling_without_history", "history_append"]
    intervals = {name: [a, b] for name, a, b in zip(names, cuts, cuts[1:])}
    return intervals, {
        "norm_indices": norms,
        "hidden_norm_indices": hidden_norms,
        "hidden_norm_detection": (
            "raw logical input width" if raw else "selected 40-core norm contract (raw unavailable)"
        ),
        "sampling_boundary_source": boundary_source,
    }


def matmul_audit(rows, raw, intervals):
    records = []
    for index, row in enumerate(rows):
        if not op_name(row).startswith("Matmul"):
            continue
        original = raw[index] if raw else {}
        attributes = original.get("ATTRIBUTES", "")
        readers = attribute_int(attributes, "num_workers_per_dram_bank")
        records.append(
            {
                "index": index,
                "segment": next(name for name, (a, b) in intervals.items() if a <= index < b),
                "op": row["OP Code"],
                "kernel_us": number(row["Device Time"]),
                "activation_dtype": row["Input 0 Datatype"],
                "weight_dtype": row["Input 1 Datatype"],
                "output_dtype": row["Output Datatype"],
                "math_fidelity": row["Math Fidelity"],
                "dram_sharded": row["DRAM Sharded"],
                "weight_memory": original.get("INPUT_1_MEMORY"),
                "weight_shape": shape(original, "INPUT_1"),
                "weight_padded_k": dimension(original, "INPUT_1", "Y", logical=False),
                "weight_padded_n": dimension(original, "INPUT_1", "X", logical=False),
                "k_block_tiles": attribute_int(attributes, "in0_block_w"),
                "output_per_core_n_tiles": attribute_int(attributes, "per_core_N"),
                "readers_per_bank": readers,
                "native_compute_workers": BANKS * readers if readers else None,
                "reported_cores": int(number(row["Cores"])),
                "attributes": attributes or None,
            }
        )
    return records


def route_audit(rows, raw, intervals):
    start, end = intervals["sampling_without_history"]
    ops = [op_name(row) for row in rows]
    topk = [i for i in range(start, end) if "Topk" in ops[i] or "TopK" in ops[i] or "ArgMax" in ops[i]]
    gathers = [i for i in range(start, end) if "AllGather" in ops[i]]
    gather_rows = [
        {
            "index": i,
            "op": ops[i],
            "input_shape": shape(raw[i], "INPUT_0") if raw else None,
            "output_shape": shape(raw[i], "OUTPUT_0") if raw else None,
            "dtype": rows[i]["Input 0 Datatype"],
        }
        for i in gathers
    ]
    return {
        "topk_operations": [{"index": i, "op": ops[i], "reported_cores": int(number(rows[i]["Cores"]))} for i in topk],
        "topk_metrics": metrics([rows[i] for i in topk]),
        "uses_large_indices_topk": any("TopkLargeIndices" in ops[i] for i in topk),
        "generic_topk_or_argmax": [ops[i] for i in topk if "LargeIndices" not in ops[i] and "Route" not in ops[i]],
        "sampling_gathers": gather_rows,
        "gather_metrics": metrics([rows[i] for i in gathers]),
        "only_small_candidate_gathers": (
            all(item["input_shape"][-1] is not None and item["input_shape"][-1] <= 32 for item in gather_rows)
            if gather_rows and raw
            else None
        ),
        "sampling_and_seed": [
            {"index": i, "op": ops[i], **metrics([rows[i]])}
            for i in range(start, end)
            if "Sampling" in ops[i] or "ManualSeed" in ops[i]
        ],
        "trace_ids": sorted({row.get("METAL TRACE ID", "") for row in raw}) if raw else None,
        "all_token_out_rows_are_trace_replays": (
            all(row.get("METAL TRACE REPLAY SESSION ID", "") not in ("", "-", "N/A") for row in raw) if raw else None
        ),
    }


def profile_geometry(raw):
    geometry = {}
    for row in raw:
        if "SdpaDecode" in op_name(row):
            page_shape = shape(row, "INPUT_4")[-2:]
            page_size = dimension(row, "INPUT_1", "Y")
            if all(value is not None for value in page_shape) and page_size:
                geometry["page_table_shape"] = page_shape
                geometry["cache_capacity"] = page_shape[-1] * page_size
        elif op_name(row) == "IndexedFillDeviceOperation":
            geometry["history_capacity"] = dimension(row, "INPUT_1", "W")
    return geometry


def device_report(device, profile_dir, config, sources):
    tables = {}
    for phase in PHASES:
        path = optional_path(profile_dir / f"device{device}_{phase}_perf_report.csv")
        if path:
            tables[phase] = sources.csv(path)
            text_path = profile_dir / f"device{device}_{phase}_perf_report.txt"
            if text_path.exists():
                sources.register(text_path)
    require("token_out" in tables, f"Missing device{device} token-out table")
    rows = tables["token_out"]
    require(rows and all(int(row["Device"]) == device for row in rows), "Table device IDs disagree")
    raw_path = optional_path(profile_dir / f"device{device}_ops.csv")
    raw = sources.csv(raw_path) if raw_path else []
    host_windows = {}
    token_raw = []
    for phase in PHASES:
        window, host_us = raw_window(raw, phase)
        if host_us is not None:
            host_windows[phase] = host_us
        if phase == "token_out":
            token_raw = window
    if token_raw:
        align_raw(rows, token_raw)
    intervals, boundary_audit = partition(rows, token_raw, config["hidden_size"])
    segments = {name: metrics(rows[a:b]) for name, (a, b) in intervals.items()}
    require(sum(item["ops"] for item in segments.values()) == len(rows), "Partition does not cover all rows")
    counts = Counter(config["layer_types"])
    weighted = {field: sum(counts[kind] * segments[kind][field] for kind in KINDS) for field in ("kernel_us", "sum_us")}
    once = {field: sum(m[field] for name, m in segments.items() if name not in KINDS) for field in weighted}
    groups = defaultdict(lambda: {"ops": 0, "kernel_us": 0.0, "incoming_gap_us": 0.0})
    for kind in KINDS:
        a, b = intervals[kind]
        for row in rows[a:b]:
            record = groups[op_name(row)]
            record["ops"] += counts[kind]
            record["kernel_us"] += counts[kind] * number(row["Device Time"])
            record["incoming_gap_us"] += counts[kind] * number(row["Op-to-Op Gap"])
    matmuls = matmul_audit(rows, token_raw, intervals)
    decoder_matmuls = [row for row in matmuls if row["segment"] in KINDS]
    head_matmuls = [row for row in matmuls if row["segment"] == "terminal_model"]
    sdpa = [
        {"index": i, "key_dtype": row.get("INPUT_1_DATATYPE"), "value_dtype": row.get("INPUT_2_DATATYPE")}
        for i, row in enumerate(token_raw)
        if "SdpaDecode" in op_name(row)
    ]
    head_indices = [item["index"] for item in matmuls if item["segment"] == "terminal_model"]
    report = {
        "device": device,
        "phases": {phase: metrics(table) for phase, table in tables.items()},
        "host_signpost_us": host_windows,
        "intervals_zero_based_half_open": intervals,
        "boundary_audit": boundary_audit,
        "persistent_geometry_from_raw": profile_geometry(token_raw),
        "segments": segments,
        "head_matmul_only": metrics([rows[i] for i in head_indices]),
        "decoder_weighted_op_groups": dict(sorted(groups.items(), key=lambda item: -item[1]["kernel_us"])),
        "matmuls": matmuls,
        "precision_and_collective_audit": {
            "decoder_bf16_times_bfp4_lofi": all(
                row["activation_dtype"] == "BFLOAT16"
                and row["weight_dtype"] == "BFLOAT4_B"
                and "LoFi" in row["math_fidelity"]
                for row in decoder_matmuls
            ),
            "head_bf16_times_bfp8_hifi2": all(
                row["activation_dtype"] == "BFLOAT16"
                and row["weight_dtype"] == "BFLOAT8_B"
                and "HiFi2" in row["math_fidelity"]
                for row in head_matmuls
            ),
            "collectives": [
                {
                    "index": i,
                    "segment": next(name for name, (a, b) in intervals.items() if a <= i < b),
                    "op": op_name(row),
                    "dtype": row["Input 0 Datatype"],
                    "input_memory": row["Input 0 Memory"],
                }
                for i, row in enumerate(rows)
                if any(name in op_name(row) for name in ("AllGather", "AllReduce", "ReduceScatter"))
            ],
        },
        "sdpa_cache_audit": sdpa,
        "route_audit": route_audit(rows, token_raw, intervals),
        "extrapolation": {
            "stack_kernel_sum_us": weighted["kernel_us"],
            "stack_with_profile_gaps_us": weighted["sum_us"],
            "nonstack_kernel_us": once["kernel_us"],
            "nonstack_with_gaps_us": once["sum_us"],
            "expanded_kernel_sum_us": weighted["kernel_us"] + once["kernel_us"],
            "expanded_with_profile_gaps_us": weighted["sum_us"] + once["sum_us"],
        },
    }
    if "prefill" in tables:
        gaps = [number(row["Op-to-Op Gap"]) for row in tables["prefill"]]
        report["prefill_gaps_gt_6us"] = {
            "count": sum(gap > 6 for gap in gaps),
            "sum_us": sum(gap for gap in gaps if gap > 6),
        }
    return report, [row["OP Code"] for row in rows]


def manifest(path, sources):
    if not path.exists():
        return {}
    return {
        name.lstrip("*"): digest
        for digest, name in (line.split(maxsplit=1) for line in sources.register(path).read_text().splitlines())
    }


def manifests_audit(profile_dir, benchmark_path, sources):
    profile_base = profile_dir.parent.parent / profile_dir.name
    profile_manifest = manifest(profile_base.with_suffix(".source.sha256"), sources)
    benchmark_manifest = manifest(benchmark_path.with_suffix(".source.sha256"), sources)
    profile_metadata = (
        sources.json(profile_base.with_suffix(".json")) if profile_base.with_suffix(".json").exists() else None
    )
    for base in (profile_base, benchmark_path.with_suffix("")):
        if base.with_suffix(".commit").exists():
            sources.register(base.with_suffix(".commit"))
    common = sorted(profile_manifest.keys() & benchmark_manifest.keys())
    changed = [name for name in common if profile_manifest[name] != benchmark_manifest[name]]
    runtime_paths = [
        str((MODEL_ROOT / f"tt/{name}.py").relative_to(REPO_ROOT))
        for name in ("model", "generator", "multichip_decoder")
    ] + ["models/common/sampling/tt_sampling.py", "ttnn/ttnn/_ttnn.so", "build/lib/_ttnncpp.so"]
    runtime_matches = {
        name: bool(profile_manifest.get(name)) and profile_manifest.get(name) == benchmark_manifest.get(name)
        for name in runtime_paths
    }
    return {
        "profile_metadata": profile_metadata,
        "profile_manifest": profile_manifest,
        "benchmark_manifest": benchmark_manifest,
        "common_entries": len(common),
        "changed_entries": changed,
        "missing_from_benchmark": sorted(profile_manifest.keys() - benchmark_manifest.keys()),
        "missing_from_profile": sorted(benchmark_manifest.keys() - profile_manifest.keys()),
        "all_recorded_sources_match": bool(common)
        and not changed
        and profile_manifest.keys() == benchmark_manifest.keys(),
        "runtime_source_matches": runtime_matches,
        "all_required_runtime_sources_match": all(runtime_matches.values()),
    }


def stored_matmul_bytes(row):
    """Bank-padded storage implied by the selected native reader geometry.

    per_core_N controls OUTPUT storage, not input-B bank width. Derive the latter
    from N and readers instead. Caller verifies Stage5 storage or the head's
    explicit per_core_N=weight.shard_width/32 contract.
    """
    readers = row["readers_per_bank"]
    k, n = row["weight_padded_k"], row["weight_padded_n"]
    require(readers and k and n and k % 32 == n % 32 == 0, "Missing/unsupported matmul tile geometry")
    require("DRAM_WIDTH_SHARDED" in (row["weight_memory"] or ""), "Read accounting requires bank-sharded weights")
    require(row["weight_dtype"] in TILE_BYTES, "Unaccounted weight dtype")
    bank_tiles = math.ceil(n / (32 * BANKS * readers)) * readers
    # Reject a tiny tail that does not consume every physical bank; storage is
    # then not the same as mandatory reads and needs a separate kernel audit.
    require(n // 32 > (BANKS - 1) * bank_tiles, "Not all weight banks active; audit required")
    return bank_tiles, (k // 32) * bank_tiles * BANKS * TILE_BYTES[row["weight_dtype"]]


def read_bound(devices, benchmark, config, manifests, sources):
    stage5 = MODEL_ROOT / "doc/optimized_multichip_decoder"
    accounting = sources.json(stage5 / "performance_accounting.json")
    capacity = sources.json(stage5 / "memory_capacity_plan.json")
    tool = sources.json(stage5 / "perf_tool_provenance.json")
    sources.register(MODEL_ROOT / "tests/optimized_multichip_accounting.py")
    projection_bytes = {
        kind: sum(projection["decode_bytes"] for projection in capacity["layers"][kind]["projections"].values())
        for kind in KINDS
    }
    for kind, profile_name in zip(KINDS, ("review_profile_l0", "review_profile_l3")):
        evidence = next(item for item in accounting if item["profile"] == profile_name)
        require(
            evidence["stored_projection_read_bytes_per_device"] == projection_bytes[kind],
            "Stage5 byte sources disagree",
        )
    decoder_path = str((MODEL_ROOT / "tt/multichip_decoder.py").relative_to(REPO_ROOT))
    decoder_hash_matches = manifests["profile_manifest"].get(decoder_path) == capacity["source_sha256"]
    inherited = {
        "stored_projection_read_bytes_per_layer_per_device": projection_bytes,
        "profile_decoder_sha_matches_stage5": decoder_hash_matches,
        "stage5_decoder_source_sha256": capacity["source_sha256"],
        "nominal_dram_bandwidth_bytes_per_second_per_device": tool["blackhole_8_bank_dram_bandwidth_gb_s"] * 1e9,
    }
    try:
        require(decoder_hash_matches, "Profile decoder manifest does not verify the inherited Stage5 source")
        all_heads = []
        for device in devices:
            for kind in KINDS:
                observed = [row for row in device["matmuls"] if row["segment"] == kind]
                expected_shapes = Counter(
                    tuple(p["local_shape"]) for p in capacity["layers"][kind]["projections"].values()
                )
                require(
                    Counter(tuple(row["weight_shape"][-2:]) for row in observed) == expected_shapes,
                    "Decoder projection shapes changed",
                )
                require(all(row["weight_dtype"] == "BFLOAT4_B" for row in observed), "Decoder precision changed")
                require(
                    sum(stored_matmul_bytes(row)[1] for row in observed) == projection_bytes[kind],
                    "Decoder bank padding changed",
                )
            require(
                len(device["sdpa_cache_audit"]) == 1
                and all(device["sdpa_cache_audit"][0][key] == "BFLOAT8_B" for key in ("key_dtype", "value_dtype")),
                "KV precision missing or changed",
            )
            heads = []
            for row in device["matmuls"]:
                if row["segment"] != "terminal_model":
                    continue
                bank_tiles, weight_bytes = stored_matmul_bytes(row)
                require(row["output_per_core_n_tiles"] == bank_tiles, "Head output/bank-width contract changed")
                require(
                    row["weight_shape"][:2] == [1, 1] and row["weight_shape"][-2] == config["hidden_size"],
                    "Head K/batch shape changed",
                )
                heads.append(
                    {
                        "logical_k_n": row["weight_shape"][-2:],
                        "dtype": row["weight_dtype"],
                        "readers_per_bank": row["readers_per_bank"],
                        "physical_n_tiles_per_bank": bank_tiles,
                        "stored_weight_bytes_per_device": weight_bytes,
                    }
                )
            require(
                heads and sum(head["logical_k_n"][1] for head in heads) == config["vocab_size"] // TP,
                "Head chunks do not cover local vocabulary",
            )
            all_heads.append(heads)
        require(all(heads == all_heads[0] for heads in all_heads), "Device head storage differs")
        counts = Counter(config["layer_types"])
        decoder_bytes = sum(counts[kind] * projection_bytes[kind] for kind in KINDS)
        head_bytes = sum(head["stored_weight_bytes_per_device"] for head in all_heads[0])
        require(
            config["num_key_value_heads"] % TP == 0 and config["head_dim"] % 32 == 0, "Unsupported local KV geometry"
        )
        kv_bytes_per_position = (
            2 * (config["num_key_value_heads"] // TP) * (config["head_dim"] // 32) * TILE_BYTES["BFLOAT8_B"] // 32
        )
        require(kv_bytes_per_position == capacity["kv_bytes_per_token_per_full_layer"], "KV byte sources disagree")
        workload = benchmark["workload"]
        contexts = list(range(workload["prompt_len"] + 1, workload["prompt_len"] + workload["gen_len"]))
        require(contexts, "No timed autoregressive decode contexts")
        kv_bytes = [
            math.ceil(context / 32) * 32 * kv_bytes_per_position * counts["full_attention"] * workload["batch"]
            for context in contexts
        ]
        total_bytes = [decoder_bytes + head_bytes + kv for kv in kv_bytes]
        bandwidth = inherited["nominal_dram_bandwidth_bytes_per_second_per_device"]
        reduced_bound = None
        metadata = manifests["profile_metadata"]
        if metadata and metadata.get("prompt") and not metadata.get("full"):
            # Additional standalone model/sample replays can advance context.
            # Do not infer live cur_pos from the allocated cache page count.
            minimum_context = len(metadata["prompt"]) + 1
            reduced_kv = math.ceil(minimum_context / 32) * 32 * kv_bytes_per_position
            reduced_bytes = sum(projection_bytes.values()) + head_bytes + reduced_kv
            reduced_bound = {
                "minimum_context_positions": minimum_context,
                "batch": 1,
                "representative_layer_counts": {kind: 1 for kind in KINDS},
                "read_bytes_per_device_lower_bound": reduced_bytes,
                "roofline_ms_per_token_lower_bound": reduced_bytes / bandwidth * 1000,
                "context_scope": "B1 run_full_model.py prompt plus at least one decode; extra standalone replays may advance context. Count minimum required KV tiles, not allocated cache capacity.",
            }
        return {
            "status": "estimated_from_verified_stored_geometry",
            **inherited,
            "decoder_projection_read_bytes_per_device": decoder_bytes,
            "selected_head_chunks": all_heads[0],
            "selected_head_read_bytes_per_device": head_bytes,
            "kv_bytes_per_position_per_full_layer_per_device": kv_bytes_per_position,
            "active_context_positions": {"first": contexts[0], "last": contexts[-1], "count": len(contexts)},
            "kv_read_bytes_per_device": {**span(kv_bytes), "mean": statistics.mean(kv_bytes)},
            "total_read_bytes_per_device": {**span(total_bytes), "mean": statistics.mean(total_bytes)},
            "roofline_ms_per_token_estimate": {
                **span([value / bandwidth * 1000 for value in total_bytes]),
                "mean": statistics.mean(total_bytes) / bandwidth * 1000,
            },
            "applies_to_benchmark_runtime": manifests["all_required_runtime_sources_match"],
            "reduced_profile_read_bound": reduced_bound,
            "formula": "(sum(layer_count[kind]*stored_projection_bytes[kind]) +selected_head_tiles +full_layer_count*batch*ceil(active_context/32)*32*kv_bytes_per_position)/nominal_device_bandwidth*1000",
            "head_storage_contract": "bank_tiles=readers*ceil(padded_N/(32*8*readers)); verify all banks active and head per_core_N equals derived physical bank width. per_core_N is not generally input-B geometry.",
            "scope": "Read-only lower bound on projection weights, head tiles and unique required BF8 K/V tiles. Excludes recurrent state, constants, embeddings, activations, writes, extra KV rereads, CCL and dispatch. Nominal bandwidth is not a measured attainable target.",
        }
    except ValueError as error:
        return {"status": "unavailable_contract_needs_audit", "reason": str(error), **inherited}


def benchmark_times(benchmark):
    result = {}
    for mode, record in benchmark.items():
        if not isinstance(record, dict) or "decode_s" not in record:
            continue
        tokens = record.get("decode_tokens", benchmark["workload"]["gen_len"] - 1)
        require(tokens > 0, f"No timed tokens in {mode}")
        result[mode] = {
            "decode_tokens": tokens,
            "ms_per_token": 1000 * record["decode_s"] / tokens,
            "tokens_per_second": record["tokens_per_second"],
            "ttft_ms": 1000 * record["ttft_s"],
        }
    require(result, "No recognized full benchmark timings")
    return result


def workload_comparison(devices, benchmark, metadata, mode):
    metadata = metadata or {}
    geometry = dict(metadata.get("persistent_geometry", {}))
    if "history_capacity" in metadata.get("perf", {}):
        geometry.setdefault("history_capacity", metadata["perf"]["history_capacity"])
    raw_geometry = [device["persistent_geometry_from_raw"] for device in devices]
    require(all(value == raw_geometry[0] for value in raw_geometry), "Profile allocation differs across devices")
    for name, value in raw_geometry[0].items():
        require(name not in geometry or geometry[name] == value, f"Profile metadata/raw {name} disagree")
        geometry[name] = value
    workload = benchmark["workload"]
    # The fresh benchmark first requests S+G-1 positions; _ensure_cache rounds
    # to 32-position pages. This is workload/source-derived, not a host timing.
    pages = math.ceil((workload["prompt_len"] + workload["gen_len"] - 1) / 32)
    expected = {
        "cache_capacity": pages * 32,
        "page_table_shape": [workload["batch"], pages],
        "history_capacity": benchmark[mode].get("history_capacity"),
    }
    matches = {
        name: geometry[name] == value if name in geometry and value is not None else None
        for name, value in expected.items()
    }
    descriptions = []
    for name, matched in matches.items():
        label = name.replace("_", " ")
        if matched is None:
            descriptions.append(f"{label} comparison unavailable")
        elif matched:
            descriptions.append(f"{label} matches ({geometry[name]})")
        else:
            descriptions.append(f"{label} differs (profile {geometry[name]}, benchmark {expected[name]})")
    return {
        "profile_persistent_geometry": geometry,
        "benchmark_expected_persistent_geometry": expected,
        "allocation_fields_match": matches,
        "benchmark_geometry_provenance": "Cache/page table derived from the fresh benchmark's S+G-1 request and generator page32 rounding; history capacity is recorded in the selected delivery mode.",
        "profile_minimum_active_decode_context": len(metadata["prompt"]) + 1 if metadata.get("prompt") else None,
        "benchmark_active_decode_context_range": [
            workload["prompt_len"] + 1,
            workload["prompt_len"] + workload["gen_len"] - 1,
        ],
        "limitation": "Reduced profile and full unprofiled benchmark differ in layer count, prompt data and active decode context range; mixed-run residual is not measured host time. Persistent allocation: "
        + "; ".join(descriptions)
        + ".",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = Sources()
    sources.register(Path(__file__))
    config = sources.json(MODEL_ROOT / "doc/functional_decoder/hf_config.json")["text_config"]
    require(set(config["layer_types"]) == set(KINDS), "Unknown HF decoder layer kind")
    benchmark = sources.json(args.benchmark_json)
    require(benchmark.get("full") and benchmark.get("mesh") == [1, TP], "Expected complete TP4 benchmark")
    require(benchmark["layers"] == list(range(config["num_hidden_layers"])), "Benchmark omits model layers")
    devices, signatures = [], []
    for device in range(TP):
        report, signature = device_report(device, args.profile_dir, config, sources)
        devices.append(report)
        signatures.append(signature)
    require(all(signature == signatures[0] for signature in signatures), "Four-device operation sequences differ")
    manifests = manifests_audit(args.profile_dir, args.benchmark_json, sources)
    delivery = benchmark_times(benchmark)
    mode = "deferred_delivery" if "deferred_delivery" in delivery else "immediate_delivery"
    require(mode in delivery, "No all-token delivery benchmark")
    comparison = workload_comparison(devices, benchmark, manifests["profile_metadata"], mode)
    full_us = delivery[mode]["ms_per_token"] * 1000
    estimates = {key: span([device["extrapolation"][key] for device in devices]) for key in devices[0]["extrapolation"]}
    estimates["full_unprofiled_minus_expanded_kernels_us"] = span(
        [full_us - device["extrapolation"]["expanded_kernel_sum_us"] for device in devices]
    )
    segments = {key: span([device["segments"][key]["sum_us"] for device in devices]) for key in devices[0]["segments"]}
    same_profile = {
        "token_out_device_span_us": span([device["phases"]["token_out"]["sum_us"] for device in devices]),
        "host_signpost_us_per_device": {
            device["device"]: device["host_signpost_us"].get("token_out") for device in devices
        },
        "host_minus_device_span_us_per_device": {
            device["device"]: device["host_signpost_us"]["token_out"] - device["phases"]["token_out"]["sum_us"]
            for device in devices
            if "token_out" in device["host_signpost_us"]
        },
        "scope": "Reduced-layer, instrumented, synchronized signpost window only; not full-model host overhead. Use token_out directly, not separately drained model+sample windows.",
    }
    roofline = read_bound(devices, benchmark, config, manifests, sources)
    same_profile["theoretical_read_lower_bound"] = roofline.get("reduced_profile_read_bound")
    differences = {}
    for left, right in (
        ("immediate_delivery", "queued_token_out"),
        ("deferred_delivery", "queued_token_out"),
        ("immediate_delivery", "deferred_delivery"),
    ):
        if left in delivery and right in delivery:
            differences[f"{left}_minus_{right}_us"] = 1000 * (
                delivery[left]["ms_per_token"] - delivery[right]["ms_per_token"]
            )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "cwd": str(Path.cwd()),
        "profile_dir": str(args.profile_dir.resolve()),
        "benchmark_json": str(args.benchmark_json.resolve()),
        "workload": benchmark["workload"],
        "layer_counts": dict(Counter(config["layer_types"])),
        "primary_delivery_mode": mode,
        "ttft_ms": delivery[mode]["ttft_ms"],
        "decode_ms_per_token_e2e": delivery[mode]["ms_per_token"],
        "decode_ms_per_token_device": None,
        "device_time_reason": "Full-model profiling intentionally replaced by real representative layers; expanded kernel/gap times are estimates, not a full-stack device measurement.",
        "roofline_ms_per_token_estimate": (
            roofline.get("roofline_ms_per_token_estimate", {}).get("mean")
            if roofline.get("applies_to_benchmark_runtime")
            else None
        ),
        "profile_and_benchmark_runtime_match": manifests["all_required_runtime_sources_match"],
        "workload_comparison": comparison,
        "methodology": {
            "units": "microseconds unless field explicitly says milliseconds",
            "partition": "Two norms per hidden stream layer, then final norm; distinguish width5120 hidden norms from width256 Q/K norms. Require seven total/five hidden norms and GDN/SDPA identity; no fixed op count or LM-head chunk count.",
            "sums": "Charge incoming gaps to the receiving segment. Weight decoder kinds by HF layer counts and add entry/terminal/sampling/history once. Subgroup metrics overlap their parent and must not be added twice.",
            "device_aggregation": "Compute each device independently; report min/median/max. Never sum four device times or per-op cross-device maxima.",
            "source_manifest_comparison": "Saved profile and benchmark manifests, not current mutable source checkout.",
        },
        "named_limitations": [
            comparison["limitation"],
            "Extrapolated profiler gaps may exceed full unprofiled latency due to instrumentation and non-additive execution; not negative host overhead.",
            "tt-perf-report DRAM matmul core count fixes the denominator at8 while native readers select16/24 workers; no utilization or saturation conclusion uses this denominator.",
            "Read bound uses stored weights and minimum unique KV tiles at nominal DRAM bandwidth; excludes other traffic and is not an achievable full-model target.",
        ],
        "devices": devices,
        "summary": {"extrapolation": estimates, "segments_us": segments},
        "same_profile_accounting": same_profile,
        "theoretical_read_bound": roofline,
        "full_benchmark": delivery,
        "whole_loop_delivery_differences": differences,
        "source_manifest_audit": manifests,
        "sources_sha256": sources.hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decode_ms_per_token_e2e": report["decode_ms_per_token_e2e"],
                "roofline": roofline["status"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
