# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize complete signposted layer windows, with explicit roofline estimates."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "", "-", "nan") else None


def tensor_shape(row, prefix, logical=False):
    values = []
    for axis in "WZYX":
        text = row.get(f"{prefix}_{axis}_PAD[LOGICAL]", row.get(f"{prefix}_{axis}", ""))
        parts = re.findall(r"\d+", text)
        if not parts:
            return ()
        values.append(int(parts[-1] if logical else parts[0]))
    return tuple(values)


def tensor_elements(row, prefix, logical=False):
    shape = tensor_shape(row, prefix, logical)
    return math.prod(shape) if shape else 0


def element_bytes(dtype):
    dtype = dtype.upper()
    if "BFLOAT16" in dtype or "FLOAT16" in dtype or "UINT16" in dtype:
        return 2
    if "FLOAT32" in dtype or "INT32" in dtype:
        return 4
    if "BFLOAT8" in dtype:
        return 1.0625
    if "BFLOAT4" in dtype:
        return 0.5625
    raise ValueError(f"Unaccounted DRAM dtype: {dtype}")


def is_native_sdpa_decode(row):
    code = re.sub(r"[^a-z]", "", row["OP CODE"].lower())
    return "sdpadecode" in code or "scaleddotproductattentiondecode" in code


def attribute_integer(row, name):
    values = {
        int(value)
        for value in re.findall(rf"\b{re.escape(name)}(?:['\"]\s*:\s*['\"]?|\s*=\s*)(\d+)", row.get("ATTRIBUTES", ""))
    }
    if len(values) > 1:
        raise ValueError(f"Conflicting native SDPA {name} metadata: {values}")
    return next(iter(values)) if values else None


def native_sdpa_cache_reads(row, position, layer_type, read_chunk_override=None):
    """Estimate this model's batch-one paged GQA K/V reads, including chunk padding.

    rt_args_common.hpp rounds both window endpoints to the effective K chunk.
    Its disjoint core workloads read each selected K/V tile once for the
    non-MLA, unsharded-query path used by NativePagedAttention.
    """
    heads, width = (8, 256) if layer_type == "sliding_attention" else (2, 512)
    if tensor_shape(row, "INPUT_0", logical=True) != (1, 1, 16, width):
        raise ValueError("Native traffic accounting requires this model's batch-one 16-query-head geometry")
    query_memory = row.get("INPUT_0_MEMORY", "")
    if "DRAM" not in query_memory or "INTERLEAVED" not in query_memory:
        raise ValueError("Native traffic accounting requires the model's unsharded DRAM query")
    if attribute_integer(row, "cache_position_modulo") not in (None, 0):
        raise ValueError("Native traffic accounting does not support circular cache addressing")
    if re.search(r"\buse_mla(?:['\"]\s*:\s*['\"]?|\s*=\s*)(?:true|1)", row.get("ATTRIBUTES", "")):
        raise ValueError("Native traffic accounting requires separate K/V caches, not MLA")
    configured_chunk = attribute_integer(row, "k_chunk_size")
    inferred_chunk = configured_chunk
    if configured_chunk == 0:
        fp32_dest = attribute_integer(row, "fp32_dest_acc_en")
        if fp32_dest is not None:
            maximum_tiles = 4 if fp32_dest else 8
            sequence_tiles = position // 32 + 1
            inferred_chunk = 32 * min(1 << (sequence_tiles - 1).bit_length(), maximum_tiles)
        else:
            inferred_chunk = None
    if read_chunk_override is None:
        if inferred_chunk is None:
            raise ValueError("Native SDPA metadata lacks effective chunk size; provide --native-sdpa-read-chunk-size")
        chunk = inferred_chunk
        chunk_basis = (
            "CSV k_chunk_size=0 and fp32_dest_acc_en; dynamic chunk capped at 4/8 tiles"
            if configured_chunk == 0
            else "CSV fixed k_chunk_size"
        )
    else:
        chunk = read_chunk_override
        chunk_basis = "explicit CLI effective read-chunk override"
        if inferred_chunk is not None and chunk != inferred_chunk:
            raise ValueError(f"Read-chunk override {chunk} contradicts native metadata ({inferred_chunk})")
    if chunk < 32 or chunk % 32 or chunk & (chunk - 1):
        raise ValueError("Native read chunk must be a positive power of two and a multiple of 32")
    window = 1024 if layer_type == "sliding_attention" else 0
    recorded_window = attribute_integer(row, "sliding_window_size")
    if recorded_window is not None and recorded_window != window:
        raise ValueError(f"Native window {recorded_window} disagrees with {layer_type} model policy ({window})")
    logical_end = position + 1
    logical_start = max(0, logical_end - window) if window else 0
    read_start = logical_start // chunk * chunk
    read_end = (logical_end + chunk - 1) // chunk * chunk
    elements, cache_bytes, allocation_bytes, dtypes = {}, 0, 0, {}
    for prefix in ("INPUT_1", "INPUT_2"):
        shape = tensor_shape(row, prefix)
        if len(shape) != 4 or shape[1:] != (heads, 32, width):
            raise ValueError(f"Unexpected native {prefix} paged cache shape: {shape}")
        if "DRAM" not in row.get(f"{prefix}_MEMORY", ""):
            raise ValueError("Native traffic accounting requires the model's DRAM K/V caches")
        if read_end > shape[0] * 32:
            raise ValueError(f"Native rounded read end {read_end} exceeds cache capacity {shape[0] * 32}")
        elements[prefix] = (read_end - read_start) * heads * width
        dtypes[prefix] = row[f"{prefix}_DATATYPE"]
        storage_bytes = element_bytes(dtypes[prefix])
        cache_bytes += elements[prefix] * storage_bytes
        allocation_bytes += math.prod(shape) * storage_bytes
    return elements, dict(
        position=position,
        logical_start=logical_start,
        logical_end_exclusive=logical_end,
        logical_tokens=logical_end - logical_start,
        read_start=read_start,
        read_end_exclusive=read_end,
        read_tokens=read_end - read_start,
        chunk_tokens=chunk,
        chunk_basis=chunk_basis,
        cache_dtypes=dtypes,
        kv_dram_bytes=cache_bytes,
        allocated_kv_bytes=allocation_bytes,
    )


def sparse_weight_groups(row):
    """Read active expert counts from native metadata without assuming top-k."""
    attrs = row.get("ATTRIBUTES", "")
    indexed = bool(re.search(r"\buse_indices(?:['\"]\s*:\s*['\"]?|\s*=\s*)(?:true|1)", attrs))
    shape = tensor_shape(row, "INPUT_1", logical=True)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError("Sparse traffic requires one expert-group dimension in input B")
    groups = shape[1]
    if indexed:
        if attribute_integer(row, "nnz") is not None:
            raise ValueError("Indexed sparse metadata must not also declare nnz")
        output = tensor_shape(row, "OUTPUT_0", logical=True)
        if len(output) != 4 or output[0] != 1:
            raise ValueError("Indexed sparse traffic requires the compact output group axis")
        active = output[1]
        basis = "CSV use_indices=true and compact OUTPUT_0 logical group dimension"
        # Optional tensor flattening may expose the indices as INPUT_3. When
        # present, require its count to agree with the compact output contract.
        if row.get("INPUT_3_DATATYPE") == "UINT16":
            ids = tensor_shape(row, "INPUT_3", logical=True)
            if math.prod(ids) != active or ids[-1] != active:
                raise ValueError("Indexed sparse ID count disagrees with compact output")
    else:
        active = attribute_integer(row, "nnz")
        if active is None:
            raise ValueError("Sparse traffic requires actual nnz or indexed compact-output metadata")
        basis = "CSV nnz"
    if not 0 < active <= groups:
        raise ValueError(f"Invalid sparse active/group count: {active}/{groups}")
    return dict(active=active, groups=groups, indexed=indexed, basis=basis)


def traffic(row, native_cache_elements=None):
    """One operand read/output write per native op, with sparse/gather corrections.

    This estimates DRAM traffic, not controller transactions. It excludes extra
    per-core rereads, NOC multicast, and profiler writes. Metadata tensor operands
    retain one generic DRAM read/write when present in the CSV. Device gaps
    remain in the latency denominator. Slice/embedding inputs use selected rows;
    the preceding whole-cache layout conversion still counts its full traffic.
    Native decode SDPA replaces only its K/V input counts with the source-derived
    per-position chunk-rounded reads; other operands retain the generic estimate.
    """
    if is_native_sdpa_decode(row) and native_cache_elements is None:
        raise ValueError("Native SDPA traffic requires per-position cache-read accounting")
    total = 0
    op = row["OP CODE"].lower()
    for key, memory in row.items():
        if not key.endswith("_MEMORY") or "DRAM" not in memory:
            continue
        prefix = key.removesuffix("_MEMORY")
        count = tensor_elements(row, prefix)
        if native_cache_elements is not None and prefix in native_cache_elements:
            count = native_cache_elements[prefix]
        if "sparsematmul" in op and prefix == "INPUT_1":
            sparse = sparse_weight_groups(row)
            count *= sparse["active"] / sparse["groups"]
        if "pagedupdatecache" in op and (prefix == "INPUT_0" or prefix.startswith("OUTPUT_")):
            # One active decode lane updates one tile row in each KV head.
            # Reader and writer transfer num_heads * width_tiles cache tiles.
            cache_pages = int(re.findall(r"\d+", row["INPUT_0_W_PAD[LOGICAL]"])[0])
            count = tensor_elements(row, "INPUT_0") / cache_pages
        if "pagedfusedupdatecache" in op and (prefix in ("INPUT_0", "INPUT_2") or prefix.startswith("OUTPUT_")):
            # The fused operation updates K and V together; each cache transfers
            # one page, not its entire allocated pool.
            cache_prefix = "INPUT_2" if prefix in ("INPUT_2", "OUTPUT_1") else "INPUT_0"
            cache_pages = int(re.findall(r"\d+", row[f"{cache_prefix}_W_PAD[LOGICAL]"])[0])
            count = tensor_elements(row, cache_prefix) / cache_pages
        if "embedding" in op and prefix == "INPUT_1":
            count = tensor_elements(row, "OUTPUT_0", logical=True)
        if ("slice" in op or "unpad" in op) and prefix == "INPUT_0":
            count = tensor_elements(row, "OUTPUT_0")
        total += count * element_bytes(row[f"{prefix}_DATATYPE"])
    return total


def useful_prefill_flops(kind):
    s, h, q, kv, d = (
        4096,
        2816,
        16,
        (8 if kind == "sliding_attention" else 2),
        (256 if kind == "sliding_attention" else 512),
    )
    pairs = s * 1024 - 1024 * 1023 // 2 if kind == "sliding_attention" else s * (s + 1) // 2
    terms = {
        "qkv_and_output_projections": 2 * s * h * (2 * q * d + kv * d * (2 if kind == "sliding_attention" else 1)),
        "shared_mlp": 6 * s * h * 2112,
        "active_experts_top8": 6 * s * h * 704 * 8,
        "router_projection": 2 * s * h * 128,
        "causal_attention_qk_and_pv": 4 * q * d * pairs,
    }
    return terms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--layer-type", choices=["sliding_attention", "full_attention"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-policy", help="Measured runtime precision/fidelity description")
    parser.add_argument("--peak-fidelity-cycles", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument(
        "--decode-start-position",
        type=int,
        default=4096,
        help="First absolute position of 128 successive traced decodes",
    )
    parser.add_argument(
        "--native-sdpa-read-chunk-size",
        type=int,
        help="Effective native read chunk in tokens when CSV metadata is absent; target FP32 dynamic path uses 128",
    )
    args = parser.parse_args()
    if args.decode_start_position < 0:
        parser.error("Decode start position must be nonnegative")
    rows = list(csv.DictReader(args.csv.open()))
    windows = {"prefill": [], "decode": []}
    phase = None
    for row in rows:
        code = row["OP CODE"]
        if code in ("PERF_PREFILL", "PERF_DECODE"):
            phase = code.removeprefix("PERF_").lower()
        elif code in ("PERF_PREFILL_END", "PERF_DECODE_END"):
            phase = None
        elif phase and row["OP TYPE"] == "tt_dnn_device":
            assert number(row, "DEVICE FW START CYCLE") is not None, "Missing device timing inside measured layer"
            windows[phase].append(row)
    assert all(windows.values()), "Both complete signposted windows are required"
    summary = {}
    per_replay = []
    native_reads = []
    for phase, selected in windows.items():
        devices = {r["DEVICE ID"] for r in selected}
        assert len(devices) == 1, devices
        ratios = [
            number(r, "DEVICE FW DURATION [ns]")
            / (number(r, "DEVICE FW END CYCLE") - number(r, "DEVICE FW START CYCLE"))
            for r in selected
            if number(r, "DEVICE FW END CYCLE") > number(r, "DEVICE FW START CYCLE")
        ]
        ns_per_cycle = statistics.median(ratios)
        groups = {}
        for row in selected:
            key = row["METAL TRACE REPLAY SESSION ID"] if phase == "decode" else "prefill"
            groups.setdefault(key, []).append(row)
        if phase == "decode":
            assert len(groups) == 128 and all(k not in ("", "-", "nan") for k in groups), list(groups)
            assert len({len(group) for group in groups.values()}) == 1, "Incomplete replay program records"
        durations, transfers = [], []
        ordered_groups = sorted(
            groups.items(), key=lambda item: min(number(r, "DEVICE FW START CYCLE") for r in item[1])
        )
        for replay_index, (session, group) in enumerate(ordered_groups):
            start = min(number(r, "DEVICE FW START CYCLE") for r in group)
            end = max(number(r, "DEVICE FW END CYCLE") for r in group)
            us = (end - start) * ns_per_cycle / 1000
            durations.append(us)
            position = args.decode_start_position + replay_index
            replay_reads = []
            transfer = 0
            if phase == "decode":
                for row in group:
                    counts = None
                    if is_native_sdpa_decode(row):
                        counts, basis = native_sdpa_cache_reads(
                            row, position, args.layer_type, args.native_sdpa_read_chunk_size
                        )
                        replay_reads.append(basis)
                        native_reads.append(dict(session=session, **basis))
                    transfer += traffic(row, counts)
            transfers.append(transfer)
            if phase == "decode":
                per_replay.append(
                    dict(
                        session=session,
                        first_cycle=start,
                        last_cycle=end,
                        device_us=us,
                        native_ops=len(group),
                        estimated_dram_bytes=transfers[-1],
                        decode_position=position,
                        native_sdpa_ops=len(replay_reads),
                        native_sdpa_kv_dram_bytes=sum(basis["kv_dram_bytes"] for basis in replay_reads),
                        native_sdpa_read_tokens=sum(basis["read_tokens"] for basis in replay_reads),
                    )
                )
        summary[phase] = dict(
            device_us=statistics.mean(durations),
            min_device_us=min(durations),
            max_device_us=max(durations),
            ns_per_cycle=ns_per_cycle,
            samples=len(groups),
            native_ops=len(selected),
            summed_kernel_device_us=sum(number(r, "DEVICE KERNEL DURATION [ns]") or 0 for r in selected)
            / 1000
            / len(groups),
        )
        if phase == "decode":
            summary[phase]["estimated_dram_bytes"] = statistics.mean(transfers)
            example = next(iter(groups.values()))
            with args.output.with_name("decode_example_ops.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(example)
    terms = useful_prefill_flops(args.layer_type)
    peak_flops = 120 * 4096 * 1.35e9 / args.peak_fidelity_cycles
    peak_dram = 512e9
    result = dict(
        layer_type=args.layer_type,
        workload=dict(
            input_tokens=4096,
            output_tokens=128,
            batch=1,
            concurrency=1,
            decode_start_position=args.decode_start_position,
        ),
        source=str(args.csv),
        whole_layer_windows=summary,
        prefill_device_us=summary["prefill"]["device_us"],
        decode_device_us=summary["decode"]["device_us"],
        prefill_useful_flops=sum(terms.values()),
        useful_flops_terms=terms,
        decode_dram_bytes=summary["decode"]["estimated_dram_bytes"],
        peak_flops_per_s=peak_flops,
        peak_dram_bytes_per_s=peak_dram,
        prefill_flops_pct=100 * sum(terms.values()) / (peak_flops * summary["prefill"]["device_us"] / 1e6),
        decode_dram_pct=100
        * summary["decode"]["estimated_dram_bytes"]
        / (peak_dram * summary["decode"]["device_us"] / 1e6),
        peak_basis=f"One P300 Blackhole ASIC: theoretical120 Tensix cores at1.35GHz,4096 FLOP/core/cycle divided by{args.peak_fidelity_cycles} fidelity cycles;512GB/s DRAM. Mixed fidelity/SFPU code, so this is a stated common useful-work roofline basis, not measured FPU utilization.",
        assumptions=[
            "Time is first native firmware start to last native firmware end, including all intervening layer operations and gaps. Decode is mean of128 individual trace replay windows; inter-replay input-copy/host gaps are excluded.",
            "Useful FLOPs count logical projection/MLP/router and causal attention work, with only8 active experts. Padding, masked attention, extra prefill expert computation, scalar normalization and transcendental work are excluded from numerator; their time remains in denominator.",
            "Estimated DRAM bytes sum one padded DRAM input read/output write per native operation. Sparse weights use recorded nnz divided by the weight expert count, or use_indices=true plus the compact output expert dimension; cache updates read/write one 32-token page across KV heads; embedding/slices count selected inputs; whole-cache conversions count the full pool. Native decode SDPA K/V reads use per-position causal/sliding spans rounded to its effective read chunk, not allocated cache capacity. Other operands, including index/page-table tensors, keep generic one-read/write counts. Extra per-core operand/table rereads, NOC/reduction and profiler traffic are excluded. This is an estimate, not hardware counters.",
            "Storage dtypes come from each CSV operand, including BFP8/BFP4 tile exponent storage. Specify measured math-fidelity and precision interpretation with --precision-policy. No percentages are clamped.",
        ],
    )
    result["sparse_weight_reads"] = [
        dict(global_call_count=row.get("GLOBAL CALL COUNT"), **sparse_weight_groups(row))
        for row in windows["decode"]
        if "sparsematmul" in row["OP CODE"].lower()
    ]
    if args.precision_policy:
        result["assumptions"][-1] = args.precision_policy + " No percentages are clamped."
    if native_reads:
        assert all(
            row["native_sdpa_ops"] == 1 for row in per_replay
        ), "Expected one native attention operation per model decode replay"
        result["native_sdpa_cache_reads"] = dict(
            basis="One read per selected K/V tile for batch-one non-MLA GQA with unsharded query; rounded spans follow rt_args_common.hpp",
            position_basis="128 successive positions assigned to replay groups ordered by first firmware cycle",
            effective_chunk_override=args.native_sdpa_read_chunk_size,
            mean_kv_dram_bytes=statistics.mean(row["kv_dram_bytes"] for row in native_reads),
            mean_read_tokens=statistics.mean(row["read_tokens"] for row in native_reads),
            per_replay=native_reads,
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".replays.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_replay[0]))
        writer.writeheader()
        writer.writerows(per_replay)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
