"""Host-only Stage 11 accounting. Never imports TTNN or synchronizes devices.

The input is the generated adapter's immutable submission/completion ledger. The
completion-ordered partition is an attribution convention for whole-serving wall
time, not an inference of device busy time. See doc/benchmark/ROOFLINE_METHOD.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO = MODEL_DIR.parents[2]
TILE_BYTES = {"bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576, "float32": 4096, "uint32": 4096}
FIDELITY_PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
PEAK_URL = "https://docs.tenstorrent.com/aibs/blackhole/installation.html"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tiled_bytes(shape, dtype):
    """TT tile payload plus BFP exponent headers, including physical padding."""
    if dtype not in TILE_BYTES or len(shape) < 2:
        raise ValueError(f"Unknown tile descriptor: {shape}, {dtype}")
    return math.prod(shape[:-2]) * math.ceil(shape[-2] / 32) * math.ceil(shape[-1] / 32) * TILE_BYTES[dtype]


def interval_union(intervals):
    result = []
    for begin, end in sorted(intervals):
        if end < begin:
            raise ValueError("Reversed host interval")
        if not result or begin > result[-1][1]:
            result.append([begin, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def partition_timeline(steps):
    if not steps:
        raise ValueError("No measured steps")
    spans = []
    for step in steps:
        begin, end = int(step["entry_ns"]), int(step["output_ready_ns"])
        if begin <= 0 or end <= begin or step["phase"] not in ("prefill", "decode"):
            raise ValueError("Incomplete/invalid actual host completion")
        spans.append((begin, end))
    start = min(x[0] for x in spans)
    stop = max(x[1] for x in spans)
    previous = start
    intervals = []
    totals = Counter()
    for step in sorted(steps, key=lambda x: (x["output_ready_ns"], x["step_id"])):
        end = step["output_ready_ns"]
        # Every tick in the serving envelope belongs exactly once. A gap is
        # assigned to the next completed operation, including scheduler work.
        intervals.append(dict(start_ns=previous, end_ns=end, phase=step["phase"], step_id=step["step_id"]))
        totals[step["phase"]] += end - previous
        previous = end
    unions = {
        phase: interval_union([(s["entry_ns"], s["output_ready_ns"]) for s in steps if s["phase"] == phase])
        for phase in ("prefill", "decode")
    }
    all_union = interval_union(spans)
    covered = sum(b - a for a, b in all_union)
    phase_union = {p: sum(b - a for a, b in u) for p, u in unions.items()}
    if sum(totals.values()) != stop - start:
        raise ValueError("Phase partition does not cover serving envelope")
    return dict(
        start_ns=start,
        stop_ns=stop,
        envelope_ns=stop - start,
        exclusive_ns=dict(totals),
        intervals=intervals,
        original_union=all_union,
        per_phase_union=unions,
        original_interval_sum_ns=sum(b - a for a, b in spans),
        original_union_ns=covered,
        internal_gap_ns=stop - start - covered,
        cross_phase_overlap_ns=sum(phase_union.values()) - covered,
    )


def request_number(request_id, concurrency):
    # OpenAI completions prefixes x-request-id with cmpl- and appends a prompt
    # index. Retain the original ID as well; reject ambiguous/nonstandard IDs.
    match = re.fullmatch(rf"(?:cmpl-)?perf-b{concurrency}-(\d+)(?:-0)?(?:-[0-9a-f]{{8}})?", request_id)
    return int(match.group(1)) if match else None


def select_measured(records, concurrency, completed):
    chosen, mapping, excluded = [], {}, Counter()
    seen = set()
    for record in records:
        if record.get("kind") != "step":
            continue
        ids = record["request_ids"]
        numbers = [request_number(x, concurrency) for x in ids]
        matches = [x is not None for x in numbers]
        if not any(matches):
            excluded[record.get("phase", "unknown")] += 1
            continue
        if not all(matches):
            raise ValueError("Measured and unmeasured requests share a physical step")
        if record["step_id"] in seen:
            raise ValueError("Duplicate measured completion")
        seen.add(record["step_id"])
        for request_id, number in zip(ids, numbers):
            if number in mapping and mapping[number] != request_id:
                raise ValueError("Multiple engine request IDs map to one measured HTTP request")
            mapping[number] = request_id
        chosen.append(record)
    if set(mapping) != set(range(completed)):
        raise ValueError(f"Measured request set mismatch: {sorted(mapping)} vs 0..{completed-1}")
    return chosen, mapping, dict(excluded)


def snapshot(server):
    """Ask the live engine's metadata thread to atomically serialize its buffer."""
    phase_path = Path(server["process"]["phase_path"])
    candidates = [server["runtime"]["snapshot_socket"]]
    reply = None
    errors = []
    for candidate in candidates:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(30)
                client.connect(candidate)
                client.sendall(b"snapshot\n")
                parts = []
                while True:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    parts.append(chunk)
                    if b"\n" in chunk:
                        break
                reply = json.loads(b"".join(parts))
            break
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if reply is None:
        raise RuntimeError("Cannot snapshot live collector: " + "; ".join(errors))
    if reply.get("errors") or reply.get("pending"):
        raise ValueError(f"Collector has lost/incomplete work: {reply}")
    path = Path(reply.get("path", phase_path))
    if path != phase_path:
        raise ValueError("Live collector snapshot path differs from server identity")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities = [r for r in records if r.get("kind") == "identity"]
    statuses = [r for r in records if r.get("kind") == "status"]
    if len(identities) != 1 or not statuses or statuses[-1].get("errors") or statuses[-1].get("pending"):
        raise ValueError("Snapshot lacks a clean identity/status pair")
    identity = identities[0]
    runtime = server["runtime"]
    for key in ("server_instance", "max_num_seqs", "layer_count"):
        if identity.get(key) != runtime.get(key):
            raise ValueError(f"Live collector identity differs from running server: {key}")
    if identity.get("clock") != "perf_counter_ns" or identity["layer_count"] != 36:
        raise ValueError("Unsupported clock or partial model")
    pid = int(identity["pid"])
    if not Path(f"/proc/{pid}").exists():
        raise ValueError("Collector engine is no longer live")
    return path, records, identity, reply


def accounting_identity(identity):
    """Validate actual loaded dimensions, precision and physical allocations."""
    if identity["max_num_seqs"] not in (1, 32) or list(identity["mesh_shape"]) != [1, 4]:
        raise ValueError("Unexpected mesh or slot capacity")
    precision = identity["precision_config"]
    expected = json.loads((MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json").read_text())
    if precision != expected:
        raise ValueError("Loaded precision differs from selected artifact")
    for key in ("weight_allocations", "head_allocations", "dimensions", "grid"):
        if not identity.get(key):
            raise ValueError(f"Missing loaded allocation/shape evidence: {key}")
    if {a["layer"] for a in identity["weight_allocations"]} != set(range(36)):
        raise ValueError("Physical weight allocation descriptors must cover all layers")
    accurate = identity.get("accurate_attention")
    if accurate is not None:
        grouped_transport = accurate.get("transport", "per_row_v1") == "grouped_rows_v1"
        packed_transport = accurate.get("transport") == "packed_gqa_rows_v1"
        expected = {
            "accounting_schema_version": 4 if packed_transport else 3 if grouped_transport else 2,
            "grouping": "largest_power_of_two_consecutive_rows",
            "query_concat": "absent" if grouped_transport or packed_transport else "materialized_dram_for_group_gt_1",
            "output_row_slice": (
                "packed_first4_rows"
                if packed_transport
                else "absent"
                if grouped_transport
                else "materialized_dram_for_group_gt_1"
            ),
            "math_fidelity": "HiFi4",
            "fp32_denominator": True,
            "fp32_output_accumulator": True,
            "grid": [8, 8],
            "q_chunk_size": 32,
            "k_chunk_size": 128,
            "offset_reads_per_core": 2,
            "offset_read_bytes": 4,
        }
        if accurate.get("transport", "per_row_v1") not in ("per_row_v1", "grouped_rows_v1", "packed_gqa_rows_v1"):
            raise ValueError("Unknown accurate-attention transport")
        if grouped_transport and accurate.get("query_buffer_factor") != "2_if_group_gt8_else1":
            raise ValueError("Unknown accurate-attention Q buffering policy")
        if packed_transport:
            expected.update(
                query_buffer_factor="1",
                logical_q_heads_per_rank=8,
                physical_q_heads_per_packed_group=2,
                logical_kv_heads_per_rank=2,
                query_pack_factor=4,
                valid_packed_rows=4,
                single_row_group="legacy_q8_aligned_offset",
            )
        limits = (1, 2, 4, 8, 16, 32) if grouped_transport or packed_transport else (1, 2, 4, 8)
        if accurate.get("max_group_size") not in limits or any(accurate.get(k) != v for k, v in expected.items()):
            raise ValueError("Unknown accurate-attention execution/accounting policy")
        for field in ("library", "provenance", "compute_kernel", "wrapper") + (
            ("packed_writer",) if packed_transport else ()
        ):
            proof = accurate.get(field, {})
            if not proof.get("path") or not re.fullmatch("[0-9a-f]{64}", proof.get("sha256", "")):
                raise ValueError(f"Missing loaded accurate-attention {field} identity")
        provenance = accurate["provenance"].get("data", {})
        if provenance.get("library_sha256") != accurate["library"]["sha256"]:
            raise ValueError("Loaded accurate-attention binding differs from build provenance")
        if packed_transport:
            mask = accurate.get("packed_mask", {})
            if (
                mask != provenance.get("packed_gqa_mask")
                or mask.get("contract") != "same_raw_position_for_all_query_rows_v1"
                or mask.get("required_position_residues") != list(range(32))
                or mask.get("writer_path") != accurate["packed_writer"]["path"]
                or mask.get("writer_sha256") != accurate["packed_writer"]["sha256"]
                or not re.fullmatch("[0-9a-f]{64}", mask.get("mask_semantics_sha256", ""))
                or not mask.get("source_dependencies")
                or not provenance.get("generated_files")
            ):
                raise ValueError("Packed attention lacks matching private mask/writer provenance")
    return precision


def prefill_work(step, identity):
    dims, precision = identity["dimensions"], identity["precision_config"]
    h, inter, q, kv, d = (
        int(dims[k])
        for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    )
    by_fidelity = Counter()
    by_component = Counter()
    for start, end in zip(step["chunk_starts"], step["chunk_ends"]):
        length = int(end) - int(start)
        if length <= 0:
            raise ValueError("Nonpositive prefill chunk")
        for index in range(identity["layer_count"]):
            policy = {**precision["layer_defaults"], **precision["layer_exceptions"].get(str(index), {})}
            terms = {
                "qkv": 2 * length * h * (q + 2 * kv) * d,
                "o": 2 * length * q * d * h,
                "mlp": 4 * length * h * inter,
                "down": 2 * length * inter * h,
                "attention": 4 * q * d * (length * start + length * (length + 1) // 2),
            }
            for role, work in terms.items():
                # 4096-token aligned prefill uses stock chunked SDPA HiFi2.
                fidelity = policy["sdpa_fidelity"] if role == "attention" else policy[role + "_fidelity"]
                by_fidelity[fidelity] += work
                by_component[role] += work
        # last_only emits a full-vocabulary head for this chunk, not every token.
        head = 2 * h * int(dims["vocab_size"])
        by_component["terminal_head"] += head
        by_fidelity[precision["head"]["compute_fidelity"]] += head
    return dict(flops=sum(by_component.values()), by_fidelity=dict(by_fidelity), by_component=dict(by_component))


def allocation_bytes(allocation):
    if isinstance(allocation, dict):
        return tiled_bytes(
            allocation.get("padded_shape", allocation["per_rank_shape"]), allocation["dtype"].split(".")[-1].lower()
        )
    role, variant, shape, dtype = allocation
    return tiled_bytes(shape, str(dtype).split(".")[-1].lower())


def attention_groups(batch, limit):
    if not 1 <= batch <= 32 or limit not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Unsupported accurate-attention grouping")
    groups = []
    start = 0
    while start < batch:
        count = 1 << (min(limit, batch - start).bit_length() - 1)
        groups.append({"row_start": start, "row_end": start + count, "batch": count})
        start += count
    return groups


def attention_geometry(groups, transport):
    result = []
    for group in groups:
        packed = transport == "packed_gqa_rows_v1" and group["batch"] > 1
        result.append(
            dict(
                group,
                logical_q_heads=8,
                physical_q_heads=2 if packed else 8,
                physical_q_rows=32,
                useful_rows_per_head=4 if packed else 1,
                query_buffer_factor=1 if packed else 2 if group["batch"] > 8 else 1,
                offset_mode="raw_clamped" if packed else "aligned32_clamped",
                mask_mode="constant_raw_position" if packed else "causal_rows",
            )
        )
    return result


def grouped_attention_transport(batch, groups, heads, dim):
    """Per-chip/layer source-informed streams; layout intermediates included.

    Tile payloads are physical padded storage. RM control rows use 32-byte
    alignment. These are modeled operator streams, not bus transaction counts.
    """
    parts = Counter()
    bf = lambda shape: tiled_bytes(shape, "bfloat16")
    ui = lambda shape: tiled_bytes(shape, "uint32")
    rm = lambda shape, size: math.prod(shape[:-1]) * math.ceil(shape[-1] * size / 32) * 32
    parts["accurate_group_q_dram_write"] = bf((1, batch, heads, dim))
    for group in groups:
        count = group["batch"]
        compact_q = bf((1, count, heads, dim))
        repeated_q = bf((count, heads, 32, dim))
        rm_q = rm((count, heads, 1, dim), 2)
        index = ui((count, heads, 1, dim))
        scalar_tile = ui((1, 1, 1, count))
        scalar_rows = ui((count, 1, 1, 1))
        rm_positions = rm((1, 1, 1, count), 4)
        rm_scalar_rows = rm((count, 1, 1, 1), 4)
        rm_index_width = rm((count, 1, 1, dim), 4)
        rm_index = rm((count, heads, 1, dim), 4)
        if count != batch:
            parts["accurate_group_q_slices"] += 2 * compact_q
            parts["accurate_group_position_slices"] += 2 * rm_positions
        parts["accurate_group_q_permute"] += compact_q + repeated_q
        # Sub-tile H=1 repeat takes TILE->RM, RM repeat, RM->TILE.
        parts["accurate_group_q_repeat_layouts"] += 4 * repeated_q + 2 * rm_q
        parts["accurate_group_attention_q_read_output_write"] += 2 * repeated_q
        # Clamp, two bitwise operations, typecast; untilize offsets separately.
        parts["accurate_group_position_index_prepare"] += 2 * rm_positions + 10 * scalar_tile
        if count > 1:
            parts["accurate_group_scalar_owner_slices"] += 2 * rm_scalar_rows
            parts["accurate_group_index_permute"] += scalar_tile + scalar_rows
        # Index repeat: TILE->RM, width repeat, head repeat, RM->TILE.
        parts["accurate_group_index_repeat_layouts"] += (
            scalar_rows + 2 * rm_scalar_rows + 2 * rm_index_width + 2 * rm_index + index
        )
        # Gather transposes values and indices, fills index padding in-place,
        # reads one Wt_input=1 tile per tile-row, writes and transposes result.
        parts["accurate_group_gather_layouts"] += 6 * repeated_q + 5 * index
        parts["accurate_group_output_permute"] += repeated_q + compact_q
    if len(groups) > 1:
        parts["accurate_group_output_concat"] = 2 * bf((1, batch, heads, dim))
    return parts


def packed_attention_transport(batch, groups, heads, kv_heads, dim):
    """Per-chip/layer DRAM transport, retaining the exact legacy count-1 path."""
    if heads != 8 or kv_heads != 2 or dim != 128:
        raise ValueError("Unsupported packed GQA tensor geometry")
    legacy_groups = [group for group in groups if group["batch"] == 1]
    parts = grouped_attention_transport(batch, legacy_groups, heads, dim)
    parts.pop("accurate_group_output_concat", None)
    for group in groups:
        count = group["batch"]
        if count == 1:
            continue
        compact_q = tiled_bytes((1, count, heads, dim), "bfloat16")
        packed_q = tiled_bytes((count, kv_heads, 32, dim), "bfloat16")
        rm_useful = count * heads * dim * 2
        rm_positions = math.ceil(count * 4 / 32) * 32
        scalar_tile = tiled_bytes((1, 1, 1, count), "uint32")
        if count != batch:
            parts["accurate_packed_q_slices"] += 2 * compact_q
            parts["accurate_packed_position_slices"] += 2 * rm_positions
        parts["accurate_packed_q_untilize"] += compact_q + rm_useful
        parts["accurate_packed_q_zero_pad"] += rm_useful + packed_q
        parts["accurate_packed_q_tilize"] += 2 * packed_q
        parts["accurate_packed_attention_q_read_output_write"] += 2 * packed_q
        # The first4-row slice still allocates/streams full32-row tiles.
        parts["accurate_packed_output_row_slice"] += 2 * packed_q
        parts["accurate_packed_output_untilize"] += packed_q + rm_useful
        parts["accurate_packed_output_tilize"] += rm_useful + compact_q
        # TILE conversion, clamp, RM conversion, distinct scalar owners.
        parts["accurate_packed_position_prepare"] += 2 * rm_positions + 4 * scalar_tile
        parts["accurate_packed_scalar_owner_slices"] += 2 * count * 32
    if len(groups) > 1:
        parts["accurate_group_output_concat"] = 2 * tiled_bytes((1, batch, heads, dim), "bfloat16")
    return parts


def decode_work(step, identity):
    chips = math.prod(identity["mesh_shape"])
    dims, precision = identity["dimensions"], identity["precision_config"]
    h, inter, q, kv, d = (
        int(dims[k])
        for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    )
    batch = int(step["generated_batch"])
    positions = [int(p) for p in step["positions"]]
    if len(positions) != batch:
        raise ValueError("Effective position ledger does not cover physical generated batch")
    active = sum(p >= 0 for p in positions)
    if active != len(step["request_ids"]):
        raise ValueError("Active positions do not match immutable request mapping")
    capacity = int(step["table_capacity"])
    fallback = capacity > 4096 * min(16, max(1, (64 // batch) // 2))
    branch = step["attention_branch"]
    accurate_policy = identity.get("accurate_attention")
    grouped_transport = bool(accurate_policy and accurate_policy.get("transport") == "grouped_rows_v1")
    packed_transport = bool(accurate_policy and accurate_policy.get("transport") == "packed_gqa_rows_v1")
    groups = (
        attention_groups(batch, int(accurate_policy["max_group_size"]) if accurate_policy else 1) if fallback else []
    )
    batched = any(group["batch"] > 1 for group in groups)
    geometry = attention_geometry(
        groups, accurate_policy.get("transport", "per_row_v1") if accurate_policy else "per_row_v1"
    )
    expected_branch = (
        "accurate_attention_packed_gqa"
        if packed_transport and batched
        else (
            "accurate_attention_batched"
            if batched
            else "accurate_attention_fallback"
            if fallback
            else "stock_paged_decode"
        )
    )
    if branch != expected_branch:
        raise ValueError("Attention branch differs from executed shape dispatch")
    if accurate_policy:
        if step.get("attention_groups") != groups:
            raise ValueError("Attention groups do not cover physical generated rows exactly once")
        if fallback and (
            step.get("attention_effective_fidelity") != "HiFi4"
            or step.get("attention_fp32_denominator") is not True
            or step.get("attention_fp32_output_accumulator") is not True
        ):
            raise ValueError("Accurate attention must retain observed HiFi4/FP32 recurrence")
        if grouped_transport and fallback:
            if step.get("attention_transport") != "grouped_rows_v1" or step.get("attention_query_buffer_factors") != [
                2 if group["batch"] > 8 else 1 for group in groups
            ]:
                raise ValueError("Grouped transport or Q-buffer geometry differs from recorded execution")
        if packed_transport and fallback:
            if (
                step.get("attention_transport") != "packed_gqa_rows_v1"
                or step.get("attention_group_geometry") != geometry
                or step.get("attention_query_buffer_factors") != [1] * len(groups)
            ):
                raise ValueError("Packed/legacy group geometry differs from recorded execution")
    parts = Counter()
    for index in range(identity["layer_count"]):
        allocations = [a for a in identity["weight_allocations"] if a["layer"] == index]
        policy = {**precision["layer_defaults"], **precision["layer_exceptions"].get(str(index), {})}
        decode_allocs = [a for a in allocations if (a.get("variant") if isinstance(a, dict) else a[1]) == "decode"]
        if len(decode_allocs) != 5:
            raise ValueError("Expected separate QKV/O/gate/up/down decode allocations")
        parts["decoder_weights"] += chips * sum(allocation_bytes(a) for a in decode_allocs)
        dtype = policy["kv"]
        # Stock SDPA partitions each KV head's rounded K128 windows across
        # workers. Accurate prefill fallback independently reads for every Q
        # head, including inactive generated rows (which clamp to offset zero).
        context_rows = [math.ceil((max(p, 0) + 1) / 128) * 128 for p in positions if fallback or p >= 0]
        stream_heads = q // chips if fallback else kv // chips
        if packed_transport and fallback:
            parts["kv_attention_reads"] += (
                chips
                * 2
                * sum(
                    group["physical_q_heads"]
                    * sum(
                        tiled_bytes((context_rows[row], d), dtype)
                        for row in range(group["row_start"], group["row_end"])
                    )
                    for group in geometry
                )
            )
        else:
            parts["kv_attention_reads"] += (
                chips * 2 * stream_heads * sum(tiled_bytes((rows, d), dtype) for rows in context_rows)
            )
        # Paged token update reads/modifies/writes the entire cache tile row.
        parts["kv_update_read_modify_write"] += chips * active * 2 * (kv // chips) * 2 * tiled_bytes((32, d), dtype)
        # Two row-parallel reduce scatters have DRAM intermediates. Model ring
        # payload with one write+read per hop; activations otherwise remain L1.
        parts["reduce_scatter_dram_staging"] += 2 * 2 * (chips - 1) * tiled_bytes((32, h), precision["dtypes"]["ccl"])
        if fallback:
            # Explicit q->DRAM, per-row query extraction/repeat, SDPA result,
            # gather/permute, and concatenation are materialized BF16 tensors.
            local_q = q // chips
            q_stream = batch * local_q * tiled_bytes((32, d), "bfloat16")
            if packed_transport:
                parts.update(
                    {
                        key: chips * value
                        for key, value in packed_attention_transport(batch, groups, local_q, kv // chips, d).items()
                    }
                )
            elif grouped_transport:
                parts.update(
                    {
                        key: chips * value
                        for key, value in grouped_attention_transport(batch, groups, local_q, d).items()
                    }
                )
            else:
                parts["accurate_attention_materialization"] += chips * (
                    2 * q_stream
                    + batch * local_q * tiled_bytes((32, d), "bfloat16") * 4
                    + 4 * tiled_bytes((32, h // chips), "bfloat16")
                )
            if accurate_policy and not grouped_transport and not packed_transport:
                # Original per-row repeated Q and gather remain. Grouping adds
                # a materialized query concat and a materialized output slice
                # for each row in groups>1; count both source read and write.
                grouped_rows = sum(group["batch"] for group in groups if group["batch"] > 1)
                row_bytes = local_q * tiled_bytes((32, d), "bfloat16")
                parts["accurate_attention_query_group_concat"] += chips * 2 * grouped_rows * row_bytes
                parts["accurate_attention_output_row_slices"] += chips * 2 * grouped_rows * row_bytes
            if accurate_policy:
                # All64 reader cores execute scalar setup once per operation,
                # including idle cores for small groups; wider groups reuse it.
                parts["accurate_attention_scalar_offset_reads"] += chips * len(groups) * 64 * 2 * 4
        # Page-table reads: one row per stock head/worker, one per accurate Q
        # head. Include update reader; aligned 32-byte row-major accesses.
        table_row_bytes = math.ceil((capacity // 32) * 4 / 32) * 32
        workers = min(16, max(1, (110 // batch) // (kv // chips)))
        readers = batch * (q // chips) if fallback else active * (kv // chips) * workers
        if fallback and packed_transport:
            readers = sum(group["batch"] * group["physical_q_heads"] for group in geometry)
        elif fallback and grouped_transport:
            readers = sum(min(64, group["batch"] * (q // chips)) for group in groups)
        legacy_position_allowance = 0 if fallback and accurate_policy else 110 * 128
        parts["page_tables_positions"] += chips * (
            readers * table_row_bytes + active * table_row_bytes + legacy_position_allowance
        )
        if fallback and accurate_policy:
            # A full-input slice aliases the existing table. Proper subranges
            # allocate and copy; generated inactive rows still participate.
            copied_table_rows = sum(group["batch"] for group in groups if group["batch"] != batch)
            parts["accurate_attention_table_slices"] += chips * 2 * copied_table_rows * table_row_bytes
    parts["head_weights"] = chips * sum(allocation_bytes(a) for a in identity["head_allocations"])
    padded_vocab = int(identity["padded_vocab"])
    # LMHead split outputs, concatenation, then local top32 and TP all-gather
    # of candidate values/indices. Greedy policy still uses split sampler.
    shard_logits = tiled_bytes((32, padded_vocab // chips), "bfloat16")
    full_logits = tiled_bytes((32, padded_vocab), "bfloat16")
    valid_logits = tiled_bytes((32, int(dims["vocab_size"])), "bfloat16")
    physical_head_outputs = sum(
        tiled_bytes((32, a["padded_shape"][-1]), "bfloat16") for a in identity["head_allocations"]
    )
    trimmed_head_outputs = sum(
        tiled_bytes((32, logical), "bfloat16")
        for a, logical in zip(identity["head_allocations"], identity.get("head_split_sizes", []))
        if a["padded_shape"][-1] != logical
    )
    parts["head_output_and_concat"] = chips * (physical_head_outputs + 2 * shard_logits + 2 * trimmed_head_outputs)
    if step.get("sampling_strategy", "split") != "split" or step["sampling_mode"] != "device":
        raise ValueError("Performance accounting requires observed split device sampling")
    candidates_local = tiled_bytes((32, 32), "bfloat16") + tiled_bytes((32, 32), "uint32")
    candidates_all = tiled_bytes((32, 32 * chips), "bfloat16") + tiled_bytes((32, 32 * chips), "uint32")
    # Tail-mask path slices valid/tail, adds a persistent tail mask, concatenates.
    invalid_width = padded_vocab - int(dims["vocab_size"])
    parts["sampler_vocab_mask"] = (
        chips * (4 * shard_logits + 3 * tiled_bytes((32, invalid_width), "bfloat16")) if invalid_width else 0
    )
    parts["sampler_topk"] = chips * (shard_logits + candidates_local)
    parts["sampler_candidate_gather"] = chips * (candidates_local + candidates_all)
    parts["sampler_index_conversion_offset_untilize"] = chips * 6 * tiled_bytes((32, 32 * chips), "uint32")
    parts["sampler_candidate_read"] = chips * candidates_all
    parts["embedding_rope"] = chips * (
        2 * 32 * (h // chips) * 2 + 2 * 2 * 32 * d * 2 + 4 * batch * tiled_bytes((32, d), "bfloat16")
    )
    parts["token_position_and_readback"] = chips * 128 * 8
    passes = int(step.get("executed_model_passes", 1))
    if passes not in (1, 2) or bool(step.get("trace_preparation", False)) != (passes == 2):
        raise ValueError("Unknown model preparation execution count")
    if passes == 2:
        # Existing _prepare_state warms the same positions, resets them, then
        # executes the requested replay. Capture itself does not execute.
        for key in list(parts):
            if key != "token_position_and_readback":
                parts[key] *= 2
        # Warmup runs split (included above) AND argmax on the warm logits.
        parts["preparation_extra_argmax"] = chips * (shard_logits + full_logits + 3 * valid_logits)
        if padded_vocab != dims["vocab_size"]:
            parts["preparation_extra_argmax"] += chips * (full_logits + valid_logits)
    return dict(
        dram_bytes=sum(parts.values()),
        by_component=dict(parts),
        attention_branch=branch,
        attention_groups=groups,
        attention_transport=(
            "packed_gqa_rows_v1"
            if packed_transport
            else "grouped_rows_v1"
            if grouped_transport
            else "per_row_v1"
            if fallback
            else None
        ),
        attention_query_buffer_factors=[group["query_buffer_factor"] for group in geometry],
        attention_group_geometry=geometry,
        attention_fidelity="HiFi4" if fallback else "selected_sdpa_fidelity",
        active_rows=active,
        generated_batch=batch,
        effective_positions=positions,
        attention_window_rows=context_rows,
        executed_model_passes=passes,
    )


def reconcile_tokens(steps, raw, mapping):
    chunks, positions = defaultdict(list), defaultdict(list)
    for step in steps:
        if step["phase"] == "prefill":
            if len(step["request_ids"]) != len(step["chunk_starts"]) or len(step["chunk_starts"]) != len(
                step["chunk_ends"]
            ):
                raise ValueError("Chunk metadata/request length mismatch")
            for rid, begin, end in zip(step["request_ids"], step["chunk_starts"], step["chunk_ends"]):
                chunks[rid].append((int(begin), int(end)))
        else:
            live_positions = [p for p in step["positions"] if p >= 0]
            for rid, pos in zip(step["request_ids"], live_positions):
                positions[rid].append(int(pos))
    results = []
    for index, rid in sorted(mapping.items()):
        intervals = sorted(chunks[rid])
        boundary = 0
        for begin, end in intervals:
            if begin != boundary:
                raise ValueError(f"Prefill chunk gap/overlap for {rid}: {intervals}")
            boundary = end
        observed_positions = sorted(positions[rid])
        if boundary != 4096 or observed_positions[:127] != list(range(4096, 4096 + 127)):
            raise ValueError(
                f"Request {rid}: incomplete 4096/128 work; prefill={boundary}, decode={observed_positions}"
            )
        if len(observed_positions) != len(set(observed_positions)) or observed_positions != list(
            range(4096, 4096 + len(observed_positions))
        ):
            raise ValueError(f"Non-contiguous effective decode position ledger for {rid}")
        results.append(
            dict(
                client_index=index,
                request_id=rid,
                input_tokens=boundary,
                delivered_output_tokens=128,
                executed_decode_steps=len(observed_positions),
                terminal_extra_steps=len(observed_positions) - 127,
                prefill_chunks=intervals,
                decode_positions=observed_positions,
            )
        )
    if raw["total_input_tokens"] != len(results) * 4096 or raw["total_output_tokens"] != len(results) * 128:
        raise ValueError("Measured API token totals differ from reconciled work")
    # Detailed raw output counts are authoritative; generation count is not
    # inferred from HTTP chunk timing (chunks may contain several tokens).
    for key, expected in (("input_lens", 4096), ("output_lens", 128)):
        if key not in raw or len(raw[key]) != len(results) or any(n != expected for n in raw[key]):
            raise ValueError(f"Missing/nonmatching detailed {key}")
    return results


def archive_sources(run, identity=None):
    folder = run / "roofline-sources"
    folder.mkdir(exist_ok=True)
    candidates = sorted((REPO / "python_env/lib").glob("python*/site-packages/tt_perf_report/perf_report.py"))
    if len(candidates) != 1:
        raise ValueError("Cannot pin tt-perf-report architecture source")
    source = candidates[0]
    metadata = source.parents[1] / "tt_perf_report-1.3.0.dist-info/METADATA"
    if not metadata.is_file() or "Version: 1.3.0\n" not in metadata.read_text():
        raise ValueError("tt-perf-report architecture registry version changed")
    shutil.copyfile(source, folder / "tt_perf_report-1.3.0.py")
    shutil.copyfile(metadata, folder / "tt_perf_report-1.3.0-METADATA")
    method = MODEL_DIR / "doc/benchmark/ROOFLINE_METHOD.md"
    shutil.copyfile(method, folder / method.name)
    result = {
        "hardware_documentation": PEAK_URL,
        "architecture_source": str(source),
        "architecture_source_sha256": sha(source),
        "method_sha256": sha(method),
        "collector_sha256": sha(__file__),
        "local_sources": {},
    }
    for reference in sorted((MODEL_DIR / "doc/benchmark").glob("roofline-reference-blackhole-*")):
        shutil.copyfile(reference, folder / reference.name)
        result["local_sources"][reference.name] = sha(reference)
    for relative in (
        "tt/multichip_decoder.py",
        "tt/model.py",
        "tt/optimized_decoder.py",
        "tt/accurate_attention/binding.cpp",
        "tt/accurate_attention/__init__.py",
        "tt/accurate_attention/build.py",
    ):
        result["local_sources"][relative] = sha(MODEL_DIR / relative)
    if identity and identity.get("accurate_attention"):
        accurate = identity["accurate_attention"]
        result["accurate_attention"] = accurate
        proof_method = MODEL_DIR / "doc/benchmark/runtime_repair/BATCHED_ACCURATE_ACCOUNTING.md"
        shutil.copyfile(proof_method, folder / proof_method.name)
        result["batched_accurate_method_sha256"] = sha(proof_method)
        if accurate.get("transport") == "grouped_rows_v1":
            grouped_method = MODEL_DIR / "doc/benchmark/runtime_repair/GROUPED_ACCURATE_ACCOUNTING.md"
            shutil.copyfile(grouped_method, folder / grouped_method.name)
            result["grouped_accurate_method_sha256"] = sha(grouped_method)
        if accurate.get("transport") == "packed_gqa_rows_v1":
            packed_method = MODEL_DIR / "doc/benchmark/runtime_repair/PACKED_GQA_ACCOUNTING.md"
            shutil.copyfile(packed_method, folder / packed_method.name)
            result["packed_accurate_method_sha256"] = sha(packed_method)
            packed_sources = {
                **accurate["provenance"]["data"]["generated_files"],
                **accurate["packed_mask"]["source_dependencies"],
            }
            result["packed_source_archives"] = {}
            for index, (relative, expected) in enumerate(sorted(packed_sources.items())):
                path = REPO / relative
                if sha(path) != expected:
                    raise ValueError(f"Loaded packed attention source changed: {relative}")
                name = f"packed-{index}-{path.name}"
                shutil.copyfile(path, folder / name)
                result["packed_source_archives"][relative] = {"sha256": expected, "archive": name}
        result["materialization_sources"] = {
            relative: sha(REPO / relative)
            for relative in (
                "ttnn/ttnn/operations/core.py",
                "ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_interleaved_start_id.cpp",
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/repeat/repeat.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/repeat_codegen_supported.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_higher_dim_rm_interleaved.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/permute/permute.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/gather/codegen/gather_codegen_supported.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/gather/codegen/gather_codegen_program_factory.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/fill_pad/fill_pad.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp",
                "ttnn/cpp/ttnn/operations/data_movement/pad/pad.cpp",
                "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp",
            )
        }
        for name in ("provenance", "compute_kernel", "wrapper") + (
            ("packed_writer",) if accurate.get("transport") == "packed_gqa_rows_v1" else ()
        ):
            proof = accurate[name]
            if sha(proof["path"]) != proof["sha256"]:
                raise ValueError(f"Loaded accurate-attention {name} changed before collection")
            suffix = Path(proof["path"]).suffix
            shutil.copyfile(proof["path"], folder / ("accurate-attention-" + name + suffix))
    write_json(folder / "sources.json", result)
    return result


def collect(run, concurrency, server, records, identity, snapshot_path):
    accounting_identity(identity)
    raw_path = run / f"perf-b{concurrency}.json"
    raw = json.loads(raw_path.read_text())
    steps, mapping, excluded = select_measured(records, concurrency, int(raw["completed"]))
    per_request = reconcile_tokens(steps, raw, mapping)
    timeline = partition_timeline(steps)
    if timeline["envelope_ns"] / 1e9 > raw["duration"]:
        raise ValueError("Whole serving phase envelope exceeds measured client duration")
    counters = {"prefill": Counter(), "decode": Counter()}
    fidelity = Counter()
    work_records = []
    for step in steps:
        work = prefill_work(step, identity) if step["phase"] == "prefill" else decode_work(step, identity)
        counters[step["phase"]].update(work["by_component"])
        fidelity.update(work.get("by_fidelity", {}))
        work_records.append(dict(step_id=step["step_id"], phase=step["phase"], **work))
    if not all(counters.values()) or not all(timeline["exclusive_ns"].get(p, 0) > 0 for p in counters):
        raise ValueError("Both phases must have observed work and positive complete wall time")
    archive_name = f"roofline-b{concurrency}-phases.jsonl"
    shutil.copyfile(snapshot_path, run / archive_name)
    sources = archive_sources(run, identity)
    chips = math.prod(identity["mesh_shape"])
    grid = identity["grid"]
    compute = grid.get("compute_with_storage", grid.get("compute", grid)) if isinstance(grid, dict) else grid
    if isinstance(compute, dict):
        compute = [compute["x"], compute["y"]]
    workers = math.prod(compute)
    lofi_peak = chips * workers * 4096 * 1.35e9
    flops = sum(fidelity.values())
    peak_flops = flops / sum(work / (lofi_peak / FIDELITY_PHASES[f]) for f, work in fidelity.items())
    evidence = dict(
        schema_version=1,
        identity=identity,
        sources=sources,
        snapshot=archive_name,
        snapshot_sha256=sha(run / archive_name),
        performance_sha256=sha(raw_path),
        server_identity_sha256=digest(server),
        excluded_step_counts=excluded,
        request_mapping=per_request,
        measured_steps=steps,
        per_step_work=work_records,
        timeline=timeline,
        component_totals={p: dict(c) for p, c in counters.items()},
        prefill_flops_by_fidelity=dict(fidelity),
        chips=chips,
        compute_workers_per_chip=workers,
        nominal_clock_hz=1.35e9,
        effective_prefill_peak_flops_per_second=peak_flops,
        dram_peak_bytes_per_second=chips * 512e9,
        estimates_are_measured_traffic=False,
    )
    evidence_name = f"roofline-b{concurrency}-accounting.json"
    write_json(run / evidence_name, evidence)
    common = dict(
        timing_scope="full_phase_wall_time",
        evidence=evidence_name,
        evidence_sha256=sha(run / evidence_name),
        peak_source=f"tt-perf-report 1.3.0 Blackhole ArchitectureSpec (archived); actual {chips} chips × {workers} compute workers; nominal 1.35 GHz; {PEAK_URL}",
        timing_method="Nonoverlapping completion-ordered partition of first runner entry through final sampled-output completion, including host work, communication and all internal gaps; async existing completion observations, no extra synchronization. Overlaps and attribution are retained in evidence.",
    )
    row = dict(
        requests=raw["completed"],
        input_tokens=raw["total_input_tokens"],
        output_tokens=raw["total_output_tokens"],
        performance_sha256=sha(raw_path),
        server_identity_sha256=digest(server),
        prefill=dict(
            **common,
            seconds=timeline["exclusive_ns"]["prefill"] / 1e9,
            flops=flops,
            peak_flops_per_second=peak_flops,
            work_method="Useful projection, causal attention and terminal full-vocabulary head FLOPs for every actual chunk across all 36 TP4 layers; mixed-fidelity harmonic peak. Scalar/elementwise work and padding excluded from FLOPs, retained in wall time.",
        ),
        decode=dict(
            **common,
            seconds=timeline["exclusive_ns"]["decode"] / 1e9,
            dram_bytes=sum(counters["decode"].values()),
            peak_dram_bytes_per_second=chips * 512e9,
            work_method="Modeled executed-batch DRAM streams: physical padded weights with BFP headers, branch-aware K128-rounded KV reads and tile updates, DRAM intermediates, collective staging, full-vocabulary sampling and token readback. Counts actual extra terminal work; not measured memory-bus traffic; coefficients/omissions in archived method.",
        ),
    )
    path = run / "roofline.json"
    rows = json.loads(path.read_text()) if path.exists() else {}
    rows[str(concurrency)] = row
    write_json(path, rows)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 32), required=True)
    parser.add_argument("--action", choices=("check", "collect"), required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    server = json.loads((run / f"perf-b{args.concurrency}-server.json").read_text())
    path, records, identity, reply = snapshot(server)
    accounting_identity(identity)
    if int(server["max_num_seqs"]) != args.concurrency:
        raise ValueError("Server slots and intended profile differ")
    if args.action == "check":
        proof = dict(
            checked_at_unix=time.time(),
            process=server["process"],
            live_identity=identity,
            socket_reply=reply,
            snapshot_sha256=sha(path),
            collector_sha256=sha(__file__),
            host_only=True,
            added_device_synchronization=False,
        )
        write_json(run / f"roofline-b{args.concurrency}-check.json", proof)
        print(
            json.dumps(
                dict(action="check", ok=True, server_instance=identity["server_instance"], records=reply.get("count"))
            )
        )
    else:
        row = collect(run, args.concurrency, server, records, identity, path)
        print(
            json.dumps(
                dict(
                    action="collect",
                    ok=True,
                    concurrency=args.concurrency,
                    prefill_seconds=row["prefill"]["seconds"],
                    decode_seconds=row["decode"]["seconds"],
                )
            )
        )


if __name__ == "__main__":
    main()
