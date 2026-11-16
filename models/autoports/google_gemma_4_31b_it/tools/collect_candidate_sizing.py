#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect candidate-specific launch sizing without touching a live device."""
import argparse
import json
import math
import re
from pathlib import Path

SUPPORTED_BATCHES = (1, 8, 32)
SMALLEST_BATCHES = (1, 8, 32)
WORKING_SET_TOKENS = 131072
IDS_PER_WORKING_SET_AGENT = 2854
IDS_PER_FULL_CONTEXT_AGENT = 4902
WORKER_BLOCK_TOKENS = 32
TILE_TOKENS = 32
# Kept in sync with generator_vllm.allocate_kv_cache_per_layer. These are
# physical tile bytes, not the approximate BFP encoding bits per value.
KV_TILE_BYTES = {"bfloat8_b": 1088, "bfloat16": 2048}
SUPPORTED_WEIGHT_LABELS = {"bfp4", "bfp4_hifi2", "bfp8", "bfp8_lofi", "bf16"}
WEIGHT_GROUPS = {"attention", "mlp", "down", "lm_head"}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_server_events(log_text):
    """Parse JSON event objects embedded in the server's ordinary log lines."""
    events = []
    decoder = json.JSONDecoder()
    for line in log_text.splitlines():
        start = line.find('{"event":')
        if start < 0:
            continue
        try:
            event, _ = decoder.raw_decode(line[start:])
        except json.JSONDecodeError:
            continue
        if event.get("event") in {
            "serving_pool_launch_validation",
            "serving_cache_allocation",
            "serving_memory",
        }:
            events.append(event)
    return events


def _single_event(events, name):
    matches = [event for event in events if event.get("event") == name]
    if len(matches) != 1:
        raise ValueError(f"Need exactly one {name} event; found {len(matches)}")
    return matches[0]


def pool_geometry(allocation, cache_dtype, num_banks):
    """Derive normalized pool IDs/bytes from actual scheduler layer shapes."""
    if cache_dtype not in KV_TILE_BYTES:
        raise ValueError(f"Unsupported serving KV dtype: {cache_dtype}")
    if not isinstance(num_banks, int) or num_banks <= 0:
        raise ValueError("Serving allocator must report a positive DRAM bank count")
    layers = allocation.get("layers")
    if not isinstance(layers, list) or len(layers) != 60:
        raise ValueError("Need cache allocation shapes for all 60 model layers")

    pool_ids = None
    storage_tiles = {}
    for row in layers:
        scheduler = row.get("scheduler_shape")
        physical = row.get("physical_shape")
        storage_id = row.get("tensor_idx")
        if (
            not isinstance(scheduler, list)
            or len(scheduler) != 4
            or not isinstance(physical, list)
            or len(physical) != 4
            or not isinstance(storage_id, int)
        ):
            raise ValueError("Malformed serving cache allocation geometry")
        blocks, heads, page, dim = scheduler
        if min(blocks, heads, page, dim) <= 0:
            raise ValueError("Cache geometry dimensions must be positive")
        if pool_ids is None:
            pool_ids = blocks
        elif pool_ids != blocks:
            raise ValueError("Layer scheduler shapes disagree on normalized pool ID count")

        per_block_tiles = heads * math.ceil(page / TILE_TOKENS) * math.ceil(dim / TILE_TOKENS)
        physical_blocks, physical_heads, physical_page, physical_dim = physical
        if physical_blocks % blocks:
            raise ValueError("Physical cache shape is not an integer number of pages per scheduler block")
        physical_tiles_per_scheduler_block = (
            (physical_blocks // blocks)
            * physical_heads
            * math.ceil(physical_page / TILE_TOKENS)
            * math.ceil(physical_dim / TILE_TOKENS)
        )
        if physical_tiles_per_scheduler_block != per_block_tiles:
            raise ValueError("Physical and scheduler cache shapes disagree per normalized pool ID")
        previous = storage_tiles.setdefault(storage_id, per_block_tiles)
        if previous != per_block_tiles:
            raise ValueError("Shared cache storage has inconsistent layer tile geometry")

    if pool_ids is None or len(storage_tiles) != 10:
        raise ValueError("Expected ten shared cache storage groups for the 60-layer hybrid model")
    # Match serving_memory.validate_pool_memory exactly: each shared storage
    # group has K/V buffers, then its complete tile count is rounded to banks.
    pool_bytes = sum(
        2 * math.ceil(pool_ids * tiles / num_banks) * num_banks * KV_TILE_BYTES[cache_dtype]
        for tiles in storage_tiles.values()
    )
    bytes_per_id = pool_bytes // pool_ids if pool_bytes % pool_ids == 0 else pool_bytes / pool_ids
    return pool_ids, pool_bytes, bytes_per_id


def size_candidate(identity, events, *, task_count, host_slots):
    """Pure host sizing from an observed runtime identity and its launch events."""
    if identity.get("max_model_len") != 262144:
        raise ValueError("Candidate must preserve the advertised 262144-token context")
    if identity.get("generator_module") != "models.autoports.google_gemma_4_31b_it.tt.generator_vllm":
        raise ValueError("Sizing identity must report this run's generated serving adapter")
    batch = identity.get("max_num_seqs")
    if batch not in SUPPORTED_BATCHES:
        raise ValueError(f"Candidate batch must be one of {SUPPORTED_BATCHES}")
    if identity.get("layer_count") != 60 or identity.get("configured_layer_count") != 60:
        raise ValueError("Sizing requires the configured full 60-layer model")

    precision = identity.get("precision")
    registration = identity.get("observed_registration") or {}
    if not isinstance(precision, dict) or registration.get("precision") != precision:
        raise ValueError("Identity precision must exactly match the observed autoport registration")
    groups = precision.get("weight_groups")
    if not isinstance(groups, dict) or set(groups) != WEIGHT_GROUPS:
        raise ValueError("Candidate precision must specify all four weight groups")
    if any(label not in SUPPORTED_WEIGHT_LABELS for label in groups.values()):
        raise ValueError("Candidate has an unsupported weight precision label")
    if precision.get("layer_exceptions") != {}:
        raise ValueError("Candidate sizing does not support layer-specific precision exceptions")
    cache_dtype = precision.get("kv_cache_dtype")
    if cache_dtype not in KV_TILE_BYTES:
        raise ValueError(f"Unsupported serving KV dtype: {cache_dtype}")

    measurement = _single_event(events, "serving_pool_launch_validation")
    if measurement.get("fits") is not True or measurement.get("pool_sizing") != "fixed":
        raise ValueError("Need a successful actual launch pool validation")
    devices = measurement.get("per_device")
    if not isinstance(devices, list) or not devices:
        raise ValueError("Launch validation has no per-device memory records")
    identity_devices = identity.get("device_ids")
    if not isinstance(identity_devices, list) or sorted(d["device_id"] for d in devices) != sorted(identity_devices):
        raise ValueError("Memory and identity device sets differ")
    pool_bytes_values = {d.get("pool_bytes") for d in devices}
    if len(pool_bytes_values) != 1 or None in pool_bytes_values:
        raise ValueError("Shared-mesh devices disagree on measured KV pool bytes")

    memory_events = [
        event
        for event in events
        if event.get("event") == "serving_memory" and event.get("phase") == "before_serving_cache"
    ]
    if len(memory_events) != 1:
        raise ValueError(f"Need exactly one pre-cache serving_memory event; found {len(memory_events)}")
    allocator = memory_events[0].get("allocator") or {}
    num_banks = allocator.get("num_banks")
    allocation = _single_event(events, "serving_cache_allocation")
    pool_ids, expected_pool_bytes, pool_id_bytes = pool_geometry(allocation, cache_dtype, num_banks)
    measured_pool_bytes = next(iter(pool_bytes_values))
    if measured_pool_bytes != expected_pool_bytes:
        raise ValueError(
            f"Measured pool bytes {measured_pool_bytes} do not match layer/dtype geometry "
            f"{expected_pool_bytes} ({pool_ids} IDs x {pool_id_bytes} bytes/ID)"
        )

    capacity = (pool_ids - 1) // IDS_PER_WORKING_SET_AGENT
    full_context_capacity = (pool_ids - 1) // IDS_PER_FULL_CONTEXT_AGENT
    agents = min(task_count, host_slots, capacity, batch)
    if agents < 1:
        raise ValueError("Measured pool/batch cannot hold one planning working set")
    if agents > batch:
        raise ValueError("Selected agent count exceeds server max_num_seqs")
    smallest_batch = next(b for b in SMALLEST_BATCHES if b >= agents)
    if smallest_batch == batch:
        batch_advisory = f"Measured batch {batch} is the smallest configured batch that can hold {agents} agents."
    else:
        batch_advisory = (
            f"Batch {smallest_batch} can hold {agents} selected agents, but this record measures batch {batch}; "
            "a separate launch is required before selecting the smaller batch."
        )

    return {
        "precision": precision,
        "pool_cache_dtype": cache_dtype,
        "pool_id_bytes": pool_id_bytes,
        "allocator_num_banks": num_banks,
        "source_identity": identity.get("source_identity", identity.get("server_identity_path")),
        "server_pid": identity.get("pid"),
        "launch_evidence": identity.get("server_log"),
        "measurement": measurement,
        "free_gb": [d["free_bytes"] / 1e9 for d in devices],
        "pool_gb": [d["pool_bytes"] / 1e9 for d in devices],
        "working_set_tokens": WORKING_SET_TOKENS,
        "pool_ids": pool_ids,
        "ids_per_agent": IDS_PER_WORKING_SET_AGENT,
        "capacity_agents": capacity,
        "pool_tokens": capacity * WORKING_SET_TOKENS,
        "agents": agents,
        "selection_bounds": {
            "task_count": task_count,
            "host_slots": host_slots,
            "pool_capacity_agents": capacity,
            "max_num_seqs": batch,
        },
        "batch": batch,
        "max_num_seqs": batch,
        "smallest_configured_batch_for_selected_agents": smallest_batch,
        "smallest_batch_is_measured": smallest_batch == batch,
        "batch_selection_advisory": batch_advisory,
        "served_max_model_len": 262144,
        "full_context_ids_per_agent": IDS_PER_FULL_CONTEXT_AGENT,
        "full_context_capacity_agents": full_context_capacity,
        "reason": measurement["pool_sizing_reason"],
        "feasible_ten_agents": capacity >= 10 and batch >= 10,
        "working_set_basis": "GPQA main131072 request envelope; SWE conservative131072 planning estimate from current phase2 measured48361 final context plus32768 output allowance rounded up, not a conversation cap. Full262144 remains served; simultaneous full-context capacity is reported separately.",
        "validity": "Actual candidate launch only. Restart, precision, pool, or batch change requires a new measurement; prior sizing artifacts remain untouched.",
    }


def _legacy_head_bfp8_policy(precision):
    return (
        precision.get("config_id") == "stage12_head_bfp8_lofi"
        and precision.get("weight_groups")
        == {"attention": "bfp4", "mlp": "bfp4", "down": "bfp4", "lm_head": "bfp8_lofi"}
        and precision.get("kv_cache_dtype") == "bfloat8_b"
    )


def output_suffix(identity, output_prefix=None):
    if output_prefix is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", output_prefix):
            raise ValueError("Output prefix must contain only letters, digits, underscores, and hyphens")
        return f"_{output_prefix}"
    if _legacy_head_bfp8_policy(identity["precision"]):
        return "_head_bfp8"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--task-count", type=positive_int, help="Compatibility shorthand setting both suite task counts"
    )
    parser.add_argument("--gpqa-task-count", type=positive_int)
    parser.add_argument("--swe-task-count", type=positive_int)
    parser.add_argument("--host-slots", type=positive_int, default=10)
    parser.add_argument("--output-prefix", help="Optional explicit suffix for suite_server_size_<prefix>.json outputs")
    args = parser.parse_args()

    identity = json.loads(args.identity.read_text())
    identity["source_identity"] = str(args.identity)
    log = Path(identity["server_log"])
    events = parse_server_events(log.read_text())
    counts = {
        "r1_gpqa_diamond": args.gpqa_task_count or args.task_count or 10,
        "swe_bench_verified": args.swe_task_count or args.task_count or 85,
    }
    suffix = output_suffix(identity, args.output_prefix)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for suite, task_count in counts.items():
        result = {**size_candidate(identity, events, task_count=task_count, host_slots=args.host_slots), "suite": suite}
        filename = "gpqa" if suite == "r1_gpqa_diamond" else "swe"
        (args.output / f"{filename}_server_size{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
    print(
        json.dumps(
            [
                {
                    "suite": result["suite"],
                    "pool_ids": result["pool_ids"],
                    "pool_id_bytes": result["pool_id_bytes"],
                    "capacity_agents": result["capacity_agents"],
                    "agents": result["agents"],
                    "batch": result["batch"],
                    "smallest_configured_batch_for_selected_agents": result[
                        "smallest_configured_batch_for_selected_agents"
                    ],
                }
                for result in results
            ]
        )
    )


if __name__ == "__main__":
    main()
