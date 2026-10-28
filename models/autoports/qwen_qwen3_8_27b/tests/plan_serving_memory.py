# SPDX-License-Identifier: Apache-2.0
"""CPU-only physical cache ledger; virtual token units are not conversation capacity."""

import argparse
import ast
import hashlib
import json
import math
import operator
from pathlib import Path


def constants(path):
    """Read arithmetic constants without importing vLLM, torch or TTNN."""
    values = {}
    wanted = {
        "STATE_BYTES_PER_LAYER",
        "ATTENTION_LAYERS",
        "RECURRENT_LAYERS",
        "RECURRENT_GROUPS",
        "ATTENTION_BYTES_PER_TOKEN",
    }
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}

    def number(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](number(node.left), number(node.right))
        raise ValueError("Unsupported cache constant expression")

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                values[target.id] = number(node.value)
    if set(values) != wanted:
        raise ValueError("Incomplete physical cache constants")
    return values


def forecast(c, budget, batch, block):
    groups = c["RECURRENT_GROUPS"]
    pools = c["RECURRENT_LAYERS"] // groups
    per_id = block * c["ATTENTION_BYTES_PER_TOKEN"] + pools * c["STATE_BYTES_PER_LAYER"]
    ids = budget // per_id
    reserved = batch * (1 + 2 * groups) + 1
    return {
        "resident_batch": batch,
        "checkpoint_tokens": block,
        "cache_budget_bytes_per_device": budget,
        "global_ids": ids,
        "bytes_per_global_id": per_id,
        "attention_backing_bytes": ids * block * c["ATTENTION_BYTES_PER_TOKEN"],
        "recurrent_snapshot_backing_bytes": ids * pools * c["STATE_BYTES_PER_LAYER"],
        "active_recurrent_state_bytes": batch * c["RECURRENT_LAYERS"] * c["STATE_BYTES_PER_LAYER"],
        "worker_reservation_ids": reserved,
        "virtual_token_hook_units": (ids - reserved) * block,
        "single_native_context_admission_floor_ids": math.ceil(262144 / block) + 2 * groups + 1 + batch,
        "screened_32_private_histories": (
            [
                {
                    "tokens_each": length,
                    "attention_ids_only": 32 * math.ceil(length / block),
                    "attention_plus_worker_reservations": 32 * math.ceil(length / block) + reserved,
                }
                for length in (16384, 32768, 65536, 131072, 262144)
            ]
            if batch == 32
            else []
        ),
    }


def measured_rows(observations, c):
    rows = []
    for label, observed in observations.items():
        a = observed["allocation"]
        ids, block = a["scheduler_blocks"], a["scheduler_block_size"]
        assert a["shared_recurrent_snapshots"] is True and a["recurrent_unique_snapshot_pools"] == 16
        assert a["selected_layers"] == list(range(64))
        assert a["attention_bytes_per_device"] == ids * block * c["ATTENTION_BYTES_PER_TOKEN"]
        assert a["recurrent_snapshot_bytes_per_device"] == ids * 16 * c["STATE_BYTES_PER_LAYER"]
        cache = a["attention_bytes_per_device"] + a["recurrent_snapshot_bytes_per_device"]
        rows.append(
            {
                "label": label,
                "source_sha256": observed["source_sha256"],
                "attention_KV_bytes": a["attention_bytes_per_device"],
                "recurrent_snapshot_bytes": a["recurrent_snapshot_bytes_per_device"],
                "active_recurrent_state_bytes": a["active_recurrent_bytes_per_device"],
                "weights_runtime_and_other_unitemized_bytes": a["allocated_bytes_per_device"]
                - cache
                - a["active_recurrent_bytes_per_device"],
                "allocator_allocated_bytes": a["allocated_bytes_per_device"],
                "allocator_free_bytes": a["free_bytes_per_device"],
                "allocator_total_bytes": a["allocated_bytes_per_device"] + a["free_bytes_per_device"],
                "preparation_checkpoints": {
                    k: v
                    for k, v in observed["preparation_memory"].items()
                    if k in ("before", "scratch_bound", "warmed_before_capture", "after")
                },
                "scope": "Measured allocator checkpoints and exact typed cache backing; not continuous peak or fragmentation proof",
            }
        )
    return rows


def union_screen(c, budget, block):
    """Unimplemented storage candidate: group-exclusive IDs may share raw backing."""
    attention = block * c["ATTENTION_BYTES_PER_TOKEN"]
    snapshots = (c["RECURRENT_LAYERS"] // c["RECURRENT_GROUPS"]) * c["STATE_BYTES_PER_LAYER"]
    native_page_bytes = 8 * 1088
    carrier_stride = math.ceil(max(attention, snapshots) / (2 * native_page_bytes)) * native_page_bytes
    per_id = 2 * carrier_stride
    return {
        "checkpoint_tokens": block,
        "cache_budget_bytes_per_device": budget,
        "conditional_bytes_per_global_id": per_id,
        "conditional_global_ids": budget // per_id,
        "existing_separate_bytes_per_global_id": attention + snapshots,
        "backing_saving_at_same_ID_count_fraction": 1 - per_id / (attention + snapshots),
        "implemented_or_measured": False,
        "requirements": [
            "Prove group-exclusive global-ID ownership and asynchronous lifetimes on exact serving sources.",
            "Translate per-layer attention tables into shared K/V carriers; preserve native logical context.",
            "Implement exact raw-byte FP32/BF16 snapshot save/restore into group-exclusive carrier regions.",
            "Prevent warmup from cloning the full shared carrier; prove scratch bounds and no cache corruption.",
            "Validate alias lifetimes, null blocks, eviction/reuse, cancellation, traces and all-layer outputs before serving.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-observations", type=Path)
    p.add_argument("--cache-budget-bytes", type=int, default=23661297664)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = Path(__file__).resolve().parents[1] / "tt/vllm_cache_specs.py"
    precision = json.loads((source.parent / "selected_precision_config.json").read_text())
    assert precision["weight_groups"]["mlp"] == "bfloat4_b"
    assert precision["kv_cache_dtype"] == "bfloat8_b"
    assert precision["recurrent_state_dtype"] == "float32"
    c = constants(source)
    assert c["ATTENTION_LAYERS"] == 16 and c["RECURRENT_LAYERS"] == 48
    # Selected BF4 TP4 MLP: K5120, N4352 per gate/up; 576 bytes per32x32 tile.
    duplicate = 64 * (5120 // 32) * (2 * 4352 // 32) * 576
    report = {
        "units": "bytes per participating TP4 device, never four-device pooled capacity",
        "cache_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "constants": c,
        "measured": (
            measured_rows(json.loads(args.prepared_observations.read_text()), c) if args.prepared_observations else []
        ),
        "forecasts": [
            forecast(c, budget, batch, block)
            for budget in (args.cache_budget_bytes, args.cache_budget_bytes + duplicate)
            for batch in (8, 16, 32)
            for block in (256, 512, 1024, 2048, 4096)
        ],
        "duplicate_MLP_candidate": {
            "potential_payload_saving_bytes": duplicate,
            "basis": "64 layers, separate BF4 gate/up plus packed BF4 gateup, TP4; source shape estimate",
            "measured_saving": False,
            "constraint": "Second budget is conditional on reclaiming the whole duplicate payload without consuming equivalent new buffers; native correctness/performance/allocation checks required",
        },
        "union_backing_design_screen": [
            union_screen(c, args.cache_budget_bytes, block) for block in (512, 1024, 2048, 4096)
        ],
        "limits": [
            "Virtual hook units are an inversion of pool sizing, not guaranteed usable conversation tokens.",
            "All attention/recurrent groups draw distinct IDs from one allocator; idle cached prefixes also consume reusable identities.",
            "32-history rows screen independent equal-length histories plus known worker reservations; they are not scheduler simulation or retention guarantees.",
            "Scratch, traces, weights, bank alignment and fragmentation must each fit every device.",
            "Current precision is BFP8 KV, FP32 recurrent state, BF16 convolution; no quantization change assumed.",
            "Harbor task permits, HTTP requests, scheduler admission and physical resident rows are separate quantities.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Wrote",
        args.output,
        "with",
        len(report["forecasts"]),
        "forecasts and",
        len(report["measured"]),
        "measured ledgers",
    )


if __name__ == "__main__":
    main()
