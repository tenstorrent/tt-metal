# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-derived per-device payload deltas at the unchanged maximum context.

This is a conservative capacity calculation, not an allocator peak measurement.
The retained weight graph follows MultichipDecoder's hybrid EP-prefill/TP-decode
path. Same-dtype QKV/output aliases and unused TP-prefill expert copies matter.
"""

import argparse
import json
import math
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tt.precision_policy import (
    baseline_precision_config,
    layer_precision_config,
    resolve_precision_config,
)

ROOT = Path(__file__).resolve().parents[1]
TILE_BYTES = {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048, "float32": 4096}
BASELINE_PEAK = 28_467_973_120
BASELINE_DECODER_WEIGHTS = 10_986_081_280
BASELINE_KV = 8_556_380_160
COMPARISON_CAPACITY = 32_000_000_000
RESERVE = 2_147_483_648
ROPE = 1_610_612_736
NORM_ROUTER_POSITION_ALLOWANCE_PER_LAYER = 8_388_608


def tile_payload(shape, dtype):
    """All supplied local matrix/cache dimensions are physical tile multiples."""
    if len(shape) < 2 or any(dim <= 0 for dim in shape) or any(dim % 32 for dim in shape[-2:]):
        raise ValueError(f"Expected positive tile-aligned physical shape, got {shape}")
    return math.prod(shape) // 1024 * TILE_BYTES[dtype]


def layer_inventory(policy, kind, config):
    """Count live local tensor payloads once, including retained unused copies."""
    sliding = kind == "sliding_attention"
    hidden, experts = config["hidden_size"], config["num_experts"]
    head_dim = config["head_dim"] if sliding else config["global_head_dim"]
    local_heads = config["num_attention_heads"] // 4
    kv_heads = config["num_key_value_heads"] if sliding else config["num_global_key_value_heads"]
    local_kv_heads = max(1, kv_heads // 4)
    local_expert = (config["moe_intermediate_size"] // 4 + 31) // 32 * 32
    local_shared = (config["intermediate_size"] // 4 + 31) // 32 * 32
    shapes = {
        "qkv": (hidden, (local_heads + 2 * local_kv_heads) * head_dim),
        "output": (local_heads * head_dim, hidden),
        "tp_gate_up": (experts, hidden, 2 * local_expert),
        "tp_down": (experts, local_expert, hidden),
        "ep_gate_up": (experts // 4, hidden, 2 * config["moe_intermediate_size"]),
        "ep_down": (experts // 4, config["moe_intermediate_size"], hidden),
        "shared_gate_up": (hidden, 2 * local_shared),
        "shared_down": (local_shared, hidden),
    }

    def size(group, dtype):
        return tile_payload(shapes[group], dtype)

    fixed = policy["fixed"]
    qkv_alias = policy["qkv_weight_dtype"] == fixed["prefill_qkv_weight_dtype"]
    output_alias = policy["output_weight_dtype"] == fixed["prefill_output_weight_dtype"]
    # Sliding's raw checkpoint upload resets the unused prefill_gate alias.
    # Full TP decode retains a separate BFP4 prefill_gate on upward recovery.
    tp_gate_alias = sliding or policy["expert_gate_dtype"] == "bfloat4_b"
    tp_down_alias = policy["expert_down_dtype"] == "bfloat4_b"
    groups = {
        "qkv_prefill": size("qkv", fixed["prefill_qkv_weight_dtype"]),
        "qkv_decode_extra": 0 if qkv_alias else size("qkv", policy["qkv_weight_dtype"]),
        "output_prefill": size("output", fixed["prefill_output_weight_dtype"]),
        "output_decode_extra": 0 if output_alias else size("output", policy["output_weight_dtype"]),
        "tp_expert_gate_up": size("tp_gate_up", policy["expert_gate_dtype"]),
        "tp_expert_down": size("tp_down", policy["expert_down_dtype"]),
        "tp_unused_prefill_gate_up": 0 if tp_gate_alias else size("tp_gate_up", "bfloat4_b"),
        "tp_unused_prefill_down": 0 if tp_down_alias else size("tp_down", "bfloat4_b"),
        "ep_prefill_gate_up": size("ep_gate_up", fixed["prefill_expert_gate_dtype"]),
        "ep_prefill_down": size("ep_down", fixed["prefill_expert_down_dtype"]),
        "shared_prefill_gate_up": size("shared_gate_up", "bfloat16"),
        "shared_prefill_down": size("shared_down", "bfloat16"),
        "shared_decode_gate_up": size("shared_gate_up", policy["shared_gate_dtype"]),
        "shared_decode_down": size("shared_down", policy["shared_down_dtype"]),
        "norm_router_position_allowance": NORM_ROUTER_POSITION_ALLOWANCE_PER_LAYER,
    }
    pages = (config["max_position_embeddings"] + 127) // 128 * 4
    cache_shape = (pages, local_kv_heads, 32, head_dim)
    return {
        "local_weight_shapes": {name: list(shape) for name, shape in shapes.items()},
        "weight_and_small_state_bytes": groups,
        "weight_and_small_state_total_bytes": sum(groups.values()),
        "kv_cache_pair_shape": [list(cache_shape)] * 2,
        "kv_cache_bytes": 2 * tile_payload(cache_shape, policy["kv_cache_dtype"]),
        "aliases": {
            "qkv_prefill_decode": qkv_alias,
            "output_prefill_decode": output_alias,
            "tp_prefill_gate_decode": tp_gate_alias,
            "tp_prefill_down_decode": tp_down_alias,
            "ep_prefill_decode": True,
            "shared_prefill_decode": False,
        },
    }


def _totals(policy, config):
    layers = {}
    ccl_roles = set()
    for index, kind in enumerate(config["layer_types"]):
        layer = layer_precision_config(policy, index, kind)
        layers[str(index)] = layer_inventory(layer, kind, config)
        ccl_roles.add((1, layer["attention_ccl_dtype"]))
        ccl_roles.add((2, layer["moe_ccl_dtype"]))
    head_shape = (config["hidden_size"], config["vocab_size"] // 4)
    # Linear pool: double full input staging, H/4 RS output, full H AG output.
    tiles_per_plane = config["hidden_size"] // 32 * 3 + config["hidden_size"] // 4 // 32
    return {
        "layers": layers,
        "decoder_weight_bytes": sum(layer["weight_and_small_state_total_bytes"] for layer in layers.values()),
        "cache_bytes": sum(layer["kv_cache_bytes"] for layer in layers.values()),
        "head_bytes": tile_payload(head_shape, policy["model"]["head_weight_dtype"]),
        "embedding_bytes": math.prod(head_shape) * 2,
        "pooled_ccl_l1_payload_bytes": sum(planes * tiles_per_plane * TILE_BYTES[dtype] for planes, dtype in ccl_roles),
    }


def precision_memory(precision_config):
    config = json.loads((ROOT / "tests/config.json").read_text())["text_config"]
    if config["max_position_embeddings"] != 262144 or config["num_hidden_layers"] != 30:
        raise ValueError("Re-audit the inherited baseline bound for this architecture/context")
    policy = resolve_precision_config(precision_config)
    baseline = _totals(baseline_precision_config(), config)
    selected = _totals(policy, config)
    if baseline["decoder_weight_bytes"] != BASELINE_DECODER_WEIGHTS or baseline["cache_bytes"] != BASELINE_KV:
        raise ValueError("Source geometry no longer reconciles with the accepted memory ledger")
    table = ((config["max_position_embeddings"] + 127) // 128 * 4) * 4
    long_buffers = 3 * config["max_position_embeddings"] * config["hidden_size"] * 2

    def peak(total):
        return (
            total["decoder_weight_bytes"]
            + total["cache_bytes"]
            + total["head_bytes"]
            + total["embedding_bytes"]
            + ROPE
            + table
            + RESERVE
            + long_buffers
        )

    if peak(baseline) != BASELINE_PEAK:
        raise ValueError("Inherited full-model bound does not reconcile")
    weight_delta = selected["decoder_weight_bytes"] - baseline["decoder_weight_bytes"]
    head_delta = selected["head_bytes"] - baseline["head_bytes"]
    cache_delta = selected["cache_bytes"] - baseline["cache_bytes"]
    selected_peak = peak(selected)
    layer_groups = {}
    for index, inventory in selected["layers"].items():
        signature = json.dumps(inventory, sort_keys=True)
        if signature not in layer_groups:
            layer_groups[signature] = {"layer_indices": [], "per_layer": inventory}
        layer_groups[signature]["layer_indices"].append(int(index))
    return {
        "config_id": policy["config_id"],
        "supported_context": config["max_position_embeddings"],
        "batch_slots": 1,
        "capability_reduction": False,
        "basis": "Source-derived tensor payloads plus inherited 2GiB reserve; not measured allocator peak",
        "baseline_conservative_peak_bound_bytes_per_device": BASELINE_PEAK,
        "decoder_weight_delta_bytes_per_device": weight_delta,
        "head_weight_delta_bytes_per_device": head_delta,
        "cache_delta_bytes_per_device": cache_delta,
        "total_dram_delta_bytes_per_device": weight_delta + head_delta + cache_delta,
        "conservative_peak_bound_bytes_per_device": selected_peak,
        "resident_bound_before_reserve_and_long_prefill_bytes_per_device": selected_peak - RESERVE - long_buffers,
        "reserve_bytes_per_device": RESERVE,
        "long_prefill_live_buffers_bytes_per_device": long_buffers,
        "comparison_capacity_decimal_bytes_per_device": COMPARISON_CAPACITY,
        "headroom_to_comparison_capacity_bytes_per_device": COMPARISON_CAPACITY - selected_peak,
        "fits_inherited_conservative_budget": selected_peak <= COMPARISON_CAPACITY,
        "no_increase_in_dram_payload": selected_peak <= BASELINE_PEAK,
        "pooled_ccl_l1_payload_bytes_per_device": selected["pooled_ccl_l1_payload_bytes"],
        "pooled_ccl_l1_delta_bytes_per_device": selected["pooled_ccl_l1_payload_bytes"]
        - baseline["pooled_ccl_l1_payload_bytes"],
        "inventory": {
            **{key: value for key, value in selected.items() if key != "layers"},
            "layer_groups": list(layer_groups.values()),
        },
        "limitations": [
            "An upper bound above physical capacity fails this budget; it does not prove allocation is impossible.",
            "Full-model maximum-context execution validates fragmentation and unmodeled transient behavior.",
            "B1 maximum context does not promise 32 simultaneous maximum-length requests.",
            "Dtype-dependent matmul CBs, cast temporaries, trace storage and allocator padding remain in the inherited reserve.",
            "L1 CCL payload excludes bank alignment and semaphore allocation; L1 is separate from DRAM.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {path.stem: precision_memory(path) for path in args.configs}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
