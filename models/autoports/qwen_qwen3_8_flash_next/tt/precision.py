# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit construction policy; no environment-selected precision or runtime fallback."""

import copy
import json
from pathlib import Path


def inherited_policy():
    return {
        "config_id": "inherited_e4_lofi",
        "weight_groups": {
            "expert_gate": "bfloat4_b",
            "expert_up": "bfloat4_b",
            "expert_down": "bfloat4_b",
            "shared": "bfloat8_b",
            "shared_down": "bfloat8_b",
            "qsa_qkv": "bfloat8_b",
            "qsa_out": "bfloat8_b",
            "gr": "bfloat16",
            "gdn_zab": "bfloat16",
            "lm_head": "bfloat16",
        },
        "compute_fidelities": {
            "experts": "LoFi",
            **{k: "HiFi4" for k in ("shared", "shared_down", "qsa_qkv", "qsa_out", "gr", "gdn_zab", "lm_head")},
        },
        "layer_exceptions": {},
        "activation_dtype": "bfloat16",
        "residual_dtype": "bfloat16",
        "ccl_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "logits_dtype": "bfloat16",
        "sampling_dtype": "bfloat16",
        "fixed_numerics": {
            "hyperconnection_norm_weights": "float32",
            "gdn_output_norm_weights": "bfloat16",
            "router_weights": "bfloat16",
            "router_logits": "float32",
            "router_topk": "float32",
            "routing_weights": "bfloat16",
            "shared_router_weights": "bfloat8_b",
            "gdn_recurrent_state": "float32",
            "hyperconnection_mixer_weights": "float32",
            "gdn_qkv_output_weights": "bfloat16",
            "exact_projection_accumulation": "float32",
            "ple_key_value_weights": "bfloat8_b",
            "ple_embedding": "group10_scale8_lloyd_packed_uint32",
            "token_embedding": "bfloat16",
            "sampling_token_ids": "uint32",
            "activation_dtype_scope": "DRAM projection inputs including LM head; other activations retain BF16/FP32 operation contracts",
        },
        "max_context": 262144,
        "kv_layout": "TILE; page32; TP4 local width256; distinct K and V",
    }


def load_policy(model_dir, value=None):
    if value is None:
        selected = Path(model_dir) / "doc/datatype_sweep/selected_precision_config.json"
        value = selected if selected.exists() else inherited_policy()
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute() and not path.exists():
            path = Path(model_dir) / path
        value = json.loads(path.read_text())
    policy = copy.deepcopy(value)
    expected = inherited_policy()
    if set(policy) != set(expected):
        raise ValueError(f"Precision fields differ: {set(policy) ^ set(expected)}")
    for name in ("weight_groups", "compute_fidelities"):
        if set(policy[name]) != set(expected[name]):
            raise ValueError(f"Incomplete {name}")
    for dtype in policy["weight_groups"].values():
        if dtype not in ("bfloat16", "bfloat8_b", "bfloat4_b"):
            raise ValueError(f"Unsupported weight dtype {dtype}")
    for fidelity in policy["compute_fidelities"].values():
        if fidelity not in ("LoFi", "HiFi2", "HiFi4"):
            raise ValueError(f"Unsupported fidelity {fidelity}")
    for name in ("activation_dtype", "ccl_dtype", "kv_cache_dtype"):
        if policy[name] not in ("bfloat16", "bfloat8_b"):
            raise ValueError(f"Unsupported {name}")
    # Exact kernels and sampling have a fixed public storage contract. Reject
    # attempts to change it instead of recording a silently ignored policy.
    for name in ("residual_dtype", "logits_dtype", "sampling_dtype", "fixed_numerics", "max_context", "kv_layout"):
        if policy[name] != expected[name]:
            raise ValueError(f"Unsupported fixed numerical contract: {name}")
    for layer, override in policy["layer_exceptions"].items():
        if str(int(layer)) != layer or not 0 <= int(layer) < 48:
            raise ValueError(f"Invalid layer exception {layer}")
        if set(override) - {"weight_groups", "compute_fidelities"}:
            raise ValueError("Layer exceptions allow weight groups and fidelities only")
        for group, entries in override.items():
            if "lm_head" in entries:
                raise ValueError("LM head is global and cannot be overridden per layer")
            absent = {"gdn_zab"} if int(layer) % 4 == 3 else {"qsa_qkv", "qsa_out"}
            if set(entries) & absent:
                raise ValueError("Layer override names a projection absent from that layer kind")
            if set(entries) - set(expected[group]):
                raise ValueError(f"Unknown layer group {group}")
            allowed = ("bfloat16", "bfloat8_b", "bfloat4_b") if group == "weight_groups" else ("LoFi", "HiFi2", "HiFi4")
            if any(v not in allowed for v in entries.values()):
                raise ValueError(f"Invalid layer override {entries}")
    for index in range(48):
        resolved = layer_policy(policy, index)["weight_groups"]
        if len({resolved["expert_" + group] for group in ("gate", "up", "down")}) != 1:
            raise ValueError("moe_fused_swiglu requires gate/up/down to share dtype")
    return policy


def layer_policy(policy, index):
    result = copy.deepcopy(policy)
    for name, overrides in policy["layer_exceptions"].get(str(index), {}).items():
        result[name].update(overrides)
    return result


def static_weight_delta(policy):
    """Exact per-rank persistent tiled-weight delta versus inherited policy.

    Dimensions include eight-bank padding in the fixed DRAM projection geometry;
    experts use128local matrices/group, each1600tiles. No runtime workspace or
    trace-memory saving is assumed from precision alone.
    """
    tile_bytes = {"bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576}
    base = inherited_policy()["weight_groups"]
    total = 0
    for index in range(48):
        weights = layer_policy(policy, index)["weight_groups"]
        counts = {"shared": 1280, "shared_down": 400, "gr": 2 * 2560}
        counts.update({"qsa_qkv": 8960, "qsa_out": 3840} if index % 4 == 3 else {"gdn_zab": 4480})
        counts.update({"expert_" + name: 128 * 1600 for name in ("gate", "up", "down")})
        total += sum(count * (tile_bytes[weights[name]] - tile_bytes[base[name]]) for name, count in counts.items())
    return total + 155520 * (tile_bytes[policy["weight_groups"]["lm_head"]] - tile_bytes[base["lm_head"]])
