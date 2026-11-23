# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serializable full-model precision contract for the TP4 model."""

import copy
import json
import os
from pathlib import Path

ROLES = ("qkv", "o", "gate", "up", "down")
SELECTED = Path(__file__).with_name("precision_config.json")


def load_precision(config=None):
    config = config or os.environ.get("GRANITE_PRECISION_CONFIG")
    if isinstance(config, dict):
        p = copy.deepcopy(config)
    elif config:
        p = json.loads(Path(config).read_text())
    elif SELECTED.exists():
        p = json.loads(SELECTED.read_text())
    else:
        raise FileNotFoundError(f"Missing required selected precision artifact: {SELECTED}")
    # These contracts require BF16/UINT32 at the current operator boundaries.
    # Reject unsupported values instead of silently recording an ignored policy.
    for key, value in dict(
        activation_dtype="bfloat16",
        residual_dtype="bfloat16",
        norm_dtype="bfloat16",
        logits_dtype="bfloat16",
        sampling_dtype="bfloat16",
        token_dtype="uint32",
        supported_context=131072,
    ).items():
        if p[key] != value:
            raise ValueError(f"{key} must be {value}, got {p[key]}")
    if p["weight_groups"]["embedding"] != "bfloat16":
        raise ValueError("row-major embedding requires BF16")
    if p["decode_matmul_input_dtype"] not in ("bfloat16", "bfloat8_b"):
        raise ValueError("decode matmul inputs support BF16 or BFP8")
    if p["ccl_dtype"] not in ("bfloat16", "bfloat8_b"):
        raise ValueError("CCL supports BF16 or BFP8")
    if p["kv_cache_dtype"] not in ("bfloat16", "bfloat8_b", "bfloat4_b"):
        raise ValueError("Unsupported KV dtype")
    for role, settings in p.get("matmul_geometry", {}).items():
        if role not in ROLES or set(settings) - {"cores", "readers", "block"}:
            raise ValueError(f"Invalid matmul geometry override: {role}")
        if any(not isinstance(v, int) or v <= 0 for v in settings.values()):
            raise ValueError("Matmul geometry values must be positive integers")
    for layer, exception in p["layer_exceptions"].items():
        if not 0 <= int(layer) < 64 or set(exception) - {"weight_groups", "compute_fidelities"}:
            raise ValueError(f"Invalid layer exception: {layer}")
        if set(exception.get("weight_groups", {})) - set(ROLES):
            raise ValueError("Layer weight exceptions must name decoder projections")
    return p


def layer_policy(p, index):
    from .decoder import multichip_policy

    policy = multichip_policy()
    w = {**p["weight_groups"], **p["layer_exceptions"].get(str(index), {}).get("weight_groups", {})}
    f = {**p["compute_fidelities"], **p["layer_exceptions"].get(str(index), {}).get("compute_fidelities", {})}
    for r in (*ROLES, "norm", "sdpa", "sdpa_prefill"):
        if r in ROLES:
            policy[r + "_dtype"] = w[r]
        policy[r + "_fidelity"] = f[r]
        policy[r + "_fp32"] = p["fp32_accumulation"][r]
    for name in ("activation_dtype", "residual_dtype", "norm_dtype"):
        policy[name] = p[name]
    if any(policy[r + "_fp32"] for r in ROLES):
        policy["prefill_subblock"] = min(policy["prefill_subblock"], 4)
    policy["cache_dtype"] = p["kv_cache_dtype"]
    policy["ccl_dtype"] = p["ccl_dtype"]
    policy["mlp_act8"] = p["decode_matmul_input_dtype"] == "bfloat8_b"
    policy["attention_act8"] = policy["mlp_act8"]
    for role, settings in p.get("matmul_geometry", {}).items():
        policy.update({role + "_" + key: value for key, value in settings.items()})
    return policy
