# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit, validated precision contract shared by all generator constructors."""

import copy
import json
from pathlib import Path

SAFE_POLICY = {
    "config_id": "bfp8_baseline",
    "weight_groups": {
        "attention": "bfloat8_b",
        "expert_gate_up": "bfloat8_b",
        "expert_down": "bfloat8_b",
        "shared_gate_up": "bfloat8_b",
        "shared_down": "bfloat8_b",
        "lm_head": "bfloat8_b",
        "embedding": "bfloat16",
        "norm": "float32",
        "router": "float32",
    },
    "compute_fidelities": {
        "attention_decode": "HiFi2",
        "attention_prefill": "HiFi4",
        "experts": "LoFi",
        "shared": "LoFi",
        "lm_head": "HiFi2",
        "router": "HiFi4",
        "norm_recurrence": "HiFi4",
        "sdpa": "HiFi2",
    },
    "layer_exceptions": [],
    "activation_dtype": "bfloat16",
    "residual_dtype": "bfloat16",
    "ccl_dtype": "bfloat16",
    "kv_cache_dtype": "bfloat8_b",
    "logits_sampling": {"logits_dtype": "bfloat16", "token_dtype": "uint32", "recurrent_state_dtype": "float32"},
}


def load_precision(config=None):
    """Reject ignored fields; a missing explicit selection retains the safe policy."""
    if config is None:
        selected = Path(__file__).parents[1] / "doc/datatype_sweep/selected_precision_config.json"
        config = selected if selected.is_file() else SAFE_POLICY
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    result = copy.deepcopy(config)
    if set(result) != set(SAFE_POLICY):
        raise ValueError("Precision config must contain exactly the complete runtime schema")
    for field in ("weight_groups", "compute_fidelities", "logits_sampling"):
        if set(result[field]) != set(SAFE_POLICY[field]):
            raise ValueError(f"Incomplete or unknown {field}")
    for group, dtype in result["weight_groups"].items():
        allowed = (
            (SAFE_POLICY["weight_groups"][group],)
            if group in ("embedding", "norm", "router")
            else ("bfloat16", "bfloat8_b", "bfloat4_b")
        )
        if dtype not in allowed:
            raise ValueError(f"Unsupported dtype {group}={dtype}")
    for group, fidelity in result["compute_fidelities"].items():
        if fidelity not in ("LoFi", "HiFi2", "HiFi4"):
            raise ValueError(f"Unknown fidelity {group}={fidelity}")
        if group in ("norm_recurrence", "sdpa") and fidelity != SAFE_POLICY["compute_fidelities"][group]:
            raise ValueError(f"{group} has a fixed accuracy-sensitive fidelity")
    for field in ("activation_dtype", "residual_dtype", "ccl_dtype", "kv_cache_dtype"):
        if result[field] not in ("bfloat16", "bfloat8_b"):
            raise ValueError(f"Unsupported {field}")
    if result["logits_sampling"] != SAFE_POLICY["logits_sampling"]:
        raise ValueError("Logits/sampling/recurrent state dtypes are fixed by their op contracts")
    for exception in result["layer_exceptions"]:
        if set(exception) - {"layers", "weight_groups", "compute_fidelities"} or "layers" not in exception:
            raise ValueError("Unknown layer exception field")
        if not exception["layers"] or any(type(i) is not int or not 0 <= i < 40 for i in exception["layers"]):
            raise ValueError("Layer exception requires indices 0..39")
        overridden = copy.deepcopy(result)
        overridden["layer_exceptions"] = []
        for field in ("weight_groups", "compute_fidelities"):
            for key, value in exception.get(field, {}).items():
                if key not in overridden[field] or key in ("lm_head", "embedding", "norm", "router"):
                    raise ValueError(f"Unsupported per-layer override {key}")
                overridden[field][key] = value
        load_precision(overridden)
    return result


def layer_precision(policy, index):
    result = copy.deepcopy(policy)
    for exception in policy["layer_exceptions"]:
        if index in exception["layers"]:
            for field in ("weight_groups", "compute_fidelities"):
                result[field].update(exception.get(field, {}))
    return result
