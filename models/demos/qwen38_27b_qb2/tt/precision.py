# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serializable full-model precision policy, shared by all construction callers."""

import copy
import json
import os
from pathlib import Path

ROLES = ("attention", "output", "gate", "up", "down")
BASELINE = {
    "config_id": "baseline_bfp4_lofi_head_bfp8_hifi2",
    "weight_groups": {**{r: "bfloat4_b" for r in ROLES}, "head": "bfloat8_b"},
    "compute_fidelities": {**{r: "LoFi" for r in ROLES}, "head": "HiFi2"},
    "fp32_dest_acc_en": True,
    "layer_exceptions": {},
    "activation_dtype": {"attention": "bfloat16", "mlp": "bfloat16"},
    "residual_dtype": "bfloat16",
    "ccl_dtype": "bfloat16",
    "kv_cache_dtype": "bfloat8_b",
    "logits_dtype": "bfloat16",
    "sampling_dtype": "bfloat16",
    "norm_dtype": "bfloat16",
    "embedding_dtype": "bfloat16",
    "recurrent_dtype": "float32",
    "convolution_dtype": "bfloat16",
    "sensitive_compute_fidelity": "HiFi4",
    "final_norm_compute_fidelity": "HiFi2",
    "token_dtype": "uint32",
    "max_context": 262144,
}


def load_precision(value=None):
    """Explicit override, environment override, selected artifact, safe baseline."""
    if value is None:
        value = os.environ.get("QWEN_PRECISION_CONFIG")
    if value is None:
        selected = Path(__file__).resolve().parents[1] / "config/precision.json"
        value = selected if selected.exists() else "baseline"
    if value == "baseline":
        policy = copy.deepcopy(BASELINE)
    elif isinstance(value, dict):
        policy = copy.deepcopy(value)
    else:
        policy = json.loads(Path(value).read_text())
    if set(policy) != set(BASELINE):
        raise ValueError("Precision policy must contain exactly the supported fields")
    # These are explicit runtime contracts of the native norm/GDN/sampler and
    # replicated residual path. Reject unsupported requests instead of ignoring them.
    for key in (
        "residual_dtype",
        "sampling_dtype",
        "norm_dtype",
        "embedding_dtype",
        "recurrent_dtype",
        "convolution_dtype",
        "sensitive_compute_fidelity",
        "final_norm_compute_fidelity",
        "token_dtype",
        "max_context",
    ):
        if policy[key] != BASELINE[key]:
            raise ValueError(f"Unsupported {key}: {policy[key]}")
    if policy["logits_dtype"] != policy["sampling_dtype"]:
        raise ValueError("Logits must match the native sampler dtype")
    for groups in (policy["weight_groups"], policy["compute_fidelities"]):
        if set(groups) != {*ROLES, "head"}:
            raise ValueError("Missing or unknown projection group")
    if set(policy["activation_dtype"]) != {"attention", "mlp"}:
        raise ValueError("Activation policy requires attention and mlp groups")
    for layer, exception in policy["layer_exceptions"].items():
        if not isinstance(layer, str) or str(int(layer)) != layer:
            raise ValueError('Layer exceptions require canonical string indices, such as "0"')
        if not 0 <= int(layer) < 64 or set(exception) - {"weight_groups", "compute_fidelities"}:
            raise ValueError("Invalid layer exception")
        for groups in exception.values():
            if set(groups) - set(ROLES):
                raise ValueError("Layer exceptions must name decoder projection groups")
    for layer in range(64):
        p = decoder_policy(policy, layer)
        if p["gate_dtype"] != p["up_dtype"] or p["gate_fidelity"] != p["up_fidelity"]:
            raise ValueError("Packed gate/up must use identical precision")
    return policy


def decoder_policy(policy, layer):
    groups = dict(policy["weight_groups"])
    fidelities = dict(policy["compute_fidelities"])
    exception = policy["layer_exceptions"].get(str(layer), {})
    groups.update(exception.get("weight_groups", {}))
    fidelities.update(exception.get("compute_fidelities", {}))
    return {
        **{r + "_dtype": groups[r] for r in ROLES},
        **{r + "_fidelity": fidelities[r] for r in ROLES},
        **{r + "_fp32": policy["fp32_dest_acc_en"] for r in ROLES},
        **{r + "_activation": d for r, d in policy["activation_dtype"].items()},
        "residual_dtype": policy["residual_dtype"],
        "ccl_dtype": policy["ccl_dtype"],
        "kv_dtype": policy["kv_cache_dtype"],
    }
