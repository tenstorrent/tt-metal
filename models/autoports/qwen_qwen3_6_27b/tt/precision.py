# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit, validated precision policy shared by every full-model constructor."""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "doc/datatype_sweep/selected_precision_config.json"
GROUPS = ("mlp_gate_up", "mlp_down", "attention_input", "attention_output")


def group_for_weight(name):
    if name.startswith("mlp."):
        return "mlp_down" if "down_proj" in name else "mlp_gate_up"
    return "attention_output" if "out_proj" in name or "o_proj" in name else "attention_input"


def baseline():
    return dict(
        schema_version=1,
        config_id="bfp8_baseline",
        weight_groups={kind: {g: "bfloat8_b" for g in GROUPS} for kind in ("linear_attention", "full_attention")},
        compute_fidelities={
            "linear_attention": {g: "HiFi2" if g.startswith("mlp") else "HiFi4" for g in GROUPS},
            "full_attention": {g: "LoFi" for g in GROUPS},
        },
        layer_exceptions={},
        activation_dtype="bfloat16",
        residual_dtype="bfloat16",
        mlp_projection_activation_dtype="bfloat16",
        ccl_dtype="bfloat16",
        ccl_scope="explicit residual/projection collectives; fused prefill MMRS and norm statistics retain native dtype",
        kv_cache_dtype="bfloat8_b",
        kv_cache_layout="TILE/page32/local_kv1/head256",
        embedding_dtype="bfloat16",
        lm_head_dtype="bfloat16",
        lm_head_fidelity="HiFi4",
        logits_dtype="bfloat16",
        sampling_dtype="bfloat16",
        norm_weight_dtype="float32",
        gdn_norm_weight_dtype="bfloat16",
        gdn_correction_dtype="bfloat8_b",
        recurrent_state_dtype="bfloat16",
        supported_context=262144,
    )


def load_precision(config=None):
    if config is None:
        config = SELECTED if SELECTED.exists() else baseline()
    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.exists():
            path = ROOT / path
        config = json.loads(path.read_text())
    result = copy.deepcopy(config)
    expected = baseline()
    if set(result) != set(expected):
        raise ValueError(f"Precision fields mismatch: {set(result) ^ set(expected)}")
    for key in (
        "schema_version",
        "activation_dtype",
        "residual_dtype",
        "kv_cache_layout",
        "embedding_dtype",
        "logits_dtype",
        "sampling_dtype",
        "norm_weight_dtype",
        "gdn_norm_weight_dtype",
        "gdn_correction_dtype",
        "recurrent_state_dtype",
        "supported_context",
        "ccl_scope",
    ):
        if result[key] != expected[key]:
            raise ValueError(f"Unsupported precision contract {key}={result[key]}")
    for key in ("ccl_dtype", "mlp_projection_activation_dtype"):
        if result[key] not in ("bfloat16", "bfloat8_b"):
            raise ValueError(key)
    for key in ("kv_cache_dtype", "lm_head_dtype"):
        if result[key] not in ("bfloat16", "bfloat8_b", "bfloat4_b"):
            raise ValueError(key)
    if result["lm_head_fidelity"] not in ("LoFi", "HiFi2", "HiFi4"):
        raise ValueError("lm_head_fidelity")
    for key in ("weight_groups", "compute_fidelities"):
        if set(result[key]) != {"linear_attention", "full_attention"}:
            raise ValueError(f"Unknown layer kind in {key}")
    if set(result["layer_exceptions"]) - {str(i) for i in range(64)}:
        raise ValueError("Layer exception keys must be canonical indices 0..63")
    for index in range(64):
        kind = "full_attention" if index % 4 == 3 else "linear_attention"
        layer_policy(result, kind, index)
    return result


def layer_policy(config, kind, index):
    weights = dict(config["weight_groups"][kind])
    fidelities = dict(config["compute_fidelities"][kind])
    exception = config["layer_exceptions"].get(str(index), {})
    if set(exception) - {"weight_groups", "compute_fidelities"}:
        raise ValueError("Unknown layer exception")
    weights.update(exception.get("weight_groups", {}))
    fidelities.update(exception.get("compute_fidelities", {}))
    if set(weights) != set(GROUPS) or set(fidelities) != set(GROUPS):
        raise ValueError("Precision group mismatch")
    if not all(v in ("bfloat16", "bfloat8_b", "bfloat4_b") for v in weights.values()):
        raise ValueError("Weight dtype")
    if not all(v in ("LoFi", "HiFi2", "HiFi4") for v in fidelities.values()):
        raise ValueError("Compute fidelity")
    policy = dict(
        weight_group_dtypes=weights,
        group_fidelities=fidelities,
        attention_fidelity=fidelities["attention_input"],
        mlp_fidelity=fidelities["mlp_gate_up"],
        mlp_activation_dtype=config["mlp_projection_activation_dtype"],
        attention_activation_dtype=config["activation_dtype"],
        **{
            key: config[key]
            for key in (
                "activation_dtype",
                "residual_dtype",
                "norm_weight_dtype",
                "gdn_norm_weight_dtype",
                "gdn_correction_dtype",
                "recurrent_state_dtype",
            )
        },
        kv_dtype=config["kv_cache_dtype"],
        # None preserves BF16 bulk payloads and FP32 statistics exactly.
        ccl_dtype=None if config["ccl_dtype"] == "bfloat16" else config["ccl_dtype"],
    )
    if kind == "linear_attention" and weights["attention_input"] != "bfloat8_b":
        policy["packed_correction_options"] = None
    return policy
