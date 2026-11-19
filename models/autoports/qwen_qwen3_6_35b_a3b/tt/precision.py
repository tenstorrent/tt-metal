# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validated construction policy; no device imports are needed to resolve it."""

import copy
import json
from pathlib import Path

SELECTED_PATH = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"
GROUPS = (
    "expert_gu",
    "expert_down",
    "expert_ep",
    "attention_input",
    "attention_output",
    "shared_packed",
    "shared_down",
    "lm_head",
)
FIXED = {
    "activation_dtype": "bf16",
    "residual_dtype": "bf16",
    "logits_dtype": "bf16",
    "sampling_dtype": "bf16",
    "embedding_dtype": "bf16",
    "linear_attention_weight_dtype": "bf16",
    "linear_attention_arithmetic": "ordered_fp32",
    "router_weight_dtype": "fp32_native_bf16_ordered_scan",
    "router_logits_dtype": "fp32",
    "routing_scores_dtype": "bf16",
    "recurrent_state_dtype": "fp32",
    "convolution_state_dtype": "bf16",
    "norm_weight_dtype": "fp32_native_bf16_sharded_helpers",
    "norm_fidelity": "HiFi4_layer_HiFi2_terminal",
    "kv_layout": "TILE",
    "shared_scalar_dtype": "bfp8",
    "shared_scalar_fidelity": "HiFi2",
}


def _compute(fidelity, fp32=True):
    return dict(math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=False, packer_l1_acc=False)


def baseline_policy():
    return dict(
        config_id="baseline_bfp8",
        schema_version=1,
        weight_groups={group: "bfp8" for group in GROUPS},
        compute_fidelities={
            group: _compute("LoFi" if group.startswith("expert") else "HiFi2", group != "expert_ep") for group in GROUPS
        },
        layer_exceptions=[],
        ccl_dtype="bf16",
        kv_cache_dtype="bf16",
        **FIXED,
    )


def _validate_groups(weights, compute, *, partial=False):
    expected = set(GROUPS)
    if set(weights) - expected or set(compute) - expected:
        raise ValueError("Unknown weight or compute group")
    if not partial and (set(weights) != expected or set(compute) != expected):
        raise ValueError("Every material weight and compute group must be specified")
    for dtype in weights.values():
        if dtype not in ("bf16", "bfp8", "bfp4"):
            raise ValueError(f"Unsupported weight dtype: {dtype}")
    for spec in compute.values():
        if not isinstance(spec, dict) or set(spec) != set(_compute("LoFi")):
            raise ValueError(
                "Each compute group requires math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc"
            )
        if spec["math_fidelity"] not in ("LoFi", "HiFi2", "HiFi4"):
            raise ValueError("Unsupported math fidelity")
        if any(type(spec[k]) is not bool for k in spec if k != "math_fidelity"):
            raise ValueError("Compute flags must be booleans")
    ep = compute.get("expert_ep", {})
    if ep.get("fp32_dest_acc_en") or ep.get("packer_l1_acc"):
        raise ValueError("Grouped EP requires BF16 destination accumulation and packer_l1_acc=False")


def resolve_precision(config=None):
    """None consumes the selected artifact, or baseline before selection.

    An explicit ``'baseline'`` always bypasses the selected artifact. Explicit
    mappings/files are complete policies: misspelled or ignored fields fail.
    """
    if config is None:
        config = SELECTED_PATH if SELECTED_PATH.exists() else "baseline"
    if isinstance(config, (str, Path)):
        config = baseline_policy() if str(config) == "baseline" else json.loads(Path(config).read_text())
    policy = copy.deepcopy(config)
    if not isinstance(policy, dict) or set(policy) != set(baseline_policy()):
        raise ValueError("Precision policy must contain exactly the complete baseline schema")
    if policy["schema_version"] != 1 or not isinstance(policy["config_id"], str) or not policy["config_id"]:
        raise ValueError("Invalid precision schema/version/id")
    for key, expected in FIXED.items():
        if policy[key] != expected:
            raise ValueError(f"{key} requires {expected}; unsupported precision boundary")
    if policy["ccl_dtype"] not in ("bf16", "bfp8") or policy["kv_cache_dtype"] not in ("bf16", "bfp8"):
        raise ValueError("CCL/KV supports bf16 or bfp8")
    _validate_groups(policy["weight_groups"], policy["compute_fidelities"])
    seen = set()
    if not isinstance(policy["layer_exceptions"], list):
        raise ValueError("layer_exceptions must be a list")
    for exception in policy["layer_exceptions"]:
        if set(exception) != {"layers", "weight_groups", "compute_fidelities"}:
            raise ValueError("Layer exception needs layers/weight_groups/compute_fidelities")
        if not isinstance(exception["layers"], list) or not exception["layers"]:
            raise ValueError("Exception layers must be nonempty")
        for layer in exception["layers"]:
            if type(layer) is not int or not 0 <= layer < 40 or layer in seen:
                raise ValueError("Exception layer must be unique and in 0..39")
            seen.add(layer)
        if "lm_head" in exception["weight_groups"] or "lm_head" in exception["compute_fidelities"]:
            raise ValueError("LM head is not a decoder-layer group")
        _validate_groups(exception["weight_groups"], exception["compute_fidelities"], partial=True)
    return policy


def layer_precision(policy, index):
    result = copy.deepcopy(policy)
    for exception in policy["layer_exceptions"]:
        if index in exception["layers"]:
            result["weight_groups"].update(exception["weight_groups"])
            result["compute_fidelities"].update(exception["compute_fidelities"])
    result["layer_exceptions"] = []
    return result


def device_dtype(name):
    import ttnn

    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}[name]


def device_compute(mesh, spec):
    import ttnn

    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, spec["math_fidelity"]),
        **{k: v for k, v in spec.items() if k != "math_fidelity"},
    )


def actual_compute(config):
    return {
        key: str(getattr(config, key)) if key == "math_fidelity" else getattr(config, key) for key in _compute("LoFi")
    }


def assert_compute(config, spec):
    import ttnn

    assert config.math_fidelity == getattr(ttnn.MathFidelity, spec["math_fidelity"])
    for key, value in spec.items():
        if key != "math_fidelity":
            assert getattr(config, key) == value, (key, getattr(config, key), value)
