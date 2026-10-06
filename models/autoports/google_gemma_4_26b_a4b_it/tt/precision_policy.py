# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strict, serializable construction policy for the accepted TP4/EP4 model."""

import copy
import json
from pathlib import Path

SELECTED_CONFIG = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"
WEIGHT_DTYPES = ("bfloat4_b", "bfloat8_b", "bfloat16")
FIDELITIES = ("LoFi", "HiFi2", "HiFi4")


def _baseline_layer(sliding):
    return {
        "qkv_weight_dtype": "bfloat8_b" if sliding else "bfloat4_b",
        "qkv_fidelity": "LoFi",
        "output_weight_dtype": "bfloat8_b",
        "output_fidelity": "LoFi",
        "expert_gate_dtype": "bfloat8_b" if sliding else "bfloat4_b",
        "expert_down_dtype": "bfloat4_b",
        "expert_fidelity": "LoFi",
        "shared_gate_dtype": "bfloat4_b",
        "shared_down_dtype": "bfloat8_b" if sliding else "bfloat4_b",
        "shared_fidelity": "LoFi",
        "qkv_input_dtype": "float32",
        "output_input_dtype": "bfloat16",
        "expert_input_dtype": "bfloat8_b",
        "shared_input_dtype": "bfloat16",
        "attention_ccl_dtype": "bfloat16" if sliding else "bfloat8_b",
        "moe_ccl_dtype": "bfloat8_b" if sliding else "bfloat16",
        "kv_cache_dtype": "bfloat8_b",
        "fixed": {
            "prefill_qkv_weight_dtype": "bfloat8_b",
            "prefill_qkv_fidelity": "HiFi4",
            "prefill_output_weight_dtype": "bfloat8_b",
            "prefill_output_fidelity": "LoFi",
            "prefill_expert_gate_dtype": "bfloat8_b" if sliding else "bfloat4_b",
            "prefill_expert_down_dtype": "bfloat4_b",
            "prefill_expert_fidelity": "LoFi",
            "prefill_shared_weight_dtype": "bfloat16",
            "prefill_shared_fidelity": "library_default",
            "router_weight_dtype": "bfloat16",
            "router_fidelity": "HiFi4" if sliding else "LoFi",
            "norm_weight_dtype": "float32",
            "norm_fidelity": "HiFi4",
            "decode_sdpa_fidelity": "HiFi4" if sliding else "LoFi",
            "prefill_sdpa_fidelity": "LoFi" if sliding else "HiFi2",
            "expert_mix_fidelity": "HiFi4",
            "expert_mix_fp32_dest_acc_en": sliding,
            "qkv_fp32_dest_acc_en": True,
            "output_fp32_dest_acc_en": True,
            "expert_fp32_dest_acc_en": False,
            "shared_fp32_dest_acc_en": False,
            "qkv_packer_l1_acc": False,
            "output_packer_l1_acc": False,
            "expert_packer_l1_acc": False,
            "shared_packer_l1_acc": True,
            "math_approx_mode": False,
        },
    }


def baseline_precision_config():
    """Return a fresh complete policy matching the pre-sweep construction path."""
    return {
        "schema_version": 1,
        "config_id": "baseline",
        "model": {
            "head_weight_dtype": "bfloat16",
            "head_fidelity": "HiFi4",
            "fixed": {
                "embedding_dtype": "bfloat16",
                "final_norm_weight_dtype": "float32",
                "logits_dtype": "bfloat16",
                "sampling_input_dtype": "bfloat16",
                "head_input_dtype": "bfloat16",
                "head_fp32_dest_acc_en": True,
                "head_packer_l1_acc": True,
                "head_math_approx_mode": False,
                "activation_dtype": "bfloat16",
                "residual_dtype": "bfloat16",
                "post_attention_residual_dtype": "float32",
                "page_size": 32,
                "cache_read_alignment": 128,
                "prefill_expert_parallel": 4,
                "decode_tensor_parallel": 4,
                "collective_topology": "Linear",
            },
        },
        "layer_types": {
            kind: _baseline_layer(kind == "sliding_attention") for kind in ("sliding_attention", "full_attention")
        },
        "layer_overrides": {},
    }


def _merge_known(base, override, path):
    if not isinstance(override, dict):
        raise ValueError(f"{path} must be an object")
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown precision field {path}.{key}")
        if isinstance(base[key], dict):
            _merge_known(base[key], value, f"{path}.{key}")
        else:
            base[key] = value


def _fixed(actual, expected, path):
    if actual != expected:
        differing = [key for key in expected if actual.get(key) != expected[key]]
        raise ValueError(f"Unsupported fixed precision fields at {path}: {differing}")


def _validate_layer(layer, kind):
    baseline = _baseline_layer(kind == "sliding_attention")
    for key in (
        "qkv_weight_dtype",
        "output_weight_dtype",
        "expert_gate_dtype",
        "expert_down_dtype",
        "shared_gate_dtype",
        "shared_down_dtype",
    ):
        if layer[key] not in WEIGHT_DTYPES:
            raise ValueError(f"Unsupported {key}: {layer[key]}")
    for key in ("qkv_fidelity", "output_fidelity", "expert_fidelity", "shared_fidelity"):
        if layer[key] not in FIDELITIES:
            raise ValueError(f"Unsupported {key}: {layer[key]}")
    for key in ("qkv_input_dtype", "output_input_dtype", "expert_input_dtype", "shared_input_dtype"):
        original = "float32" if key == "qkv_input_dtype" else "bfloat16"
        if layer[key] not in (original, "bfloat8_b"):
            raise ValueError(f"Unsupported {key}: {layer[key]}")
    for key in ("moe_ccl_dtype", "kv_cache_dtype"):
        if layer[key] not in ("bfloat16", "bfloat8_b"):
            raise ValueError(f"Unsupported {key}: {layer[key]}")
    if layer["attention_ccl_dtype"] not in ("float32", "bfloat16", "bfloat8_b"):
        raise ValueError("Attention CCL must use FP32, BF16 or BFP8")
    _fixed(layer["fixed"], baseline["fixed"], f"{kind}.fixed")


def resolve_precision_config(value=None):
    """Read a path/dict; None selects the committed artifact when it exists."""
    if value is None:
        value = SELECTED_CONFIG if SELECTED_CONFIG.exists() else {}
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, dict):
        raise ValueError("precision_config must be a JSON path or dictionary")
    value = copy.deepcopy(value)
    overrides = value.pop("layer_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("layer_overrides must be an object keyed by layer index")
    result = baseline_precision_config()
    _merge_known(result, value, "precision")
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("Unsupported precision schema version")
    if not isinstance(result["config_id"], str) or not result["config_id"]:
        raise ValueError("config_id must be a nonempty string")
    model = result["model"]
    if model["head_weight_dtype"] not in WEIGHT_DTYPES or model["head_fidelity"] not in FIDELITIES:
        raise ValueError("Unsupported head weight dtype or fidelity")
    _fixed(model["fixed"], baseline_precision_config()["model"]["fixed"], "model.fixed")
    for kind, layer in result["layer_types"].items():
        _validate_layer(layer, kind)
    for index, override in overrides.items():
        if (
            not isinstance(index, str)
            or not index.isdigit()
            or str(int(index)) != index
            or not isinstance(override, dict)
        ):
            raise ValueError("Layer overrides require nonnegative canonical string indices and object values")
        # Validate all keys immediately; kind-specific fixed values are checked on resolution.
        _merge_known(copy.deepcopy(result["layer_types"]["sliding_attention"]), override, f"layer_overrides.{index}")
    result["layer_overrides"] = overrides
    return result


def layer_precision_config(config, index, kind):
    layer = copy.deepcopy(config["layer_types"][kind])
    _merge_known(layer, config["layer_overrides"].get(str(index), {}), f"layer_overrides.{index}")
    _validate_layer(layer, kind)
    return layer


def dtype(value):
    import ttnn

    return getattr(ttnn, value)


def fidelity(value):
    import ttnn

    return getattr(ttnn.MathFidelity, value)


def dtype_name(value):
    for name in (*WEIGHT_DTYPES, "float32"):
        if value == dtype(name):
            return name
    raise ValueError(f"Unrecognized runtime dtype {value}")


def fidelity_name(value):
    for name in FIDELITIES:
        if value == fidelity(name):
            return name
    raise ValueError(f"Unrecognized runtime fidelity {value}")


def assert_precision_matches(actual, expected, path="precision"):
    """Compare actual tensor/kernel attributes with the resolved policy."""
    for key, value in expected.items():
        if isinstance(value, dict):
            assert_precision_matches(actual[key], value, f"{path}.{key}")
        elif actual[key] != value:
            raise ValueError(f"Precision mismatch {path}.{key}: requested {value!r}, runtime {actual[key]!r}")
