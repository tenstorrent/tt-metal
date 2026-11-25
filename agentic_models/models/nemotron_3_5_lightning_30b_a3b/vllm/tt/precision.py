# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validated precision construction contract shared by every full-model caller."""

import copy
import json
from pathlib import Path

ROLES = ("qkv", "o", "in_proj", "out_proj", "up", "down", "shared_up", "shared_down", "lm_head")
MODEL_DIR = Path(__file__).resolve().parents[1]
SELECTED = MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json"


def baseline():
    return {
        "config_id": "bfp8_lofi_baseline",
        "weight_groups": {role: "bfloat8_b" for role in ROLES},
        "compute_fidelities": {role: "HiFi4" if role == "lm_head" else "LoFi" for role in ROLES},
        "layer_exceptions": {},
        "activation_dtype": "bfloat16",
        "residual_dtype": "bfloat16",
        "ccl_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat8_b",
        "logits_dtype": "bfloat16",
        "sampling_dtype": "bfloat16",
        "norm_dtype": "bfloat16",
        "router_dtype": "float32",
        "recurrent_dtype": "float32",
        "fp32_dest_acc_en": True,
        "math_approx_mode": False,
    }


def load_precision(config=None):
    if config is None:
        import os

        # perf/multiuser: optional override for precision experiments (default: the selected config).
        override = os.environ.get("NEMOTRON_PRECISION_CONFIG")
        config = override if override else (SELECTED if SELECTED.exists() else baseline())
    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.exists():
            path = MODEL_DIR / path
        config = json.loads(path.read_text())
    result = copy.deepcopy(config)
    if set(result) != set(baseline()):
        raise ValueError("Precision config must contain exactly the complete construction contract")
    for field in ("weight_groups", "compute_fidelities"):
        if set(result[field]) != set(ROLES):
            raise ValueError(f"{field} must specify every projection group")
    for role in ROLES:
        _validate_group(result["weight_groups"][role], result["compute_fidelities"][role])
    valid_layer_keys = {str(index) for index in range(52)}
    for layer, overrides in result["layer_exceptions"].items():
        if layer not in valid_layer_keys:
            raise ValueError("Layer exception keys must be canonical decimal strings from '0' to '51'")
        if set(overrides) - {"weight_groups", "compute_fidelities"}:
            raise ValueError("Invalid precision layer exception")
        for field, groups in overrides.items():
            if "lm_head" in groups:
                raise ValueError("lm_head is global and cannot have a layer exception")
            if set(groups) - set(ROLES):
                raise ValueError("Unknown exception projection")
        for role in ROLES:
            _validate_group(
                overrides.get("weight_groups", {}).get(role, result["weight_groups"][role]),
                overrides.get("compute_fidelities", {}).get(role, result["compute_fidelities"][role]),
            )
    for field in ("activation_dtype", "residual_dtype", "ccl_dtype", "kv_cache_dtype"):
        if result[field] not in ("bfloat16", "bfloat8_b"):
            raise ValueError(f"Unsupported {field}")
    # These sensitive paths have a single supported representation. Reject an
    # unsupported declaration instead of silently ignoring a policy field.
    for field in (
        "logits_dtype",
        "sampling_dtype",
        "norm_dtype",
        "router_dtype",
        "recurrent_dtype",
        "fp32_dest_acc_en",
        "math_approx_mode",
    ):
        if result[field] != baseline()[field]:
            raise ValueError(f"Unsupported {field}: {result[field]}")
    return result


def _validate_group(dtype, fidelity):
    if dtype not in ("bfloat16", "bfloat8_b", "bfloat4_b") or fidelity not in ("LoFi", "HiFi2", "HiFi4"):
        raise ValueError("Unsupported projection dtype/fidelity")


def layer_precision(config, index):
    """Resolve family-scoped defaults for the projections present in this layer.

    Exceptions may name decoder projection families absent from a layer; those
    entries are inapplicable there. This permits common first/last-layer policies
    without encoding model topology in the loader. The LM head is global and
    only consumes the top-level policy, so layer-local head overrides are invalid.
    """
    result = copy.deepcopy(config)
    for field, values in config["layer_exceptions"].get(str(index), {}).items():
        result[field].update(values)
    return result
