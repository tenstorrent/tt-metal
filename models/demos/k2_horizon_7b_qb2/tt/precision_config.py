# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Required stage-8 precision artifact, shared by all generator construction paths.

An explicit path (or mapping) overrides the selected artifact. Geometry stays
with the optimized full-model implementation; this file owns numerical policy.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

from .optimized_full_model_policy import optimized_full_model_policy

SELECTED_CONFIG = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"
DTYPES = {"bfloat16", "bfloat8_b", "bfloat4_b"}
FIDELITIES = {"LoFi", "HiFi2", "HiFi4"}
LAYER_FIELDS = {
    "attention",
    "qkv_dtype",
    "o_dtype",
    "mlp",
    "down",
    "kv",
    "attention_fidelity",
    "qkv_fidelity",
    "o_fidelity",
    "mlp_fidelity",
    "down_fidelity",
    "sdpa_fidelity",
    "attention_activation",
    "mlp_activation",
    "prefill_qkv_activation",
    "prefill_mlp_activation",
}
FIXED_DTYPES = {
    "embedding": "bfloat16",
    "norm": "bfloat16",
    "residual": "bfloat16",
    "matmul_output": "bfloat16",
    "logits": "bfloat16",
    "sampling_values": "bfloat16",
    "token_ids": "uint32",
    "rope": "bfloat16",
}


def load_precision_config(value=None):
    if value is None:
        value = SELECTED_CONFIG
    config = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else copy.deepcopy(value)
    required = {
        "schema_version",
        "config_id",
        "model",
        "base_policy",
        "layer_defaults",
        "layer_exceptions",
        "dtypes",
        "head",
        "accumulation",
        "max_context",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("Incomplete or unknown precision configuration fields")
    if config["model"] != "IFM/K2-Horizon-7B" or config["base_policy"] != "optimized_full_model_stage7":
        raise ValueError("Precision artifact targets another model or geometry")
    if config["max_context"] != 524288:
        raise ValueError("Precision selection must preserve the model context contract")
    if set(config["dtypes"]) != set(FIXED_DTYPES) | {"ccl"}:
        raise ValueError("Incomplete dtype contract")
    for key, dtype in FIXED_DTYPES.items():
        if config["dtypes"][key] != dtype:
            raise ValueError(f"{key} currently requires {dtype}; received {config['dtypes'][key]}")
    if config["dtypes"]["ccl"] not in {"bfloat16", "bfloat8_b"}:
        raise ValueError("CCL supports BF16 or BFP8")
    if config["accumulation"] != {
        "decode_fp32": True,
        "prefill_fp32": True,
        "head_fp32": True,
        "norm_fidelity": "HiFi4",
        "math_approx_mode": False,
        "packer_l1_acc": True,
    }:
        raise ValueError("Unsupported accumulation policy")
    if set(config["head"]) != {"weight_dtype", "compute_fidelity"}:
        raise ValueError("Incomplete head policy")
    if config["head"]["weight_dtype"] not in DTYPES or config["head"]["compute_fidelity"] not in FIDELITIES:
        raise ValueError("Unsupported head precision")
    for layer, changes in [(None, config["layer_defaults"]), *config["layer_exceptions"].items()]:
        if layer is not None and (str(int(layer)) != layer or not 0 <= int(layer) < 36):
            raise ValueError("Layer exception must be in [0,36)")
        if changes.keys() - LAYER_FIELDS:
            raise ValueError(f"Unknown layer fields: {changes.keys() - LAYER_FIELDS}")
        for key, item in changes.items():
            if item not in (FIDELITIES if key.endswith("fidelity") else DTYPES):
                raise ValueError(f"Invalid precision value {key}={item}")
    return config


def layer_policy(config, index):
    changes = {**config["layer_defaults"], **config["layer_exceptions"].get(str(index), {})}
    base = optimized_full_model_policy(index)
    changes["prefill_fp32"] = config["accumulation"]["prefill_fp32"]
    for role in ("qkv", "o", "mlp", "down"):
        field = role + "_geometry"
        changes[field] = replace(getattr(base, field), fp32=config["accumulation"]["decode_fp32"])
    return replace(base, **changes)
