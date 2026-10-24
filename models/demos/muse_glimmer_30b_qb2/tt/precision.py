# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Required, validated precision policy shared by model and generator construction."""

import copy
import json
import os
from pathlib import Path

GROUPS = ("qkv", "self_attn.o_proj", "mlp.gate_up", "mlp.down_proj")
KINDS = ("sliding_attention", "full_attention")
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
SELECTED = CONFIG_ROOT / "selected_precision_config.json"


def baseline_config():
    kinds = {}
    for kind in KINDS:
        sliding = kind == "sliding_attention"
        kinds[kind] = {
            "weight_groups": {
                name: {"dtype": "bfloat4_b", "fidelity": "LoFi", "fp32": name == "qkv" and sliding} for name in GROUPS
            },
            "activations": {
                "attention_act_dtype": "bfloat16" if sliding else "bfloat8_b",
                "mlp_act_dtype": "bfloat8_b",
                "prefill_attention_act_dtype": "bfloat16" if sliding else "bfloat8_b",
                "prefill_mlp_act_dtype": "bfloat16" if sliding else "bfloat8_b",
            },
            "ccl": {
                "ccl_dtype": "bfloat16" if sliding else "bfloat8_b",
                "decode.mlp.down_proj.ccl_dtype": "bfloat8_b",
            },
            "compute": {
                "sdpa_fidelity": "HiFi2",
                "norm_fidelity": "HiFi2" if sliding else "LoFi",
                "norm_fp32": sliding,
                **(
                    {
                        "input_layernorm.fidelity": "HiFi4",
                        "pre_feedforward_layernorm.fidelity": "LoFi",
                        "pre_feedforward_layernorm.fp32": False,
                    }
                    if sliding
                    else {}
                ),
            },
        }
    return {
        "schema_version": 1,
        "config_id": "baseline_bfp4_lofi_head_bfp8_hifi2",
        "model_id": "meta-models/Muse-Glimmer-30B",
        "revision": "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        "max_context": 131072,
        "default_physical_pages": 4096,
        "decoder_kinds": kinds,
        "layer_exceptions": {},
        "kv_cache_dtype": "bfloat8_b",
        "residual_dtype": "bfloat16",
        "embedding_dtype": "bfloat16",
        "norm_weight_dtype": "bfloat16",
        "rope_dtype": "bfloat16",
        "logits_dtype": "bfloat16",
        "sampling_parameter_dtype": "bfloat16",
        "sampling_index_dtype": "uint32",
        "final_norm_fidelity": "HiFi4",
        "final_norm_fp32": True,
        "short_prefill_sliding_input_norm": {"fidelity": "LoFi", "fp32": False},
        "terminal": {
            "weight_dtype": "bfloat8_b",
            "fidelity": "HiFi2",
            "fp32": True,
            "block_w": 1,
            "readers": 3,
            "splits": 1,
            "input_cores": 52,
        },
        "fixed_runtime_assumptions": {
            "short_prefill_matmul_output": "bfloat16",
            "normalized_activations": "bfloat16",
            "qkv_head_split_input": "bfloat16",
            "cache_update_input": "bfloat16",
            "sampling": "Sampling1D split-greedy/top-k32; BF16 logits/parameters; uint32 indices/seeds",
            "moe": "none: dense gated MLP; qkv includes attention gate projection",
        },
    }


def load_precision_config(value=None):
    """An explicit dict/path overrides the selected artifact; 'baseline' is a safe rollback."""
    source = value if value is not None else os.environ.get("MUSE_PRECISION_CONFIG", str(SELECTED))
    config = (
        baseline_config()
        if source == "baseline"
        else (copy.deepcopy(source) if isinstance(source, dict) else json.loads(Path(source).read_text()))
    )
    expected = baseline_config()
    if set(config) != set(expected):
        raise ValueError(f"Precision fields differ: {set(config) ^ set(expected)}")
    for key in (
        "schema_version",
        "model_id",
        "revision",
        "max_context",
        "default_physical_pages",
        "residual_dtype",
        "embedding_dtype",
        "norm_weight_dtype",
        "rope_dtype",
        "logits_dtype",
        "sampling_parameter_dtype",
        "sampling_index_dtype",
        "fixed_runtime_assumptions",
    ):
        if config[key] != expected[key]:
            raise ValueError(f"Unsupported precision contract {key}: {config[key]!r}")
    if set(config["decoder_kinds"]) != set(KINDS):
        raise ValueError("Require both decoder kinds")
    for idx in config["layer_exceptions"]:
        if not 0 <= int(idx) < 52:
            raise ValueError(f"Invalid layer exception: {idx}")
    for kind in KINDS:
        decoder_policy(config, kind, -1)
    return config


def decoder_policy(config, kind, layer_idx):
    spec = copy.deepcopy(config["decoder_kinds"][kind])
    spec.setdefault("prefill_geometry", {})
    exception = config["layer_exceptions"].get(str(layer_idx), {})
    for section, values in exception.items():
        if section not in spec:
            raise ValueError(f"Unknown layer exception section {section}")
        spec[section].update(values)
    if set(spec) != {"weight_groups", "activations", "ccl", "compute", "prefill_geometry"} or set(
        spec["weight_groups"]
    ) != set(GROUPS):
        raise ValueError("Incomplete decoder precision policy")
    result = {
        "kv_dtype": config["kv_cache_dtype"],
        "norm_weight_dtype": config["norm_weight_dtype"],
        "rope_dtype": config["rope_dtype"],
    }
    for name, values in spec["weight_groups"].items():
        if set(values) != {"dtype", "fidelity", "fp32"}:
            raise ValueError(f"Incomplete matmul group {name}")
        result.update({f"{name}.{key}": value for key, value in values.items()})
    allowed = {
        "activations": {
            "attention_act_dtype",
            "mlp_act_dtype",
            "prefill_attention_act_dtype",
            "prefill_mlp_act_dtype",
        },
        "ccl": {
            "ccl_dtype",
            "decode.mlp.down_proj.ccl_dtype",
            "decode.self_attn.o_proj.ccl_dtype",
            "prefill.mlp.down_proj.ccl_dtype",
            "prefill.self_attn.o_proj.ccl_dtype",
        },
        "compute": {
            "sdpa_fidelity",
            "norm_fidelity",
            "norm_fp32",
            "qk_norm.fidelity",
            "qk_norm.fp32",
            "input_layernorm.fidelity",
            "pre_feedforward_layernorm.fidelity",
            "pre_feedforward_layernorm.fp32",
        },
    }
    for section in ("activations", "ccl", "compute"):
        if set(spec[section]) - allowed[section]:
            raise ValueError(f"Unknown {section} precision field")
        result.update(spec[section])
    geometry_keys = {"minimal_m", "minimal_k", "minimal_n"} | {
        f"{name}.minimal_{axis}" for name in GROUPS for axis in ("m", "k", "n")
    }
    if set(spec["prefill_geometry"]) - geometry_keys:
        raise ValueError("Unknown prefill geometry field")
    if any(not isinstance(v, int) or v <= 0 for v in spec["prefill_geometry"].values()):
        raise ValueError("Prefill block sizes must be positive integers")
    result.update(spec["prefill_geometry"])
    return result
