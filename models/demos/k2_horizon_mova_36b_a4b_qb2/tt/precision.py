# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Versioned constructor precision policy. Overrides never mutate a live model."""

import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    "schema_version": 1,
    "config_id": "baseline_bfp4_lofi",
    "weight_groups": {
        "attention_dtype": "bfloat4_b",
        "attention_decode_dtype": "bfloat4_b",
        "output_weight_dtype": "bfloat4_b",
        "mlp_dtype": "bfloat4_b",
        "down_weight_dtype": "bfloat4_b",
        "experts_dtype": "bfloat4_b",
        "expert_down_dtype": "bfloat4_b",
        "value_dtype": "bfloat4_b",
    },
    "layer_exceptions": [{"layers": [0, 1, 2], "weight_groups": {"attention_dtype": "bfloat8_b"}}],
    "compute_fidelities": {"attention": "LoFi", "mlp": "LoFi", "experts": "LoFi", "value": "LoFi"},
    "activations": {
        "attention_activation": "bfloat8_b",
        "mlp_activation": "bfloat8_b",
        "expert_activation": "bfloat8_b",
        "expanded_expert_activation": "bfloat8_b",
        "prefill_value_activation": "bfloat8_b",
        "ep_value_activation": "bfloat8_b",
        "prefill_norm_dtype": "bfloat16",
        "dense_prefill_hidden_dtype": "bfloat8_b",
        "ep_expert_activation": "bfloat8_b",
        "value_activation": "bfloat16",
    },
    "residual_dtype": "bfloat16",
    "ccl_dtype": "bfloat8_b",
    "prefill_ccl_dtype": "bfloat16",
    "kv_cache_dtype": "bfloat8_b",
    "lm_head_dtype": "bfloat16",
    "lm_head_fidelity": "HiFi4",
    "embedding_dtype": "bfloat16",
    "norm_dtype": "bfloat16",
    "norm_fidelity": "HiFi4",
    "logits_dtype": "bfloat16",
    "sampling_dtype": "bfloat16",
    "routing": "packed projection precision; BF16 routing scores/top-k; weighting follows activation dtype except BFP4 promotes to BF16; active value4/MLP8",
    "sdpa_fidelities": {"prefill": "HiFi2", "short": "LoFi", "middle": "HiFi2", "long": "HiFi2"},
    "max_context": 524288,
}


def load_precision(config=None):
    if config is None:
        config = os.environ.get("K2_PRECISION_CONFIG")
    if config is None:
        # The committed artifact is part of model construction. A missing
        # deployment artifact must not silently restore an older policy.
        config = ROOT / "doc/datatype_sweep/selected_precision_config.json"
    if config == "baseline":
        config = BASELINE
    policy = copy.deepcopy(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
    optional = {"lm_head_in0_block_w"}
    if set(BASELINE) - set(policy) or set(policy) - set(BASELINE) - optional:
        raise ValueError(f"Incomplete/unknown precision fields: {set(policy) ^ set(BASELINE)}")
    if policy.get("lm_head_in0_block_w", 1) not in (1, 2, 5, 10):
        raise ValueError("LM-head block must divide the 10-tile input shard")
    for key in (
        "schema_version",
        "max_context",
        "embedding_dtype",
        "norm_dtype",
        "logits_dtype",
        "sampling_dtype",
        "routing",
    ):
        if policy[key] != BASELINE[key]:
            raise ValueError(f"Unsupported {key}: {policy[key]}")
    for key in ("weight_groups", "compute_fidelities", "activations", "sdpa_fidelities"):
        if set(policy[key]) != set(BASELINE[key]):
            raise ValueError(f"Incomplete {key}")
    return policy


def layer_options(policy, index):
    options = {**policy["weight_groups"], **policy["activations"]}
    options.update({key + "_fidelity": value for key, value in policy["compute_fidelities"].items()})
    options.update(
        cache_dtype=policy["kv_cache_dtype"],
        ccl_dtype=policy["ccl_dtype"],
        prefill_ccl_dtype=policy["prefill_ccl_dtype"],
        residual_dtype=policy["residual_dtype"],
        norm_dtype=policy["norm_dtype"],
        norm_fidelity=policy["norm_fidelity"],
        sdpa_fidelities=policy["sdpa_fidelities"],
    )
    for exception in policy["layer_exceptions"]:
        if index in exception["layers"]:
            options.update(exception.get("weight_groups", {}))
            options.update({k + "_fidelity": v for k, v in exception.get("compute_fidelities", {}).items()})
    return options


def runtime_summary(model):
    """Read constructed tensors/config objects, rather than echoing policy JSON."""
    return {
        "config_id": model.precision["config_id"],
        "embedding": str(model.embedding.dtype),
        "head": str(model.lm_head.dtype),
        "decode_head": [str(w.dtype) for w in model.decode_lm_heads],
        "head_fidelity": str(model.head_compute.math_fidelity),
        "head_in0_block_w": model.decode_head_program.in0_block_w,
        "norm_fidelity": str(model.compute.math_fidelity),
        "layers": [
            {
                "index": layer.layer_idx,
                "weights": {k: str(v.dtype) for k, v in layer.weights.items()},
                "weight_manifest": layer.weight_manifest,
                "decode_shapes": {k: list(v.shape) for k, v in layer.decode_weights.items()},
                "decode_weights": {k: str(v.dtype) for k, v in layer.decode_weights.items()},
                "fidelities": {k: str(v.math_fidelity) for k, v in layer.matmul_compute.items()},
                "sdpa_fidelities": {k: str(v.math_fidelity) for k, v in layer.sdpa_compute.items()},
                "ccl": str(layer.ar_dtype),
                "ccl_scratch": [str(tensor.dtype) for tensor, _ in layer.ar_slots],
                "cache": str(layer.kv_cache_dtype),
                "options": layer.options,
            }
            for layer in model.layers
        ],
    }
