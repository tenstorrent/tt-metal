# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validated, construction-time precision policy shared by every model caller."""

import copy
import json
import os
from pathlib import Path

GROUPS = ("attention_input", "attention_output", "mlp_gate_up", "mlp_down", "lm_head")
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"
BASELINE = {
    "id": "bfp8_mlp_lofi",
    "weight_groups": dict.fromkeys(GROUPS, "bfloat8_b"),
    "compute_fidelities": {
        "attention_input": "HiFi2",
        "attention_output": "HiFi2",
        "mlp_gate_up": "LoFi",
        "mlp_down": "LoFi",
        "lm_head": "HiFi2",
    },
    "layer_exceptions": {},
    "activation_dtype": {"decode_projection_input": "bfloat16", "prefill_norm_large": "bfloat8_b"},
    "residual_dtype": "bfloat16",
    "ccl_dtype": "bfloat8_b",
    "kv_cache_dtype": "bfloat8_b",
    "logits_dtype": "bfloat16",
    "sampling_dtype": "bfloat16",
    "norm_weight_dtype": "bfloat16",
    "embedding_dtype": "bfloat16",
    "recurrent_state_dtype": "float32",
    "norm_fidelity": "HiFi4",
    "terminal_norm_fidelity": "HiFi2",
    "sdpa_decode_fidelity": "HiFi2",
    "fp32_dest_acc_en": True,
    "max_context": 262144,
}


def load_policy(value=None):
    if value is None and os.environ.get("ORNITH_PRECISION_CONFIG"):
        # Explicit opt-in override (e.g. bf16_hifi4 accuracy A/B); unset keeps the selected production policy.
        value = os.environ["ORNITH_PRECISION_CONFIG"]
    if value is None:
        value = DEFAULT_PATH if DEFAULT_PATH.exists() else BASELINE
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists() and not path.is_absolute():
            path = DEFAULT_PATH.parents[2] / path
        value = json.loads(path.read_text())
    policy = copy.deepcopy(value)
    if set(policy) != set(BASELINE):
        raise ValueError(f"Precision policy keys must be exactly {sorted(BASELINE)}")
    for key in ("weight_groups", "compute_fidelities"):
        if set(policy[key]) != set(GROUPS):
            raise ValueError(f"Missing/unknown {key}")
    if any(v not in ("bfloat16", "bfloat8_b", "bfloat4_b") for v in policy["weight_groups"].values()):
        raise ValueError("Unsupported weight precision")
    if any(v not in ("LoFi", "HiFi2", "HiFi4") for v in policy["compute_fidelities"].values()):
        raise ValueError("Unsupported matmul fidelity")
    if set(policy["activation_dtype"]) != set(BASELINE["activation_dtype"]):
        raise ValueError("Activation policy must specify both phases")
    for value in policy["activation_dtype"].values():
        if value not in ("bfloat16", "bfloat8_b"):
            raise ValueError("Unsupported activation dtype")
    for key in ("ccl_dtype", "kv_cache_dtype"):
        if policy[key] not in ("bfloat16", "bfloat8_b"):
            raise ValueError(f"Unsupported {key}")
    # These kernel/state contracts are intentionally fixed. Reject a request
    # rather than silently ignoring a precision field that cannot be honored.
    fixed = (
        "residual_dtype",
        "logits_dtype",
        "sampling_dtype",
        "norm_weight_dtype",
        "embedding_dtype",
        "recurrent_state_dtype",
        "norm_fidelity",
        "terminal_norm_fidelity",
        "sdpa_decode_fidelity",
        "fp32_dest_acc_en",
        "max_context",
    )
    for key in fixed:
        if policy[key] != BASELINE[key]:
            raise ValueError(f"The current kernel contract requires {key}={BASELINE[key]}")
    for index, exception in policy["layer_exceptions"].items():
        if not str(index).isdigit() or not 0 <= int(index) < 32:
            raise ValueError("Layer exceptions must name indices 0..31")
        if set(exception) - {"weight_groups", "compute_fidelities"}:
            raise ValueError("Layer exceptions may override weights and compute only")
        merged = copy.deepcopy(policy)
        merged["layer_exceptions"] = {}
        for key, overrides in exception.items():
            if set(overrides) - (set(GROUPS) - {"lm_head"}):
                raise ValueError("Unknown decoder group in layer exception")
            merged[key].update(overrides)
        load_policy(merged)
    return policy


def layer_policy(policy, index):
    result = copy.deepcopy(policy)
    for key, overrides in policy["layer_exceptions"].get(str(index), {}).items():
        result[key].update(overrides)
    return result


def group_for(name):
    if name in ("mlp.gate_proj", "mlp.up_proj", "packed_mlp", "gate_up"):
        return "mlp_gate_up"
    if name == "mlp.down_proj":
        return "mlp_down"
    return "attention_input" if name == "packed_input" else "attention_output"


def prefill_l1_row_limit(activation_dtype, weight_dtypes):
    """Bound retained prefill tensors by their effective, per-layer precision.

    At 4096+, BF16 tensors plus the fixed minimal-matmul dataflow buffers
    overlap in L1. DRAM placement preserves geometry, precision and chunk caps.
    The reviewed BFP8 placement continues through 8192 physical rows.
    """
    return 4095 if "bfloat16" in (activation_dtype, *weight_dtypes) else 8192


def prefill_uses_dram(logical_rows, l1_limit):
    """Account for tile padding even for direct, unbucketed decoder inputs."""
    return ((logical_rows + 31) // 32) * 32 > l1_limit
