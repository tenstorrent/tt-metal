# SPDX-License-Identifier: Apache-2.0
"""Validated construction policy shared by full-model callers."""

import copy
import hashlib
import json
from pathlib import Path

POLICIES = {
    "bfp4": ("bfloat4_b", "LoFi"),
    "bfp4_hifi2": ("bfloat4_b", "HiFi2"),
    "bfp8": ("bfloat8_b", "HiFi2"),
    "bfp8_lofi": ("bfloat8_b", "LoFi"),
    "bf16": ("bfloat16", "HiFi4"),
}
BASELINE = {
    "config_id": "baseline_bfp4_lofi",
    "weight_groups": {"attention": "bfp4", "mlp": "bfp4", "down": "bfp4", "lm_head": "bfp8"},
    "layer_exceptions": {},
    "compute_fidelities": {"attention": "LoFi", "mlp": "LoFi", "down": "LoFi", "lm_head": "HiFi2"},
    "activation_dtype": "bfloat8_b",
    "residual_dtype": "bfloat16",
    "ccl_dtype": {"prefill": "bfloat8_b", "decode": "bfloat16"},
    "kv_cache_dtype": "bfloat8_b",
    "logits_dtype": "bfloat16",
    "sampling_dtype": "bfloat16",
    "embedding_dtype": "bfloat16",
    "norm_dtype": "bfloat16",
    "max_seq_len": 262144,
}


# Selected full-model policy; evidence remains in doc/datatype_sweep.
SELECTED = {
    "config_id": "head_bfp4",
    "weight_groups": {"attention": "bfp4", "mlp": "bfp4", "down": "bfp4", "lm_head": "bfp4"},
    "layer_exceptions": {},
    "compute_fidelities": {"attention": "LoFi", "mlp": "LoFi", "down": "LoFi", "lm_head": "LoFi"},
    "activation_dtype": "bfloat8_b",
    "residual_dtype": "bfloat16",
    "ccl_dtype": {"prefill": "bfloat8_b", "decode": "bfloat16"},
    "kv_cache_dtype": "bfloat8_b",
    "logits_dtype": "bfloat16",
    "sampling_dtype": "bfloat16",
    "embedding_dtype": "bfloat16",
    "norm_dtype": "bfloat16",
    "max_seq_len": 262144,
}


def load_precision(value=None):
    if value is None:
        value = SELECTED
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    policy = copy.deepcopy(value)
    if set(policy) != set(BASELINE):
        raise ValueError("Precision policy fields must exactly match the construction schema")
    if set(policy["weight_groups"]) != set(BASELINE["weight_groups"]) or set(policy["compute_fidelities"]) != set(
        BASELINE["compute_fidelities"]
    ):
        raise ValueError("Every material projection group needs a dtype and fidelity")
    if set(policy["ccl_dtype"]) != {"prefill", "decode"}:
        raise ValueError("CCL requires separate prefill and decode policies")
    if any(str(int(i)) != i or not 0 <= int(i) < 60 for i in policy["layer_exceptions"]):
        raise ValueError("Layer exceptions require indices 0 through 59")
    for group, name in policy["weight_groups"].items():
        if policy["compute_fidelities"][group] != POLICIES[name][1]:
            raise ValueError("Weight policy and compute fidelity disagree")
    for overrides in policy["layer_exceptions"].values():
        if not set(overrides) <= {"attention", "mlp", "down"}:
            raise ValueError("Unknown layer precision override")
        for name in overrides.values():
            if name not in POLICIES:
                raise ValueError("Unknown layer policy")
    for key in ("residual_dtype", "logits_dtype", "sampling_dtype", "embedding_dtype", "norm_dtype", "max_seq_len"):
        if policy[key] != BASELINE[key]:
            raise ValueError(f"{key} is a fixed runtime contract: {BASELINE[key]}")
    if policy["ccl_dtype"]["decode"] not in ("bfloat16", "bfloat8_b"):
        raise ValueError("Persistent decode collectives support BF16 or BFP8 backing storage")
    for dtype in (policy["activation_dtype"], policy["kv_cache_dtype"], policy["ccl_dtype"]["prefill"]):
        if dtype not in ("bfloat16", "bfloat8_b"):
            raise ValueError("Unsupported activation/cache/CCL dtype")
    return policy


EXPERIMENTAL_DECODER_BFP8_SHA256 = "751c757733be109423b5ed3bcdcd5856f6be0d2e55f1a96f250fca499cc58b71"


def load_serving_precision(selector=None, expected_sha256=None):
    """Require an explicit pinned opt-in for an unaccepted diagnostic policy.

    The digest covers canonical JSON (sorted keys, compact separators), not the
    historical configuration file's whitespace. Normal serving keeps SELECTED.
    """
    if selector is None:
        if expected_sha256 is not None:
            raise ValueError("Experimental precision digest requires a selector")
        return load_precision()
    if selector != "decoder_bfp8":
        raise ValueError("Unknown experimental serving precision")
    if expected_sha256 != EXPERIMENTAL_DECODER_BFP8_SHA256:
        raise ValueError("Experimental serving precision digest mismatch")
    policy = load_precision()
    policy["config_id"] = "decoder_bfp8"
    policy["weight_groups"] = dict.fromkeys(policy["weight_groups"], "bfp8")
    policy["compute_fidelities"] = dict.fromkeys(policy["compute_fidelities"], "HiFi2")
    policy = load_precision(policy)
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != EXPERIMENTAL_DECODER_BFP8_SHA256:
        raise ValueError("Experimental serving policy contents changed")
    return policy
