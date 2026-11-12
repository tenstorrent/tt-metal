# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free configuration for the GPT-OSS full-depth common-runner gate: two 1K chunks per slot."""

import json
import math
import os
from pathlib import Path

MANIFEST = Path(__file__).parent / "manifests/gpt_oss_120b_2k.json"


def prefill_runner_scenario():
    env = json.loads(MANIFEST.read_text())["env"]
    for key in (
        "PREFILL_NUM_LAYERS",
        "PREFILL_NUM_USERS",
        "PREFILL_MAX_SEQ_LEN",
        "PREFILL_CHUNK_SIZE",
        "PREFILL_SP",
        "PREFILL_TP",
        "GPT_OSS_BOUNDED_SLIDING_KV",
    ):
        if os.environ.get(key, env[key]) != env[key]:
            raise ValueError(f"GPT-OSS acceptance requires {key}={env[key]}")
    threshold = float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.85"))
    if not math.isfinite(threshold) or not 0.85 <= threshold <= 1.0:
        raise ValueError("GPT-OSS acceptance PCC must be finite and at least the common CI floor 0.85")
    env.update(
        PREFILL_LAYER_ACK_D2H="0",
        PREFILL_USE_TRACE="0",
        PREFILL_KV_ONLY_LAST_LAYER="0",
        PREFILL_STANDALONE_CHUNKED_PCC=str(threshold),
    )
    return {
        "users": int(env["PREFILL_NUM_USERS"]),
        "layers": int(env["PREFILL_NUM_LAYERS"]),
        "max_seq_len": int(env["PREFILL_MAX_SEQ_LEN"]),
        "expected_slots": int(env["PREFILL_NUM_USERS"]),
        "env": env,
    }


def validate_prefill_slot_traces(spec, scenario):
    from models.demos.common.prefill.runners.trace_utils import golden_key_frame, validate_gqa_trace

    spec = spec or os.environ.get("PREFILL_TRACE_DIR", "")
    paths = [Path(value.strip()) for value in spec.split(",") if value.strip()]
    if len(paths) not in (1, scenario["expected_slots"]):
        raise ValueError("set one shared or two per-slot golden directories in PREFILL_PRODUCER_SLOT_TRACES")
    length = scenario["max_seq_len"]
    for path in paths:
        metadata = json.loads((path / "metadata.json").read_text())
        tokens = metadata["token_ids"]
        if len(tokens) < length or any(type(token) is not int or not 0 <= token < 201088 for token in tokens[:length]):
            raise ValueError(f"golden must contain a valid {length}-token GPT-OSS prefix: {path}")
        if golden_key_frame(path) != "hf":
            raise ValueError("GPT-OSS producer expects HF half-split post-RoPE keys")
        if metadata.get("disable_sliding_window", False) or metadata.get("zero_sinks", False):
            raise ValueError(
                "diagnostic goldens with modified sliding attention or sinks are not acceptance references"
            )
        validate_gqa_trace(path, length, num_layers=scenario["layers"], num_kv_heads=8, head_dim=64)


def validate_kv_for_pcc(golden, actual):
    """PCC must never hide a missing tensor, a shape error, NaN, or infinity."""
    import torch

    if golden.shape != actual.shape or golden.numel() == 0:
        raise ValueError(f"incompatible GPT-OSS KV tensors: golden={golden.shape}, actual={actual.shape}")
    if not torch.isfinite(golden).all() or not torch.isfinite(actual).all():
        raise ValueError("GPT-OSS KV comparison contains non-finite values")
