# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free configuration for the GPT-OSS 1K, full-depth common-runner gate."""

import json
import math
import os
from pathlib import Path


def prefill_runner_scenario():
    manifest = Path(__file__).parent / "manifests/gpt_oss_120b_1k.json"
    env = json.loads(manifest.read_text())["env"]
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
            raise ValueError(f"GPT-OSS 1K acceptance requires {key}={env[key]}")
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
        "users": 2,
        "layers": 36,
        "max_seq_len": 1024,
        "expected_slots": 2,
        "require_clean_shutdown": True,
        "ready_timeout_s": 3600,
        "producer_timeout_s": 1800,
        "env": env,
        "producer": {
            "PREFILL_NUM_USERS": "2",
            "PREFILL_PRODUCER_CHUNKS": "1",
            "PREFILL_PRODUCER_MAX_REQUESTS": "2",
            "PREFILL_PRODUCER_DURATION_S": "inf",
            "PREFILL_PRODUCER_WARMUP_CHUNKS": "0",
            "PREFILL_PRODUCER_MULTI_TURN_PROB": "0",
            "PREFILL_PRODUCER_INTERLEAVE": "round_robin",
            "PREFILL_PRODUCER_P_GAP": "0",
            "PREFILL_PRODUCER_P_BURST": "0",
            "PREFILL_SEND_SHUTDOWN": "1",
        },
    }


def validate_prefill_slot_traces(spec, scenario):
    from models.demos.common.prefill.runners.trace_utils import golden_key_frame, validate_gqa_trace

    spec = spec or os.environ.get("PREFILL_TRACE_DIR", "")
    paths = [Path(value.strip()) for value in spec.split(",") if value.strip()]
    if len(paths) not in (1, scenario["expected_slots"]):
        raise ValueError("set one shared or two per-slot golden directories in PREFILL_PRODUCER_SLOT_TRACES")
    for path in paths:
        metadata = json.loads((path / "metadata.json").read_text())
        tokens = metadata["token_ids"]
        if len(tokens) < 1024 or any(type(token) is not int or token < 0 or token >= 201088 for token in tokens[:1024]):
            raise ValueError(f"golden must contain a valid 1024-token GPT-OSS prefix: {path}")
        if golden_key_frame(path) != "hf":
            raise ValueError("GPT-OSS producer expects HF half-split post-RoPE keys")
        if metadata.get("disable_sliding_window", False) or metadata.get("zero_sinks", False):
            raise ValueError(
                "diagnostic goldens with modified sliding attention or sinks are not acceptance references"
            )
        validate_gqa_trace(path, 1024, num_layers=36, num_kv_heads=8, head_dim=64)


def validate_kv_for_pcc(golden, actual):
    """PCC must never hide a missing tensor, a shape error, NaN, or infinity."""
    import torch

    if golden.shape != actual.shape or golden.numel() == 0:
        raise ValueError(f"incompatible GPT-OSS KV tensors: golden={golden.shape}, actual={actual.shape}")
    if not torch.isfinite(golden).all() or not torch.isfinite(actual).all():
        raise ValueError("GPT-OSS KV comparison contains non-finite values")


def require_layer_acks(received, expected):
    """A valid cache does not excuse missing layer-completion notifications."""
    if received != expected or expected <= 0:
        raise RuntimeError(f"GPT-OSS layer completion gate failed: received {received}/{expected} ACKs")


def require_clean_runner_exit(returncode):
    """None means the runner needed forced cleanup instead of draining its shutdown sentinel."""
    if returncode != 0:
        raise RuntimeError(f"GPT-OSS runner did not drain and exit cleanly: returncode={returncode}")
