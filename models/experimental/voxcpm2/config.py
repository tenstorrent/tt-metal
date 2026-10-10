# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint contracts; architecture dimensions always come from config.json."""

import json
import math
from pathlib import Path

UPSTREAM_SOURCE_REVISION = "f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629"
MODEL_ID = "openbmb/VoxCPM2"


def validate_minicpm_config(config):
    required = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "rms_norm_eps",
        "max_position_embeddings",
    )
    for key in required:
        if key not in config:
            raise ValueError(f"Missing MiniCPM config: {key}")
        if (
            not isinstance(config[key], (int, float))
            or isinstance(config[key], bool)
            or not math.isfinite(config[key])
            or config[key] <= 0
        ):
            raise ValueError(f"MiniCPM {key} must be positive")
    for key in (*required[:-2], "max_position_embeddings"):
        if not isinstance(config[key], int):
            raise ValueError(f"MiniCPM {key} must be an integer")
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    if heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    channels = config.get("kv_channels")
    if channels is None:
        if config["hidden_size"] % heads:
            raise ValueError(
                "hidden_size must be divisible by query heads without kv_channels"
            )
        channels = config["hidden_size"] // heads
    if (
        not isinstance(channels, int)
        or isinstance(channels, bool)
        or channels <= 0
        or channels % 2
    ):
        raise ValueError("Attention head dimension must be a positive even integer")
    if not config.get("no_rope", False):
        rope = config.get("rope_scaling", {})
        original = rope.get("original_max_position_embeddings", 0)
        if original <= 1 or config["max_position_embeddings"] < original:
            raise ValueError(
                "LongRoPE requires max_position_embeddings >= original > 1"
            )
        if config.get("rope_theta", 0) <= 0:
            raise ValueError("rope_theta must be positive")
        for key in ("long_factor", "short_factor"):
            factors = rope.get(key, [])
            if len(factors) != channels // 2 or any(
                not math.isfinite(x) or x <= 0 for x in factors
            ):
                raise ValueError(
                    f"LongRoPE {key} must contain head_dim/2 positive finite factors"
                )
    return channels


def load_config(checkpoint):
    path = Path(checkpoint).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    validate_minicpm_config(config["lm_config"])
    for key in ("patch_size", "feat_dim", "residual_lm_num_layers"):
        if (
            not isinstance(config.get(key), int)
            or not math.isfinite(config[key])
            or config[key] <= 0
        ):
            raise ValueError(f"Missing or invalid VoxCPM2 {key}")
    return config
