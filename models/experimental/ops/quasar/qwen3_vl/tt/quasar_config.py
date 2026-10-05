# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar bring-up configuration for the Qwen3-VL copy: bf16, device-derived grids, truncation."""
import math


def vision_padded_seq_len(n: int) -> int:
    # vision_attention needs a multiple of 128, and of 2048 once the sequence exceeds 2048.
    step = 128 if n <= 2048 else 2048
    return math.ceil(n / step) * step


def truncate_hf_config(config, vision_layers, text_layers, deepstack_at=None):
    config.vision_config.depth = vision_layers
    config.text_config.num_hidden_layers = text_layers
    if deepstack_at is not None:
        config.vision_config.deepstack_visual_indexes = [deepstack_at]
    return config
