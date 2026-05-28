# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K2.6 model configuration.

Overrides DeepSeekV3Config with the values from Kimi K2.6's
text_config (https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json).
"""

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config


class KimiK26Config(DeepSeekV3Config):
    # MoE: routing
    NUM_ROUTED_EXPERTS = 384
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    ROUTE_SCALE = 2.827

    # Model architecture
    NUM_DENSE_LAYERS = 1
    VOCAB_SIZE = 163840

    # MLA
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 64

    # Norm / RoPE
    RMS_NORM_EPS = 1e-5
    ROPE_THETA = 50000.0
    MAX_POSITION_EMBEDDINGS = 262144

    # YaRN scaling
    ROPE_SCALING_FACTOR = 64.0
    ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS = 4096
    ROPE_SCALING_BETA_FAST = 32.0
    ROPE_SCALING_BETA_SLOW = 1.0
    ROPE_SCALING_MSCALE = 1.0
    ROPE_SCALING_MSCALE_ALL_DIM = 1.0
