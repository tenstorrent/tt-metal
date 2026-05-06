# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K2.6 model configuration.

Overrides DeepSeekV3Config with the values from Kimi K2.6's
text_config (https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json).
Kimi K2.6 reuses the DeepSeek V3 architecture (architectures =
["DeepseekV3ForCausalLM"]); only hyperparameters differ.

Fields not listed here inherit the DeepSeek V3 value.
"""

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config


class KimiK26Config(DeepSeekV3Config):
    """Kimi K2.6 hyperparameters as overrides over DeepSeek V3."""

    # MoE: routing
    NUM_ROUTED_EXPERTS = 384  # vs 256
    NUM_EXPERT_GROUPS = 1  # vs 8 (Kimi uses a single group)
    NUM_LIMITED_GROUPS = 1  # vs 4 (topk_group = 1)
    ROUTE_SCALE = 2.827  # vs 2.5 (routed_scaling_factor)

    # Model architecture
    NUM_DENSE_LAYERS = 1  # vs 3 (first_k_dense_replace)
    VOCAB_SIZE = 163840  # vs 129280

    # MLA
    NUM_ATTENTION_HEADS = 64  # vs 128
    NUM_KEY_VALUE_HEADS = 64  # not present in DSv3 config; same as num_attention_heads here

    # Norm / RoPE
    RMS_NORM_EPS = 1e-5  # vs 1e-6
    ROPE_THETA = 50000.0
    MAX_POSITION_EMBEDDINGS = 262144

    # YaRN scaling (Kimi uses yarn factor=64, original_max_position_embeddings=4096)
    ROPE_SCALING_FACTOR = 64.0
    ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS = 4096
    ROPE_SCALING_BETA_FAST = 32.0
    ROPE_SCALING_BETA_SLOW = 1.0
    ROPE_SCALING_MSCALE = 1.0
    ROPE_SCALING_MSCALE_ALL_DIM = 1.0
