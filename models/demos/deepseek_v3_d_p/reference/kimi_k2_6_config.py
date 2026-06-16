# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K2.6 Model Configuration (text tower only).

Single source of truth for model dimension constants.
Values from HuggingFace config.json for Kimi-K2.6 (text_config).
"""


class KimiK26Config:
    """Kimi K2.6 model dimensions."""

    # Core dimensions
    EMB_SIZE = 7168  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 18432  # Dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 384
    NUM_EXPERTS_PER_TOKEN = 8
    NUM_SHARED_EXPERTS = 1
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    ROUTE_SCALE = 2.827

    # Model architecture
    NUM_LAYERS = 61
    NUM_DENSE_LAYERS = 1  # first_k_dense_replace
    VOCAB_SIZE = 163840

    # MLA dimensions
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 64
    Q_LORA_RANK = 1536
    KV_LORA_RANK = 512
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 128

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
