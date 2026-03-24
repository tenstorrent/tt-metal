# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3/R1 671B Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for DeepSeek-V3.
"""


class DeepSeekV3Config:
    """DeepSeek V3/R1 671B model dimensions."""

    # Core dimensions
    EMB_SIZE = 7168  # embedding dimension
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 18432  # Dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    NUM_SHARED_EXPERTS = 1
    NUM_EXPERT_GROUPS = 8
    NUM_LIMITED_GROUPS = 4

    # Model architecture
    NUM_LAYERS = 61
    NUM_DENSE_LAYERS = 3
    VOCAB_SIZE = 129280

    # MLA dimensions
    NUM_ATTENTION_HEADS = 128
    Q_LORA_RANK = 1536
    KV_LORA_RANK = 512
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 128

    # Other
    RMS_NORM_EPS = 1e-6
    ROUTE_SCALE = 2.5
