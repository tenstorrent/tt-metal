# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniMax M2.7 Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for MiniMax-M2.7.
"""


class MiniMaxM27Config:
    """MiniMax M2.7 model dimensions."""

    # Core dimensions
    EMB_SIZE = 3072  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 1536  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 1536  # Dense FFN hidden dimension (same as MoE)

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    NUM_SHARED_EXPERTS = 0  # shared_intermediate_size = 0

    # Model architecture
    NUM_LAYERS = 62
    VOCAB_SIZE = 200064
    HEAD_DIM = 128

    # Attention dimensions
    NUM_ATTENTION_HEADS = 48
    NUM_KEY_VALUE_HEADS = 8
    ROTARY_DIM = 64

    # Other
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 5000000
