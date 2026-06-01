# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V4 Pro Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for DeepSeek-V4-Pro.
"""


class DeepSeekV4ProConfig:
    """DeepSeek V4 Pro model dimensions."""

    # Core dimensions
    EMB_SIZE = 7168  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 3072  # MoE FFN hidden dimension
    HEAD_DIM = 512

    # MoE configuration
    NUM_ROUTED_EXPERTS = 384
    NUM_EXPERTS_PER_TOKEN = 6
    NUM_SHARED_EXPERTS = 1

    # Model architecture
    NUM_LAYERS = 61
    NUM_HASH_LAYERS = 3
    VOCAB_SIZE = 129280
    SLIDING_WINDOW = 128

    # MLA dimensions
    NUM_ATTENTION_HEADS = 128
    NUM_KEY_VALUE_HEADS = 1
    Q_LORA_RANK = 1536
    O_LORA_RANK = 1024
    O_GROUPS = 16
    QK_ROPE_HEAD_DIM = 64

    # Indexer / sparse attention
    INDEX_N_HEADS = 64
    INDEX_HEAD_DIM = 128
    INDEX_TOPK = 1024
    # Compressed attention config
    COMPRESS_RATES = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
    COMPRESS_ROPE_THETA = 160000.0
    HC_MULT = 4
    HC_SINKHORN_ITERS = 20
    HC_EPS = 1.0e-6
    # Other
    RMS_NORM_EPS = 1e-6
    ROUTE_SCALE = 2.5
    ROPE_THETA = 10000
    SWIGLU_LIMIT = 10.0
    MAX_POSITION_EMBEDDINGS = 1048576
