# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V4 Flash Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for DeepSeek-V4-Flash.
"""


class DeepSeekV4FlashConfig:
    """DeepSeek V4 Flash model dimensions."""

    # Core dimensions
    EMB_SIZE = 4096  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    # Routed-expert hybrid split: experts with <= this many active tokens go to
    # moe_fused_swiglu, the rest to unified_routed_expert_moe. Measured crossover on the
    # 4096x2048 routed-expert shape (1.04x at 768, 0.92x at 896); it is the last full M_BLOCK
    # boundary before the fused op opens another block while the composite's chunk
    # schedule stays flat, so the two costs cross just above it.
    # Not enabled: only Kimi K2.6/K2.7 and GLM 5.1/5.2 dispatch both routed-expert ops today.
    # The measured crossover is kept under _MEASURED so it is not re-derived; rename it back to
    # ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD to turn the split on, which is all the readers look for.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED = 768
    HEAD_DIM = 512

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 6
    NUM_SHARED_EXPERTS = 1
    # V4 drops V3's expert-group routing: a single group means the gate collapses to a plain top-k.
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    # V4 replaces V3/Kimi's sigmoid router affinity with sqrt(softplus(.)).
    SCORE_FUNC = "sqrtsoftplus"

    # Model architecture
    NUM_LAYERS = 43
    NUM_HASH_LAYERS = 3
    VOCAB_SIZE = 129280
    SLIDING_WINDOW = 128

    # MLA dimensions
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 1
    Q_LORA_RANK = 1024
    O_LORA_RANK = 1024
    O_GROUPS = 8
    QK_ROPE_HEAD_DIM = 64

    # Indexer / sparse attention (NSA-style)
    INDEX_N_HEADS = 64
    INDEX_HEAD_DIM = 128
    INDEX_TOPK = 512
    # Compressed attention config
    COMPRESS_RATES = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
    COMPRESS_ROPE_THETA = 160000.0
    HC_MULT = 4
    HC_SINKHORN_ITERS = 20
    HC_EPS = 1.0e-6

    # Other
    RMS_NORM_EPS = 1e-6
    ROUTE_SCALE = 1.5
    ROPE_THETA = 10000
    SWIGLU_LIMIT = 10.0
    MAX_POSITION_EMBEDDINGS = 1048576
