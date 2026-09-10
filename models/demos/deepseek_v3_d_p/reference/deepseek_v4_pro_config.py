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
    # Routed-expert hybrid split: experts with <= this many active tokens go to
    # moe_fused_swiglu, the rest to unified_routed_expert_moe. On the 7168x3072 routed-expert
    # shape the composite already wins from 256 and gives the band back only at 576, where its
    # tail per_core_M rounds 18 tile-rows up to 32. 128 is the aggregate-optimal cut over that
    # sawtooth (+0.02% against a per-count oracle, worst cell +3.8% at 576).
    # Not enabled: only Kimi K2.6/K2.7 and GLM 5.1/5.2 dispatch both routed-expert ops today.
    # The measured crossover is kept under _MEASURED so it is not re-derived; rename it back to
    # ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD to turn the split on, which is all the readers look for.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED = 128
    HEAD_DIM = 512

    # MoE configuration
    NUM_ROUTED_EXPERTS = 384
    NUM_EXPERTS_PER_TOKEN = 6
    NUM_SHARED_EXPERTS = 1
    # V4 drops V3's expert-group routing: a single group means the gate collapses to a plain top-k.
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    # V4 replaces V3/Kimi's sigmoid router affinity with sqrt(softplus(.)).
    SCORE_FUNC = "sqrtsoftplus"

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
