# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniMax M3 Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for MiniMax-M3.
"""


class MiniMaxM3Config:
    """MiniMax-M3 text-backbone model dimensions and hyperparameters."""

    # Core dimensions
    EMB_SIZE = 6144
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # Implementation-specific; keep in sync with migration code

    # FFN dimensions
    MOE_INTERMEDIATE_SIZE = 3072  # Routed-expert FFN hidden dimension
    # Routed-expert hybrid split: experts with <= this many active tokens go to
    # moe_fused_swiglu, the rest to unified_routed_expert_moe. The two ops cross once on the
    # 6144x3072 routed-expert shape, between 128 and 160. 128 matches a per-count oracle exactly
    # for DRAM-interleaved and ND-sharded weights alike. Measured under SwiGluOai, the activation
    # these experts actually run.
    # Not enabled: only Kimi K2.6/K2.7 and GLM 5.1/5.2 dispatch both routed-expert ops today.
    # The measured crossover is kept under _MEASURED so it is not re-derived; rename it back to
    # ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD to turn the split on, which is all the readers look for.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED = 128

    # DRAM ND-sharded routed-expert weights: a core's whole K-row weight slice arrives in one
    # NoC request instead of one per tile. Worth 1.21x at 64 active tokens and 1.20x at 128 on this
    # 6144x3072 shape. Unusually, the fused op keeps 1.04x all the way to 5120 here, where the
    # composite has gone flat -- so this shape gains at long sequences too if the split is on.
    ROUTED_EXPERT_WEIGHTS_DRAM_SHARDED = True
    SHARED_INTERMEDIATE_SIZE = 3072  # Always-on shared expert
    INTERMEDIATE_SIZE = 12288  # Dense FFN hidden dimension

    # Model architecture
    NUM_LAYERS = 60
    NUM_DENSE_LAYERS = 3
    NUM_MOE_LAYERS = NUM_LAYERS - NUM_DENSE_LAYERS
    VOCAB_SIZE = 200064
    MAX_POSITION_EMBEDDINGS = 1_048_576

    # Attention
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 4
    HEAD_DIM = 128
    ROTARY_DIM = 64
    ROPE_THETA = 5_000_000
    PARTIAL_ROTARY_FACTOR = 0.5
    ATTENTION_DROPOUT = 0.0
    USE_QK_NORM = True
    QK_NORM_TYPE = "per_head"
    USE_GEMMA_NORM = True
    ATTENTION_OUTPUT_GATE = False

    # MoE routing
    NUM_ROUTED_EXPERTS = 128
    NUM_EXPERTS_PER_TOKEN = 4
    NUM_SHARED_EXPERTS = 1
    SCORING_FUNC = "sigmoid"
    USE_ROUTING_BIAS = True
    ROUTE_SCALE = 2.0

    # Layer schedules: 0 = dense/full-attention, 1 = MoE/sparse-attention
    MOE_LAYER_FREQ = (0, 0, 0) + (1,) * 57
    SPARSE_ATTENTION_FREQ = (0, 0, 0) + (1,) * 57

    # Sparse attention / MSA
    USE_SPARSE_ATTENTION = True
    SPARSE_INDEX_DIM = 128
    SPARSE_NUM_INDEX_HEADS = 4
    SPARSE_BLOCK_SIZE = 128
    SPARSE_TOPK_BLOCKS = 16
    SPARSE_SCORE_TYPE = "max"
    SPARSE_INIT_BLOCK = 0
    SPARSE_LOCAL_BLOCK = 1

    # SwiGLU-OAI activation
    HIDDEN_ACT = "swigluoai"
    SWIGLU_ALPHA = 1.702
    SWIGLU_LIMIT = 7.0

    # Other model settings
    RMS_NORM_EPS = 1e-6
    TIE_WORD_EMBEDDINGS = False
    USE_CACHE = True
    NUM_MTP_MODULES = 7
    NUM_NEXTN_PREDICT_LAYERS = 1
