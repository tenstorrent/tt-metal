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
    # moe_fused_swiglu, the rest to unified_routed_expert_moe. On the 6144x3072 routed-expert
    # shape the composite already wins from 256 and gives the band back only at 576-640, where
    # its tail per_core_M rounds 18 tile-rows up to 32. 128 is the aggregate-optimal cut over
    # that sawtooth (+0.03% against a per-count oracle, worst cell +6.8% at 576). Measured under
    # SwiGluOai, the activation these experts actually run.
    # Not enabled: the M3 MoE builds TtRoutedExpert directly and forwards no threshold, so nothing
    # reads this. Nothing on the op side blocks it any more -- moe_fused_swiglu carries SwiGluOai,
    # and M3's only bias is the router's e_score_correction_bias, not an expert-FFN bias, so there
    # is none to lose. Kept under _MEASURED so it is not re-derived; rename it back to
    # ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD once that path forwards one.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED = 128
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
