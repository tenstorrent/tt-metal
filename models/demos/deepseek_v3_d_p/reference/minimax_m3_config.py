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
    # No routed-expert hybrid threshold. moe_fused_swiglu does not BUILD on this shape since the
    # gate/up accumulators went bf16: the CB layout needs 1_610_112 bytes against a 1_461_248 budget
    # on an 11x8 grid, so the op TT_FATALs at every token count rather than merely being slower at
    # some of them. A threshold would be fiction -- these experts run on unified_routed_expert_moe
    # only. Re-derive one once the fused op fits again; phase_cb_alias() is now dead for every shape
    # (cb_gather_gate is bf16 while cb_h_slice and cb_out_tiles are bfp8, so the three page sizes can
    # never agree), and that lost allocation is the L1 to recover.
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
