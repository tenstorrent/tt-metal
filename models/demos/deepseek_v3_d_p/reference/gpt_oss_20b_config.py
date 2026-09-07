# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS 20B Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for gpt-oss-20b.
"""


class GptOss20BConfig:
    """GPT-OSS 20B model configuration."""

    # Core dimensions
    EMB_SIZE = 2880
    MOE_INTERMEDIATE_SIZE = 2880
    # Routed-expert hybrid split: experts with <= this many active tokens go to
    # moe_fused_swiglu, the rest to unified_routed_expert_moe. Measured under SwiGLU-OAI -- the
    # activation these experts actually run -- on the 2880x2880 routed-expert shape: the fused op
    # wins only at 256 (0.75x) and loses from 512 on, so 256 is both the last M_BLOCK boundary it
    # wins at and the aggregate-optimal cut (+0.04% against a per-count oracle).
    # The 120B shares this routed-expert shape exactly, so the crossover is the same.
    # Not enabled, for two independent reasons. Nothing forwards a threshold: the gpt-oss MoE
    # builds TtRoutedExpert directly and passes none. And moe_fused_swiglu has no bias inputs,
    # so TtRoutedExpert rejects a threshold outright whenever use_expert_bias is on -- the fused
    # band would silently drop the gate/up/down biases the composite band applies. The measured
    # crossover is kept under _MEASURED so it is not re-derived.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED = 256
    INTERMEDIATE_SIZE = 2880
    HEAD_DIM = 64

    # Model architecture
    NUM_LAYERS = 24
    VOCAB_SIZE = 201088
    MAX_POSITION_EMBEDDINGS = 131072
    INITIAL_CONTEXT_LENGTH = 4096

    # Attention
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 8
    ATTENTION_BIAS = True
    ATTENTION_DROPOUT = 0.0
    SLIDING_WINDOW = 128

    LAYER_TYPES = (
        "sliding_attention",
        "full_attention",
    ) * 12

    # RoPE / YaRN
    ROPE_THETA = 150000
    ROPE_PARAMETERS = {
        "rope_type": "yarn",
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": False,
        "original_max_position_embeddings": 4096,
    }

    # MoE
    NUM_ROUTED_EXPERTS = 32
    NUM_EXPERTS_PER_TOKEN = 4
    NUM_SHARED_EXPERTS = 0  # Derived: no shared-expert block

    # GPT-OSS SwiGLU
    HIDDEN_ACT = "silu"
    SWIGLU_ALPHA = 1.702
    SWIGLU_LIMIT = 7.0

    # Normalization / initialization
    RMS_NORM_EPS = 1e-5
    INITIALIZER_RANGE = 0.02

    # Miscellaneous model config
    ROUTER_AUX_LOSS_COEF = 0.9
    OUTPUT_ROUTER_LOGITS = False
    USE_CACHE = True
    TIE_WORD_EMBEDDINGS = False
    PAD_TOKEN_ID = 199999
    EOS_TOKEN_ID = 200002

    # Weight format, not an architectural hyperparameter
    QUANT_METHOD = "mxfp4"

    # Implementation-specific, not from the HF model config
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
