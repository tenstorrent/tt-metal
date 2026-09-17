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
    # moe_fused_swiglu, the rest to unified_routed_expert_moe. Measured under SwiGLU-OAI WITH the
    # expert biases these FFNs carry, on the 2880x2880 routed-expert shape: the fused op holds the
    # band to 384, the composite takes 416-512, gives it back at 576-640 where its tail per_core_M
    # rounds 18 tile-rows up to 32, and wins outright from 768. 384 is the aggregate-optimal cut
    # over that sawtooth (+0.10% against a per-count oracle, worst cell +14% at 576).
    # Biasing is what places this cut so high: it costs the composite far more than the fused op,
    # which adds gate/up bias on a pack-and-reload pass it already needed while the composite pays
    # a full extra broadcast pass. Measuring this bias-free would send the low counts to the
    # slower op.
    # Not enabled: the gpt-oss MoE builds TtRoutedExpert directly and forwards no threshold, so
    # nothing reads this. Both op-side blockers are gone -- moe_fused_swiglu carries SwiGluOai and
    # the per-expert bias, and TtRoutedExpert no longer refuses a threshold on biased experts.
    # Kept under _MEASURED so it is not re-derived; rename it back to
    # ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD once that path forwards one.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED = 384
    # The 120B shares this routed-expert shape exactly, so the crossover is the same.
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
