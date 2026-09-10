# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS 120B Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for gpt-oss-120b.
"""


class GptOss120BConfig:
    """GPT-OSS 120B model dimensions."""

    # Core dimensions
    EMB_SIZE = 2880  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2880  # MoE FFN hidden dimension
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
    INTERMEDIATE_SIZE = 2880  # Dense FFN hidden dimension (same as MoE)
    HEAD_DIM = 64

    # MoE configuration
    NUM_ROUTED_EXPERTS = 128
    NUM_EXPERTS_PER_TOKEN = 4
    NUM_SHARED_EXPERTS = 0
    # GPT-OSS routes with no expert groups; selection is a plain top-k over all experts.
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    # Weights are a softmax over the selected top-k logits; no extra route scaling.
    ROUTE_SCALE = 1.0

    # Model architecture
    NUM_LAYERS = 36
    VOCAB_SIZE = 201088
    SLIDING_WINDOW = 128
    # TODO: HF config defines `layer_types` interleaving `sliding_attention` and `full_attention`
    # per layer. Decide whether to encode that pattern here or hardcode it at the call site.

    # Attention dimensions
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 8

    # Other
    RMS_NORM_EPS = 1e-5
    ROPE_THETA = 150000
    SWIGLU_LIMIT = 7.0
