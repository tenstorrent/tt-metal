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
    # moe_fused_swiglu, the rest to unified_routed_expert_moe. Measured under SwiGLU-OAI -- the
    # activation these experts actually run -- on the 2880x2880 routed-expert shape: the fused op
    # wins only at 256 (0.75x) and loses from 512 on, so 256 is both the last M_BLOCK boundary it
    # wins at and the aggregate-optimal cut (+0.04% against a per-count oracle).
    # Not enabled, for two independent reasons. Nothing forwards a threshold: the gpt-oss MoE
    # builds TtRoutedExpert directly and passes none. And moe_fused_swiglu has no bias inputs,
    # so TtRoutedExpert rejects a threshold outright whenever use_expert_bias is on -- the fused
    # band would silently drop the gate/up/down biases the composite band applies. The measured
    # crossover is kept under _MEASURED so it is not re-derived.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED = 256
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
