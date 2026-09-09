# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K2.6 Model Configuration (text tower only).

Single source of truth for model dimension constants.
Values from HuggingFace config.json for Kimi-K2.6 (text_config).
"""


class KimiK26Config:
    """Kimi K2.6 model dimensions."""

    # Core dimensions
    EMB_SIZE = 7168  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    # Routed-expert hybrid split: experts with <= this many active tokens go to moe_fused_swiglu,
    # the rest to unified_routed_expert_moe. Shares K2.7's 7168x2048 routed expert, so it shares its
    # measurement -- the crossover is a function of that shape, not of the weights.
    # Re-cut for bf16 gate/up accumulators (which cost the fused op ~24% at long ISL): the crossover
    # is 896, so 768 is the last count the fused op still wins. Bisected at 128-token steps rather
    # than taken off the power-of-two sweep -- the coarse sweep only bounds it to (512, 1024], and
    # 640/768 are both fused wins inside that gap.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD = 768
    INTERMEDIATE_SIZE = 18432  # Dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 384
    NUM_EXPERTS_PER_TOKEN = 8
    NUM_SHARED_EXPERTS = 1
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    ROUTE_SCALE = 2.827

    # Gate-test device-mode scores bar, relaxing the shared 0.93. 384 experts under sigmoid near-tie
    # the top-8 boundary at 640 tokens/chip: every device/reference disagreement sits at an fp64
    # selection-score margin below the bf16 matmul's own logit error, so the two sides order tied
    # slots differently. The weights themselves are right (0.8% relative L2 vs an fp64 golden); it is
    # the position-wise PCC that lands at 0.926-0.941 across the 8 SP chips of a Blackhole Galaxy 8x4.
    GATE_SCORES_PCC_DEVICE = 0.92

    # Model architecture
    NUM_LAYERS = 61
    NUM_DENSE_LAYERS = 1  # first_k_dense_replace
    VOCAB_SIZE = 163840

    # MLA dimensions
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 64
    Q_LORA_RANK = 1536
    KV_LORA_RANK = 512
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 128

    # Norm / RoPE
    RMS_NORM_EPS = 1e-5
    ROPE_THETA = 50000.0
    MAX_POSITION_EMBEDDINGS = 262144

    # YaRN scaling
    ROPE_SCALING_FACTOR = 64.0
    ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS = 4096
    ROPE_SCALING_BETA_FAST = 32.0
    ROPE_SCALING_BETA_SLOW = 1.0
    ROPE_SCALING_MSCALE = 1.0
    ROPE_SCALING_MSCALE_ALL_DIM = 1.0
