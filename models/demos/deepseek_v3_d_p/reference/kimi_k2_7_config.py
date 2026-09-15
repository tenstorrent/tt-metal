# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K2.7-Code Model Configuration (text tower only).

Single source of truth for model dimension constants.
Values from HuggingFace config.json for Kimi-K2.7-Code (``text_config``).

Deliberately standalone rather than subclassing ``KimiK26Config``. The two generations agree on every
dimension today, so inheritance would have been shorter -- but it would also mean an edit made for
K2.6 silently moved K2.7, and it would hide which values K2.7 actually asserts.
"""


class KimiK27Config:
    """Kimi K2.7-Code model dimensions."""

    # Core dimensions
    EMB_SIZE = 7168  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    # Routed-expert hybrid split: experts with <= this many active tokens go to
    # moe_fused_swiglu, the rest to unified_routed_expert_moe. The two ops cross twice on the
    # 7168x2048 routed-expert shape: the composite's cost is flat inside an M chunk while the
    # fused op's rises with the count, so the composite wins 320-512, loses 576-768 where the
    # tail per_core_M rounds 18 tile-rows up to 32, and wins outright from 896. 768 is the
    # aggregate-optimal cut over that sawtooth (+0.13% against a per-count oracle, worst cell
    # +21% at 512), not a single crossing -- there is none.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD = 768
    INTERMEDIATE_SIZE = 18432  # Dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 384
    NUM_EXPERTS_PER_TOKEN = 8
    NUM_SHARED_EXPERTS = 1
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    ROUTE_SCALE = 2.827

    # Gate-test device-mode scores bar. pcc_scores sorts both sides, so this measures the
    # selected-weight distribution rather than slot alignment; 384 experts, top-8 floors at
    # 0.9935 on a 2x4 Blackhole mesh, the tightest reachable shape.
    GATE_SCORES_PCC_DEVICE = 0.983

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
