# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniMax M2.7 Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for MiniMax-M2.7.
"""


class MiniMaxM27Config:
    """MiniMax M2.7 model dimensions."""

    # Core dimensions
    EMB_SIZE = 3072  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 1536  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 1536  # Dense FFN hidden dimension (same as MoE)

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    # Pages per combine_fabric2d forwarding channel, trading DRAM against throughput. Swept at ring8 / 5k:
    # hot_expert reads 1145, 1052, 958, 913, 909, 909, 911 us over 32, 48, 64, 80, 96, 112, 134 pages, flat
    # from 80 up, and 64 already costs 5%. Balanced never fills a channel, so only hot_expert constrains it.
    COMBINE_FWD_VC_PAGES = 80
    # Pages per dispatch_fabric2d forwarding channel. Shallowest depth inside 5% at
    # ring8 / 5k: -1.3% balanced, +0.2% hot_expert.
    DISPATCH_FWD_VC_PAGES = 32
    NUM_SHARED_EXPERTS = 0  # shared_intermediate_size = 0
    # MiniMax routes with a single expert group -> gate collapses to plain top-k.
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    # Reference route_tokens_to_experts sum-normalizes with no extra scaling.
    ROUTE_SCALE = 1.0

    # Gate-test device-mode scores bar. pcc_scores sorts both sides, so this measures the
    # selected-weight distribution rather than slot alignment; 256 experts, top-8 floors at
    # 0.9977 on a 2x4 Blackhole mesh, the tightest reachable shape.
    GATE_SCORES_PCC_DEVICE = 0.987

    # Model architecture
    NUM_LAYERS = 62
    VOCAB_SIZE = 200064
    HEAD_DIM = 128

    # Attention dimensions
    NUM_ATTENTION_HEADS = 48
    NUM_KEY_VALUE_HEADS = 8
    ROTARY_DIM = 64

    # Other
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 5000000
