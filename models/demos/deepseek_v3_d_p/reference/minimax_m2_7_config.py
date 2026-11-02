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
    # Divisor on combine_fabric2d's forwarding region -- see DeepSeekV3Config.COMBINE_FWD_REDUCTION for
    # what it trades. Held at 8 where the larger models take 16: measured 8x4, going to 16 costs this
    # model 4.8% on hot_expert traffic to save only 20 -> 10 MB a chip.
    COMBINE_FWD_REDUCTION = 8
    # Pages one of dispatch_fabric2d's forwarding channels holds. The region is split by movement class,
    # so no depth can deadlock the ring -- this trades DRAM against throughput and nothing else. A channel
    # is a pipeline buffer, so the figure is a page count rather than anything scaled from the sequence
    # length. Measured 8x4 at ring8 / 5k over 24..96 pages: the shallowest depth inside 5% of the dense
    # region on both traffic shapes, at -1.3% balanced and +0.2% hot_expert.
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
