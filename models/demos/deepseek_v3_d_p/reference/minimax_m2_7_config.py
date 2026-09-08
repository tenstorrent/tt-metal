# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniMax M2.7 Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for MiniMax-M2.7.
"""


from models.demos.common.prefill.fabric import moe_fabric_payload_size


class MiniMaxM27Config:
    """MiniMax M2.7 model dimensions."""

    # Core dimensions
    EMB_SIZE = 3072  # embedding dimension
    FABRIC_PAYLOAD_SIZE = moe_fabric_payload_size(EMB_SIZE)
    MOE_INTERMEDIATE_SIZE = 1536  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 1536  # Dense FFN hidden dimension (same as MoE)

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
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
