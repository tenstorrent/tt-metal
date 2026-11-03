# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3/R1 671B Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for DeepSeek-V3.
"""


class DeepSeekV3Config:
    """DeepSeek V3/R1 671B model dimensions."""

    # Core dimensions
    EMB_SIZE = 7168  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 18432  # Dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    # Pages one MoE forwarding channel holds, for dispatch_fabric2d, combine_fabric2d and the routed expert
    # overlapped with combine alike: their buffers come out of the same free DRAM at different times, so one
    # value, the larger of the ops' needs, costs nothing over two. Deeper trades DRAM for throughput. Picked as
    # the shallowest depth with both ops within 5% of their best on hot_expert traffic, measured 8x4 ring8 / 5k
    # over 32..96 pages; balanced traffic is flat in depth.
    # 64 costs dispatch 5.3% on hot_expert. Dispatch 35 MB a chip, combine 27 MB.
    MOE_FWD_VC_PAGES = 80
    NUM_SHARED_EXPERTS = 1
    NUM_EXPERT_GROUPS = 8
    NUM_LIMITED_GROUPS = 4

    # Gate-test device-mode scores bar. pcc_scores sorts both sides, so this measures the
    # selected-weight distribution rather than slot alignment; 256 experts, top-8, group-limited floors at
    # 0.9950 on a 2x4 Blackhole mesh, the tightest reachable shape.
    GATE_SCORES_PCC_DEVICE = 0.985

    # Model architecture
    NUM_LAYERS = 61
    NUM_DENSE_LAYERS = 3
    VOCAB_SIZE = 129280

    # MLA dimensions
    NUM_ATTENTION_HEADS = 128
    Q_LORA_RANK = 1536
    KV_LORA_RANK = 512
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 128

    # Other
    RMS_NORM_EPS = 1e-6
    ROUTE_SCALE = 2.5
