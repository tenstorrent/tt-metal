# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A small DeepSeek-V4.1 configuration for fast device iteration.

Same schedule rules, block types and code paths as the released model, with every dimension shrunk while
keeping the device constraints (hidden and q_lora split over TP=4 in whole tiles, >= 32 heads after the
head->sequence reshard, 32-wide QDQ groups, experts at the smallest in-tree-tested count, top-k a multiple of 16).
Oracles at these dims take seconds on CPU; real dims stay the acceptance runs.
"""

from dataclasses import replace

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig


class SmallV41Config(DeepSeekV41FlashConfig):
    EMB_SIZE = 1024
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    MOE_INTERMEDIATE_SIZE = 256
    NUM_ROUTED_EXPERTS = 128  # smallest count the in-tree MoE dispatch is exercised with (32 hung in routing setup)
    NUM_ATTENTION_HEADS = 32
    HEAD_DIM = 128
    Q_LORA_RANK = 256
    O_LORA_RANK = 128
    INDEX_N_HEADS = 32
    INDEX_HEAD_DIM = 128
    INDEX_TOPK = 128  # selected / visible rows ~ 0.5 at S=512, ratio 2 (real: 512 of 1024 at S=2048)
    CANDIDATE_TOPK_BLOCKS = 32  # 32 x 8 = 256 >= INDEX_TOPK rows, half of the 64 visible blocks


def small_spec(layers: tuple[int, ...], seq_len: int, *, seed: int = 0):
    """An oracle spec for V4.1 layers ``layers`` (their real roles) at SmallV41Config dims, synthetic weights."""
    c = SmallV41Config
    spec = orc.real_spec(layers, seq_len, candidate_topk_blocks=c.CANDIDATE_TOPK_BLOCKS, seed=seed)
    args = replace(
        spec.args,
        dim=c.EMB_SIZE,
        moe_inter_dim=c.MOE_INTERMEDIATE_SIZE,
        n_routed_experts=c.NUM_ROUTED_EXPERTS,
        n_heads=c.NUM_ATTENTION_HEADS,
        head_dim=c.HEAD_DIM,
        q_lora_rank=c.Q_LORA_RANK,
        o_lora_rank=c.O_LORA_RANK,
        index_n_heads=c.INDEX_N_HEADS,
        index_head_dim=c.INDEX_HEAD_DIM,
        index_topk=c.INDEX_TOPK,
    )
    return replace(spec, args=args, model_args=replace(spec.model_args, dim=c.EMB_SIZE))
