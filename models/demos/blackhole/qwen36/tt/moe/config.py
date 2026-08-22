# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE config, derived from the parsed HF text config via Qwen36ModelArgs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    """Per-layer MoE parameters. Attribute names (num_experts / top_k / hidden_size /
    moe_intermediate_size) match what the sparse expert forward passes expect."""

    hidden_size: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    shared_intermediate_size: int  # 0/None when the checkpoint has no shared expert
    norm_topk_prob: bool
    num_devices: int

    @classmethod
    def from_args(cls, args) -> "MoEConfig":
        return cls(
            hidden_size=args.dim,
            num_experts=args.moe_num_experts,
            top_k=args.moe_top_k,
            moe_intermediate_size=args.moe_intermediate_size,
            shared_intermediate_size=args.moe_shared_intermediate_size or 0,
            norm_topk_prob=args.moe_norm_topk_prob,
            num_devices=getattr(args, "num_devices", 1),
        )
