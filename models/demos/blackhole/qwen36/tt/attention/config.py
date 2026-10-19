# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    norm_eps: float
    max_seq_len: int
    # Projection weight dtype; None = the shipped bfloat8_b. args.weight_dtype (bf16) under QWEN36_EXACT_PROFILE=1.
    weight_dtype: object = None
    # QWEN36_EXACT_PROFILE=1: HiFi2 matmuls (bf16 operands) instead of the shipped LoFi.
    exact_profile: bool = False

    @classmethod
    def from_args(cls, args) -> "AttentionConfig":
        return cls(
            num_heads=args.n_heads,
            num_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            norm_eps=args.norm_eps,
            max_seq_len=args.max_seq_len,
            weight_dtype=args.weight_dtype if getattr(args, "exact_profile", False) else None,
            exact_profile=bool(getattr(args, "exact_profile", False)),
        )
