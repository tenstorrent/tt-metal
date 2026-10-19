# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static configuration for the Qwen3.5-9B Gated DeltaNet (linear attention) layer."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GDNConfig:
    num_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    q_dim: int
    k_dim: int
    v_dim: int
    long_prefill_chunk_size: int = 128
    # Projection weight dtype (qkv / in_proj_a / in_proj_b / in_proj_z / out_proj and their fused copies); None = the
    # shipped bfloat8_b. args.weight_dtype (bf16) under QWEN36_EXACT_PROFILE=1.
    weight_dtype: object = None
    # QWEN36_EXACT_PROFILE=1: HiFi2 projections (bf16 operands) instead of the shipped LoFi.
    exact_profile: bool = False

    @classmethod
    def from_args(cls, args) -> "GDNConfig":
        return cls(
            num_heads=args.linear_num_key_heads,
            num_v_heads=args.linear_num_value_heads,
            head_k_dim=args.linear_key_head_dim,
            head_v_dim=args.linear_value_head_dim,
            conv_kernel_size=args.linear_conv_kernel_dim,
            norm_eps=args.norm_eps,
            q_dim=args.linear_q_dim,
            k_dim=args.linear_k_dim,
            v_dim=args.linear_v_dim,
            weight_dtype=args.weight_dtype if getattr(args, "exact_profile", False) else None,
            exact_profile=bool(getattr(args, "exact_profile", False)),
        )
