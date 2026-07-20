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
        )
