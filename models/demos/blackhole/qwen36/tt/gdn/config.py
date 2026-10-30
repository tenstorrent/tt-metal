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
    # Decode conv state kept as K-1 tiled [1, 1, D] rows (no slice/re-tilize per tap); single-P150 2B/4B only.
    split_conv_state: bool = False
    # Decode delta rule with heads on the batch dim (gated_delta_rule_decode_head_batch_ttnn); same scope.
    decode_head_batch: bool = False
    # Mega in-projection with a and b zero-padded to whole tiles (no untilize/re-tilize to split b); same scope.
    tile_aligned_mega: bool = False
    # 1D decode matmul core counts for the mega in-projection and o_proj; 0 keeps ttnn-auto. Same scope.
    decode_in_cores: int = 0
    decode_o_cores: int = 0

    @classmethod
    def from_args(cls, args) -> "GDNConfig":
        p150_opt = getattr(args, "p150_decode_opt", False)
        proj_cores = getattr(args, "p150_decode_proj_cores", {})
        return cls(
            split_conv_state=p150_opt,
            decode_head_batch=p150_opt,
            tile_aligned_mega=p150_opt,
            decode_in_cores=proj_cores.get("gdn_in", 0),
            decode_o_cores=proj_cores.get("o", 0),
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
