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
    # 1D decode matmul core counts (single-P150 Qwen3.5-2B / 4B); 0 keeps ttnn-auto.
    decode_q_cores: int = 0
    decode_kv_cores: int = 0
    decode_o_cores: int = 0

    @classmethod
    def from_args(cls, args) -> "AttentionConfig":
        proj_cores = getattr(args, "p150_decode_proj_cores", {})
        return cls(
            num_heads=args.n_heads,
            num_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            norm_eps=args.norm_eps,
            max_seq_len=args.max_seq_len,
            decode_q_cores=proj_cores.get("attn_q", 0),
            decode_kv_cores=proj_cores.get("attn_kv", 0),
            decode_o_cores=proj_cores.get("o", 0),
        )
