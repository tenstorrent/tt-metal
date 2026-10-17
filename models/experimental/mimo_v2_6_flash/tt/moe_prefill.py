# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo MoE prefill experts on one TP=8 row (``TtMiMoMoE.forward``): x [1, 1, T, hidden] and the dense routing
weights [1, 1, T, 256] -> this chip's partial output [1, 1, T, hidden], the routing-weighted sum over every token's
top-8 experts of its 256-wide slice of the intermediate dim. The caller all-reduces it across the row.

The experts are the layer's BFP4 tensors, shared with decode: w_gate_up [1, 256, hidden, 2 * 256] per chip
(columns [gate | up] of this chip's slice) and w_down [1, 256, 256, hidden], DRAM interleaved tiles.

Tokens are grouped 32 per tile row; ``sparse_matmul`` computes an expert for a group only if some token in the
group routes to it (the union mask), and rows of tokens that did not pick that expert get routing weight 0. A
group of 32 tokens touches ~160 of the 256 experts, so every 32 tokens read most of the expert weights again.
"""

import ttnn

from .common import TILE


class PrefillExperts:
    TOKENS_PER_CALL = 128  # bounds the [groups, 256, 32, hidden] expert-output buffer

    def __init__(self, moe):
        self.moe = moe

    def __call__(self, x, dense):
        """x [1, 1, T, hidden], dense [1, 1, T, E] (T a multiple of 32) -> partial [1, 1, T, hidden]."""
        num_tokens, hidden, experts = x.shape[-2], self.moe.hidden, self.moe.num_experts
        step = self.TOKENS_PER_CALL
        if num_tokens <= step:
            return self._experts(x, dense)
        parts = []
        for s in range(0, num_tokens, step):
            e = min(s + step, num_tokens)
            xs = ttnn.slice(x, (0, 0, s, 0), (1, 1, e, hidden))
            ds = ttnn.slice(dense, (0, 0, s, 0), (1, 1, e, experts))
            parts.append(self._experts(xs, ds))
            xs.deallocate(True)
            ds.deallocate(True)
        out = ttnn.concat(parts, dim=2)
        for p in parts:
            p.deallocate(True)
        return out

    def _experts(self, x, dense):
        """x [1, 1, T, hidden], dense [1, 1, T, E] with T a multiple of 32 -> partial [1, 1, T, hidden]."""
        moe = self.moe
        num_tokens = x.shape[-2]
        groups = num_tokens // TILE
        E, H, I = moe.num_experts, moe.hidden, moe.inter_local

        union = ttnn.max(ttnn.reshape(dense, (1, groups, TILE, E)), dim=2, keepdim=True)  # [1, G, 1, E]
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        union.deallocate(True)

        x4 = ttnn.reshape(x, (1, groups, TILE, H))
        out_tile = ttnn.Tile([TILE, TILE])
        mm_kwargs = dict(
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=out_tile,
            dtype=ttnn.bfloat16,
            compute_kernel_config=moe.expert_config,
        )
        gate_up = ttnn.sparse_matmul(x4, moe.w_gate_up, program_config=moe.gate_up_program, **mm_kwargs)
        act = moe._swiglu(ttnn.reshape(gate_up, (groups, E, TILE, 2 * I)))

        down_sparsity = ttnn.reshape(sparsity, (1, 1, groups, E))
        down = ttnn.sparse_matmul(
            act,
            moe.w_down,
            sparsity=down_sparsity,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=out_tile,
            dtype=ttnn.bfloat16,
            compute_kernel_config=moe.expert_config,
            program_config=moe.down_program,
        )  # [G, E, 32, H]
        act.deallocate(True)
        sparsity.deallocate(True)

        rw = ttnn.transpose(ttnn.reshape(dense, (groups, TILE, E)), -2, -1)  # [G, E, 32]
        rw = ttnn.reshape(rw, (groups, E, TILE, 1))
        weighted = ttnn.mul(down, rw, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down.deallocate(True)
        rw.deallocate(True)
        out = ttnn.sum(weighted, dim=1, keepdim=True)  # [G, 1, 32, H]
        weighted.deallocate(True)
        return ttnn.reshape(out, (1, 1, num_tokens, H))
