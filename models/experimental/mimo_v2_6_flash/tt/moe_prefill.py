# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo MoE prefill experts on one TP=8 row (``TtMiMoMoE.forward``): x [1, 1, T, hidden] and every token's top-8
experts and routing weights -> this chip's partial output [1, 1, T, hidden], the routing-weighted sum over the
token's experts of its 256-wide slice of the intermediate dim. The caller all-reduces it across the row.

The experts are the layer's BFP4 tensors, shared with decode: w_gate_up [1, 256, hidden, 2 * 256] per chip
(columns [gate | up] of this chip's slice) and w_down [1, 256, 256, hidden], DRAM interleaved tiles.

Expert-major: the T x 8 (token, k) pairs are sorted by expert on the host (the routing is read back once per
call) and each expert's pairs are padded to whole 32-row groups, so every group belongs to one expert. The
token rows are gathered into that order (``embedding``), and an indexed ``sparse_matmul`` runs each group against
its own expert's weights (gate|up, SwiGLU, x routing weight, down), at most 256 groups per call. The pairs' rows
are gathered back k-major and summed over k. Each expert's weights are read once per 32-row group of its pairs
(~1100 groups at T = 4096) instead of once per 32 tokens that touch it.
"""

import torch

import ttnn

from .common import TILE


class PrefillExperts:
    def __init__(self, moe):
        self.moe = moe
        self.all_experts = None  # the sparsity operand indexed sparse_matmul takes (not read)

    def _upload(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        moe = self.moe
        return ttnn.from_torch(
            t, device=moe.plan.mesh_device, dtype=dtype, layout=layout, mesh_mapper=moe.plan.replicate()
        )

    def plan(self, indices, weights, num_tokens):
        """Host plan from chip 0's routing (identical on every chip): per packed row its token and routing weight
        (padding rows: token 0, weight 0), each group's expert, and each (k, token) pair's packed row."""
        E, K = self.moe.num_experts, indices.shape[-1]
        ids = ttnn.to_torch(ttnn.get_device_tensors(indices)[0]).reshape(-1).long()
        wts = ttnn.to_torch(ttnn.get_device_tensors(weights)[0]).reshape(-1).float()
        n = ids.numel()
        order = torch.argsort(ids, stable=True)
        counts = torch.bincount(ids, minlength=E)
        groups = (counts + TILE - 1) // TILE
        first = torch.cumsum(counts, 0) - counts  # first sorted pair of each expert
        first_row = (torch.cumsum(groups, 0) - groups) * TILE  # first packed row of each expert
        sorted_ids = ids[order]
        dest = first_row[sorted_ids] + torch.arange(n) - first[sorted_ids]
        rows = int(groups.sum()) * TILE
        row_token = torch.zeros(rows, dtype=torch.int32)
        row_token[dest] = (order // K).to(torch.int32)
        row_weight = torch.zeros(rows)
        row_weight[dest] = wts[order]
        pair_row = torch.empty(n, dtype=torch.int32)
        pair_row[order] = dest.to(torch.int32)
        pair_row = pair_row.reshape(num_tokens, K).T.reshape(1, -1)  # k-major: entry k * T + t
        group_expert = torch.repeat_interleave(torch.arange(E, dtype=torch.int32), groups)
        return row_token, row_weight, group_expert, pair_row

    def __call__(self, x, indices, weights):
        """x [1, 1, T, hidden] (T a multiple of 32), indices [1, 1, T, 8] (uint), weights [1, 1, T, 8] (fp32,
        normalized) -> partial [1, 1, T, hidden] bf16."""
        moe = self.moe
        T, H, I, E = x.shape[-2], moe.hidden, moe.inter_local, moe.num_experts
        K = indices.shape[-1]
        if self.all_experts is None:
            self.all_experts = self._upload(torch.ones(1, 1, 1, E), ttnn.bfloat16)
        row_token, row_weight, group_expert, pair_row = self.plan(indices, weights, T)
        G = group_expert.numel()

        row_token_tt = self._upload(row_token.reshape(1, -1), ttnn.uint32)
        row_weight_tt = self._upload(row_weight.reshape(1, G, TILE, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        x_rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (T, H))
        packed = ttnn.embedding(row_token_tt, x_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, G * 32, H]
        packed = ttnn.reshape(packed, (1, G, TILE, H))
        x_rm.deallocate(True)
        row_token_tt.deallocate(True)

        outs = []
        for g0 in range(0, G, E):  # indexed sparse_matmul takes at most E groups per call
            g1 = min(G, g0 + E)
            ids = self._upload(group_expert[g0:g1].reshape(1, 1, 1, -1), ttnn.uint16)
            mm = dict(
                sparsity=self.all_experts,
                indices=ids,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=ttnn.Tile([TILE, TILE]),
                dtype=ttnn.bfloat16,
                compute_kernel_config=moe.expert_config,
            )
            xa = ttnn.slice(packed, (0, g0, 0, 0), (1, g1, TILE, H))
            gate_up = ttnn.sparse_matmul(xa, moe.w_gate_up, program_config=moe.gate_up_program, **mm)
            xa.deallocate(True)
            act = moe._swiglu(ttnn.reshape(gate_up, (1, g1 - g0, TILE, 2 * I)))
            w = ttnn.slice(row_weight_tt, (0, g0, 0, 0), (1, g1, TILE, 1))
            act_w = ttnn.mul(act, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            act.deallocate(True)
            w.deallocate(True)
            down = ttnn.sparse_matmul(act_w, moe.w_down, program_config=moe.down_program, **mm)  # [1, g, 32, H]
            act_w.deallocate(True)
            ids.deallocate(True)
            outs.append(down)
        packed.deallocate(True)
        row_weight_tt.deallocate(True)

        y = ttnn.concat(outs, dim=1) if len(outs) > 1 else outs[0]
        if len(outs) > 1:
            for t in outs:
                t.deallocate(True)
        y_rm = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), (G * TILE, H))
        y.deallocate(True)
        pair_row_tt = self._upload(pair_row, ttnn.uint32)
        pairs = ttnn.embedding(pair_row_tt, y_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, K * T, H]
        y_rm.deallocate(True)
        pair_row_tt.deallocate(True)
        out = ttnn.sum(ttnn.reshape(pairs, (K, T, H)), dim=0, keepdim=True)
        pairs.deallocate(True)
        return ttnn.reshape(out, (1, 1, T, H))
