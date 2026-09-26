# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 dense MLP (GeGLU, intermediate 2112) on a 1x4 mesh, tensor parallel over the intermediate dim.

    gate_up = x @ [up_c | gate_c]           column-parallel, one fused matmul per chip [H, 2 * 544]
    h       = gelu(gate_c) * up_c           apply_geglu (Accurate gelu; tanh vs exact is below bf16 noise here)
    out     = all_reduce(h @ down_c)        row-parallel down [544, H] per chip

2112 / 4 = 528 is not tile aligned, so each chip's slice is padded to 544: zero columns in gate/up (gelu(0) * 0 = 0)
and zero rows in down, so the pad contributes nothing. The all-reduce comes before post_mlp_norm (the norm is nonlinear).

Adapted from models/demos/gemma4/tt/shared_mlp.py (per-chip padding and [up_i | gate_i] interleave) and
models/demos/gemma4/tt/experts/operations.py:apply_geglu. Weights bf16 (plan.yaml: dense mlp bf16), matmuls HiFi4 + fp32 acc.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.gemma4.tt.experts.operations import apply_geglu

NUM_CHIPS = 4
TILE = 32


def _hifi4():
    # HiFi4, not HiFi2: HiFi2 shrank every token's output norm by 0.3-1.4% (rel L2 0.0083 vs 0.0033 on the layer-0 golden).
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


class TtDenseMLP:
    def __init__(self, mesh, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, dtype=ttnn.bfloat16):
        """HF weights: w_gate, w_up [I, H]; w_down [H, I]."""
        self.mesh = mesh
        n = NUM_CHIPS
        inter = w_gate.shape[0]
        per = -(-inter // n)
        self.per = -(-per // TILE) * TILE  # 528 -> 544
        gate_t, up_t, down_t = w_gate.float().T, w_up.float().T, w_down.float().T  # [H, I], [H, I], [I, H]
        hidden = gate_t.shape[0]

        gu, dn = [], []
        for c in range(n):
            lo, hi = c * per, min((c + 1) * per, inter)
            g = torch.zeros(hidden, self.per)
            u = torch.zeros(hidden, self.per)
            d = torch.zeros(self.per, hidden)
            g[:, : hi - lo] = gate_t[:, lo:hi]
            u[:, : hi - lo] = up_t[:, lo:hi]
            d[: hi - lo] = down_t[lo:hi]
            gu.append(torch.cat([u, g], dim=-1))
            dn.append(d)
        self.w_gate_up = self._shard(torch.stack(gu)[:, None], dtype)  # [4, 1, H, 1088] -> [1, 1, H, 1088] per chip
        self.w_down = self._shard(torch.stack(dn)[:, None], dtype)  # [4, 1, 544, H] -> [1, 1, 544, H] per chip

    def _shard(self, t, dtype):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE (ffn_norm output). Returns replicated [1, 1, S, H] (all-reduced)."""
        s = x.shape[-2]
        p = self.per
        gate_up = ttnn.linear(x, self.w_gate_up, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, s, p])
        gate = ttnn.slice(gate_up, [0, 0, 0, p], [1, 1, s, 2 * p])
        ttnn.deallocate(gate_up)
        h = apply_geglu(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        o = ttnn.linear(h, self.w_down, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        out = ttnn.all_reduce(o, cluster_axis=1)
        ttnn.deallocate(o)
        return out
