# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 clamped SwiGLU MLP on a 2x2 mesh, TP=4 over the intermediate dim: the dense MLP (layers 0-2, intermediate
12288, 3072 per chip) and the MoE layers' shared expert (``name="mlp.shared_experts"``, intermediate 2048, 512 per chip).

    g_c   = x @ Wg_c, u_c = x @ Wu_c                 column-parallel [H, I/4] per chip, fp32 out
    h_c   = silu(min(g_c, L)) * clamp(u_c, -L, L)    L = swiglu_limit (10)
    out   = all_reduce(h_c @ Wd_c)                   row-parallel [I/4, H] per chip, cluster_axis=None (axis 1, then 0)

Intermediates and the down partials stay fp32 through the all_reduce; the output is cast to bf16.

Chip d (row-major device order) holds intermediate columns [d I/4, (d + 1) I/4). Weights: fp8 e4m3 + 128x128 block
scale dequantized by the reference loader, stored bf16. Every matmul HiFi4 + fp32 acc.
From models/demos/mimo_v2_6_d_p_2x2/tt/mlp.py:TtDenseMLP (fused silu -> explicit clamps).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.glm53_flash_d_p.reference.weights import PREFIX
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, mm_config
from models.demos.glm53_flash_d_p.tt.mm_configs import linear_config

TILE = 32


class TtDenseMLP:
    def __init__(self, mesh, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, limit: float = 10.0):
        """HF weights: w_gate, w_up [I, H]; w_down [H, I]."""
        self.mesh = mesh
        self.limit = float(limit)
        n = mesh.get_num_devices()
        inter, hidden = w_gate.shape
        assert inter % (n * TILE) == 0, f"intermediate {inter} not tile aligned over {n} chips"
        self.per = inter // n
        # gate and up as one [H, 2 I/n] weight per chip (chip c: its gate columns, then its up columns): one matmul with
        # N = 2 I/n (shared expert 2 x 0.297 -> 0.261 ms, tests/test_matmul_tune.py), outputs sliced apart
        gt, ut = w_gate.float().T, w_up.float().T  # [H, I]
        per = self.per
        gu = torch.cat(
            [torch.cat([gt[:, c * per : (c + 1) * per], ut[:, c * per : (c + 1) * per]], -1) for c in range(n)], -1
        )
        self.w_gu = self._shard(gu.reshape(1, 1, hidden, 2 * inter), -1)
        self.w_down = self._shard(w_down.float().T.reshape(1, 1, inter, hidden), -2)
        self.cfg = hifi4_config()

    def _shard(self, t, dim):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=dim),
        )

    def __call__(self, x: ttnn.Tensor, split: bool = False) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE bf16 (ffn_norm output). Returns replicated [1, 1, S, H] bf16.
        split: return this chip's [1, 1, S/4, H] quarter instead (reduce_scatter on both axes, fp32)."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        lim = self.limit
        gu = ttnn.linear(
            x,
            self.w_gu,
            dtype=ttnn.float32,
            program_config=linear_config(x, self.w_gu, ttnn.float32),
            compute_kernel_config=mm_config(self.cfg),
            memory_config=mc,
        )
        s_, per = gu.shape[-2], self.per
        g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, s_, per), memory_config=mc)
        u = ttnn.slice(gu, (0, 0, 0, per), (1, 1, s_, 2 * per), memory_config=mc)
        ttnn.deallocate(gu)
        gc = ttnn.minimum(g, lim, memory_config=mc)
        ttnn.deallocate(g)
        a = ttnn.silu(gc, memory_config=mc)
        ttnn.deallocate(gc)
        uc = ttnn.clamp(u, min=-lim, max=lim, memory_config=mc)
        ttnn.deallocate(u)
        h = ttnn.multiply(a, uc, dtype=ttnn.float32, memory_config=mc)
        ttnn.deallocate(a)
        ttnn.deallocate(uc)
        # split + bf16 fabric reduce-scatter: emit the partial in bf16 (it is cast to bf16 for the reduce-scatter anyway;
        # halves the 5120 x 4096 output write that bounds this K=256 matmul)
        odt = (
            ttnn.bfloat16
            if split and os.environ.get("GLM_SCATTER_OP", "fabric_ring") in ("fabric_bf16", "fabric_ring")
            else ttnn.float32
        )
        o = ttnn.linear(
            h,
            self.w_down,
            dtype=odt,
            program_config=linear_config(h, self.w_down, odt),
            compute_kernel_config=mm_config(self.cfg),
            memory_config=mc,
        )
        ttnn.deallocate(h)
        # reduce over both mesh axes: split -> scatter_rows (bf16 fabric_reduce_scatter by default, see common.py); the
        # replicated path keeps an fp32 all_reduce (a bf16 all_reduce scaled the sum by +0.19% on the 2x2 mesh).
        if split:
            from models.demos.glm53_flash_d_p.tt.common import scatter_rows

            r = scatter_rows(o)
        else:
            r = ttnn.all_reduce(o, cluster_axis=None, memory_config=mc)
        ttnn.deallocate(o)
        if r.dtype == ttnn.bfloat16:
            return r
        out = ttnn.typecast(r, ttnn.bfloat16, memory_config=mc)
        ttnn.deallocate(r)
        return out


def build_mlp(mesh, loader, cfg, layer: int, name: str = "mlp") -> TtDenseMLP:
    p = f"{PREFIX}layers.{layer}.{name}."
    w = [loader.weight(p + f"{n}_proj.weight") for n in ("gate", "up", "down")]
    return TtDenseMLP(mesh, *w, limit=cfg.swiglu_limit)
