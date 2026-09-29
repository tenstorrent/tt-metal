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

import torch

import ttnn
from models.demos.glm53_flash_d_p.reference.weights import PREFIX
from models.demos.glm53_flash_d_p.tt.common import hifi4_config

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
        self.w_gate = self._shard(w_gate.float().T.reshape(1, 1, hidden, inter), -1)
        self.w_up = self._shard(w_up.float().T.reshape(1, 1, hidden, inter), -1)
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
        g = ttnn.linear(x, self.w_gate, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=mc)
        gc = ttnn.minimum(g, lim, memory_config=mc)
        ttnn.deallocate(g)
        a = ttnn.silu(gc, memory_config=mc)
        ttnn.deallocate(gc)
        u = ttnn.linear(x, self.w_up, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=mc)
        uc = ttnn.clamp(u, min=-lim, max=lim, memory_config=mc)
        ttnn.deallocate(u)
        h = ttnn.multiply(a, uc, dtype=ttnn.float32, memory_config=mc)
        ttnn.deallocate(a)
        ttnn.deallocate(uc)
        o = ttnn.linear(h, self.w_down, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=mc)
        ttnn.deallocate(h)
        # 2x2: reduce over both mesh axes (all 4 TP ranks), in fp32: a bf16 all_reduce scales the sum by +0.19%.
        if split:
            from models.demos.glm53_flash_d_p.tt.common import scatter_rows

            r = scatter_rows(o)
        else:
            r = ttnn.all_reduce(o, cluster_axis=None, memory_config=mc)
        ttnn.deallocate(o)
        out = ttnn.typecast(r, ttnn.bfloat16, memory_config=mc)
        ttnn.deallocate(r)
        return out


def build_mlp(mesh, loader, cfg, layer: int, name: str = "mlp") -> TtDenseMLP:
    p = f"{PREFIX}layers.{layer}.{name}."
    w = [loader.weight(p + f"{n}_proj.weight") for n in ("gate", "up", "down")]
    return TtDenseMLP(mesh, *w, limit=cfg.swiglu_limit)
