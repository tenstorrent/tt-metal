# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 dense SwiGLU MLP (layers 0-1, intermediate 9216) on the 4x2 mesh: TP=2 over the columns (axis 1), SP=4
over the rows (axis 0), per plan.md. Also usable for the MoE layers' shared expert (prefix ``mlp.shared_experts.``).

    gate_c = x @ Wg_c                        column-parallel [3584, 4608] on chip column c, bf16 W, fp32 out
    up_c   = x @ Wu_c                        column-parallel [3584, 4608], fp32 out
    h_c    = silu(gate_c) * up_c             ttnn.multiply with a SILU input activation, fp32
    part_c = h_c @ Wd_c                      row-parallel down [4608, 3584], fp32 partial [S/4, 3584]
    out    = reduce_scatter(part, dim 3, axis 1)   -> [S/4, 1792] fp32, the residual's column split

Chip (r, c) holds intermediate columns [4608c, 4608c + 4608) of gate / up and the matching rows of down (replicated
over the rows). Every matmul runs HiFi4 + fp32 dest acc.

From models/demos/hy4_preview_d_p/tt/mlp.py:TtDenseMLP (TP=2 over axis 1, same structure; here SP=4 rows).
No host work in __call__.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.xing40_a4b_d_p.tt.settings import settings

TILE = 32


class TtDenseMLP:
    def __init__(
        self, mesh, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, tp_axis: int = 1, mid=ttnn.float32
    ):
        """HF weights: w_gate, w_up [I, H]; w_down [H, I]. ``mid``: dtype of gate / up / h (float32; bfloat16 for
        comparison)."""
        self.mesh, self.tp_axis, self.mid = mesh, tp_axis, mid
        tp = mesh.shape[tp_axis]
        inter, hidden = w_gate.shape
        assert inter % (tp * TILE) == 0 and hidden % (tp * TILE) == 0, (inter, hidden, tp)
        self.per = inter // tp
        self.w_gate = self._shard(w_gate.T.reshape(1, 1, hidden, inter), 3)
        self.w_up = self._shard(w_up.T.reshape(1, 1, hidden, inter), 3)
        self.w_down = self._shard(w_down.T.reshape(1, 1, inter, hidden), 2)
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _shard(self, t: torch.Tensor, dim: int) -> ttnn.Tensor:
        dims = [None, None]
        dims[self.tp_axis] = dim
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=tuple(dims)),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: ffn_norm [1, 1, S/4, H] TILE per chip (rows split over axis 0, replicated over the TP axis).
        Returns mlp_out [1, 1, S/4, H/2] fp32 (rows over axis 0, columns over the TP axis)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        g = ttnn.linear(x, self.w_gate, dtype=self.mid, compute_kernel_config=self.ckc, memory_config=dram)
        u = ttnn.linear(x, self.w_up, dtype=self.mid, compute_kernel_config=self.ckc, memory_config=dram)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=self.mid, memory_config=dram)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        part = ttnn.linear(h, self.w_down, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(h)
        out = ttnn.reduce_scatter(
            part, dim=3, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear, memory_config=dram
        )
        ttnn.deallocate(part)
        return out


def build_mlp(mesh, loader, cfg, layer: int, prefix: str = "mlp.") -> TtDenseMLP:
    p = f"model.layers.{layer}.{prefix}"
    return TtDenseMLP(
        mesh,
        loader.get(p + "gate_proj.weight").float(),
        loader.get(p + "up_proj.weight").float(),
        loader.get(p + "down_proj.weight").float(),
        tp_axis=1,
    )
