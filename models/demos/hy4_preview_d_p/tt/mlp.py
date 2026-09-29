# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 dense SwiGLU MLP (layer 0, intermediate 18432, unclamped) on the 2x2 mesh: TP=2 over the columns (axis 1),
SP=2 over the rows (axis 0), per plan.md.

    gate_c = x @ Wg_c                        column-parallel [H, 9216] on chip column c, fp32 out
    up_c   = x @ Wu_c                        column-parallel [H, 9216], fp32 out
    h_c    = silu(gate_c) * up_c             ttnn.multiply with a SILU input activation, fp32
    part_c = h_c @ Wd_c                      row-parallel down [9216, H], fp32 partial [S/2, H]
    out    = reduce_scatter(part, dim 3, axis 1)   -> [S/2, H/2] fp32, the residual's column split

Chip (r, c) holds intermediate columns [9216c, 9216c + 9216) of gate / up and the matching rows of down (replicated
over the rows). Weights are bf16 as stored. Every matmul runs HiFi4 + fp32 dest acc, and gate / up / h stay fp32
(known issues: HiFi2 shrinks output norms; bf16 intermediates widen the per-token norm ratio). No clamp: the
swiglu_limit applies only to the routed experts.

From models/demos/mimo_v2_6_d_p_2x2/tt/mlp.py:TtDenseMLP (flat TP=4 + all_reduce) -> TP=2 over axis 1 with the
input's rows split over axis 0 and a reduce_scatter over axis 1 (tt/attention.py's o_proj epilogue).
No host work in __call__.
"""

from __future__ import annotations

import torch

import ttnn

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
        # Column-parallel gate / up: [H, I] split on the last dim over the TP axis, replicated over the other.
        self.w_gate = self._shard(w_gate.T.reshape(1, 1, hidden, inter), 3)
        self.w_up = self._shard(w_up.T.reshape(1, 1, hidden, inter), 3)
        # Row-parallel down: [I, H] split on dim -2 over the TP axis.
        self.w_down = self._shard(w_down.T.reshape(1, 1, inter, hidden), 2)
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
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
        """x: ffn_norm [1, 1, S/2, H] TILE per chip (rows split over axis 0, replicated over the TP axis).
        Returns mlp_out [1, 1, S/2, H/2] fp32 (rows over axis 0, columns over the TP axis)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        g = ttnn.linear(x, self.w_gate, dtype=self.mid, compute_kernel_config=self.ckc, memory_config=dram)
        u = ttnn.linear(x, self.w_up, dtype=self.mid, compute_kernel_config=self.ckc, memory_config=dram)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=self.mid, memory_config=dram)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        part = ttnn.linear(h, self.w_down, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(h)
        out = ttnn.reduce_scatter(part, dim=3, cluster_axis=self.tp_axis, memory_config=dram)
        ttnn.deallocate(part)
        return out
