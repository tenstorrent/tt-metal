# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 dense SwiGLU MLP (layer 0, intermediate 16384) on a 2x2 mesh, tensor parallel (TP=4) over the intermediate dim.

    gate_c = silu(x @ Wg_c)                 column-parallel [H, 4096] per chip (silu fused into the matmul)
    up_c   = x @ Wu_c                       column-parallel [H, 4096] per chip
    out    = all_reduce((gate_c * up_c) @ Wd_c)   row-parallel down [4096, H] per chip, one all_reduce [S, H]

16384 / 4 = 4096 is tile aligned, so no padding. Weights: the checkpoint's fp8 e4m3 + 128x128 block scale, dequantized
by the reference loader and stored bf16 (plan.yaml). Matmuls HiFi4 + fp32 acc (known issue: HiFi2 shrinks norms over
chained matmuls). From models/demos/gemma4_a4b_d_p/tt/mlp.py:TtDenseMLP (GeGLU -> SwiGLU, no pad).

2x2 port of models/demos/mimo_v2_6_d_p/tt/mlp.py (1x4). Changed: chip d = 2*row + col (ShardTensorToMesh over the
row-major device order) holds intermediate columns [4096d, 4096d + 4096), and the down reduce is
ttnn.all_reduce(cluster_axis=None), which runs axis 1 then axis 0 (all_reduce.cpp), i.e. over all 4 TP ranks.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


class TtDenseMLP:
    def __init__(self, mesh, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, dtype=ttnn.bfloat16):
        """HF weights: w_gate, w_up [I, H]; w_down [H, I]."""
        self.mesh = mesh
        n = mesh.get_num_devices()
        inter, hidden = w_gate.shape
        assert inter % (n * TILE) == 0, f"intermediate {inter} not tile aligned over {n} chips"
        self.per = inter // n
        # Column-parallel gate/up: chip c holds output columns [c * per, (c + 1) * per) -> shard [H, I] on dim -1.
        self.w_gate = self._shard(w_gate.float().T.reshape(1, 1, hidden, inter), -1, dtype)
        self.w_up = self._shard(w_up.float().T.reshape(1, 1, hidden, inter), -1, dtype)
        # Row-parallel down: chip c holds input rows [c * per, (c + 1) * per) -> shard [I, H] on dim -2.
        self.w_down = self._shard(w_down.float().T.reshape(1, 1, inter, hidden), -2, dtype)
        self.cfg = _hifi4()

    def _shard(self, t, dim, dtype):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=dim),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE (ffn_norm output). Returns replicated [1, 1, S, H] (all-reduced)."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        gate = ttnn.linear(x, self.w_gate, activation="silu", compute_kernel_config=self.cfg, memory_config=mc)
        up = ttnn.linear(x, self.w_up, compute_kernel_config=self.cfg, memory_config=mc)
        h = ttnn.mul(gate, up, memory_config=mc)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        o = ttnn.linear(h, self.w_down, compute_kernel_config=self.cfg, memory_config=mc)
        ttnn.deallocate(h)
        # 2x2: reduce over both mesh axes (all 4 TP ranks); one axis alone would drop half the shards.
        out = ttnn.all_reduce(o, cluster_axis=None)
        ttnn.deallocate(o)
        return out
