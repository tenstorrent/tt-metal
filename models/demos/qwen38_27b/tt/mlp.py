# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dense SwiGLU MLP of Qwen3.5 (both block types): out = down(silu(gate(x)) * up(x)).

The input is the residual-stream layout [1, 1, 640, 5120] bf16 per chip (sequence split over mesh axis 0, replicated
over axis 1). TP over the mesh columns:
  gate / up column-parallel (17408 / 4 = 4352 intermediate columns per chip), fp32 outputs -> silu(gate) * up -> bf16
  -> down row-parallel (each chip holds its 4352 input rows), fp32 partial [1, 1, 640, 5120]
  -> all_reduce axis 1 in fp32 (a bf16 all_reduce scales the sum up by ~0.2%, known issue) -> typecast bf16.
No SP collective: the MLP is token-local.

bfloat8_b shares one exponent per 16 values along a tile row. gate / up are stored [K, N] (blocks along the outputs,
so the residual stream's few huge input channels keep their weights' bits); down is stored [N, K] and read with
transpose_b (blocks along K, so a large output column does not take its neighbours' exponent), as out_proj in
linear_attention.py.
"""

from __future__ import annotations

import ttnn

from .norm import compute_config
from .settings import settings


class TtMlp:
    def __init__(self, mesh, w: dict, tp_axis: int = 1):
        rows, cols = tuple(mesh.shape)
        self.tp_axis = tp_axis
        dram = ttnn.DRAM_MEMORY_CONFIG
        col_mapper = lambda dim: ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, dim))  # noqa: E731

        def up(t, dim):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=dram,
                mesh_mapper=col_mapper(dim),
            )

        gate = w["mlp.gate_proj.weight"].float()  # [I, H]
        upw = w["mlp.up_proj.weight"].float()  # [I, H]
        down = w["mlp.down_proj.weight"].float()  # [H, I]
        assert gate.shape[0] % cols == 0, f"intermediate {gate.shape[0]} not divisible by {cols} TP columns"
        self.w_gate = up(gate.T, -1)  # [H, I/cols] per chip
        self.w_up = up(upw.T, -1)
        self.w_down = up(down, -1)  # [H, I/cols] per chip, read transposed
        self.cfg = compute_config(settings.get("MLP_FIDELITY"), settings.get("MLP_FP32_ACC"))

    def __call__(self, x):
        dram = ttnn.DRAM_MEMORY_CONFIG
        g = ttnn.linear(x, self.w_gate, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=dram)
        u = ttnn.linear(x, self.w_up, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=dram)
        s = ttnn.silu(g, memory_config=dram)
        ttnn.deallocate(g)
        h = ttnn.multiply(s, u, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(s)
        ttnn.deallocate(u)
        partial = ttnn.linear(
            h,
            self.w_down,
            transpose_b=True,
            dtype=ttnn.float32,
            compute_kernel_config=self.cfg,
            memory_config=dram,
        )
        ttnn.deallocate(h)
        out = ttnn.all_reduce(partial, cluster_axis=self.tp_axis, memory_config=dram)
        ttnn.deallocate(partial)
        res = ttnn.typecast(out, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(out)
        return res
