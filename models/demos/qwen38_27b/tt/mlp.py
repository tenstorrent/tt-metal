# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dense SwiGLU MLP (the `mlp` step of both block types) of Qwen3.8-27B on the 8 x 4 mesh.

Per chip (SP row r, TP column c), on its own 640 rows ([1, 1, 640, 5120] bf16, replicated over TP):
  1. column-parallel [gate | up]: intermediate channels [4352 c, 4352 (c + 1)) of each, in one matmul (fp32 out).
  2. silu(gate) * up -> bf16 [1, 1, 640, 4352].
  3. row-parallel down_proj on the column's 4352 inputs (fp32 partials) -> all_reduce over axis 1 -> bf16 replicated
     residual layout.
"""

import torch

import ttnn

from .settings import settings


def compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("MLP_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("MLP_FP32_ACC"),
        packer_l1_acc=False,
    )


class TtMLP:
    """mlp of one layer. Weights from the checkpoint dict ``w`` (``mlp.*`` keys, bf16)."""

    def __init__(self, mesh, cfg, w: dict, prefix: str = "mlp."):
        self.mesh = mesh
        tp = mesh.shape[1]
        self.ci = cfg.inter // tp  # 4352 intermediate channels per column
        ci = self.ci
        self.ckc = compute_config()
        gate, up = (
            w[prefix + "gate_proj.weight"],
            w[prefix + "up_proj.weight"],
        )  # bf16 on the host (half of fp32's 1 GB)
        # Weights stay [out, in] (HF layout) and run with transpose_b: the bfloat8_b exponent blocks then run along the
        # reduction dim (see gdn.py). Column c: [gate rows of c | up rows of c] = [2 * 4352, hidden].
        gu = torch.cat([torch.cat([gate[c * ci : (c + 1) * ci], up[c * ci : (c + 1) * ci]]) for c in range(tp)], 0)
        self.w_gu = self._shard(gu, dim=2)
        # row-parallel down_proj [hidden, inter]: column c takes inputs [4352 c, 4352 (c + 1)): [hidden, 4352]
        self.w_down = self._shard(w[prefix + "down_proj.weight"], dim=3)

    def _shard(self, t: torch.Tensor, dim: int) -> ttnn.Tensor:
        """2-D t as [1, 1, *t.shape] bfloat8_b, split in tp equal blocks along dim (2 = rows, 3 = columns): block c on
        TP column c, replicated over the SP rows."""
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            device=self.mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, dim)),
        )

    def _linear(self, x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        """x @ w^T (w stored [out, in]), fp32 out."""
        return ttnn.linear(
            x,
            w,
            transpose_b=True,
            compute_kernel_config=self.ckc,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        dram, ci = ttnn.DRAM_MEMORY_CONFIG, self.ci
        gu = self._linear(x, self.w_gu)  # [1, 1, rows, 2 * 4352] fp32
        rows = gu.shape[2]
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, rows, ci], memory_config=dram)
        up = ttnn.slice(gu, [0, 0, 0, ci], [1, 1, rows, 2 * ci], memory_config=dram)
        ttnn.deallocate(gu)
        sg = ttnn.silu(gate, memory_config=dram)
        ttnn.deallocate(gate)
        h = ttnn.multiply(sg, up, dtype=ttnn.bfloat16, memory_config=dram)
        for t in (sg, up):
            ttnn.deallocate(t)
        part = self._linear(h, self.w_down)  # [1, 1, rows, hidden] fp32 partial of this column's inputs
        ttnn.deallocate(h)
        red = ttnn.all_reduce(part, cluster_axis=1, memory_config=dram)
        ttnn.deallocate(part)
        out = ttnn.typecast(red, ttnn.bfloat16)
        ttnn.deallocate(red)
        return out
