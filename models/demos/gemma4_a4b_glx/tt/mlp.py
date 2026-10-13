# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 dense MLP (GeGLU, intermediate 2112) for prefill on the 8x4 Galaxy mesh.

out = down(gelu_tanh(gate(x)) * up(x)). TP=4 over the mesh columns: gate/up are column-parallel, down is
row-parallel, then all_reduce over the columns (cluster_axis=1). The input is replicated within each row.
2112 / 4 = 528 is not tile aligned, so each chip's slice is zero-padded to 544: the padded gate/up columns give
gelu(0) * 0 = 0 and the padded down rows are zero, so the padding adds nothing.
"""

import torch

import ttnn

from .settings import settings

_FIDELITY = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}
_GELU = {"Tanh": ttnn.GeluVariant.Tanh, "Accurate": ttnn.GeluVariant.Accurate}

TP_AXIS = 1
TILE = 32


class TtMLP:
    def __init__(self, mesh, w_gate, w_up, w_down):
        """w_gate, w_up: HF [I, H]; w_down: HF [H, I]."""
        self.mesh = mesh
        tp = mesh.shape[TP_AXIS]
        inter = w_gate.shape[0]
        per = inter // tp
        assert per * tp == inter, f"intermediate {inter} over TP={tp}"
        per_pad = ((per + TILE - 1) // TILE) * TILE
        self.per_pad = per_pad

        def pad_cols(w_t):  # [H, I] -> [H, tp * per_pad], chip c's columns in block c
            out = torch.zeros(w_t.shape[0], tp * per_pad, dtype=torch.float32)
            for c in range(tp):
                out[:, c * per_pad : c * per_pad + per] = w_t[:, c * per : (c + 1) * per].float()
            return out

        gate = pad_cols(w_gate.t())
        up = pad_cols(w_up.t())
        down = pad_cols(w_down).t().contiguous()  # [tp * per_pad, H]
        self.w_gate = self._weight(gate, dim=-1)
        self.w_up = self._weight(up, dim=-1)
        self.w_down = self._weight(down, dim=-2)

        self.mm_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=_FIDELITY[settings.get("MLP_MATMUL_FIDELITY")],
            math_approx_mode=False,
            fp32_dest_acc_en=settings.get("MLP_MATMUL_FP32_ACC"),
            packer_l1_acc=False,
        )
        self.gelu = _GELU[settings.get("MLP_GELU_VARIANT")]

    def _weight(self, w, dim):
        return ttnn.from_torch(
            w.reshape(1, 1, *w.shape),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, tuple(self.mesh.shape), dims=(None, dim)),
        )

    def __call__(self, x):
        """x: [1, 1, S, H] bf16, replicated within each row. Returns [1, 1, S, H] bf16, replicated within each row."""
        gate = ttnn.linear(x, self.w_gate, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        up = ttnn.linear(x, self.w_up, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        act = ttnn.gelu(gate, variant=self.gelu)
        ttnn.deallocate(gate)
        h = ttnn.mul(act, up)
        ttnn.deallocate(act)
        ttnn.deallocate(up)
        partial = ttnn.linear(h, self.w_down, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(h)
        out = ttnn.all_reduce(partial, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        return out
