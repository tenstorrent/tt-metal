# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dense SwiGLU MLP of MiMo layer 0 on the [8, 4] mesh: intermediate split over TP axis 1.

Per chip, one chunk of 5120 tokens:
    x [1, 1, 160, 4096] local rows -> all_gather over TP -> [640, 4096] (the SP row's tokens)
    -> gate / up linear with this rank's 4096 intermediate rows -> silu(gate) * up [640, 4096]
    -> down linear on this rank's 4096 input columns (partial sums, fp32) -> reduce_scatter over TP
    -> [1, 1, 160, 4096] bf16

Weights are the checkpoint's fp8 block-scaled tensors dequantized, rounded to bf16 and then to bfloat8_b, kept in the
[out, in] orientation and run with transpose_b, so the bfp8 blocks run along K (attention's lesson: [in, out] blocks
put single output columns far off).
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl
from .settings import settings


def rank_weights(ck, layer_idx: int, tp: int):
    """bf16 [out, in] per-TP-rank weights: gate, up [tp, I / tp, H] (rank r's intermediate rows) and down
    [tp, H, I / tp] (rank r's input columns).

    The dequantized weights are rounded to bf16 (the reference's weight dtype) before ttnn makes them bfloat8_b: its
    fp32 -> bfloat8_b conversion lands farther from the weights than bf16 -> bfloat8_b (layer 0, host emulation:
    output column 525 rel 0.029 from fp32 vs 0.009 from bf16)."""
    p = f"model.layers.{layer_idx}.mlp"
    wg, wu, wd = (ck.fp8(f"{p}.{n}").to(torch.bfloat16) for n in ("gate_proj", "up_proj", "down_proj"))
    inter, hidden = wg.shape
    li = inter // tp
    return (
        wg.view(tp, li, hidden).contiguous(),
        wu.view(tp, li, hidden).contiguous(),
        wd.view(hidden, tp, li).permute(1, 0, 2).contiguous(),
    )


class DenseMLP:
    def __init__(self, mesh, ck, layer_idx: int):
        self.mesh = mesh
        self.sp, self.tp = tuple(mesh.shape)
        wg, wu, wd = rank_weights(ck, layer_idx, self.tp)
        self.w_gate = self._sharded_weight(wg.unsqueeze(0))  # [1, tp, N, K] -> per chip [1, 1, N, K]
        self.w_up = self._sharded_weight(wu.unsqueeze(0))
        self.w_down = self._sharded_weight(wd.unsqueeze(0))
        self.mm_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("MLP_FIDELITY")),
            math_approx_mode=False,
            fp32_dest_acc_en=settings.get("MLP_FP32_ACC"),
            packer_l1_acc=True,
        )
        self.inter_dtype = ttnn.float32 if settings.get("MLP_INTER_FP32") else ttnn.bfloat16
        self.out_fp32 = settings.get("MLP_OUT_FP32")

    def _sharded_weight(self, w: torch.Tensor):
        return ttnn.from_torch(
            w,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, 1)),
        )

    def _linear(self, x, w, dtype):
        return ttnn.linear(
            x,
            w,
            transpose_b=True,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_cfg,
        )

    def __call__(self, x):
        """x [1, 1, 160, H] bf16 (this chip's ffn_norm rows) -> [1, 1, 160, H] bf16."""
        xg = ccl.gather_rows(x)
        g = self._linear(xg, self.w_gate, self.inter_dtype)
        u = self._linear(xg, self.w_up, self.inter_dtype)
        xg.deallocate(True)
        h = ttnn.multiply(
            g,
            u,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)],
            dtype=self.inter_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        g.deallocate(True)
        u.deallocate(True)
        partial = self._linear(h, self.w_down, ttnn.float32 if self.out_fp32 else ttnn.bfloat16)
        h.deallocate(True)
        y = ccl.scatter_rows(partial)
        partial.deallocate(True)
        if self.out_fp32:  # the TP partials summed in fp32, rounded to bf16 once
            y32 = y
            y = ttnn.typecast(y32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y32.deallocate(True)
        return y
