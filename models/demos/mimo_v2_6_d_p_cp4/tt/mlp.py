# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 dense SwiGLU MLP (layer 0, intermediate 16384) under CP=4 on the 1x4 mesh, TP=1.

    out = (silu(x @ Wg) * (x @ Wu)) @ Wd       on chip c's own CP slice x [1, 1, S/4, H]

Every chip holds the whole gate/up [H, 16384] and down [16384, H] (bf16, 0.38 GiB); rows are independent, so no CCL.
Weights: the checkpoint's fp8 e4m3 + 128x128 block scale, dequantized to bf16 at load. Matmuls HiFi4 + fp32 acc
(known issue: HiFi2 shrinks norms over chained matmuls). From models/demos/mimo_v2_6_d_p/tt/mlp.py:TtDenseMLP
(column/row-parallel shards and the all_reduce removed).
"""

from __future__ import annotations

import torch

import ttnn


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


class TtDenseMLP:
    def __init__(self, mesh, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, dtype=ttnn.bfloat16):
        """HF weights: w_gate, w_up [I, H]; w_down [H, I]. Replicated on every chip."""
        self.mesh = mesh
        inter, hidden = w_gate.shape
        self.w_gate = self._replicate(w_gate.float().T.reshape(1, 1, hidden, inter), dtype)
        self.w_up = self._replicate(w_up.float().T.reshape(1, 1, hidden, inter), dtype)
        self.w_down = self._replicate(w_down.float().T.reshape(1, 1, inter, hidden), dtype)
        self.cfg = _hifi4()

    def _replicate(self, t, dtype):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: chip c's CP slice [1, 1, S/4, H] TILE (ffn_norm output). Returns the same shape and sharding."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        gate = ttnn.linear(x, self.w_gate, activation="silu", compute_kernel_config=self.cfg, memory_config=mc)
        up = ttnn.linear(x, self.w_up, compute_kernel_config=self.cfg, memory_config=mc)
        h = ttnn.mul(gate, up, memory_config=mc)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(h, self.w_down, compute_kernel_config=self.cfg, memory_config=mc)
        ttnn.deallocate(h)
        return out


def build_mlp(mesh, loader, layer: int) -> TtDenseMLP:
    """TtDenseMLP for the dense layer; fp8 + 128x128 block scale dequantized at load."""
    from models.demos.mimo_v2_6_d_p.reference.weights import fp8_weight

    p = f"model.layers.{layer}.mlp."
    wg, wu, wd = (fp8_weight(loader, p + f"{n}.weight", torch.float32) for n in ("gate_proj", "up_proj", "down_proj"))
    return TtDenseMLP(mesh, wg, wu, wd)
