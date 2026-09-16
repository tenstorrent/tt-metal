# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel SwiGLU MLP (layer-0 dense MLP and the MoE shared expert). Returns the row-parallel PARTIAL sum."""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.weights import as_device_tensor, linear_weight


class TPSwiGLU:
    """gate/up column-parallel [H, I/tp]; down row-parallel [I/tp, H]. ``forward`` returns the per-chip partial."""

    def __init__(self, mesh_device, w_gate, w_up, w_down, *, name: str, cache_path: Path | None, dtype=ttnn.bfloat8_b):
        self.mesh_device = mesh_device
        kw = dict(cache_path=cache_path, dtype=dtype)
        self.gate = as_device_tensor(
            mesh_device, None if w_gate is None else linear_weight(w_gate), name=f"{name}.gate", shard_dim=-1, **kw
        )
        self.up = as_device_tensor(
            mesh_device, None if w_up is None else linear_weight(w_up), name=f"{name}.up", shard_dim=-1, **kw
        )
        self.down = as_device_tensor(
            mesh_device, None if w_down is None else linear_weight(w_down), name=f"{name}.down", shard_dim=-2, **kw
        )
        arch = mesh_device.arch()
        self.compute = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,S,H] replicated -> partial [1,1,S,H] (needs an all-reduce across the mesh)."""
        g = ttnn.linear(
            x, self.gate, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG, activation="silu"
        )
        u = ttnn.linear(x, self.up, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.multiply(g, u, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(h, self.down, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        return out
