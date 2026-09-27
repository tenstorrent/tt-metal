# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 RMSNorm on a 1xN mesh, replicated: y = w * (x * (mean(x^2) + eps)^-0.5) (plain w, not 1 + w).

From models/demos/gemma4_a4b_d_p/tt/rms_norm.py: the weight is a TILE [1, 1, 1, H] gamma, the kernel runs HiFi4 with
fp32 accumulation (the normalization is in fp32 inside the kernel, as HF computes it in fp32). ttnn.rms_norm multiplies
by the weight as given, so the checkpoint weight is used unchanged.
"""

from __future__ import annotations

import os

import torch

import ttnn


class TtRMSNorm:
    def __init__(self, mesh, weight: torch.Tensor | None, eps: float = 1e-6, dtype=ttnn.bfloat16):
        self.mesh = mesh
        self.eps = eps
        self.dtype = dtype
        self.weight = None
        if weight is not None:
            self.weight = ttnn.from_torch(
                weight.float().reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE on device. Returns the same shape, replicated.

        MIMO_NORM_IMPL=bringup (default) runs ttnn.bringup.rms_norm, the AI-generated perf-optimized drop-in
        (ttnn/ttnn/bringup/rms_norm_ttnn); MIMO_NORM_IMPL=native runs ttnn.rms_norm."""
        op = ttnn.rms_norm if os.environ.get("MIMO_NORM_IMPL", "bringup") == "native" else ttnn.bringup.rms_norm
        return op(
            x,
            weight=self.weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def to_device_replicated(mesh, h: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Host [S, H] -> replicated device [1, 1, S, H] TILE (harness boundary only, never inside a forward)."""
    return ttnn.from_torch(
        h.reshape(1, 1, *h.shape[-2:]).to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def replicated_to_host(t: ttnn.Tensor) -> torch.Tensor:
    """Replicated device [1, 1, S, H] -> host [S, H] (chip 0's copy; harness boundary only)."""
    first = ttnn.get_device_tensors(t)[0]
    return ttnn.to_torch(first).reshape(t.shape[-2], t.shape[-1])
