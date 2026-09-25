# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 RMSNorm on a 1xN mesh, replicated: y = x * (mean(x^2) + eps)^-0.5 * w (plain w, not 1 + w).

Adapted from models/demos/gemma4/tt/rms_norm.py (prefill path): the weight is a TILE [1, 1, 1, H] gamma (bounds L1 for
the fp32 intermediates on wide prefill rows) and the kernel runs HiFi4 with fp32 accumulation. ttnn.rms_norm multiplies
by the weight as given, so the checkpoint weight is used unchanged.
"""

from __future__ import annotations

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
        """x: replicated [1, 1, S, H] TILE on device. Returns the same shape, replicated."""
        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def to_device_replicated(mesh, h: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Host [S, H] -> replicated device [1, 1, S, H] TILE."""
    return ttnn.from_torch(
        h.reshape(1, 1, *h.shape[-2:]).to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def replicated_to_host(t: ttnn.Tensor) -> torch.Tensor:
    """Replicated device [1, 1, S, H] -> host [S, H] (chip 0's copy)."""
    first = ttnn.get_device_tensors(t)[0]
    return ttnn.to_torch(first).reshape(t.shape[-2], t.shape[-1])
