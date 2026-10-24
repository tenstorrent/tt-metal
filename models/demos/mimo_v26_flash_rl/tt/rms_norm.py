# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm on the device: ``w * x / rms(x)`` over the hidden dim (MiMo: plain ``w``, not ``1 + w``), eps from config.

Runs on each chip's local rows of the residual stream ([1, 1, 160, 4096] bf16 on the 8x4 mesh); the weight is
replicated bf16. The all_gather over TP that the column-parallel projection after the norm needs belongs to that
projection's module, so the norm output stays on the local rows like the residual.
"""

from __future__ import annotations

import torch

import ttnn

from .settings import settings


def compute_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("NORM_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("NORM_FP32_ACC"),
        packer_l1_acc=False,
    )


class RMSNorm:
    def __init__(self, mesh, weight: torch.Tensor, eps: float):
        self.eps = eps
        self.weight = ttnn.from_torch(
            weight.reshape(1, 1, 1, -1).to(torch.bfloat16),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.compute_kernel_config = compute_config(mesh)

    def __call__(self, x, memory_config=None):
        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.eps,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
