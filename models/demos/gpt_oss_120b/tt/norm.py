# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS RMSNorm (attn_norm, ffn_norm, final_norm): out = w * x / sqrt(mean(x^2) + eps), plain weight (no +1).

The residual stream holds the full hidden (2880 = 90 tiles) on every chip (plan.md: split by sequence over axis 0,
replicated over axis 1), so the norm is local: one ttnn.rms_norm, no collective. Weight bf16 ROW_MAJOR [90, 32],
replicated; output bf16. Compute precision from tt/settings.py (NORM_FIDELITY, NORM_FP32_ACC).
"""

import torch

import ttnn

from .settings import settings


def norm_compute_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("NORM_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("NORM_FP32_ACC"),
        packer_l1_acc=False,
    )


class TtRmsNorm:
    def __init__(self, mesh, weight: torch.Tensor, eps: float):
        self.eps = float(eps)
        self.weight = ttnn.from_torch(
            weight.reshape(1, 1, -1, ttnn.TILE_SIZE).to(torch.bfloat16),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.compute = norm_compute_config(mesh)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )
