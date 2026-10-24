# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Zero-centred RMSNorm of Qwen3.5 (attn_norm, mlp_norm, final norm): out = x * rsqrt(mean(x^2) + eps) * (1 + w).

The residual stream is [1, 1, 640, 5120] bf16 per chip (sequence split over mesh axis 0, replicated over axis 1),
so the norm is local: one fused ttnn.rms_norm, no collective. The ``1 +`` is added in fp32 at load (the reference
adds it in fp32 at use), and the gamma is replicated on every chip as a [1, 1, 1, H] TILE tensor: a ROW_MAJOR gamma
turns off the op's streaming (large-tensor) kernel for a TILE input, and a 5120-wide row then overflows L1.
"""

from __future__ import annotations

import torch

import ttnn

from .settings import settings


def compute_config(fidelity: str, fp32_acc: bool):
    return ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE,
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


class TtRmsNorm:
    def __init__(self, mesh, weight: torch.Tensor, eps: float):
        self.eps = eps
        dim = weight.shape[-1]
        gamma = (1.0 + weight.float()).reshape(1, 1, 1, dim)
        wdtype = ttnn.float32 if settings.get("NORM_WEIGHT_DTYPE") == "float32" else ttnn.bfloat16
        self.weight = ttnn.from_torch(
            gamma,
            dtype=wdtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.cfg = compute_config(settings.get("NORM_FIDELITY"), settings.get("NORM_FP32_ACC"))

    def __call__(self, x):
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.cfg,
        )
