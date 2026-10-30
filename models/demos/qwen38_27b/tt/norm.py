# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decoder RMSNorm of Qwen3.8-27B on the device: x * rsqrt(mean(x^2) + eps) * (1 + w).

Token-local: each chip normalises its own SP rows [1, 1, 640, 5120] (replicated over the 4 TP columns) against the
full replicated weight; no collective. The (1 + w) offset is folded into the weight once at load time.
"""

import torch

import ttnn

from .settings import settings


def compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("NORM_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("NORM_FP32_ACC"),
        packer_l1_acc=False,
    )


class TtRMSNorm:
    """input_layernorm / post_attention_layernorm / final norm (weight stored as w, applied as 1 + w)."""

    def __init__(self, mesh, weight: torch.Tensor, eps: float):
        dim = weight.numel()
        self.eps = eps
        # TILE gamma: a ROW_MAJOR gamma disables rms_norm's streaming (large-tensor) kernel, and the whole-row
        # kernel's buffers for a 5120-wide row exceed L1
        gamma = (1.0 + weight.float()).reshape(1, 1, 1, dim)
        self.weight = ttnn.from_torch(
            gamma,
            device=mesh,
            dtype=getattr(ttnn, settings.get("NORM_WEIGHT_DTYPE")),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.cfg = compute_config()

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.cfg,
        )
