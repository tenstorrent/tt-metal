# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3 vision patch embedding as one matmul.

Hugging Face runs a Conv3d whose stride is its kernel, with no bias. The image processor already flattens each patch
into a row of 1176 values. So the Conv3d is a matmul of those rows with its weight flattened to [1280, 1176]. There
is no position embedding and no class token.
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.common import as_linear_weight, linear, linear_compute_config


class M3VLPatchEmbed(LightweightModule):
    """Patch embedding as one matmul with no bias: (L, 1176) -> (L, 1280)."""

    def __init__(
        self,
        mesh_device,
        proj_weight: torch.Tensor,  # [out_dim, 3, 2, 14, 14] (Conv3d)
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        # Flattened to [out_dim, 1176] in the order of a patch row's values, then ttnn.linear convention [in, out].
        self.weight = as_linear_weight(mesh_device, proj_weight.flatten(1), dtype)
        self.compute_kernel_config = linear_compute_config()

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_proj: torch.nn.Conv3d,
        dtype=ttnn.bfloat16,
    ) -> "M3VLPatchEmbed":
        assert (
            isinstance(ref_proj, torch.nn.Conv3d) and ref_proj.stride == ref_proj.kernel_size and ref_proj.bias is None
        ), f"expected a Conv3d with stride equal to its kernel and no bias, got {ref_proj}"
        return cls(mesh_device=mesh_device, proj_weight=ref_proj.weight.data, dtype=dtype)

    def forward(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """Project each patch row: (..., L, 1176) -> (..., L, out_dim)."""
        return linear(x_flat, self.weight, None, self.compute_kernel_config)
