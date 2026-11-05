# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL patch embedding (Conv3d-as-Linear).

The HF reference patch embed is `nn.Conv3d(3, 1280, kernel=(2,14,14),
stride=(2,14,14), bias=False)`. The M3 image processor (Qwen2-VL lineage)
pre-cuts each image into non-overlapping spatio-temporal patches and
*flattens* them, so `pixel_values` already arrives as `(L, 1176)` where
1176 = 3 (channels) * 2 (temporal_patch) * 14 * 14. On such a patch the
Conv3d reduces to a single dot product per output channel, that is a
per-patch Linear from 1176 to 1280, with no bias.

We implement exactly that: flatten the Conv3d weight `[1280, 3, 2, 14, 14]`
to `[1280, 1176]` and run one `ttnn.linear`. `from_torch` takes the Conv3d,
such as Hugging Face's, whose shape gives the patch geometry.

There is no learned position embedding (M3-VL is RoPE-only) and no class
token, so the projection output feeds straight into `pre_layrnorm`.

Reference: `MiniMaxM3VLVisionEmbeddings.proj` (Conv3d).
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.common import as_linear_weight, linear, linear_compute_config


class M3VLPatchEmbed(LightweightModule):
    """Conv3d-as-Linear patch projection: (L, 1176) -> (L, 1280), no bias."""

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
        """Apply the per-patch projection.

        Args:
            x_flat: device tensor of shape (..., L, 1176).

        Returns:
            device tensor of shape (..., L, out_dim).
        """
        return linear(x_flat, self.weight, None, self.compute_kernel_config)
