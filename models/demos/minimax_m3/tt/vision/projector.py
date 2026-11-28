# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3 vision projector, as Hugging Face's MiniMaxM3VLMultiModalProjector computes it:

    x = mlp(x)                # each patch: 1280 -> 6144 -> 6144
    x = fold_patches(x, 4)    # each 2 x 2 block of patches into one row of 4 x 6144 = 24576
    x = merge_mlp(x)          # each vision token: 24576 -> 6144 -> 6144

Both MLPs are M3VLMLP: linears with bias and the exact GELU. The image processor puts the 4 patches of each 2 x 2
block next to each other, so the fold is a reshape.
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP


def fold_patches(x: ttnn.Tensor, merge_factor: int) -> ttnn.Tensor:
    """Join each merge_factor consecutive rows into one: (1, 1, L, D) -> (1, 1, L / merge_factor, merge_factor * D).

    Frees x. The reshape runs on a row-major copy, as it crosses tile rows.
    """
    rows, width = x.shape[-2], x.shape[-1]
    folded = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(x)
    folded = ttnn.reshape(folded, (1, 1, rows // merge_factor, width * merge_factor))
    return ttnn.to_layout(folded, ttnn.TILE_LAYOUT)


class M3VLProjector(LightweightModule):
    """MLP per patch, then each block of 4 patches folded into one row, then the merge MLP."""

    def __init__(self, mlp: M3VLMLP, merge_mlp: M3VLMLP, spatial_merge_size: int):
        super().__init__()
        self.mlp = mlp
        self.merge_mlp = merge_mlp
        self.merge_factor = int(spatial_merge_size) ** 2

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_projector: torch.nn.Module,  # Hugging Face's MiniMaxM3VLMultiModalProjector: linear_1/2, merge_linear_1/2
        dtype=ttnn.bfloat16,
    ) -> "M3VLProjector":
        return cls(
            mlp=M3VLMLP.from_linears(mesh_device, ref_projector.linear_1, ref_projector.linear_2, dtype=dtype),
            merge_mlp=M3VLMLP.from_linears(
                mesh_device, ref_projector.merge_linear_1, ref_projector.merge_linear_2, dtype=dtype
            ),
            spatial_merge_size=ref_projector.spatial_merge_size,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: (1, 1, L, hidden) -> (1, 1, L//merge_factor, text_hidden)."""
        x = self.mlp(x)
        x = fold_patches(x, self.merge_factor)
        return self.merge_mlp(x)
