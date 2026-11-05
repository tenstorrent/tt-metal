# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL multimodal projector (spatial merge folds into it).

Per `MiniMaxM3VLMultiModalProjector.forward`:
    x = linear_2(gelu(linear_1(x)))                    # per-patch: 1280 -> 6144 -> 6144
    x = x.reshape(L // merge**2, -1)                   # group 4 patches -> 24576
    x = merge_linear_2(merge_act(merge_linear_1(x)))   # 24576 -> 6144 -> 6144

so here an M3VLMLP, fold_patches and a second M3VLMLP.

There is no pre-norm. The 4 grouped patches are consecutive in the
sequence (the image processor pre-orders patches so each group of
spatial_merge**2 is a 2x2 spatial block, the ordering the 3D RoPE coords
use). All four linears carry bias; the activation is the accurate (erf)
GELU (`ACT2FN["gelu"]`).
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP


def fold_patches(x: ttnn.Tensor, merge_factor: int) -> ttnn.Tensor:
    """(1, 1, L, D) tile tensor -> (1, 1, L // merge_factor, merge_factor * D): each merge_factor consecutive patches
    become one row. The reshape crosses tile rows, so it runs on a row-major copy.

    Consumes x: it is freed once its row-major copy exists, so the fold holds two copies of the activations at a time,
    not three.
    """
    rows, width = x.shape[-2], x.shape[-1]
    folded = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(x)
    folded = ttnn.reshape(folded, (1, 1, rows // merge_factor, width * merge_factor))
    return ttnn.to_layout(folded, ttnn.TILE_LAYOUT)


class M3VLProjector(LightweightModule):
    """Per-patch MLP -> spatial-merge fold (4 patches) -> merge MLP."""

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
