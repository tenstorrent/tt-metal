# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Full MiniMax-M3-VL vision tower (+ optional projector).

Forward pipeline (matches `MiniMaxM3VLVisionModel.forward` then
`MiniMaxM3VLMultiModalProjector`):

    pixel_values (L, 1176) host
        -> patch_embed       (Conv3d-as-Linear 1176 -> 1280)          device
        -> pre_layrnorm
        -> 32 x encoder blocks
              (LN1 -> attn(3D RoPE, full SDPA) -> residual; LN2 -> MLP -> residual)
        -> [tower output (L, 1280)]
        -> projector         (per-patch MLP -> merge 4 patches -> merge MLP)
        -> vision tokens (L // 4, 6144)

There is NO post-encoder LayerNorm and NO attention windowing — the tower
attends fully. The 3D-RoPE cos/sin are precomputed host-side from
`image_grid_thw` and pushed to device once per forward (shared by all
blocks). Weights come from the `_m3_loader` reference module tree.
"""
from __future__ import annotations

from typing import List, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3_vl.tt.block import M3VLBlock
from models.demos.minimax_m3_vl.tt.layernorm import M3VLLayerNorm
from models.demos.minimax_m3_vl.tt.patch_embed import M3VLPatchEmbed
from models.demos.minimax_m3_vl.tt.projector import M3VLProjector
from models.demos.minimax_m3_vl.tt.rope import rope_cos_sin_padded


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


class M3VLVisionModel(LightweightModule):
    """Full M3-VL vision tower + optional multimodal projector."""

    def __init__(
        self,
        mesh_device,
        model_args,
        patch_embed: M3VLPatchEmbed,
        pre_layrnorm: M3VLLayerNorm,
        blocks: List[M3VLBlock],
        projector: Optional[M3VLProjector] = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.args = model_args
        self.dtype = dtype
        self.patch_embed = patch_embed
        self.pre_layrnorm = pre_layrnorm
        self.blocks = blocks
        self.projector = projector

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        reference,  # VisionReference from _m3_loader.build_reference
        model_args,
        with_projector: bool = True,
        dtype=ttnn.bfloat16,
    ) -> "M3VLVisionModel":
        patch_embed = M3VLPatchEmbed.from_torch(mesh_device, reference.patch_embed, dtype=dtype)
        pre_layrnorm = M3VLLayerNorm.from_torch(mesh_device, reference.pre_layrnorm, dtype=dtype)
        blocks = [
            M3VLBlock.from_torch(
                mesh_device,
                ref_layer,
                hidden_size=model_args.hidden_size,
                num_heads=model_args.num_attention_heads,
                head_dim=model_args.head_dim,
                dtype=dtype,
            )
            for ref_layer in reference.layers
        ]
        projector = (
            M3VLProjector.from_torch(
                mesh_device, reference.projector, spatial_merge_size=model_args.spatial_merge_size, dtype=dtype
            )
            if with_projector
            else None
        )
        return cls(
            mesh_device=mesh_device,
            model_args=model_args,
            patch_embed=patch_embed,
            pre_layrnorm=pre_layrnorm,
            blocks=blocks,
            projector=projector,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    def _push_pixel_values(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """Push flattened pixel_values (L, 1176) to device as (1, 1, L, 1176)."""
        L, D = pixel_values.shape
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if _is_mesh_device(self.device) else None
        return ttnn.from_torch(
            pixel_values.to(torch.bfloat16).view(1, 1, L, D).contiguous(),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def _stage_rope(self, image_grid_thw: torch.Tensor):
        cos_pt, sin_pt = rope_cos_sin_padded(
            image_grid_thw.to(torch.int64),
            head_dim=self.args.head_dim,
            padded_head_dim=self.args.padded_head_dim,
            theta=self.args.rope_theta,
            spatial_merge_size=self.args.spatial_merge_size,
        )
        return self.blocks[0].attention.stage_cos_sin(cos_pt, sin_pt)

    # ------------------------------------------------------------------
    def forward_tower(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Run patch_embed -> pre_layrnorm -> 32 blocks. Returns (1, 1, L, 1280)."""
        x_tt = self._push_pixel_values(pixel_values)
        x_tt = self.patch_embed(x_tt)
        x_tt = self.pre_layrnorm(x_tt)
        cos_tt, sin_tt = self._stage_rope(image_grid_thw)
        for blk in self.blocks:
            x_tt = blk(x_tt, cos_tt, sin_tt)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
        return x_tt

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Full forward. With projector: (1, 1, L//4, 6144); else the tower out (1, 1, L, 1280)."""
        x_tt = self.forward_tower(pixel_values, image_grid_thw)
        if self.projector is None:
            return x_tt
        out_tt = self.projector(x_tt)
        ttnn.deallocate(x_tt)
        return out_tt
