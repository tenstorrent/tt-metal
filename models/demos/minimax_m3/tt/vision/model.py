# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Full MiniMax-M3-VL vision tower (+ optional projector).

Forward pipeline (matches `MiniMaxM3VLVisionModel.forward` then
`MiniMaxM3VLMultiModalProjector`):

    pixel_values (L, 1176) host
        -> patch_embed       (Conv3d-as-Linear 1176 -> 1280)          device
        -> pre_layrnorm
        -> encoder blocks, one per reference layer (32 in the checkpoint)
              (LN1 -> attn(3D RoPE, full SDPA) -> residual; LN2 -> MLP -> residual)
        -> [tower output (L, 1280)]
        -> projector         (per-patch MLP -> merge 4 patches -> merge MLP)
        -> vision tokens (L // 4, 6144)

There is NO post-encoder LayerNorm and NO attention windowing — the tower
attends fully. The 3D-RoPE cos/sin depend only on `image_grid_thw`: they are
computed on host and pushed to DRAM with the first image of a grid, then
kept for the next images of that grid; each forward copies them to L1 once
for all blocks. Weights come from the Torch reference (reference/vision.py):
seeded random weights in the unit tests, the checkpoint's through
loader.build_reference.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.block import M3VLBlock
from models.demos.minimax_m3.tt.vision.common import to_device
from models.demos.minimax_m3.tt.vision.image_input import check_image_grid, check_image_input
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed
from models.demos.minimax_m3.tt.vision.projector import M3VLProjector
from models.demos.minimax_m3.tt.vision.rope import rope_tensors

# The RoPE tables of the ROPE_CACHE_GRIDS (at least 1) image grids used last stay in DRAM. 16 is not tuned to any
# mix of image sizes. It is cheap: cos and sin of a 672x672 image, the largest that the image processor makes (2304
# patches), take 0.88 MB of DRAM, so 14 MB for 16 such grids.
ROPE_CACHE_GRIDS = 16


class M3VLVisionModel(LightweightModule):
    """Full M3-VL vision tower + optional multimodal projector.

    from_torch's dtype is the dtype of the weights. The pixel values go to the device as bf16 whatever it is, so the
    activations are bf16, the only dtype the RoPE op (rotary_embedding_llama) takes.
    """

    def __init__(
        self,
        mesh_device,
        config: MiniMaxM3VisionConfig,
        padded_head_dim: int,
        patch_embed: M3VLPatchEmbed,
        pre_layrnorm: M3VLLayerNorm,
        blocks: List[M3VLBlock],
        projector: Optional[M3VLProjector] = None,
    ):
        super().__init__()
        self.device = mesh_device
        self.config = config
        self.padded_head_dim = padded_head_dim
        self.patch_embed = patch_embed
        self.pre_layrnorm = pre_layrnorm
        self.blocks = blocks
        self.projector = projector
        self._rope_cache = OrderedDict()  # (t, h, w) -> (cos, sin) on device, the least recently used first

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        reference,  # MiniMaxM3VisionEncoder from reference/vision.py, whose config gives the dimensions
        with_projector: bool = True,
        dtype=ttnn.bfloat16,
    ) -> "M3VLVisionModel":
        config = reference.config
        # q, k and v heads padded to whole tiles on device (80 -> 96), in the weights and the RoPE tables.
        padded_head_dim = nearest_32(config.head_dim)
        patch_embed = M3VLPatchEmbed.from_torch(mesh_device, reference.patch_embed, dtype=dtype)
        pre_layrnorm = M3VLLayerNorm.from_torch(mesh_device, reference.pre_layrnorm, dtype=dtype)
        blocks = [
            M3VLBlock.from_torch(mesh_device, ref_layer, padded_head_dim, dtype=dtype) for ref_layer in reference.layers
        ]
        projector = M3VLProjector.from_torch(mesh_device, reference.projector, dtype=dtype) if with_projector else None
        return cls(
            mesh_device=mesh_device,
            config=config,
            padded_head_dim=padded_head_dim,
            patch_embed=patch_embed,
            pre_layrnorm=pre_layrnorm,
            blocks=blocks,
            projector=projector,
        )

    # ------------------------------------------------------------------
    def push_pixel_values(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Check one image's input and push its pixel values to device as forward_device takes them.

        Takes one image, pixel_values (L, 1176) and image_grid_thw (1, 3), and raises ValueError before any device work
        unless they pass check_image_input. On the device, rows past the grid's patches would get zero q and k from the
        RoPE op (rotary_embedding_llama) without an error, one NaN or Inf value would reach every patch through
        attention, and the patches of several images would attend to each other, as in Hugging Face 5.12.1; vLLM and
        SGLang keep attention inside each image. Returns the pixel values on device as bf16 (1, 1, L, 1176), whatever
        the weights' dtype.
        """
        check_image_input(pixel_values, image_grid_thw, self.config.patch_dim)
        return to_device(pixel_values.reshape(1, 1, *pixel_values.shape), self.device)

    def _stage_rope(self, image_grid_thw: torch.Tensor):
        """New cos and sin tables of the grid on device (rope_tensors), which the caller owns."""
        return rope_tensors(
            self.device,
            image_grid_thw,
            head_dim=self.config.head_dim,
            padded_head_dim=self.padded_head_dim,
            theta=self.config.rope_theta,
            spatial_merge_size=self.config.spatial_merge_size,
        )

    def _cached_rope(self, image_grid_thw: torch.Tensor):
        """The cos and sin tables of the grid, staged with its first image and kept for the next images of that grid.

        Holds the tables of the ROPE_CACHE_GRIDS grids used last and frees the two tables of the grid it drops. Only
        _tower_device uses them, to copy them to L1 for the blocks, so no caller holds a table that this frees.
        """
        key = tuple(image_grid_thw[0].tolist())
        if key in self._rope_cache:
            self._rope_cache.move_to_end(key)
            return self._rope_cache[key]
        if len(self._rope_cache) >= ROPE_CACHE_GRIDS:
            for table in self._rope_cache.popitem(last=False)[1]:
                ttnn.deallocate(table)
        self._rope_cache[key] = self._stage_rope(image_grid_thw)
        return self._rope_cache[key]

    def _rope_to_l1(self, cos: ttnn.Tensor, sin: ttnn.Tensor):
        """L1 copies of the cached cos and sin for one forward, which the caller frees.

        At 672x672 the RoPE op reads the cos and sin of each sequence tile again for each of the 16 heads, two thirds
        of what it reads; with the tables uploaded to L1 once, so without these copies, the image ran 1.7 ms faster on
        a P150. The cache stays in DRAM: in L1 it would hold the tables of up to ROPE_CACHE_GRIDS grids at all times,
        in room that circular buffers need.
        """
        return tuple(ttnn.to_memory_config(table, ttnn.L1_MEMORY_CONFIG) for table in (cos, sin))

    # ------------------------------------------------------------------
    def forward_tower(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Run patch_embed -> pre_layrnorm -> the blocks on one image (push_pixel_values). Returns (1, 1, L, 1280)."""
        return self._tower_device(self.push_pixel_values(pixel_values, image_grid_thw), image_grid_thw)

    def _tower_device(self, x_tt: ttnn.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """forward_tower's device ops, on the pixel values that push_pixel_values gave, which stay allocated."""
        x_tt = self.patch_embed(x_tt)
        x_tt = self.pre_layrnorm(x_tt)
        cos_tt, sin_tt = self._rope_to_l1(*self._cached_rope(image_grid_thw))
        for blk in self.blocks:
            x_tt = blk(x_tt, cos_tt, sin_tt)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
        return x_tt

    def forward_device(self, x_tt: ttnn.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Every device op of forward, on the pixel values that push_pixel_values gave for image_grid_thw. They stay
        allocated, so one upload serves several calls. With projector: (1, 1, L//4, 6144); else the tower out (1, 1, L,
        1280).

        Raises ValueError before any device work unless image_grid_thw passes check_image_grid and x_tt has the shape
        that push_pixel_values gives for it, (1, 1, h * w, patch_dim), which it reads from the tensor's host metadata.
        Pixel values pushed for another grid of the same patch count, such as 32 x 72 for 48 x 48 or 46 x 34 for
        34 x 46, pass and give wrong tokens without an error.
        """
        expected_shape = (1, 1, check_image_grid(image_grid_thw), self.config.patch_dim)
        if tuple(x_tt.shape) != expected_shape:
            raise ValueError(
                f"x_tt must be the pixel values that push_pixel_values gives for grid_thw {image_grid_thw.tolist()},"
                f" of shape {expected_shape}, got {tuple(x_tt.shape)}"
            )
        x_tt = self._tower_device(x_tt, image_grid_thw)
        if self.projector is None:
            return x_tt
        out_tt = self.projector(x_tt)
        ttnn.deallocate(x_tt)
        return out_tt

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Full forward of one image: push_pixel_values, then forward_device."""
        return self.forward_device(self.push_pixel_values(pixel_values, image_grid_thw), image_grid_thw)
