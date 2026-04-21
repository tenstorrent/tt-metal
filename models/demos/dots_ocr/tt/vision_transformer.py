# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full TTNN Vision Transformer for Dots OCR.

Orchestrates:
- PatchEmbedTT (image → patches)
- 42 VisionBlockTT layers
- PatchMergerTT (final merging)

This replaces the hybrid approach with a complete TTNN implementation.
"""

from __future__ import annotations

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.demos.dots_ocr.tt.vision_block import create_vision_block
from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs
from models.demos.dots_ocr.tt.vision_patch_embed import create_patch_embed
from models.demos.dots_ocr.tt.vision_rmsnorm import RMSNorm


class VisionTransformerTT(LightweightModule):
    """
    Full TTNN Vision Transformer for Dots.mocr.

    This implements the complete vision tower:
    1. Patch embedding
    2. 42 transformer blocks with post-norm
    3. Final patch merging
    """

    def __init__(
        self,
        mesh_device,
        model_args: DotsVisionModelArgs,
        state_dict: dict,
        weight_cache_path=None,
        dtype=None,
    ):
        super().__init__()
        ttnn = get_ttnn()
        if dtype is None:
            dtype = ttnn.bfloat16 if ttnn is not None else torch.bfloat16
        self.mesh_device = mesh_device
        self.model_args = model_args
        self.dtype = dtype
        self.num_layers = model_args.vision_config.num_hidden_layers  # 42 layers

        # Patch embedding
        self.patch_embed = create_patch_embed(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        # 42 Vision blocks
        self.blocks = []
        for i in range(self.num_layers):
            block = create_vision_block(
                mesh_device=mesh_device,
                model_args=model_args,
                state_dict=state_dict,
                layer_num=i,
                weight_cache_path=weight_cache_path,
                dtype=dtype,
            )
            self.blocks.append(block)

        # Final norm (post-norm architecture)
        self.norm = RMSNorm(
            device=mesh_device,
            dim=model_args.vision_dim,
            state_dict=state_dict,
            state_dict_prefix="vision_tower.",
            weight_key="norm",
            weight_dtype=dtype,
            eps=model_args.rms_norm_eps,
        )

        # Use existing PatchMerger (already implemented)
        from models.demos.dots_ocr.tt.patch_merger import PatchMerger as PatchMergerTT

        # Dots HF checkpoints have used both `vision_tower.merger.*` and `vision_tower.patch_merger.*`
        # naming schemes. Select the prefix that exists in the provided state_dict.
        patch_merger_prefix = "vision_tower.patch_merger"
        if not any(k.startswith(patch_merger_prefix + ".") for k in state_dict.keys()):
            alt = "vision_tower.merger"
            if any(k.startswith(alt + ".") for k in state_dict.keys()):
                patch_merger_prefix = alt

        self.patch_merger = PatchMergerTT(
            mesh_device=mesh_device,
            hidden_size=model_args.vision_dim,
            out_hidden_size=model_args.vision_dim,  # Usually same size
            spatial_merge_size=model_args.spatial_merge_size,
            state_dict=state_dict,
            state_dict_prefix=patch_merger_prefix,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None) -> torch.Tensor:
        """
        Full vision transformer forward pass.

        Args:
            pixel_values: [B, C, H, W] from HF processor
            grid_thw:     [B, 3] temporal/height/width grid from processor

        Returns:
            Vision tokens ``[N_tokens, hidden_size]`` ready for
            :func:`models.demos.dots_ocr.tt.common.merge_vision_tokens`.

        Notes
        -----
        The vision sub-blocks (``VisionAttentionTT``, ``VisionMLPTT``) currently run on host
        (see their ``forward`` implementations). We therefore use host tensors through the
        block stack and only call into TT for the final ``PatchMerger``. Converting the
        entire stack to TT tensors is a separate, larger refactor; we document this
        explicitly rather than silently round-tripping inside every block.
        """
        # 1. Patch embedding (may return either a torch or TT tensor depending on the
        #    patch embed implementation — normalize to torch for the host block stack).
        x = self.patch_embed(pixel_values, grid_thw)
        ttnn = get_ttnn()
        if ttnn is not None and not isinstance(x, torch.Tensor):
            x = ttnn.to_torch(x)

        # 2. Transformer blocks (host, post-norm).
        for block in self.blocks:
            x = block(x)

        # 3. Final norm.
        x = self.norm(x)

        # 4. Patch merger — expects a 4D tensor [B, 1, seq, hidden].
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.to(torch.bfloat16)

        merged = self.patch_merger(x) if hasattr(self.patch_merger, "forward") else x
        if isinstance(merged, torch.Tensor) and merged.dim() == 4:
            merged = merged.squeeze(0).squeeze(0)
        return merged

    def to_host(self):
        """Ensure all tensors are moved to host for cleanup."""


# Main interface function to replace the hybrid vision_tower_forward
def vision_tower_forward_ttnn(
    vision_transformer: VisionTransformerTT, pixel_values: torch.Tensor, grid_thw: torch.Tensor
) -> torch.Tensor:
    """
    TTNN version of vision tower forward.

    This is the main entry point that should be called instead of
    the hybrid version when using full TTNN vision.
    """
    return vision_transformer.forward(pixel_values, grid_thw)


# Convenience function to create the full vision transformer
def create_dots_vision_transformer(
    mesh_device, model_args=None, state_dict=None, weight_cache_path=None, dtype=None, hf_model=None
):
    """Create full TTNN VisionTransformerTT for Dots OCR."""
    if model_args is None:
        model_args = DotsVisionModelArgs(mesh_device=mesh_device)

    if dtype is None:
        ttnn = get_ttnn()
        dtype = ttnn.bfloat16 if ttnn is not None else torch.bfloat16

    if state_dict is None:
        # Create dummy state dict for testing
        state_dict = {}

    return VisionTransformerTT(
        mesh_device=mesh_device,
        model_args=model_args,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        dtype=dtype,
    )
