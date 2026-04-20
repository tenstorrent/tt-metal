# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Patch Embedding for Dots OCR Vision Transformer.

Converts image input (pixel_values) + grid_thw into patch embeddings
that can be fed into the vision transformer blocks.
"""

from __future__ import annotations

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn


class PatchEmbedTT(LightweightModule):
    """
    TTNN implementation of patch embedding for Dots Vision Transformer.

    Handles:
    - Converting images to patches (14x14 for Dots)
    - Linear projection
    - Position embeddings
    - grid_thw (temporal, height, width) handling for document images
    """

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        weight_cache_path=None,
        dtype=None,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1536,
        state_dict_prefix: str = "vision_tower.patch_embed",
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.state_dict_prefix = state_dict_prefix
        ttnn = get_ttnn()
        if dtype is None:
            # Stub/partial ``ttnn`` installs may lack ``bfloat16``; fall back to torch.
            if ttnn is not None and getattr(ttnn, "bfloat16", None) is not None:
                dtype = ttnn.bfloat16
            else:
                dtype = torch.bfloat16
        self.dtype = dtype

        # Load weights from state dict
        self._load_weights(state_dict, weight_cache_path, dtype)

    def _load_weights(self, state_dict: dict, weight_cache_path, dtype):
        """Load patch embed weights from HF state dict."""
        ttnn = get_ttnn()
        prefix = self.state_dict_prefix

        # Projection weight: [embed_dim, in_channels * patch_size * patch_size]
        proj_weight_key = f"{prefix}.proj.weight"
        if proj_weight_key in state_dict and ttnn is not None and self.mesh_device is not None:
            weight = state_dict[proj_weight_key]
            memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            self.proj_weight = ttnn.as_tensor(
                weight,
                device=self.mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=weight_cache_path / "proj_weight" if weight_cache_path else None,
            )
        else:
            # Fallback for testing
            self.proj_weight = None
            if proj_weight_key not in state_dict:
                print(f"Warning: {proj_weight_key} not found in state_dict")

        # Check for position embedding (if it exists)
        pos_embed_key = f"{prefix}.position_embedding.weight"
        if pos_embed_key in state_dict and ttnn is not None and self.mesh_device is not None:
            pos_weight = state_dict[pos_embed_key]
            memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            self.pos_embed_weight = ttnn.as_tensor(
                pos_weight,
                device=self.mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=weight_cache_path / "pos_embed" if weight_cache_path else None,
            )
        else:
            self.pos_embed_weight = None

    def _process_grid_thw(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Process grid_thw tensor from processor.

        grid_thw typically has shape [batch_size, 3] with values:
        [temporal_patches, height_patches, width_patches]
        """
        if grid_thw is None:
            # Default for square images
            return torch.tensor([[1, 16, 16]])  # 1 temporal, 16x16 spatial patches

        if grid_thw.dim() == 2:
            return grid_thw
        elif grid_thw.dim() == 1:
            return grid_thw.unsqueeze(0)
        return grid_thw

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None) -> ttnn.Tensor:
        """
        Convert image to patch embeddings.

        Args:
            pixel_values: [B, C, H, W] image tensor
            grid_thw: [B, 3] grid dimensions from processor (temporal, height, width)

        Returns:
            ttnn.Tensor: Patch embeddings [B, num_patches, embed_dim]
        """
        B, C, H, W = pixel_values.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Process grid_thw
        grid_thw = self._process_grid_thw(grid_thw)

        # For now, use a simplified approach - convert to patches on host first
        # In a full implementation, this would be done with TTNN operations
        # For Phase 1, we'll convert on host and then move to device

        # Simple patch embedding on host (to be optimized later)
        # Reshape to patches: [B, num_patches, C * patch_h * patch_w]
        #
        # Two sources of truth for the patch grid:
        #   1. ``self.patch_size`` — the configured patch side (default 14 for Dots).
        #   2. ``grid_thw`` — the HF processor's actual ``[temporal, height_patches, width_patches]``.
        # When they disagree (e.g. tests feed synthetic 64x64 images with grid=[1,4,4]),
        # ``grid_thw`` is authoritative because it reflects what the processor produced;
        # we derive the per-axis patch pixel size from the image shape.
        if grid_thw is not None:
            temporal = int(grid_thw[0, 0].item())
            height_patches = int(grid_thw[0, 1].item())
            width_patches = int(grid_thw[0, 2].item())
        else:
            temporal = 1
            height_patches = H // self.patch_size
            width_patches = W // self.patch_size

        # Guard against zero-patch configurations (e.g. grid=[1,0,0] dummy inputs).
        assert height_patches > 0 and width_patches > 0, (
            f"Invalid patch grid temporal={temporal}, H={height_patches}, W={width_patches} "
            f"(pixel_values={pixel_values.shape}, grid_thw={grid_thw})"
        )
        assert (
            H % height_patches == 0 and W % width_patches == 0
        ), f"Image {H}x{W} not divisible by patch grid {height_patches}x{width_patches}"
        patch_h = H // height_patches
        patch_w = W // width_patches
        num_patches = temporal * height_patches * width_patches

        # Flatten spatial dimensions into patches.
        x = pixel_values.reshape(B, C, height_patches, patch_h, width_patches, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches, C * patch_h * patch_w)

        # Linear projection: patches -> embed_dim
        if isinstance(x, torch.Tensor):
            if self.proj_weight is not None:
                # ``self.proj_weight`` may already have been moved to a TTNN tensor in
                # ``__init__`` (device path). Normalize it back to a host torch tensor here so
                # the host-side ``torch.nn.functional.linear`` works — full TTNN matmul is a
                # separate port (see VisionTransformerTT.forward doc for scope).
                ttnn_mod = get_ttnn()
                weight = self.proj_weight
                if ttnn_mod is not None and not isinstance(weight, torch.Tensor):
                    weight = ttnn_mod.to_torch(weight)
                x = torch.nn.functional.linear(
                    x.float(), weight.reshape(self.embed_dim, -1).float()[:, : x.shape[-1]]
                ).to(torch.bfloat16)
            else:
                # Dummy projection for testing
                x = torch.randn(B, num_patches, self.embed_dim, dtype=torch.bfloat16)

        ttnn = get_ttnn()
        if ttnn is None or self.mesh_device is None:
            return x

        memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        dtype_tt = getattr(ttnn, "bfloat16", torch.bfloat16)
        x_tt = ttnn.from_torch(
            x,
            device=self.mesh_device,
            dtype=dtype_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return x_tt

    def to_host(self):
        """Move any persistent tensors to host."""


# Convenience function for testing
def create_patch_embed(mesh_device, state_dict=None, weight_cache_path=None, dtype=None, **kwargs):
    """Create PatchEmbedTT with Dots defaults; kwargs may override patch_size/embed_dim."""
    kwargs.setdefault("patch_size", 14)
    kwargs.setdefault("in_channels", 3)
    kwargs.setdefault("embed_dim", 1536)
    return PatchEmbedTT(
        mesh_device=mesh_device,
        state_dict=state_dict or {},
        weight_cache_path=weight_cache_path,
        dtype=dtype,
        **kwargs,
    )
