# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""SigLIP2-NaFlex vision embeddings for LFM2.5-VL.

Unlike Gemma3's SigLIP tower (Conv2d patch embed over ``[B, C, H, W]``), SigLIP2-NaFlex
expects already-patchified pixels ``[B, max_num_patches, C * P * P]`` plus per-image
``spatial_shapes`` used to bicubically resize the fixed ``num_patches`` position table.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32


class TtSiglip2VisionEmbeddings(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        patch_size,
        num_channels,
        hidden_dim,
        num_patches,
        bias=True,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.position_embedding_size = int(round(num_patches**0.5))
        self.patch_dim = num_channels * patch_size * patch_size

        weight_key = f"{state_dict_prefix}patch_embedding._linear.weight"
        if weight_key not in state_dict:
            # Accept either converted (_linear) or raw HF Linear keys.
            weight_key = f"{state_dict_prefix}patch_embedding.weight"
        weight = state_dict[weight_key]
        if weight.ndim == 4:
            weight = weight.view(hidden_dim, -1)
        assert weight.shape == (hidden_dim, self.patch_dim), (weight.shape, hidden_dim, self.patch_dim)

        pad_len = nearest_32(weight.shape[-1]) - weight.shape[-1]
        if pad_len:
            weight = torch.cat([weight, torch.zeros(hidden_dim, pad_len, dtype=weight.dtype)], dim=-1)
        padded_weight = weight.permute(1, 0).reshape(1, 1, -1, hidden_dim)

        self._linear_weight = ttnn.as_tensor(
            padded_weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        bias_key = f"{state_dict_prefix}patch_embedding._linear.bias"
        if bias_key not in state_dict:
            bias_key = f"{state_dict_prefix}patch_embedding.bias"
        self._linear_bias = None
        if bias and bias_key in state_dict:
            self._linear_bias = ttnn.as_tensor(
                state_dict[bias_key].reshape(1, -1),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        pos_key = f"{state_dict_prefix}position_embedding.positional_embedding"
        if pos_key not in state_dict:
            pos_key = f"{state_dict_prefix}position_embedding.weight"
        self._pos_embedding_host = state_dict[pos_key].float()  # [num_patches, hidden]

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @staticmethod
    def patchify_images(images: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ``[B, C, H, W]`` images into SigLIP2 patchified tensors + spatial shapes."""
        bsz, channels, height, width = images.shape
        assert height % patch_size == 0 and width % patch_size == 0, (height, width, patch_size)
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = (
            images.reshape(bsz, channels, grid_h, patch_size, grid_w, patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(bsz, grid_h * grid_w, channels * patch_size * patch_size)
        )
        spatial_shapes = torch.tensor([[grid_h, grid_w]] * bsz, dtype=torch.long, device=images.device)
        return patches, spatial_shapes

    def resize_positional_embeddings(self, spatial_shapes: torch.Tensor, max_length: int) -> torch.Tensor:
        """Bicubic-resize the fixed position table to each image's ``(H, W)`` grid (NaFlex)."""
        embed_dim = self._pos_embedding_host.shape[-1]
        pos = self._pos_embedding_host.reshape(self.position_embedding_size, self.position_embedding_size, embed_dim)
        # (1, dim, H, W) for interpolate
        pos = pos.permute(2, 0, 1).unsqueeze(0).float()
        batch_size = spatial_shapes.shape[0]
        out = torch.empty((batch_size, max_length, embed_dim), dtype=torch.float32)
        for i in range(batch_size):
            height, width = int(spatial_shapes[i, 0]), int(spatial_shapes[i, 1])
            resized = F.interpolate(pos, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
            resized = resized.reshape(embed_dim, height * width).transpose(0, 1)
            out[i, : height * width] = resized
            if height * width < max_length:
                out[i, height * width :] = resized[0]
        return out

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            pixel_values: either ``[B, C, H, W]`` or already-patchified ``[B, N, C*P*P]``
            spatial_shapes: ``[B, 2]`` required when pixel_values are patchified; inferred when 4D
        Returns:
            ttnn embeddings ``[B, N, hidden_dim]``
        """
        if pixel_values.ndim == 4:
            pixel_values, spatial_shapes = self.patchify_images(pixel_values, self.patch_size)
        elif spatial_shapes is None:
            raise ValueError("spatial_shapes is required for patchified SigLIP2 pixel_values")

        batch, num_tokens, patch_dim = pixel_values.shape
        pad_len = nearest_32(patch_dim) - patch_dim
        if pad_len:
            pixel_values = torch.cat(
                [pixel_values, torch.zeros(batch, num_tokens, pad_len, dtype=pixel_values.dtype)], dim=-1
            )

        tt_pixels = ttnn.from_torch(
            pixel_values.reshape(1, 1, batch * num_tokens, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        patch_embeds = ttnn.linear(
            tt_pixels,
            self._linear_weight,
            bias=self._linear_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        patch_embeds = ttnn.reshape(patch_embeds, (batch, num_tokens, self.hidden_dim))

        pos = self.resize_positional_embeddings(spatial_shapes.cpu(), max_length=num_tokens)
        tt_pos = ttnn.from_torch(
            pos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return ttnn.add(patch_embeds, tt_pos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
