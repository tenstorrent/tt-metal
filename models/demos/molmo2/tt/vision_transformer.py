# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Vision Transformer encoder for Molmo2.

This implements the full ViT encoder with:
- Patch embedding (14x14 patches from 378x378 images -> 729 tokens)
- Learned positional embedding (with bicubic interpolation for non-native sizes)
- 25 transformer blocks (27 total, 25 used)
- Multi-layer output collection for vision adapter

Key dimensions:
- Image size: 378x378
- Patch size: 14x14
- Number of patches: 729 (27x27)
- Hidden dim: 1152
- Heads: 16
- Head dim: 72
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.vision_block import VisionBlock


class VisionTransformer(LightweightModule):
    """
    Molmo2 Vision Transformer encoder.

    Processes images through patch embedding, positional embedding,
    and a stack of transformer blocks. Returns hidden states from
    all layers for multi-scale feature extraction.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        num_layers: int = 25,
        hidden_dim: int = 1152,
        intermediate_dim: int = 4304,
        num_heads: int = 16,
        head_dim: int = 72,
        patch_size: int = 14,
        image_size: int = 378,
        layer_norm_eps: float = 1e-6,
        weight_cache_path=None,
        state_dict_prefix: str = "model.vision_backbone.image_vit",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize VisionTransformer.

        Args:
            mesh_device: TTNN mesh device
            state_dict: Model state dict containing weights
            num_layers: Number of transformer blocks to use (25 for Molmo2)
            hidden_dim: Model hidden dimension (1152)
            intermediate_dim: MLP intermediate dimension
            num_heads: Number of attention heads (16)
            head_dim: Dimension per head (72)
            patch_size: Size of image patches (14)
            image_size: Expected input image size (378)
            layer_norm_eps: Epsilon for LayerNorm
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.dtype = dtype

        # Calculate patch grid dimensions
        self.num_patches_per_side = image_size // patch_size  # 27
        self.num_patches = self.num_patches_per_side**2  # 729

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Patch embedding: Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        # We store the weight and bias, actual embedding done on CPU or via TTNN fold+linear
        patch_embed_weight = state_dict[f"{state_dict_prefix}.patch_embedding.weight"]
        patch_embed_bias = state_dict[f"{state_dict_prefix}.patch_embedding.bias"]

        # Reshape conv weight for linear: [hidden_dim, 3*patch_size*patch_size]
        # Original shape: [hidden_dim, 3, patch_size, patch_size]
        self.patch_embed_weight_torch = patch_embed_weight.reshape(hidden_dim, -1).transpose(-2, -1)
        self.patch_embed_bias_torch = patch_embed_bias

        # Store on device for TTNN operations
        self.patch_embed_weight = ttnn.as_tensor(
            self.patch_embed_weight_torch.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("patch_embedding.weight"),
        )

        self.patch_embed_bias = ttnn.as_tensor(
            patch_embed_bias,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("patch_embedding.bias"),
        )

        # Positional embedding: [1, num_patches, hidden_dim]
        # Shape in state dict: [num_patches, hidden_dim]
        self.positional_embedding_torch = state_dict[f"{state_dict_prefix}.positional_embedding"]
        self.base_num_patches_per_side = self.num_patches_per_side  # For interpolation

        self.positional_embedding = ttnn.as_tensor(
            self.positional_embedding_torch.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("positional_embedding"),
        )

        # Transformer blocks
        self.blocks = []
        for layer_num in range(num_layers):
            block = VisionBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                layer_num=layer_num,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                layer_norm_eps=layer_norm_eps,
                weight_cache_path=weight_cache_path,
                state_dict_prefix=f"{state_dict_prefix}.transformer.resblocks.{layer_num}",
                dtype=dtype,
            )
            self.blocks.append(block)

    def interpolate_pos_embedding(
        self,
        target_patches_per_side: int,
    ) -> torch.Tensor:
        """
        Interpolate positional embedding to a different grid size.

        Uses bicubic interpolation (matching HuggingFace implementation).

        Args:
            target_patches_per_side: Target number of patches per side

        Returns:
            Interpolated positional embedding tensor
        """
        if target_patches_per_side == self.base_num_patches_per_side:
            return self.positional_embedding_torch

        # Reshape to 2D grid: [num_patches, hidden_dim] -> [1, h, w, hidden_dim]
        pos_embed = self.positional_embedding_torch.reshape(
            self.base_num_patches_per_side,
            self.base_num_patches_per_side,
            self.hidden_dim,
        )
        pos_embed = pos_embed.unsqueeze(0).permute(0, 3, 1, 2)  # [1, hidden_dim, h, w]

        # Bicubic interpolation
        pos_embed = F.interpolate(
            pos_embed,
            size=(target_patches_per_side, target_patches_per_side),
            mode="bicubic",
            align_corners=False,
        )

        # Reshape back: [1, hidden_dim, h, w] -> [num_patches, hidden_dim]
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_dim)

        return pos_embed

    def patch_embed_cpu(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Apply patch embedding on CPU.

        Args:
            pixel_values: Input images [batch, 3, height, width]

        Returns:
            Patch embeddings [batch, num_patches, hidden_dim]
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        # Unfold into patches: [batch, 3, patches_h, patch_size, patches_w, patch_size]
        x = pixel_values.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)

        # Reshape: [batch, patches_h * patches_w, patch_size * patch_size * 3]
        # After unfolds: [batch, C, patches_h, patches_w, patch_size, patch_size]
        # Need: [batch, patches_h, patches_w, patch_size, patch_size, C] (HWC order to match HF)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(
            batch_size, patches_h * patches_w, self.patch_size * self.patch_size * channels
        )

        # Linear projection (weight is already transposed to [588, 1152])
        x = torch.matmul(x, self.patch_embed_weight_torch) + self.patch_embed_bias_torch

        return x

    def forward(
        self,
        x: ttnn.Tensor,
        patch_grid: Tuple[int, int] = None,
        return_all_hidden_states: bool = True,
    ) -> List[ttnn.Tensor]:
        """
        Forward pass through the vision transformer.

        Args:
            x: Input tensor of shape [1, 1, num_patches, hidden_dim]
               (after patch embedding and positional embedding have been applied)
            patch_grid: Tuple of (patches_h, patches_w) for positional embedding interpolation
            return_all_hidden_states: If True, return hidden states from all layers

        Returns:
            List of hidden states from all layers if return_all_hidden_states=True,
            otherwise just the final hidden state
        """
        hidden_states = []

        # Add positional embedding if not already added
        # (Typically done before calling forward, but can be done here)

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_all_hidden_states:
                # Clone tensor to preserve it for multi-scale feature extraction
                hidden_states.append(ttnn.clone(x))

        if return_all_hidden_states:
            return hidden_states
        else:
            return [x]

    def forward_with_patch_embed(
        self,
        pixel_values: torch.Tensor,
        return_all_hidden_states: bool = True,
    ) -> List[ttnn.Tensor]:
        """
        Full forward pass including patch embedding (CPU) and transformer (TTNN).

        Args:
            pixel_values: Input images [batch, 3, height, width] as torch tensor
            return_all_hidden_states: If True, return hidden states from all layers

        Returns:
            List of hidden states from requested layers
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        # Patch embedding on CPU
        x = self.patch_embed_cpu(pixel_values)  # [batch, num_patches, hidden_dim]

        # Interpolate positional embedding if needed
        if patches_h != self.base_num_patches_per_side or patches_w != self.base_num_patches_per_side:
            # For simplicity, assume square patches
            assert patches_h == patches_w, "Non-square patch grids not yet supported"
            pos_embed = self.interpolate_pos_embedding(patches_h)
        else:
            pos_embed = self.positional_embedding_torch

        # Add positional embedding
        x = x + pos_embed.unsqueeze(0)

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        # Process each crop separately (nlp_create_qkv_heads requires batch_size=1)
        # Then concatenate hidden states from all crops
        if batch_size > 1:
            all_crop_hidden_states = []

            for crop_idx in range(batch_size):
                # Get single crop: [1, num_patches, hidden_dim]
                crop_x = x[crop_idx : crop_idx + 1]
                crop_x = crop_x.unsqueeze(0)  # [1, 1, num_patches, hidden_dim]

                crop_x_ttnn = ttnn.from_torch(
                    crop_x,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                )

                # Forward through transformer blocks
                crop_hidden_states = self.forward(crop_x_ttnn, return_all_hidden_states=return_all_hidden_states)
                all_crop_hidden_states.append(crop_hidden_states)

            # Combine hidden states from all crops
            # Each layer's hidden states from all crops should be concatenated
            # crop_hidden_states[i] is layer i's output: [1, 1, num_patches, hidden_dim]
            # We want: [1, 1, batch*num_patches, hidden_dim]
            combined_hidden_states = []
            num_layers_out = len(all_crop_hidden_states[0])

            for layer_idx in range(num_layers_out):
                # Collect this layer's outputs from all crops
                layer_outputs = [crop_states[layer_idx] for crop_states in all_crop_hidden_states]
                # Concatenate along sequence dimension (dim=2)
                combined = ttnn.concat(layer_outputs, dim=2)
                combined_hidden_states.append(combined)

                # Clean up individual crop tensors
                for crop_tensor in layer_outputs:
                    ttnn.deallocate(crop_tensor)

            return combined_hidden_states
        else:
            # Single crop - original path
            x = x.unsqueeze(0)  # [1, batch, num_patches, hidden_dim]

            x_ttnn = ttnn.from_torch(
                x,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

            # Forward through transformer blocks
            return self.forward(x_ttnn, return_all_hidden_states=return_all_hidden_states)
