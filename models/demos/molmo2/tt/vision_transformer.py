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
        num_devices = mesh_device.get_num_devices() if is_mesh_device else 1

        self.patch_embed_compute_memory_config = (
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if is_mesh_device and num_devices >= 8 else ttnn.DRAM_MEMORY_CONFIG
        )

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

    @staticmethod
    def _max_divisor_at_most(n: int, limit: int) -> int:
        for d in range(limit, 0, -1):
            if n % d == 0:
                return d
        return 1

    def _patch_embed_matmul_1d_program_config(self, num_m_rows: int):
        tile = ttnn.TILE_SIZE
        m_pad = ttnn.core.roundup(num_m_rows, tile)
        k_el = self.patch_size * self.patch_size * 3
        n_el = self.hidden_dim
        k_pad = ttnn.core.roundup(k_el, tile)
        n_pad = ttnn.core.roundup(n_el, tile)
        k_tiles = k_pad // tile
        n_tiles = n_pad // tile
        gs = self.mesh_device.compute_with_storage_grid_size()
        gx, gy = gs.x, gs.y
        num_cores = gx * gy
        if n_tiles // num_cores < 1:
            gy = max(1, n_tiles // gx)
            num_cores = gx * gy
        per_core_m = m_pad // tile
        per_core_n = ttnn.core.divup(n_pad, tile * num_cores)
        in0_block_w = self._max_divisor_at_most(k_tiles, 8)
        max_sb = 8
        out_subblock_w = max(i for i in range(1, max_sb + 1) if per_core_n % i == 0)
        out_subblock_h = max(
            (h for h in range(1, max_sb + 1) if per_core_m % h == 0 and h * out_subblock_w <= max_sb),
            default=1,
        )
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

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

    def patch_embed_ttnn(
        self,
        pixel_values: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Apply patch embedding with linear projection on TTNN.

        CPU handles only the unfold/permute (pure reshape, no compute).
        TTNN handles the linear projection and positional embedding add.

        Args:
            pixel_values: Input images [batch, 3, height, width]

        Returns:
            Embedded patches [1, 1, batch*num_patches, hidden_dim] on device
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size
        num_patches = patches_h * patches_w

        # CPU: unfold + permute -- pure data layout reorganization, no arithmetic
        x = pixel_values.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(
            1, 1, batch_size * num_patches, self.patch_size * self.patch_size * channels
        )

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        # Transfer raw (unprocessed) patches to device
        x_ttnn = ttnn.from_torch(
            x,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        mm_kwargs = {"memory_config": self.patch_embed_compute_memory_config}
        if self.patch_embed_compute_memory_config != ttnn.DRAM_MEMORY_CONFIG:
            mm_kwargs["program_config"] = self._patch_embed_matmul_1d_program_config(batch_size * num_patches)
        embedded = ttnn.matmul(x_ttnn, self.patch_embed_weight, **mm_kwargs)
        ttnn.deallocate(x_ttnn)

        # Add bias
        embedded = ttnn.add(embedded, self.patch_embed_bias, memory_config=self.patch_embed_compute_memory_config)

        # Add positional embedding: [1, 1, num_patches, 1152]
        if batch_size == 1:
            # Shapes match directly -- [1, 1, N, 1152] + [1, 1, N, 1152]
            embedded = ttnn.add(
                embedded,
                self.positional_embedding,
                memory_config=self.patch_embed_compute_memory_config,
            )
        else:
            # Tile positional embedding for multi-crop: [1, 1, B*N, 1152]
            pos_tiles = [self.positional_embedding] * batch_size
            pos_tiled = ttnn.concat(pos_tiles, dim=2)
            embedded = ttnn.add(embedded, pos_tiled, memory_config=self.patch_embed_compute_memory_config)
            ttnn.deallocate(pos_tiled)

        # Preserve existing interface expectations for downstream blocks.
        if self.patch_embed_compute_memory_config != ttnn.DRAM_MEMORY_CONFIG:
            embedded = ttnn.to_memory_config(embedded, ttnn.DRAM_MEMORY_CONFIG)

        return embedded

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
                # No clone needed - block() creates a new tensor, so old x is still valid
                # This follows tt_transformers' pattern for traceable ViT
                hidden_states.append(x)

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
        Full forward pass: patch + positional embedding, then ViT blocks on TTNN.

        For the native patch grid (Molmo2 27x27), patch/pos use ``patch_embed_ttnn`` so
        T3K can run matmul/add in L1 width-sharded mode (see ``patch_embed_compute_memory_config``).

        Non-native grids still use CPU interpolation + H2D because learned pos is resized on CPU.
        """
        batch_size, channels, height, width = pixel_values.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        native_grid = patches_h == self.base_num_patches_per_side and patches_w == self.base_num_patches_per_side

        def forward_one_crop_ttnn(crop_pixels: torch.Tensor) -> List[ttnn.Tensor]:
            """crop_pixels: [1, 3, H, W]; returns hidden states from self.forward."""
            x_ttnn = self.patch_embed_ttnn(crop_pixels)
            return self.forward(x_ttnn, return_all_hidden_states=return_all_hidden_states)

        if native_grid:
            # Multi-crop: one ViT forward per crop so attention stays within each crop; patch embed uses L1 on T3K.
            if batch_size > 1:
                all_crop_hidden_states = []
                for crop_idx in range(batch_size):
                    crop_pixels = pixel_values[crop_idx : crop_idx + 1]
                    all_crop_hidden_states.append(forward_one_crop_ttnn(crop_pixels))

                combined_hidden_states = []
                num_layers_out = len(all_crop_hidden_states[0])
                for layer_idx in range(num_layers_out):
                    layer_outputs = [crop_states[layer_idx] for crop_states in all_crop_hidden_states]
                    combined = ttnn.concat(layer_outputs, dim=2)
                    combined_hidden_states.append(combined)
                    for crop_tensor in layer_outputs:
                        ttnn.deallocate(crop_tensor)
                return combined_hidden_states

            return forward_one_crop_ttnn(pixel_values)

        # Non-native patch grid: HF torch path for interp pos, then H2D (DRAM; blocks expect interleaved DRAM).
        assert patches_h == patches_w, "Non-square patch grids not yet supported"
        x = self.patch_embed_cpu(pixel_values)
        pos_embed = self.interpolate_pos_embedding(patches_h)
        x = x + pos_embed.unsqueeze(0)

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        if batch_size > 1:
            all_crop_hidden_states = []
            for crop_idx in range(batch_size):
                crop_x = x[crop_idx : crop_idx + 1].unsqueeze(0)
                crop_x_ttnn = ttnn.from_torch(
                    crop_x,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                )
                all_crop_hidden_states.append(
                    self.forward(crop_x_ttnn, return_all_hidden_states=return_all_hidden_states)
                )

            combined_hidden_states = []
            num_layers_out = len(all_crop_hidden_states[0])
            for layer_idx in range(num_layers_out):
                layer_outputs = [crop_states[layer_idx] for crop_states in all_crop_hidden_states]
                combined = ttnn.concat(layer_outputs, dim=2)
                combined_hidden_states.append(combined)
                for crop_tensor in layer_outputs:
                    ttnn.deallocate(crop_tensor)
            return combined_hidden_states

        x_ttnn = ttnn.from_torch(
            x.unsqueeze(0),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return self.forward(x_ttnn, return_all_hidden_states=return_all_hidden_states)
