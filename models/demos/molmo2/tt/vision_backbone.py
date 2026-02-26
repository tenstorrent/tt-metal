# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Vision Backbone for Molmo2.

Combines the Vision Transformer encoder with the image pooling and projector
to produce visual embeddings ready for the language model.

Pipeline:
    1. ViT encoder: images -> multi-scale hidden states (layers 18, 24)
    2. Concat features on hidden dim: [B*T, N, 1152] x 2 -> [B, T*N, 2304]
    3. Gather features using pooled_patches_idx
    4. Cross-attention pooling: [B*N_out, K_pool, 2304] -> [B, N_out, 1152]
    5. SwiGLU projection: [B, N_out, 1152] -> [B, N_out, 4096]
    6. Filter by valid_token mask: [valid_tokens, 4096]
"""

from typing import Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.image_pooling import ImagePooling
from models.demos.molmo2.tt.image_projector import ImageProjector
from models.demos.molmo2.tt.vision_transformer import VisionTransformer


class VisionBackbone(LightweightModule):
    """
    Complete vision backbone for Molmo2.

    Processes images through ViT, pools features using cross-attention,
    and projects to the language model hidden dimension.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        # ViT config
        vit_num_layers: int = 25,
        vit_hidden_dim: int = 1152,
        vit_intermediate_dim: int = 4304,
        vit_num_heads: int = 16,
        vit_head_dim: int = 72,
        patch_size: int = 14,
        image_size: int = 378,
        # Feature layers (0-indexed)
        feature_layers: Tuple[int, int] = (18, 24),
        # Adapter config
        adapter_hidden_dim: int = 1152,
        adapter_intermediate_dim: int = 12288,
        adapter_num_heads: int = 16,
        adapter_head_dim: int = 72,
        # Output config
        output_dim: int = 4096,
        # Common config
        layer_norm_eps: float = 1e-6,
        weight_cache_path=None,
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize VisionBackbone.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing all weights
            vit_num_layers: Number of ViT layers to use (25)
            vit_hidden_dim: ViT hidden dimension (1152)
            vit_intermediate_dim: ViT MLP intermediate dimension (4304)
            vit_num_heads: ViT attention heads (16)
            vit_head_dim: ViT head dimension (72)
            patch_size: Image patch size (14)
            image_size: Expected image size (378)
            feature_layers: ViT layers to extract features from (18, 24)
            adapter_hidden_dim: Adapter hidden dimension (1152)
            adapter_intermediate_dim: Projector intermediate dimension (12288)
            adapter_num_heads: Pooling attention heads (16)
            adapter_head_dim: Pooling head dimension (72)
            output_dim: Output dimension for language model (4096)
            layer_norm_eps: Epsilon for LayerNorm
            weight_cache_path: Path to cache weights
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.feature_layers = feature_layers
        self.vit_hidden_dim = vit_hidden_dim
        self.adapter_hidden_dim = adapter_hidden_dim
        self.output_dim = output_dim

        # Pool input dimension is concat of feature layers
        pool_input_dim = vit_hidden_dim * len(feature_layers)  # 2304

        # Vision Transformer encoder
        self.image_vit = VisionTransformer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            num_layers=vit_num_layers,
            hidden_dim=vit_hidden_dim,
            intermediate_dim=vit_intermediate_dim,
            num_heads=vit_num_heads,
            head_dim=vit_head_dim,
            patch_size=patch_size,
            image_size=image_size,
            layer_norm_eps=layer_norm_eps,
            weight_cache_path=weight_cache_path,
            state_dict_prefix="model.vision_backbone.image_vit",
            dtype=dtype,
        )

        # Image pooling (cross-attention)
        self.image_pooling_2d = ImagePooling(
            mesh_device=mesh_device,
            state_dict=state_dict,
            input_dim=pool_input_dim,
            hidden_dim=adapter_hidden_dim,
            num_heads=adapter_num_heads,
            head_dim=adapter_head_dim,
            weight_cache_path=weight_cache_path,
            state_dict_prefix="model.vision_backbone.image_pooling_2d",
            dtype=dtype,
        )

        # Image projector (SwiGLU)
        self.image_projector = ImageProjector(
            mesh_device=mesh_device,
            state_dict=state_dict,
            input_dim=adapter_hidden_dim,
            intermediate_dim=adapter_intermediate_dim,
            output_dim=output_dim,
            weight_cache_path=weight_cache_path,
            state_dict_prefix="model.vision_backbone.image_projector",
            dtype=dtype,
        )

    def encode_image(
        self,
        images_embedded: ttnn.Tensor,
        num_crops: int = 1,
    ) -> ttnn.Tensor:
        """
        Encode images through ViT and extract multi-scale features.

        Args:
            images_embedded: Embedded image patches [B*T, N, hidden_dim]
                             after patch embedding and positional embedding
            num_crops: Number of crops per image (T)

        Returns:
            Concatenated multi-scale features [B, T*N, pool_input_dim]
        """
        # Run through ViT and collect all hidden states
        hidden_states = self.image_vit.forward(
            images_embedded,
            return_all_hidden_states=True,
        )

        # Extract features from specified layers and concat
        features = []
        for layer_idx in self.feature_layers:
            features.append(hidden_states[layer_idx])

        # Concatenate on hidden dimension
        # Each feature is [1, 1, B*T*N, hidden_dim]
        image_features = ttnn.concat(features, dim=-1)

        return image_features

    def encode_image_from_pixels(
        self,
        pixel_values: torch.Tensor,
        num_crops: int = 1,
    ) -> ttnn.Tensor:
        """
        Encode images from raw pixel values through ViT.

        Args:
            pixel_values: Raw pixel values [B, C, H, W] as torch tensor
            num_crops: Number of crops per image (T)

        Returns:
            Concatenated multi-scale features [B, T*N, pool_input_dim]
        """
        # Run through ViT with patch embedding
        hidden_states = self.image_vit.forward_with_patch_embed(
            pixel_values,
            return_all_hidden_states=True,
        )

        # Extract features from specified layers and concat
        features = []
        for layer_idx in self.feature_layers:
            features.append(hidden_states[layer_idx])

        # Concatenate on hidden dimension
        image_features = ttnn.concat(features, dim=-1)

        return image_features

    def forward(
        self,
        images_embedded,
        pooled_patches_idx: torch.Tensor,
        num_crops: int = 1,
        use_attention_mask: bool = True,
    ) -> ttnn.Tensor:
        """
        Full forward pass through vision backbone.

        Args:
            images_embedded: Either:
                - Embedded image patches [1, 1, B*T*N, hidden_dim] (ttnn.Tensor)
                - Raw pixel values [B, C, H, W] (torch.Tensor)
            pooled_patches_idx: Indices for gathering patch features (on CPU)
                                Shape: [B, N_out, K_pool]
            num_crops: Number of crops per image (T)
            use_attention_mask: Whether to use attention mask in pooling

        Returns:
            Visual embeddings for language model [valid_tokens, output_dim]
        """
        batch_size = pooled_patches_idx.shape[0]

        # 1. Encode image through ViT
        # Check if input is raw pixels (torch.Tensor) or already embedded (ttnn.Tensor)
        if isinstance(images_embedded, torch.Tensor):
            image_features = self.encode_image_from_pixels(images_embedded, num_crops)
        else:
            image_features = self.encode_image(images_embedded, num_crops)

        # 2. Convert to torch for gathering (CPU operation)
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        if is_mesh_device:
            image_features_torch = ttnn.to_torch(
                image_features, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[0]
        else:
            image_features_torch = ttnn.to_torch(image_features)

        image_features_torch = image_features_torch.squeeze(0).squeeze(0)  # [B*T*N, pool_dim]

        # 3. Gather features using pooled_patches_idx (CPU)
        # pooled_patches_idx: [B, N_out, K_pool] - indices into flattened features
        pool_dim = image_features_torch.shape[-1]
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        # Identify valid indices (>= 0)
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)  # [B, N_out]

        # Gather features
        # Flatten features: [B*T*N, pool_dim] -> use as lookup table
        batch_idx = torch.arange(batch_size, dtype=torch.long)
        batch_idx = batch_idx.view(batch_size, 1, 1).expand(-1, n_out, k_pool)

        # Clip negative indices to 0 for gathering (will be masked later)
        clipped_idx = torch.clip(pooled_patches_idx, min=0)

        # Reshape features for batched indexing: [B, T*N, pool_dim]
        total_patches = image_features_torch.shape[0] // batch_size
        features_batched = image_features_torch.reshape(batch_size, total_patches, pool_dim)

        # Gather: [B, N_out, K_pool, pool_dim]
        to_pool = features_batched[batch_idx, clipped_idx]

        # Mask invalid positions
        to_pool = to_pool * valid.unsqueeze(-1).float()

        # Reshape for attention: [B*N_out, K_pool, pool_dim]
        to_pool = to_pool.reshape(-1, k_pool, pool_dim)

        # 4. Compute query (mean of valid features)
        if use_attention_mask:
            valid_flat = valid.reshape(-1, k_pool).float()
            denom = valid_flat.sum(-1, keepdim=True)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            query = to_pool.sum(-2, keepdim=True) / denom.unsqueeze(-1)
            attn_mask = valid.reshape(-1, 1, 1, k_pool).float()
            # Convert mask to additive form for SDPA
            attn_mask = torch.where(attn_mask == 0, float("-inf"), 0.0)
        else:
            query = to_pool.mean(-2, keepdim=True)
            attn_mask = None

        # 5. Convert back to TTNN for pooling
        query_ttnn = ttnn.from_torch(
            query.unsqueeze(0),  # [1, B*N_out, 1, pool_dim]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        to_pool_ttnn = ttnn.from_torch(
            to_pool.unsqueeze(0),  # [1, B*N_out, K_pool, pool_dim]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if attn_mask is not None:
            attn_mask_ttnn = ttnn.from_torch(
                attn_mask,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_mask_ttnn = None

        # 6. Cross-attention pooling
        pooled_features = self.image_pooling_2d(
            query=query_ttnn,
            key_value=to_pool_ttnn,
            attn_mask=attn_mask_ttnn,
        )

        ttnn.deallocate(query_ttnn)
        ttnn.deallocate(to_pool_ttnn)
        if attn_mask_ttnn is not None:
            ttnn.deallocate(attn_mask_ttnn)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features, [1, 1, batch_size * n_out, -1])

        # 7. Project to language model dimension
        visual_embeddings = self.image_projector(pooled_features)

        # 8. Filter by valid tokens (return only valid embeddings)
        # Convert to torch for filtering
        if is_mesh_device:
            visual_embeddings_torch = ttnn.to_torch(
                visual_embeddings, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[0]
        else:
            visual_embeddings_torch = ttnn.to_torch(visual_embeddings)

        visual_embeddings_torch = visual_embeddings_torch.squeeze(0).squeeze(0)  # [B*N_out, output_dim]

        # Apply valid token mask
        valid_embeddings = visual_embeddings_torch[valid_token.flatten()]

        return valid_embeddings

    def forward_encode_only(
        self,
        images_embedded: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Encode images only (without pooling/projection).

        Useful for testing the ViT encoder separately.

        Args:
            images_embedded: Embedded image patches [1, 1, N, hidden_dim]

        Returns:
            Concatenated multi-scale features [1, 1, N, pool_input_dim]
        """
        return self.encode_image(images_embedded)
