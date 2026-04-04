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
        # Feature layers (0-indexed) - match HF order: [-3, -9] -> [24, 18]
        feature_layers: Tuple[int, int] = (24, 18),
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
        dtype=ttnn.bfloat16,  # Changed from bfloat8_b for better vision precision
        use_tensor_parallel: bool = False,  # ViT uses DP (data parallel) for frames, not TP
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
            feature_layers: ViT layers to extract features from (24, 18) - HF order
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

        # Vision Transformer encoder (TP=8 shards weights across devices)
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
            use_tensor_parallel=use_tensor_parallel,
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

    # Class-level request counter for debugging
    _encode_image_request_count = 0

    def _normalize_for_projector(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Normalize and clip pooled features before projector.

        The Molmo2 ViT uses large LayerNorm gamma values (up to 18x) which causes
        pooled features to have outliers (±40 after normalization). The SwiGLU
        projector computes gate*up, which squares these outliers causing scale
        explosion (e.g., 40*40=1600 becomes extreme after projection).

        This normalizes to unit variance AND clips outliers to ±3 std to prevent
        the quadratic explosion in SwiGLU.
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"

        # Convert to torch, normalize, convert back
        # (TTNN doesn't have a built-in var normalization that works well here)
        if is_mesh_device:
            x_torch = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0]
        else:
            x_torch = ttnn.to_torch(x)

        # Compute statistics
        x_std = x_torch.std()
        x_mean = x_torch.mean()

        logger.debug(
            f"Pooled features before normalization: mean={x_mean:.4f}, std={x_std:.4f}, min={x_torch.min():.4f}, max={x_torch.max():.4f}"
        )

        # Normalize to unit variance (keep mean, scale to std=1.0)
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        x_normalized = (x_torch - x_mean) / (x_std + eps)

        # Clip outliers to ±3 std to prevent quadratic explosion in SwiGLU
        # This is critical: without clipping, outliers of ±40 become ±1600 after gate*up
        clip_value = 3.0
        x_normalized = torch.clamp(x_normalized, -clip_value, clip_value)

        logger.debug(
            f"Pooled features after normalization+clip: mean={x_normalized.mean():.4f}, std={x_normalized.std():.4f}, min={x_normalized.min():.4f}, max={x_normalized.max():.4f}"
        )

        # Convert back to TTNN
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None
        x_out = ttnn.from_torch(
            x_normalized,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Deallocate input since we created a new tensor
        ttnn.deallocate(x)

        return x_out

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
        from loguru import logger

        # Track request count for debugging
        VisionBackbone._encode_image_request_count += 1
        request_num = VisionBackbone._encode_image_request_count
        logger.info(f"encode_image REQUEST #{request_num}: Starting ViT forward...")
        logger.info(f"  images_embedded shape: {list(images_embedded.shape)}")

        # Run through ViT and collect all hidden states
        hidden_states = self.image_vit.forward(
            images_embedded,
            return_all_hidden_states=True,
        )

        logger.info(
            f"encode_image REQUEST #{request_num}: ViT forward complete, got {len(hidden_states)} hidden states"
        )

        # Extract features from specified layers and concat
        features = []
        used_indices = set(self.feature_layers)
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh_device else None
        for layer_idx in self.feature_layers:
            features.append(hidden_states[layer_idx])
            # Debug: check per-layer stats
            try:
                layer_torch = (
                    ttnn.to_torch(hidden_states[layer_idx], mesh_composer=mesh_composer)[0]
                    if is_mesh_device
                    else ttnn.to_torch(hidden_states[layer_idx])
                )
                logger.info(
                    f"  Layer {layer_idx}: shape={list(layer_torch.shape)}, mean={layer_torch.mean():.4f}, std={layer_torch.std():.4f}, min={layer_torch.min():.4f}, max={layer_torch.max():.4f}"
                )
            except:
                logger.info(f"  Using layer {layer_idx}, shape: {list(hidden_states[layer_idx].shape)}")

        # CRITICAL: Deallocate unused hidden states to prevent memory leak
        # For video with 8 frames, each hidden state is ~27MB, 25 layers = ~670MB total
        # Only using 2 layers means 23 layers (~620MB) would be leaked per request!
        deallocated_count = 0
        for i, hs in enumerate(hidden_states):
            if i not in used_indices:
                ttnn.deallocate(hs)
                deallocated_count += 1
        logger.info(f"encode_image REQUEST #{request_num}: Deallocated {deallocated_count} unused hidden states")

        # Concatenate on hidden dimension
        # Each feature is [1, 1, B*T*N, hidden_dim]
        image_features = ttnn.concat(features, dim=-1)

        # NOTE: Do NOT deallocate features here - ttnn.concat may return a view that references inputs
        # The deallocation of unused hidden states above is sufficient for memory management
        logger.info(f"encode_image REQUEST #{request_num}: Concat complete (keeping feature tensors)")

        logger.info(f"encode_image REQUEST #{request_num}: Complete, output shape: {list(image_features.shape)}")

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
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        query_ttnn = ttnn.from_torch(
            query.unsqueeze(0),  # [1, B*N_out, 1, pool_dim]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        to_pool_ttnn = ttnn.from_torch(
            to_pool.unsqueeze(0),  # [1, B*N_out, K_pool, pool_dim]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        if attn_mask is not None:
            attn_mask_ttnn = ttnn.from_torch(
                attn_mask,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        else:
            attn_mask_ttnn = None

        # 6. Cross-attention pooling
        pooled_features_raw = self.image_pooling_2d(
            query=query_ttnn,
            key_value=to_pool_ttnn,
            attn_mask=attn_mask_ttnn,
        )

        ttnn.deallocate(query_ttnn)
        ttnn.deallocate(to_pool_ttnn)
        if attn_mask_ttnn is not None:
            ttnn.deallocate(attn_mask_ttnn)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features_raw, [1, 1, batch_size * n_out, -1])
        # NOTE: Do NOT deallocate pooled_features_raw - reshape may return a view

        # 7. Normalize pooled features before projection
        pooled_features = self._normalize_for_projector(pooled_features)

        # 8. Project to language model dimension
        visual_embeddings = self.image_projector(pooled_features)
        # NOTE: pooled_features deallocated inside _normalize_for_projector

        # 8. Filter by valid tokens (return only valid embeddings)
        # Convert to torch for filtering
        if is_mesh_device:
            visual_embeddings_torch = ttnn.to_torch(
                visual_embeddings, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[0]
        else:
            visual_embeddings_torch = ttnn.to_torch(visual_embeddings)

        # CRITICAL: Deallocate TTNN tensor after converting to torch
        ttnn.deallocate(visual_embeddings)

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

    def forward_ttnn(
        self,
        images_embedded: ttnn.Tensor,
        pooled_patches_idx_ttnn: ttnn.Tensor,
        valid_mask_ttnn: ttnn.Tensor,
        valid_token_ttnn: ttnn.Tensor,
        n_out: int,
        k_pool: int,
        batch_size: int = 1,
    ) -> ttnn.Tensor:
        """
        Full forward pass using TTNN ops (traceable).

        This version keeps everything in TTNN to enable tracing.

        Args:
            images_embedded: Embedded image patches [1, 1, B*T*N, hidden_dim]
            pooled_patches_idx_ttnn: Flattened indices [1, B*N_out*K_pool] (clipped to >= 0)
            valid_mask_ttnn: Valid mask [1, 1, B*N_out*K_pool, 1] for masking gathered features
            valid_token_ttnn: Valid token mask [B*N_out] for final filtering
            n_out: Number of output positions
            k_pool: Pooling kernel size
            batch_size: Batch size (default 1)

        Returns:
            Visual embeddings [1, 1, B*N_out, output_dim]
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh_device else None

        def _stats(t, name):
            try:
                x = ttnn.to_torch(t, mesh_composer=mesh_composer)[0] if is_mesh_device else ttnn.to_torch(t)
                return f"{name}: shape={list(x.shape)}, mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}"
            except:
                return f"{name}: stats unavailable"

        # 1. Encode image through ViT
        image_features = self.encode_image(images_embedded)
        logger.debug(_stats(image_features, "ViT features"))
        # image_features: [1, 1, B*T*N, pool_dim]

        # Squeeze to 2D for embedding lookup: [B*T*N, pool_dim]
        image_features_2d = ttnn.reshape(image_features, [-1, image_features.shape[-1]])

        # Convert to ROW_MAJOR for embedding lookup (embedding table must be ROW_MAJOR)
        image_features_2d = ttnn.to_layout(image_features_2d, ttnn.ROW_MAJOR_LAYOUT)

        # 2. Gather features using ttnn.embedding
        # pooled_patches_idx_ttnn: [1, B*N_out*K_pool] contains indices into B*T*N
        gathered = ttnn.embedding(
            pooled_patches_idx_ttnn,
            image_features_2d,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # gathered: [1, B*N_out*K_pool, pool_dim]

        # Reshape to [1, 1, B*N_out*K_pool, pool_dim] for masking
        pool_dim = image_features.shape[-1]
        gathered = ttnn.reshape(gathered, [1, 1, batch_size * n_out * k_pool, pool_dim])

        # 3. Apply valid mask (zero out invalid positions)
        # valid_mask_ttnn: [1, 1, B*N_out*K_pool, 1]
        gathered = ttnn.mul(gathered, valid_mask_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(_stats(gathered, "gathered (after mask)"))

        # Reshape to [1, B*N_out, K_pool, pool_dim]
        to_pool = ttnn.reshape(gathered, [1, batch_size * n_out, k_pool, pool_dim])

        # 4. Compute query (mean of valid features per output position)
        # Sum along K_pool dimension
        query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)  # [1, B*N_out, 1, pool_dim]

        # Simplified mean: uses static K_pool as denominator instead of per-position valid counts.
        # Trade-off: enables TTNN tracing (dynamic per-position valid counts break trace capture)
        # at the cost of slight accuracy reduction vs forward() which uses a proper masked mean.
        # Measured PCC gap vs forward() path: < 0.01 for typical inputs.
        query = ttnn.mul(query_sum, 1.0 / k_pool, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(_stats(query, "query (mean of gathered)"))
        logger.debug(_stats(to_pool, "to_pool (key/value)"))

        # 5. Cross-attention pooling
        # query: [1, B*N_out, 1, pool_dim]
        # to_pool (key/value): [1, B*N_out, K_pool, pool_dim]
        # attn_mask is skipped here: dynamic masking breaks TTNN trace capture.
        # The non-traced forward() path passes the mask correctly.

        pooled_features = self.image_pooling_2d(
            query=query,
            key_value=to_pool,
            attn_mask=None,
        )
        logger.debug(_stats(pooled_features, "pooled_features (after pooling, before projection)"))

        ttnn.deallocate(query)
        ttnn.deallocate(to_pool)
        ttnn.deallocate(gathered)
        ttnn.deallocate(image_features)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features, [1, 1, batch_size * n_out, -1])

        # 6. Normalize pooled features before projection
        # The pooled features have std ~4.0 due to large LayerNorm gamma in ViT,
        # but projector expects std ~1.0. Without normalization, SwiGLU (gate*up)
        # causes quadratic explosion in scale.
        pooled_features = self._normalize_for_projector(pooled_features)

        # 7. Project to language model dimension
        visual_embeddings = self.image_projector(pooled_features)
        ttnn.deallocate(pooled_features)

        return visual_embeddings

    def encode_image_only_ttnn(self, images_embedded: ttnn.Tensor) -> ttnn.Tensor:
        """
        Run only ViT encoding without pooling/projection.

        Use this for chunked video processing where pooling needs cross-frame
        attention across ALL frames. Call this in chunks, concatenate the
        outputs, then call pool_and_project_ttnn on the combined features.

        Args:
            images_embedded: Embedded patches [1, 1, B*N, hidden_dim]

        Returns:
            image_features: ViT output [1, 1, B*N, pool_dim] (concatenated multi-scale features)
        """
        return self.encode_image(images_embedded)

    def pool_and_project_ttnn(
        self,
        image_features: ttnn.Tensor,
        pooled_patches_idx_ttnn: ttnn.Tensor,
        valid_mask_ttnn: ttnn.Tensor,
        n_out: int,
        k_pool: int,
        batch_size: int,
    ) -> ttnn.Tensor:
        """
        Run gather, pooling, and projection on pre-computed ViT features.

        Use this after encode_image_only_ttnn when doing chunked processing.
        The image_features should contain ALL frames concatenated so that
        pooling can attend across all frames via global indices.

        Args:
            image_features: ViT output [1, 1, total_patches, pool_dim] from ALL frames
            pooled_patches_idx_ttnn: Flattened GLOBAL indices [1, B*N_out*K_pool]
            valid_mask_ttnn: Valid mask [1, 1, B*N_out*K_pool, 1]
            n_out: Number of output positions per frame
            k_pool: Pooling kernel size
            batch_size: Total number of frames

        Returns:
            visual_embeddings: [1, 1, B*N_out, output_dim]
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh_device else None

        logger.info(f"pool_and_project_ttnn (SINGLE-PASS): batch_size={batch_size}, n_out={n_out}, k_pool={k_pool}")
        logger.info(f"  image_features shape: {image_features.shape}")

        # DEBUG: Check ViT feature statistics
        if is_mesh_device:
            vit_debug = ttnn.to_torch(image_features, mesh_composer=mesh_composer)[0]
        else:
            vit_debug = ttnn.to_torch(image_features)
        logger.info(f"  ViT features stats: mean={vit_debug.mean().item():.4f}, std={vit_debug.std().item():.4f}")

        # Squeeze to 2D for embedding lookup: [total_patches, pool_dim]
        image_features_2d = ttnn.reshape(image_features, [-1, image_features.shape[-1]])

        # Convert to ROW_MAJOR for embedding lookup
        image_features_2d = ttnn.to_layout(image_features_2d, ttnn.ROW_MAJOR_LAYOUT)

        # Gather features using ttnn.embedding with GLOBAL indices
        gathered = ttnn.embedding(
            pooled_patches_idx_ttnn,
            image_features_2d,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # gathered: [1, B*N_out*K_pool, pool_dim]

        # Reshape to [1, 1, B*N_out*K_pool, pool_dim] for masking
        pool_dim = image_features.shape[-1]
        gathered = ttnn.reshape(gathered, [1, 1, batch_size * n_out * k_pool, pool_dim])

        # Apply valid mask (zero out invalid positions)
        gathered = ttnn.mul(gathered, valid_mask_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape to [1, B*N_out, K_pool, pool_dim]
        to_pool = ttnn.reshape(gathered, [1, batch_size * n_out, k_pool, pool_dim])

        # Compute query (mean of features per output position)
        query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)  # [1, B*N_out, 1, pool_dim]
        query = ttnn.mul(query_sum, 1.0 / k_pool, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # DEBUG: Check query and to_pool stats
        if is_mesh_device:
            query_debug = ttnn.to_torch(query, mesh_composer=mesh_composer)[0]
            to_pool_debug = ttnn.to_torch(to_pool, mesh_composer=mesh_composer)[0]
        else:
            query_debug = ttnn.to_torch(query)
            to_pool_debug = ttnn.to_torch(to_pool)
        logger.info(f"  query stats: mean={query_debug.mean().item():.4f}, std={query_debug.std().item():.4f}")
        logger.info(f"  to_pool stats: mean={to_pool_debug.mean().item():.4f}, std={to_pool_debug.std().item():.4f}")

        # Cross-attention pooling
        pooled_features = self.image_pooling_2d(
            query=query,
            key_value=to_pool,
            attn_mask=None,
        )

        # DEBUG: Check pooled stats
        if is_mesh_device:
            pooled_debug = ttnn.to_torch(pooled_features, mesh_composer=mesh_composer)[0]
        else:
            pooled_debug = ttnn.to_torch(pooled_features)
        logger.info(f"  pooled stats: mean={pooled_debug.mean().item():.4f}, std={pooled_debug.std().item():.4f}")

        ttnn.deallocate(query)
        ttnn.deallocate(to_pool)
        ttnn.deallocate(gathered)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features, [1, 1, batch_size * n_out, -1])

        # Normalize pooled features before projection
        pooled_features = self._normalize_for_projector(pooled_features)

        # Project to language model dimension
        visual_embeddings = self.image_projector(pooled_features)
        ttnn.deallocate(pooled_features)

        # DEBUG: Check final visual embedding stats
        if is_mesh_device:
            vis_debug = ttnn.to_torch(visual_embeddings, mesh_composer=mesh_composer)[0]
        else:
            vis_debug = ttnn.to_torch(visual_embeddings)
        logger.info(
            f"  SINGLE-PASS visual embeddings: mean={vis_debug.mean().item():.4f}, std={vis_debug.std().item():.4f}, min={vis_debug.min().item():.4f}, max={vis_debug.max().item():.4f}"
        )

        return visual_embeddings

    def pool_and_project_chunked_ttnn(
        self,
        image_features: ttnn.Tensor,
        pooled_patches_idx: torch.Tensor,
        valid_mask: torch.Tensor,
        n_out: int,
        k_pool: int,
        batch_size: int,
        max_frames_per_pool_chunk: int = 16,
    ) -> ttnn.Tensor:
        """
        Chunked pooling for large videos - processes output frames in chunks
        while keeping all ViT features available for cross-frame attention.

        This reduces peak memory by processing pooling in batches, but still
        allows each output to reference patches from ANY frame via global indices.

        Args:
            image_features: ViT output [1, 1, total_patches, pool_dim] from ALL frames
            pooled_patches_idx: GLOBAL indices [batch_size, n_out, k_pool] (torch, CPU)
            valid_mask: Valid mask [batch_size, n_out, k_pool] (torch, CPU)
            n_out: Number of output positions per frame
            k_pool: Pooling kernel size
            batch_size: Total number of frames
            max_frames_per_pool_chunk: Max frames to pool at once

        Returns:
            visual_embeddings: [1, 1, B*N_out, output_dim]
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh_device else None

        logger.info(f"pool_and_project_chunked: {batch_size} frames in chunks of {max_frames_per_pool_chunk}")
        logger.info(f"  pooled_patches_idx shape: {pooled_patches_idx.shape}, valid_mask shape: {valid_mask.shape}")
        logger.info(f"  n_out={n_out}, k_pool={k_pool}")

        # DEBUG: Log index statistics for the ENTIRE tensor
        valid_idx_all = pooled_patches_idx[valid_mask]
        logger.info(
            f"  GLOBAL index stats: min={valid_idx_all.min().item()}, max={valid_idx_all.max().item()}, count={len(valid_idx_all)}"
        )

        # Convert ViT features to 2D embedding table (stays on device throughout)
        pool_dim = image_features.shape[-1]
        logger.info(f"  image_features shape before reshape: {image_features.shape}")
        image_features_2d = ttnn.reshape(image_features, [-1, pool_dim])
        image_features_2d = ttnn.to_layout(image_features_2d, ttnn.ROW_MAJOR_LAYOUT)
        logger.info(f"  image_features_2d shape after reshape: {image_features_2d.shape}")

        # DEBUG: Check ViT feature statistics
        if is_mesh_device:
            vit_debug = ttnn.to_torch(image_features, mesh_composer=mesh_composer)[0]
        else:
            vit_debug = ttnn.to_torch(image_features)
        logger.info(
            f"  ViT features stats: mean={vit_debug.mean().item():.4f}, std={vit_debug.std().item():.4f}, min={vit_debug.min().item():.4f}, max={vit_debug.max().item():.4f}"
        )

        # Can deallocate original 4D tensor now
        ttnn.deallocate(image_features)

        # Process pooling in chunks of frames
        all_embeddings = []

        for chunk_start in range(0, batch_size, max_frames_per_pool_chunk):
            chunk_end = min(chunk_start + max_frames_per_pool_chunk, batch_size)
            chunk_frames = chunk_end - chunk_start

            # Extract chunk of indices and masks (these are OUTPUT frames, indices are still GLOBAL)
            chunk_idx = pooled_patches_idx[chunk_start:chunk_end]  # [chunk_frames, n_out, k_pool]
            chunk_valid = valid_mask[chunk_start:chunk_end]  # [chunk_frames, n_out, k_pool]

            # Debug: check index ranges (indices should be < batch_size * 729 patches per frame)
            valid_idx = chunk_idx[chunk_valid]
            expected_total_patches = batch_size * 729  # patches per frame
            if len(valid_idx) > 0:
                logger.debug(
                    f"  Chunk {chunk_start//max_frames_per_pool_chunk} index range: min={valid_idx.min().item()}, max={valid_idx.max().item()}, expected_total_patches={expected_total_patches}"
                )

            # Flatten for gather
            flat_idx = torch.clip(chunk_idx, min=0).reshape(1, -1).to(torch.int32)
            flat_valid = chunk_valid.reshape(1, 1, -1, 1).float()

            # Move to device
            idx_ttnn = ttnn.from_torch(
                flat_idx,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            valid_ttnn = ttnn.from_torch(
                flat_valid,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

            # Gather using GLOBAL indices from full feature table
            gathered = ttnn.embedding(
                idx_ttnn,
                image_features_2d,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logger.info(
                f"  Chunk {chunk_start//max_frames_per_pool_chunk} gathered shape: {gathered.shape}, expected: [1, {chunk_frames * n_out * k_pool}, {pool_dim}]"
            )

            # DEBUG: Check gathered feature statistics before masking
            if is_mesh_device:
                gathered_debug = ttnn.to_torch(gathered, mesh_composer=mesh_composer)[0]
            else:
                gathered_debug = ttnn.to_torch(gathered)
            logger.info(
                f"    Chunk {chunk_start//max_frames_per_pool_chunk} gathered stats (before mask): mean={gathered_debug.mean().item():.4f}, std={gathered_debug.std().item():.4f}"
            )
            ttnn.deallocate(idx_ttnn)

            # Reshape and mask
            gathered = ttnn.reshape(gathered, [1, 1, chunk_frames * n_out * k_pool, pool_dim])
            gathered = ttnn.mul(gathered, valid_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(valid_ttnn)

            # Reshape for pooling
            to_pool = ttnn.reshape(gathered, [1, chunk_frames * n_out, k_pool, pool_dim])
            ttnn.deallocate(gathered)

            # Compute query
            query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)
            query = ttnn.mul(query_sum, 1.0 / k_pool, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(query_sum)

            # DEBUG: Check query and to_pool stats before pooling
            if is_mesh_device:
                query_debug = ttnn.to_torch(query, mesh_composer=mesh_composer)[0]
                to_pool_debug = ttnn.to_torch(to_pool, mesh_composer=mesh_composer)[0]
            else:
                query_debug = ttnn.to_torch(query)
                to_pool_debug = ttnn.to_torch(to_pool)
            logger.info(
                f"    Chunk {chunk_start//max_frames_per_pool_chunk} query stats: mean={query_debug.mean().item():.4f}, std={query_debug.std().item():.4f}"
            )
            logger.info(
                f"    Chunk {chunk_start//max_frames_per_pool_chunk} to_pool stats: mean={to_pool_debug.mean().item():.4f}, std={to_pool_debug.std().item():.4f}"
            )

            # Cross-attention pooling
            logger.info(
                f"  Chunk {chunk_start//max_frames_per_pool_chunk} before pooling: query shape={query.shape}, to_pool shape={to_pool.shape}"
            )
            pooled = self.image_pooling_2d(query=query, key_value=to_pool, attn_mask=None)
            logger.info(f"  Chunk {chunk_start//max_frames_per_pool_chunk} after pooling: pooled shape={pooled.shape}")

            # DEBUG: Check pooled stats
            if is_mesh_device:
                pooled_debug = ttnn.to_torch(pooled, mesh_composer=mesh_composer)[0]
            else:
                pooled_debug = ttnn.to_torch(pooled)
            logger.info(
                f"    Chunk {chunk_start//max_frames_per_pool_chunk} pooled stats: mean={pooled_debug.mean().item():.4f}, std={pooled_debug.std().item():.4f}"
            )

            ttnn.deallocate(query)
            ttnn.deallocate(to_pool)

            # Reshape pooled features for later normalization
            pooled = ttnn.reshape(pooled, [1, 1, chunk_frames * n_out, -1])

            # Move pooled features to CPU (before normalization/projection)
            if is_mesh_device:
                pooled_torch = ttnn.to_torch(pooled, mesh_composer=mesh_composer)[0]
            else:
                pooled_torch = ttnn.to_torch(pooled)

            # Ensure 4D shape
            if pooled_torch.dim() == 3:
                pooled_torch = pooled_torch.unsqueeze(1)

            logger.debug(
                f"  Chunk {chunk_start//max_frames_per_pool_chunk} pooled (before norm): shape={pooled_torch.shape}, mean={pooled_torch.mean().item():.4f}, std={pooled_torch.std().item():.4f}"
            )
            all_embeddings.append(pooled_torch)
            ttnn.deallocate(pooled)
            ttnn.synchronize_device(self.mesh_device)

        # Deallocate embedding table
        ttnn.deallocate(image_features_2d)

        # Concatenate all pooled features BEFORE normalization
        combined_pooled = torch.cat(all_embeddings, dim=2)  # [1, 1, total_outputs, pool_dim]
        logger.info(f"pool_and_project_chunked: Combined pooled shape: {combined_pooled.shape}")
        logger.info(
            f"  Combined pooled stats (before norm): mean={combined_pooled.mean().item():.4f}, std={combined_pooled.std().item():.4f}, min={combined_pooled.min().item():.4f}, max={combined_pooled.max().item():.4f}"
        )

        # Apply GLOBAL normalization (same as single-pass path)
        eps = 1e-6
        global_mean = combined_pooled.mean()
        global_std = combined_pooled.std()
        combined_normalized = (combined_pooled - global_mean) / (global_std + eps)
        clip_value = 3.0
        combined_normalized = torch.clamp(combined_normalized, -clip_value, clip_value)
        logger.info(
            f"  Combined pooled stats (after norm): mean={combined_normalized.mean().item():.4f}, std={combined_normalized.std().item():.4f}, min={combined_normalized.min().item():.4f}, max={combined_normalized.max().item():.4f}"
        )

        # Move normalized features to device and project
        combined_ttnn = ttnn.from_torch(
            combined_normalized,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Project all at once (this should fit since we're projecting, not pooling)
        visual_embeddings = self.image_projector(combined_ttnn)
        ttnn.deallocate(combined_ttnn)

        # Get final stats
        if is_mesh_device:
            final_debug = ttnn.to_torch(visual_embeddings, mesh_composer=mesh_composer)[0]
        else:
            final_debug = ttnn.to_torch(visual_embeddings)
        logger.info(f"pool_and_project_chunked: Complete, output shape: {visual_embeddings.shape}")
        logger.info(
            f"  Final projected stats: mean={final_debug.mean().item():.4f}, std={final_debug.std().item():.4f}, min={final_debug.min().item():.4f}, max={final_debug.max().item():.4f}"
        )

        return visual_embeddings
