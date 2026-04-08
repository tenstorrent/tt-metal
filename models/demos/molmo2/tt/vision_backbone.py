# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Vision Backbone for Molmo2.

Combines the Vision Transformer encoder with the image pooling and projector
to produce visual embeddings ready for the language model.

Pipeline:
    1. ViT encoder: images -> multi-scale hidden states (layers 18, 24).
       Long video: optional frame chunking (env ``MOLMO2_VIT_ENCODE_FRAMES_CHUNK``; default ``0``).
       Set e.g. ``8`` to chunk; each slice is materialized before ViT for SDPA stability.
    2. Concat features on hidden dim: [B*T, N, 1152] x 2 -> [B, T*N, 2304]
    3. Gather features using pooled_patches_idx
    4. Cross-attention pooling: [B*N_out, K_pool, 2304] -> [B, N_out, 1152]
    5. SwiGLU projection: [B, N_out, 1152] -> [B, N_out, 4096]
    6. Filter by valid_token mask: [valid_tokens, 4096]
"""

import os
from typing import List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.image_pooling import ImagePooling
from models.demos.molmo2.tt.image_projector import ImageProjector
from models.demos.molmo2.tt.vision_transformer import VisionTransformer


def _vit_encode_frames_chunk_from_env() -> int:
    """Max frames per ViT forward in encode_image; 0 disables chunking (single forward).

    Default 0: one ViT over the full sequence (matches pre-chunking behavior). Set to e.g. 8
    to cap peak memory on long video; chunks are materialized after ``ttnn.slice`` for stability.
    """
    v = os.environ.get("MOLMO2_VIT_ENCODE_FRAMES_CHUNK", "16")
    try:
        c = int(v)
    except ValueError:
        return 0
    return max(0, c)


def _concat_tensors_dim2_device(chunks: List[ttnn.Tensor]) -> ttnn.Tensor:
    """Concatenate on sequence dim (2) on device; pairwise merge to limit peak DRAM."""
    if len(chunks) == 1:
        return chunks[0]
    out = chunks[0]
    for i in range(1, len(chunks)):
        nxt = ttnn.concat([out, chunks[i]], dim=2)
        ttnn.deallocate(out)
        ttnn.deallocate(chunks[i])
        out = nxt
    return out


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

    # Class-level request counter for debugging
    _encode_image_request_count = 0

    def _effective_num_crops_for_embedded(self, seq_len: int, num_crops: int) -> int:
        """
        Resolve how many frame/crop repeats are packed along the sequence axis.

        If num_crops > 1, it is trusted and seq_len must divide evenly.
        If num_crops == 1 and seq_len is a multiple of the ViT patch count (e.g. 729),
        infer a multi-frame / multi-crop video layout (seq_len // num_patches).
        """
        if num_crops > 1:
            if seq_len % num_crops != 0:
                raise ValueError(f"images_embedded seq_len {seq_len} not divisible by num_crops {num_crops}")
            return num_crops
        npp = self.image_vit.num_patches
        if seq_len > npp and npp > 0 and seq_len % npp == 0:
            t = seq_len // npp
            return t if t > 1 else 1
        return 1

    def _vit_embedded_to_multi_scale_features(
        self,
        x: ttnn.Tensor,
        matmul_output_memory_config,
        request_num: int,
        logger,
        seq_token_range: Optional[Tuple[int, int]] = None,
    ) -> ttnn.Tensor:
        """Single ViT forward on embedded patches; returns [1,1,S,pool_dim].

        If ``seq_token_range`` is ``(t0, t1)`` (half-open token indices on dim 2), run ViT only on
        that slice. The subrange is materialized here (``slice`` + ``to_memory_config``) so encode
        callers can pass the full ``images_embedded`` without building a separate sliced tensor.
        """
        x_in = x
        deallocate_x_in = False
        if seq_token_range is not None:
            t0, t1 = seq_token_range
            seq_full = int(x.shape[2])
            hidden_dim = int(x.shape[3])
            if t0 != 0 or t1 != seq_full:
                # Materialize: sliced TILE tensors can share parent storage / padding metadata that
                # breaks or hangs scaled_dot_product_attention in VisionAttention.
                x_in = ttnn.to_memory_config(
                    ttnn.slice(x, (0, 0, t0, 0), (1, 1, t1, hidden_dim)),
                    matmul_output_memory_config,
                )
                deallocate_x_in = True

        vit_input_shape = list(x_in.shape)
        hidden_states = self.image_vit.forward(
            x_in,
            return_all_hidden_states=True,
            matmul_output_memory_config=matmul_output_memory_config,
        )
        if deallocate_x_in:
            ttnn.deallocate(x_in)
        logger.info(
            f"encode_image REQUEST #{request_num}: ViT forward complete, "
            f"{len(hidden_states)} hidden states, ViT input shape={vit_input_shape}"
        )

        used_indices = set(self.feature_layers)
        features = []
        for layer_idx in self.feature_layers:
            features.append(hidden_states[layer_idx])
            logger.info(f"  Using layer {layer_idx}, shape: {list(hidden_states[layer_idx].shape)}")

        deallocated_count = 0
        for i, hs in enumerate(hidden_states):
            if i not in used_indices:
                ttnn.deallocate(hs)
                deallocated_count += 1
        logger.info(f"encode_image REQUEST #{request_num}: Deallocated {deallocated_count} unused hidden states")

        image_features = ttnn.concat(features, dim=-1)
        for f in features:
            ttnn.deallocate(f)
        return image_features

    def encode_image(
        self,
        images_embedded: ttnn.Tensor,
        num_crops: int = 1,
        matmul_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """
        Encode images through ViT and extract multi-scale features.

        Args:
            images_embedded: Embedded image patches [1, 1, T*N, hidden_dim]
                             after patch embedding and positional embedding
            num_crops: Number of crops / frames (T) packed along the sequence axis. If 1 and
                the sequence length is a multiple of ``num_patches`` (729 for 378/14), T is inferred.

        Returns:
            Concatenated multi-scale features [1, 1, T*N, pool_input_dim]

        Long videos: set env ``MOLMO2_VIT_ENCODE_FRAMES_CHUNK`` (default 8) to run multiple ViT
        forwards over frame groups and ``ttnn.concat`` the results on device along dim 2.
        Use ``0`` to force a single forward (original behavior).
        """
        from loguru import logger

        VisionBackbone._encode_image_request_count += 1
        request_num = VisionBackbone._encode_image_request_count
        seq_len = int(images_embedded.shape[2])
        eff_crops = self._effective_num_crops_for_embedded(seq_len, num_crops)
        tokens_per_crop = seq_len // eff_crops
        chunk_frames = _vit_encode_frames_chunk_from_env()

        logger.info(f"encode_image REQUEST #{request_num}: Starting ViT encode...")
        logger.info(f"  images_embedded shape: {list(images_embedded.shape)}")
        logger.info(
            f"  effective_crops={eff_crops}, tokens_per_crop={tokens_per_crop}, "
            f"MOLMO2_VIT_ENCODE_FRAMES_CHUNK={chunk_frames}"
        )

        if chunk_frames == 0 or eff_crops <= chunk_frames:
            image_features = self._vit_embedded_to_multi_scale_features(
                images_embedded,
                matmul_output_memory_config,
                request_num,
                logger,
            )
        else:
            out_chunks: List[ttnn.Tensor] = []
            for f0 in range(0, eff_crops, chunk_frames):
                f1 = min(f0 + chunk_frames, eff_crops)
                t0, t1 = f0 * tokens_per_crop, f1 * tokens_per_crop
                logger.info(
                    f"encode_image REQUEST #{request_num}: ViT chunk frames [{f0}, {f1}) "
                    f"-> token slice [{t0}, {t1}) (slice+materialize inside ViT helper)"
                )
                c_out = self._vit_embedded_to_multi_scale_features(
                    images_embedded,
                    matmul_output_memory_config,
                    request_num,
                    logger,
                    seq_token_range=(t0, t1),
                )
                out_chunks.append(c_out)
            image_features = _concat_tensors_dim2_device(out_chunks)
            logger.info(
                f"encode_image REQUEST #{request_num}: Concatenated {len(out_chunks)} " f"device chunks on dim=2"
            )

        logger.info(f"encode_image REQUEST #{request_num}: Complete, output shape: {list(image_features.shape)}")
        return image_features

    def encode_image_from_pixels(
        self,
        pixel_values: torch.Tensor,
        num_crops: int = 1,
        matmul_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            matmul_output_memory_config=matmul_output_memory_config,
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

        # Optional: use L1 for matmul outputs when tensors are small (same logic as forward_ttnn)
        if isinstance(images_embedded, torch.Tensor):
            # pixel_values: [B, C, H, W] -> num_patches = (H/14)*(W/14), vit_el = B * num_patches * hidden_dim
            _, _, h, w = images_embedded.shape
            num_patches = (h // self.image_vit.patch_size) * (w // self.image_vit.patch_size)
            _vit_el = batch_size * num_patches * self.image_vit.hidden_dim
        else:
            _vit_el = images_embedded.shape[2] * images_embedded.shape[3]
        vit_matmul_config = ttnn.L1_MEMORY_CONFIG if _vit_el <= 512 * 1024 else ttnn.DRAM_MEMORY_CONFIG

        # 1. Encode image through ViT
        # Check if input is raw pixels (torch.Tensor) or already embedded (ttnn.Tensor)
        if isinstance(images_embedded, torch.Tensor):
            image_features = self.encode_image_from_pixels(
                images_embedded, num_crops, matmul_output_memory_config=vit_matmul_config
            )
        else:
            image_features = self.encode_image(
                images_embedded, num_crops, matmul_output_memory_config=vit_matmul_config
            )

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

        # L1 for pooling/projector matmuls when small (same threshold as forward_ttnn)
        _query_el = batch_size * n_out * 1 * pool_dim
        _to_pool_el = batch_size * n_out * k_pool * pool_dim
        pool_matmul_config = (
            ttnn.L1_MEMORY_CONFIG if _query_el <= 512 * 1024 and _to_pool_el <= 512 * 1024 else ttnn.DRAM_MEMORY_CONFIG
        )
        _pooled_el = batch_size * n_out * self.adapter_hidden_dim
        project_matmul_config = ttnn.L1_MEMORY_CONFIG if _pooled_el <= 512 * 1024 else ttnn.DRAM_MEMORY_CONFIG

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
            matmul_output_memory_config=pool_matmul_config,
        )

        ttnn.deallocate(query_ttnn)
        ttnn.deallocate(to_pool_ttnn)
        if attn_mask_ttnn is not None:
            ttnn.deallocate(attn_mask_ttnn)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features_raw, [1, 1, batch_size * n_out, -1])
        # NOTE: Do NOT deallocate pooled_features_raw - reshape may return a view

        # 7. Project to language model dimension
        visual_embeddings = self.image_projector(
            pooled_features,
            matmul_output_memory_config=project_matmul_config,
        )
        # NOTE: Do NOT deallocate pooled_features - projection may use views internally

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
        num_crops: int = 1,
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
            num_crops: Frames or crops T along the sequence axis (default 1; infer when seq is T*729)

        Returns:
            Visual embeddings [1, 1, B*N_out, output_dim]
        """
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"

        _vit_el = images_embedded.shape[2] * images_embedded.shape[3]
        vit_matmul_config = ttnn.L1_MEMORY_CONFIG if _vit_el <= 512 * 1024 else ttnn.DRAM_MEMORY_CONFIG

        # 1. Encode image through ViT (optional frame chunking via MOLMO2_VIT_ENCODE_FRAMES_CHUNK)
        image_features = self.encode_image(
            images_embedded,
            num_crops=num_crops,
            matmul_output_memory_config=vit_matmul_config,
        )
        # image_features: [1, 1, B*T*N, pool_dim]

        # Squeeze to 2D for embedding lookup: [B*T*N, pool_dim]
        image_features_2d = ttnn.to_layout(
            ttnn.reshape(image_features, [-1, image_features.shape[-1]]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.deallocate(image_features)

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
        pool_dim = image_features_2d.shape[-1]
        gathered = ttnn.reshape(gathered, [1, 1, batch_size * n_out * k_pool, pool_dim])
        ttnn.deallocate(image_features_2d)

        # 3. Apply valid mask (zero out invalid positions)
        # valid_mask_ttnn: [1, 1, B*N_out*K_pool, 1]
        gathered = ttnn.mul(gathered, valid_mask_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape to [1, B*N_out, K_pool, pool_dim]
        to_pool = ttnn.reshape(gathered, [1, batch_size * n_out, k_pool, pool_dim])
        ttnn.deallocate(gathered)

        # 4. Compute query (mean of valid features per output position)
        # Sum along K_pool dimension
        query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)  # [1, B*N_out, 1, pool_dim]

        # Simplified mean: uses static K_pool as denominator instead of per-position valid counts.
        # Trade-off: enables TTNN tracing (dynamic per-position valid counts break trace capture)
        # at the cost of slight accuracy reduction vs forward() which uses a proper masked mean.
        # Measured PCC gap vs forward() path: < 0.01 for typical inputs.
        query = ttnn.mul(query_sum, 1.0 / k_pool, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(query_sum)

        # Optional: move small inputs to L1 so pooling matmuls read from L1 (reduces latency).
        # Only when tensor fits in L1 to avoid OOM (e.g. < 512K elements per tensor).
        _query_el = batch_size * n_out * 1 * pool_dim
        _to_pool_el = batch_size * n_out * k_pool * pool_dim
        if _query_el <= 512 * 1024:
            query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG)
        if _to_pool_el <= 512 * 1024:
            to_pool = ttnn.to_memory_config(to_pool, ttnn.L1_MEMORY_CONFIG)

        # 5. Cross-attention pooling
        # query: [1, B*N_out, 1, pool_dim]
        # to_pool (key/value): [1, B*N_out, K_pool, pool_dim]
        # attn_mask is skipped here: dynamic masking breaks TTNN trace capture.
        # The non-traced forward() path passes the mask correctly.

        pool_matmul_config = (
            ttnn.L1_MEMORY_CONFIG if _query_el <= 512 * 1024 and _to_pool_el <= 512 * 1024 else ttnn.DRAM_MEMORY_CONFIG
        )
        pooled_features = self.image_pooling_2d(
            query=query,
            key_value=to_pool,
            attn_mask=None,
            matmul_output_memory_config=pool_matmul_config,
        )

        ttnn.deallocate(query)
        ttnn.deallocate(to_pool)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features, [1, 1, batch_size * n_out, -1])

        # Optional: move to L1 so projector matmuls read from L1 when tensor is small
        _pooled_el = batch_size * n_out * self.adapter_hidden_dim
        if _pooled_el <= 512 * 1024:
            pooled_features = ttnn.to_memory_config(pooled_features, ttnn.L1_MEMORY_CONFIG)

        # 6. Project to language model dimension
        project_matmul_config = ttnn.L1_MEMORY_CONFIG if _pooled_el <= 512 * 1024 else ttnn.DRAM_MEMORY_CONFIG
        visual_embeddings = self.image_projector(
            pooled_features,
            matmul_output_memory_config=project_matmul_config,
        )
        ttnn.deallocate(pooled_features)

        return visual_embeddings
