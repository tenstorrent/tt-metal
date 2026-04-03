# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B Full Model Implementation.

Combines the Vision Backbone (ViT + Adapter) and Text Model (LM) into a
unified multimodal model for visual question answering and image captioning.

Architecture:
    1. Image Processing: Preprocess images and compute pooled_patches_idx
    2. Vision Backbone: ViT encoder -> multi-scale features -> pooling -> projection
    3. Embedding Fusion: Insert visual embeddings into text sequence
    4. Text Model: Decoder-only transformer for autoregressive generation
"""

from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.text_model import TextModel
from models.demos.molmo2.tt.vision_backbone import VisionBackbone


class Molmo2Model(LightweightModule):
    """
    Full Molmo2-8B multimodal model.

    Combines vision encoder, adapter, and language model for
    image-conditioned text generation.
    """

    def __init__(
        self,
        mesh_device,
        state_dict: Dict[str, torch.Tensor],
        # Vision config
        vit_num_layers: int = 25,
        vit_hidden_dim: int = 1152,
        vit_intermediate_dim: int = 4304,
        vit_num_heads: int = 16,
        vit_head_dim: int = 72,
        patch_size: int = 14,
        image_size: int = 378,
        feature_layers: Tuple[int, int] = (24, 18),  # HF order: [-3, -9]
        # Adapter config
        adapter_hidden_dim: int = 1152,
        adapter_intermediate_dim: int = 12288,
        adapter_num_heads: int = 16,
        adapter_head_dim: int = 72,
        # Text config
        text_num_layers: int = 36,
        text_hidden_dim: int = 4096,
        text_intermediate_dim: int = 12288,
        text_num_heads: int = 32,
        text_num_kv_heads: int = 8,
        text_head_dim: int = 128,
        vocab_size: int = 152064,
        max_seq_len: int = 8192,
        max_batch_size: int = 1,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-5,
        # Common config
        layer_norm_eps: float = 1e-6,
        weight_cache_path=None,
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize Molmo2Model.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Complete model state dict
            vit_*: Vision transformer configuration
            adapter_*: Vision adapter configuration
            text_*: Language model configuration
            weight_cache_path: Path to cache weights
            dtype: Data type for weights
        """
        super().__init__()

        from loguru import logger

        self.mesh_device = mesh_device
        self.text_hidden_dim = text_hidden_dim
        self.dtype = dtype

        # Special token IDs
        self.image_patch_id = 151938
        self.bos_token_id = 151643
        self.eos_token_id = 151645

        logger.info("Creating VisionBackbone...")
        # Vision backbone (ViT + Adapter)
        # Use bfloat16 for vision backbone to avoid numerical precision issues
        # that cause overflow with bfloat8_b (visual embeddings with values ~37000-61000)
        vision_dtype = ttnn.bfloat16  # Higher precision for vision computations
        logger.info(f"Using vision_dtype={vision_dtype} (model dtype={dtype})")
        self.vision_backbone = VisionBackbone(
            mesh_device=mesh_device,
            state_dict=state_dict,
            vit_num_layers=vit_num_layers,
            vit_hidden_dim=vit_hidden_dim,
            vit_intermediate_dim=vit_intermediate_dim,
            vit_num_heads=vit_num_heads,
            vit_head_dim=vit_head_dim,
            patch_size=patch_size,
            image_size=image_size,
            feature_layers=feature_layers,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_intermediate_dim=adapter_intermediate_dim,
            adapter_num_heads=adapter_num_heads,
            adapter_head_dim=adapter_head_dim,
            output_dim=text_hidden_dim,
            layer_norm_eps=layer_norm_eps,
            weight_cache_path=weight_cache_path,
            dtype=vision_dtype,
        )
        logger.info("VisionBackbone created")

        logger.info("Creating TextModel...")
        # Text model (Language Model)
        self.text_model = TextModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            num_layers=text_num_layers,
            hidden_dim=text_hidden_dim,
            intermediate_dim=text_intermediate_dim,
            num_heads=text_num_heads,
            num_kv_heads=text_num_kv_heads,
            head_dim=text_head_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def embed_image(
        self,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        max_frames_per_chunk: int = 8,
        use_data_parallel: bool = False,
        frames_per_device: int = 8,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """
        Process image through vision backbone (fully on TTNN).

        For videos with many frames, processes in chunks to avoid OOM.
        With use_data_parallel=True, uses all 8 devices to process frames in parallel.

        Args:
            pixel_values: Preprocessed image tensor [B, C, H, W]
            pooled_patches_idx: Patch indices for pooling [B, N_out, K_pool]
            max_frames_per_chunk: Max frames to process at once (default 8, used when data_parallel=False)
            use_data_parallel: Use data parallelism across devices (default False)
            frames_per_device: Frames per device per pass when using data parallel (default 8)

        Returns:
            Tuple of:
              - visual_embeddings: [1, 1, N_out, hidden_dim] on device (unfiltered)
              - valid_token: [B, N_out] bool tensor on CPU
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        batch_size = pooled_patches_idx.shape[0]
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        # Patch embedding on TTNN: CPU unfold only, linear+pos_embed on device
        vit = self.vision_backbone.image_vit
        patch_features = vit.patch_size * vit.patch_size * 3  # 14*14*3 = 588

        # Check if we need data parallel or chunked processing for video frames
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        num_devices = self.mesh_device.get_num_devices() if is_mesh_device else 1

        # Auto-enable data parallel for videos with enough frames
        # DP provides ~2x speedup by distributing frames across devices
        auto_dp = is_mesh_device and batch_size >= num_devices and batch_size > 1

        if (use_data_parallel or auto_dp) and is_mesh_device and batch_size >= num_devices:
            # Data parallel: shard frames across all devices
            # Auto-calculate frames_per_device if batch divides evenly, else use provided value
            effective_frames_per_device = frames_per_device
            if batch_size % num_devices == 0:
                # Perfect split: each device gets batch_size/num_devices frames
                effective_frames_per_device = max(1, batch_size // num_devices)
            logger.info(
                f"embed_image: Auto-enabled data parallel for {batch_size} frames across {num_devices} devices, {effective_frames_per_device} frames/device"
            )
            return self._embed_image_data_parallel(
                pixel_values, pooled_patches_idx, effective_frames_per_device, num_devices
            )
        elif batch_size > max_frames_per_chunk:
            logger.info(f"embed_image: Chunked processing {batch_size} frames in chunks of {max_frames_per_chunk}")
            return self._embed_image_chunked(pixel_values, pooled_patches_idx, max_frames_per_chunk)

        # Single-batch processing (original path for images and small videos)
        # Detect input format:
        # - Pre-unfolded from vLLM: [num_crops, num_patches, 588] - 3D with last dim == 588
        # - Raw image format: [B, C, H, W] - 4D
        if pixel_values.dim() == 3 and pixel_values.shape[-1] == patch_features:
            # Pre-unfolded patch format from vLLM [num_crops, num_patches, 588]
            embedded_ttnn = vit.patch_embed_from_patches_ttnn(pixel_values)
        else:
            embedded_ttnn = vit.patch_embed_ttnn(pixel_values)

        # Prepare gather indices and masks (CPU, fast)
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)  # [B, N_out] bool
        clipped_idx = torch.clip(pooled_patches_idx, min=0)
        flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)
        valid_mask = valid.reshape(1, 1, -1, 1).float()

        idx_ttnn = ttnn.from_torch(
            flat_idx,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        valid_mask_ttnn = ttnn.from_torch(
            valid_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        valid_token_ttnn = ttnn.from_torch(
            valid_token.flatten().float(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        visual_embeddings = self.vision_backbone.forward_ttnn(
            images_embedded=embedded_ttnn,
            pooled_patches_idx_ttnn=idx_ttnn,
            valid_mask_ttnn=valid_mask_ttnn,
            valid_token_ttnn=valid_token_ttnn,
            n_out=n_out,
            k_pool=k_pool,
            batch_size=batch_size,
        )

        # Debug: check visual embeddings values
        from loguru import logger

        try:
            is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
            if is_mesh_device:
                mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                vis_torch = ttnn.to_torch(visual_embeddings, mesh_composer=mesh_composer)[0]
            else:
                vis_torch = ttnn.to_torch(visual_embeddings)
            logger.info(f"embed_image DEBUG: visual_embeddings shape={vis_torch.shape}")
            logger.info(
                f"embed_image DEBUG: visual_embeddings stats: min={vis_torch.min().item():.4f}, max={vis_torch.max().item():.4f}, mean={vis_torch.mean().item():.4f}, std={vis_torch.std().item():.4f}"
            )
            logger.info(f"embed_image DEBUG: visual_embeddings first 5 values: {vis_torch.flatten()[:5].tolist()}")
        except Exception as e:
            logger.warning(f"embed_image DEBUG: Failed to inspect visual_embeddings: {e}")

        ttnn.deallocate(embedded_ttnn)
        ttnn.deallocate(idx_ttnn)
        ttnn.deallocate(valid_mask_ttnn)
        ttnn.deallocate(valid_token_ttnn)

        return visual_embeddings, valid_token

    def _embed_image_chunked(
        self,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        max_frames_per_chunk: int = 8,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """
        Process video frames in chunks with two-stage processing.

        Stage 1: Run ViT encoding in chunks (no cross-frame attention needed in ViT)
        Stage 2: Run pooling on ALL concatenated features (preserves cross-frame attention)

        This allows processing videos with many frames (up to 384) while preserving
        the cross-frame attention in image_pooling_2d that the model requires.

        Args:
            pixel_values: [B, C, H, W] where B is number of frames
            pooled_patches_idx: [B, N_out, K_pool] pooling indices (GLOBAL indices)
            max_frames_per_chunk: Max frames per ViT chunk

        Returns:
            Tuple of (visual_embeddings, valid_token)
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh_device else None

        batch_size = pooled_patches_idx.shape[0]  # Total number of frames
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        vit = self.vision_backbone.image_vit
        patch_features = vit.patch_size * vit.patch_size * 3  # 588

        logger.info(
            f"embed_image_chunked: Two-stage processing {batch_size} frames in chunks of {max_frames_per_chunk}"
        )

        # ============================================================
        # STAGE 1: Run ViT encoding in chunks
        # ViT has no cross-frame attention, so chunking is safe here
        # ============================================================
        all_vit_features = []

        for chunk_start in range(0, batch_size, max_frames_per_chunk):
            chunk_end = min(chunk_start + max_frames_per_chunk, batch_size)
            chunk_size = chunk_end - chunk_start

            logger.debug(f"  Stage 1: ViT encoding frames {chunk_start}-{chunk_end} ({chunk_size} frames)")

            # Extract chunk of pixel values
            if pixel_values.dim() == 3 and pixel_values.shape[-1] == patch_features:
                chunk_pixels = pixel_values[chunk_start:chunk_end]
            else:
                chunk_pixels = pixel_values[chunk_start:chunk_end]

            # Embed and run ViT only (no pooling yet)
            if pixel_values.dim() == 3 and pixel_values.shape[-1] == patch_features:
                embedded_ttnn = vit.patch_embed_from_patches_ttnn(chunk_pixels)
            else:
                embedded_ttnn = vit.patch_embed_ttnn(chunk_pixels)

            # Run ViT encoding only
            vit_features = self.vision_backbone.encode_image_only_ttnn(embedded_ttnn)

            # Move to CPU for concatenation
            if is_mesh_device:
                vit_features_torch = ttnn.to_torch(vit_features, mesh_composer=mesh_composer)[0]
            else:
                vit_features_torch = ttnn.to_torch(vit_features)
            logger.debug(
                f"    ViT chunk {chunk_start//max_frames_per_chunk} shape: {vit_features_torch.shape}, expected seq_len: {chunk_size * 729}, mean={vit_features_torch.mean().item():.4f}, std={vit_features_torch.std().item():.4f}"
            )
            all_vit_features.append(vit_features_torch)

            # Cleanup
            ttnn.deallocate(embedded_ttnn)
            ttnn.deallocate(vit_features)
            ttnn.synchronize_device(self.mesh_device)

        # ============================================================
        # Concatenate ALL ViT features
        # to_torch returns [1, seq, hidden], concatenate along seq (dim=1)
        # Then reshape to [1, 1, total_patches, pool_dim] for pool_and_project_ttnn
        # ============================================================
        combined_vit_features = torch.cat(all_vit_features, dim=1)  # [1, total_seq, pool_dim]
        combined_vit_features = combined_vit_features.unsqueeze(1)  # [1, 1, total_seq, pool_dim]
        logger.info(f"  Combined ViT features: {combined_vit_features.shape}")

        # ============================================================
        # STAGE 2: Pool + Project on ALL features together
        # This preserves cross-frame attention via GLOBAL indices
        # Use chunked pooling for large videos to avoid OOM
        # ============================================================
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)  # [batch_size, N_out]

        # Threshold for using chunked pooling (based on memory analysis)
        max_frames_for_single_pool = 32

        if batch_size <= max_frames_for_single_pool:
            # Small video: use single pooling pass
            logger.debug(f"  Stage 2: Single-pass pooling on {batch_size} frames")

            clipped_idx = torch.clip(pooled_patches_idx, min=0)
            flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)
            valid_mask = valid.reshape(1, 1, -1, 1).float()

            combined_features_ttnn = ttnn.from_torch(
                combined_vit_features,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

            idx_ttnn = ttnn.from_torch(
                flat_idx,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

            valid_mask_ttnn = ttnn.from_torch(
                valid_mask,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

            visual_embeddings = self.vision_backbone.pool_and_project_ttnn(
                image_features=combined_features_ttnn,
                pooled_patches_idx_ttnn=idx_ttnn,
                valid_mask_ttnn=valid_mask_ttnn,
                n_out=n_out,
                k_pool=k_pool,
                batch_size=batch_size,
            )

            ttnn.deallocate(combined_features_ttnn)
            ttnn.deallocate(idx_ttnn)
            ttnn.deallocate(valid_mask_ttnn)
        else:
            # Large video: use chunked pooling to reduce memory
            logger.debug(f"  Stage 2: Chunked pooling on {batch_size} frames")

            combined_features_ttnn = ttnn.from_torch(
                combined_vit_features,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

            visual_embeddings = self.vision_backbone.pool_and_project_chunked_ttnn(
                image_features=combined_features_ttnn,
                pooled_patches_idx=pooled_patches_idx,  # torch tensor, not flattened
                valid_mask=valid,  # torch tensor
                n_out=n_out,
                k_pool=k_pool,
                batch_size=batch_size,
                max_frames_per_pool_chunk=32,
            )
            # Note: combined_features_ttnn is deallocated inside pool_and_project_chunked_ttnn

        logger.info(f"embed_image_chunked: Complete, output shape: {visual_embeddings.shape}")

        return visual_embeddings, valid_token

    def _embed_image_data_parallel(
        self,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        frames_per_device: int = 8,
        num_devices: int = 8,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """
        Process video frames with DATA PARALLELISM across devices.

        Instead of replicating work across devices (wasteful), this method
        SHARDS frames across devices so each device processes different frames
        in parallel.

        For 64 frames with 8 devices and frames_per_device=8:
          - Device 0: frames 0-7
          - Device 1: frames 8-15
          - ...
          - Device 7: frames 56-63
          - All 64 frames processed in ONE parallel pass!

        For 384 frames: 384 / (8 devices * 8 frames) = 6 sequential passes

        Args:
            pixel_values: [total_frames, C, H, W] where total_frames can be large
            pooled_patches_idx: [total_frames, N_out, K_pool] pooling indices (GLOBAL)
            frames_per_device: Frames each device processes per pass (default: 8)
            num_devices: Number of devices in mesh (default: 8)

        Returns:
            Tuple of (visual_embeddings, valid_token)
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        if not is_mesh_device:
            logger.warning("Data parallel requires mesh device, falling back to chunked")
            return self._embed_image_chunked(pixel_values, pooled_patches_idx, frames_per_device)

        total_frames = pooled_patches_idx.shape[0]
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        vit = self.vision_backbone.image_vit
        num_patches_per_frame = (vit.image_size // vit.patch_size) ** 2  # 729
        patch_features = vit.patch_size * vit.patch_size * 3  # 588
        pool_dim = vit.hidden_dim * 2  # 2304 (concat of 2 layers)

        frames_per_pass = frames_per_device * num_devices
        num_passes = (total_frames + frames_per_pass - 1) // frames_per_pass

        logger.info(
            f"embed_image_data_parallel: {total_frames} frames, "
            f"{frames_per_device} frames/device, {num_devices} devices, "
            f"{frames_per_pass} frames/pass, {num_passes} passes"
        )

        # Mesh mappers for sharding and gathering
        shard_mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        replicate_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)

        # ============================================================
        # STAGE 1: Run ViT encoding with data parallelism
        # ============================================================
        all_vit_features = []

        for pass_idx in range(num_passes):
            pass_start = pass_idx * frames_per_pass
            pass_end = min(pass_start + frames_per_pass, total_frames)
            actual_frames_this_pass = pass_end - pass_start

            logger.debug(
                f"  Pass {pass_idx + 1}/{num_passes}: frames {pass_start}-{pass_end} "
                f"({actual_frames_this_pass} frames)"
            )

            # Get frames for this pass
            pass_pixels = pixel_values[pass_start:pass_end]  # [actual_frames, C, H, W]

            # Pad to full frames_per_pass if needed for even sharding
            if actual_frames_this_pass < frames_per_pass:
                pad_frames = frames_per_pass - actual_frames_this_pass
                # Pad with zeros (will be masked out later)
                if pass_pixels.dim() == 4:
                    pad_shape = (pad_frames,) + pass_pixels.shape[1:]
                else:
                    pad_shape = (pad_frames,) + pass_pixels.shape[1:]
                padding = torch.zeros(pad_shape, dtype=pass_pixels.dtype)
                pass_pixels = torch.cat([pass_pixels, padding], dim=0)

            # Reshape for sharding: [num_devices, frames_per_device, C, H, W]
            pass_pixels = pass_pixels.reshape(num_devices, frames_per_device, *pass_pixels.shape[1:])

            # CPU preprocessing: unfold patches
            # Each device's data: [frames_per_device, C, H, W] -> [frames_per_device, num_patches, patch_features]
            device_patches_list = []
            for dev_idx in range(num_devices):
                dev_pixels = pass_pixels[dev_idx]  # [frames_per_device, C, H, W]
                if dev_pixels.dim() == 4:
                    # Unfold patches on CPU
                    B, C, H, W = dev_pixels.shape
                    x = dev_pixels.unfold(2, vit.patch_size, vit.patch_size)
                    x = x.unfold(3, vit.patch_size, vit.patch_size)
                    x = x.permute(0, 2, 3, 4, 5, 1).reshape(B * num_patches_per_frame, patch_features)
                    device_patches_list.append(x)
                else:
                    device_patches_list.append(dev_pixels.reshape(-1, patch_features))

            # Stack for sharding: [num_devices, frames_per_device * num_patches, patch_features]
            all_patches = torch.stack(device_patches_list, dim=0).float()
            # Reshape to [num_devices, 1, frames_per_device * num_patches, patch_features]
            all_patches = all_patches.unsqueeze(1)

            # Transfer to devices with SHARDING (each device gets its own frames)
            patches_ttnn = ttnn.from_torch(
                all_patches,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard_mapper,
            )

            # Run patch embedding projection on each device (weights are replicated)
            embedded = ttnn.matmul(patches_ttnn, vit.patch_embed_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(patches_ttnn)
            embedded = ttnn.add(embedded, vit.patch_embed_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Add positional embedding (tiled for frames_per_device frames)
            pos_tiles = [vit.positional_embedding] * frames_per_device
            pos_tiled = ttnn.concat(pos_tiles, dim=2)
            embedded = ttnn.add(embedded, pos_tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pos_tiled)

            # Run ViT forward on each device's data
            vit_output = self.vision_backbone.encode_image(embedded)
            ttnn.deallocate(embedded)

            # Gather results from all devices
            # Output shape per device: [1, 1, frames_per_device * num_patches, pool_dim]
            # After gather: [num_devices, 1, frames_per_device * num_patches, pool_dim]
            vit_features_torch = ttnn.to_torch(vit_output, mesh_composer=mesh_composer)
            ttnn.deallocate(vit_output)

            # Reshape: [num_devices, 1, frames*patches, dim] -> [1, 1, total_patches, dim]
            vit_features_torch = vit_features_torch.reshape(1, 1, -1, pool_dim)

            # Trim padding if we padded earlier
            actual_patches = actual_frames_this_pass * num_patches_per_frame
            vit_features_torch = vit_features_torch[:, :, :actual_patches, :]

            all_vit_features.append(vit_features_torch)
            ttnn.synchronize_device(self.mesh_device)

        # Concatenate all passes
        combined_vit_features = torch.cat(all_vit_features, dim=2)
        logger.info(f"  Combined ViT features: {combined_vit_features.shape}")

        # ============================================================
        # STAGE 2: Pool + Project (same as chunked version)
        # ============================================================
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)

        max_frames_for_single_pool = 32
        batch_size = total_frames

        if batch_size <= max_frames_for_single_pool:
            logger.debug(f"  Stage 2: Single-pass pooling on {batch_size} frames")

            clipped_idx = torch.clip(pooled_patches_idx, min=0)
            flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)
            valid_mask = valid.reshape(1, 1, -1, 1).float()

            combined_features_ttnn = ttnn.from_torch(
                combined_vit_features,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper,
            )

            idx_ttnn = ttnn.from_torch(
                flat_idx,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper,
            )

            valid_mask_ttnn = ttnn.from_torch(
                valid_mask,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper,
            )

            visual_embeddings = self.vision_backbone.pool_and_project_ttnn(
                image_features=combined_features_ttnn,
                pooled_patches_idx_ttnn=idx_ttnn,
                valid_mask_ttnn=valid_mask_ttnn,
                n_out=n_out,
                k_pool=k_pool,
                batch_size=batch_size,
            )

            ttnn.deallocate(combined_features_ttnn)
            ttnn.deallocate(idx_ttnn)
            ttnn.deallocate(valid_mask_ttnn)
        else:
            logger.debug(f"  Stage 2: Chunked pooling on {batch_size} frames")

            combined_features_ttnn = ttnn.from_torch(
                combined_vit_features,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper,
            )

            visual_embeddings = self.vision_backbone.pool_and_project_chunked_ttnn(
                image_features=combined_features_ttnn,
                pooled_patches_idx=pooled_patches_idx,
                valid_mask=valid,
                n_out=n_out,
                k_pool=k_pool,
                batch_size=batch_size,
                max_frames_per_pool_chunk=32,
            )

        logger.info(f"embed_image_data_parallel: Complete, output shape: {visual_embeddings.shape}")

        return visual_embeddings, valid_token

    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        visual_embeddings_ttnn: ttnn.Tensor,
        valid_token: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Fuse text and visual embeddings on device using selector matmul.

        No CPU roundtrip: text embed + selector matmul + add all on device.

        Args:
            input_ids: Token IDs with image_patch_id placeholders [batch, seq_len]
            visual_embeddings_ttnn: Visual embeddings [1, 1, N_out, hidden_dim] on device
            valid_token: [B, N_out] bool mask for which visual tokens are valid (CPU)

        Returns:
            Fused embeddings [1, 1, seq_len, hidden_dim] on device
        """
        batch_size, seq_len = input_ids.shape
        # Note: batch_size > 1 is now supported
        hidden_dim = self.text_hidden_dim

        # DEBUG: Track request count
        if not hasattr(self, "_multimodal_request_count"):
            self._multimodal_request_count = 0
        self._multimodal_request_count += 1
        logger.info(f"prepare_inputs_for_multimodal: REQUEST #{self._multimodal_request_count}")
        logger.info(f"  input_ids.shape={input_ids.shape}, seq_len={seq_len}")
        logger.info(f"  visual_embeddings_ttnn shape={visual_embeddings_ttnn.shape}")
        logger.info(f"  valid_token.shape={valid_token.shape}, sum={valid_token.sum().item()}")

        # FIX: vLLM places image tokens BEFORE the chat template (wrong position)
        # We need to reorder: move image tokens to after <|im_start|>user\n
        im_start_token = 151644  # <|im_start|>
        user_token = 872  # user
        newline_token = 198  # \n

        # Check if image tokens are at start (wrong) and chat template comes after (also wrong)
        image_positions = (input_ids[0] == self.image_patch_id).nonzero(as_tuple=True)[0]
        if len(image_positions) > 0 and image_positions[0] == 0:
            # Image tokens at start - find where chat template is
            im_start_pos = (input_ids[0] == im_start_token).nonzero(as_tuple=True)[0]
            if len(im_start_pos) > 0:
                chat_start = im_start_pos[0].item()
                # Check if this is the user message (followed by user token and newline)
                if (
                    chat_start + 2 < seq_len
                    and input_ids[0, chat_start + 1] == user_token
                    and input_ids[0, chat_start + 2] == newline_token
                ):
                    # Find how many image tokens are at the start
                    num_leading_images = 0
                    for i in range(chat_start):
                        if input_ids[0, i] == self.image_patch_id:
                            num_leading_images += 1
                        else:
                            break  # Stop at first non-image token

                    if num_leading_images > 0 and num_leading_images == chat_start:
                        # All tokens before chat template are image tokens - reorder!
                        logger.info(
                            f"  FIX: Reordering {num_leading_images} image tokens from start to after chat template"
                        )
                        # New order: [<|im_start|>user\n, IMAGE_TOKENS, rest...]
                        image_tokens = input_ids[0, :num_leading_images]
                        chat_and_rest = input_ids[0, num_leading_images:]
                        # Insert image tokens after <|im_start|>user\n (3 tokens)
                        new_input_ids = torch.cat(
                            [
                                chat_and_rest[:3],  # <|im_start|>user\n
                                image_tokens,  # image tokens
                                chat_and_rest[3:],  # rest of the message
                            ]
                        ).unsqueeze(0)
                        input_ids = new_input_ids
                        logger.info(f"  FIX: Reordered input_ids, new shape={input_ids.shape}")
                        logger.info(f"  FIX: New input_ids first 20: {input_ids[0][:20].tolist()}")

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        # Get text embeddings on device
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        text_embeddings_ttnn = self.text_model.embed_tokens(input_ids_ttnn)
        ttnn.deallocate(input_ids_ttnn)

        # Filter valid visual embeddings on device via ttnn.embedding (gather)
        valid_indices = valid_token.flatten().nonzero(as_tuple=True)[0].to(torch.int32)
        num_valid = len(valid_indices)

        if num_valid == 0:
            return text_embeddings_ttnn

        valid_indices_ttnn = ttnn.from_torch(
            valid_indices.unsqueeze(0),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        # visual_embeddings_ttnn: [1, 1, N_out, hidden_dim] -> [1, N_out, hidden_dim] for gather
        visual_for_gather = ttnn.reshape(visual_embeddings_ttnn, [1, -1, hidden_dim])
        valid_visual_ttnn = ttnn.embedding(valid_indices_ttnn, visual_for_gather)
        ttnn.deallocate(valid_indices_ttnn)
        # NOTE: Do NOT deallocate visual_for_gather - reshape may return a view

        # valid_visual_ttnn: [1, num_valid, hidden_dim] -> [1, 1, num_valid, hidden_dim]
        valid_visual_ttnn = ttnn.reshape(valid_visual_ttnn, [1, 1, num_valid, hidden_dim])
        # NOTE: Do NOT deallocate original - reshape may return a view

        # Find image positions in input_ids
        image_positions = (input_ids[0] == self.image_patch_id).nonzero(as_tuple=True)[0]
        logger.info(
            f"prepare_inputs_for_multimodal: image_patch_id={self.image_patch_id}, found {len(image_positions)} image positions, num_valid={num_valid}"
        )
        if len(image_positions) != num_valid:
            logger.warning(
                f"  MISMATCH! len(image_positions)={len(image_positions)} != num_valid={num_valid}, returning text-only embeddings!"
            )
            ttnn.deallocate(valid_visual_ttnn)
            return text_embeddings_ttnn

        # O(n) index-based fusion instead of O(n²) selector matrix
        # This avoids creating a [seq_len, num_valid] dense matrix which would be
        # 5GB+ for 256 frames (50k visual tokens)
        #
        # Move to CPU, scatter visual embeddings at image positions, move back
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh_device else None

        # Get text embeddings on CPU
        if is_mesh_device:
            text_emb_cpu = ttnn.to_torch(text_embeddings_ttnn, mesh_composer=mesh_composer)[0]
        else:
            text_emb_cpu = ttnn.to_torch(text_embeddings_ttnn)
        ttnn.deallocate(text_embeddings_ttnn)

        # Get visual embeddings on CPU
        if is_mesh_device:
            visual_cpu = ttnn.to_torch(valid_visual_ttnn, mesh_composer=mesh_composer)[0]
        else:
            visual_cpu = ttnn.to_torch(valid_visual_ttnn)
        ttnn.deallocate(valid_visual_ttnn)

        # text_emb_cpu shape: [1, seq_len, hidden_dim] (squeezed from [1, 1, seq_len, hidden_dim])
        # visual_cpu shape: [1, num_valid, hidden_dim] (squeezed from [1, 1, num_valid, hidden_dim])
        # Ensure 4D for consistency
        if text_emb_cpu.dim() == 3:
            text_emb_cpu = text_emb_cpu.unsqueeze(1)  # [1, 1, seq_len, hidden_dim]
        if visual_cpu.dim() == 3:
            visual_cpu = visual_cpu.unsqueeze(1)  # [1, 1, num_valid, hidden_dim]

        logger.info(f"  O(n) fusion: text_emb_cpu shape={text_emb_cpu.shape}, visual_cpu shape={visual_cpu.shape}")
        logger.info(
            f"  O(n) fusion: visual_cpu stats: mean={visual_cpu.mean().item():.4f}, std={visual_cpu.std().item():.4f}"
        )
        logger.info(
            f"  O(n) fusion: text_emb_cpu stats (before): mean={text_emb_cpu.mean().item():.4f}, std={text_emb_cpu.std().item():.4f}"
        )

        # Fuse: ADD visual to text at image positions (O(n) indexing instead of O(n²) matmul)
        # Reference: x.view(-1, x.shape[-1])[is_image_patch] += image_features
        text_emb_cpu[0, 0, image_positions, :] += visual_cpu[0, 0, :, :]
        logger.info(
            f"  O(n) fusion: text_emb_cpu stats (after): mean={text_emb_cpu.mean().item():.4f}, std={text_emb_cpu.std().item():.4f}"
        )

        # Move fused embeddings back to device
        fused_ttnn = ttnn.from_torch(
            text_emb_cpu,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        return fused_ttnn

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
        page_table: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through the full Molmo2 model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            pixel_values: Optional image tensor for visual input
            pooled_patches_idx: Optional patch indices for pooling
            attention_mask: Optional attention mask
            kv_caches: Optional KV cache for incremental decoding
            start_pos: Starting position for KV cache
            page_table: Optional page table for paged attention (vLLM)

        Returns:
            Tuple of (logits, new_kv_caches)
        """
        # Process images if provided -- fully on TTNN, no CPU roundtrip
        if pixel_values is not None and pooled_patches_idx is not None:
            visual_embeddings_ttnn, valid_token = self.embed_image(pixel_values, pooled_patches_idx)
            hidden_states_ttnn = self.prepare_inputs_for_multimodal(input_ids, visual_embeddings_ttnn, valid_token)
            ttnn.deallocate(visual_embeddings_ttnn)
        else:
            # Text-only forward
            is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

            input_ids_ttnn = ttnn.from_torch(
                input_ids,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            hidden_states_ttnn = self.text_model.embed_tokens(input_ids_ttnn)
            ttnn.deallocate(input_ids_ttnn)

        # Forward through text model (handles both prefill and decode via KV cache)
        logits, new_kv_caches = self.text_model(
            hidden_states=hidden_states_ttnn,
            start_pos=start_pos,
            attn_mask=None,
            kv_caches=kv_caches,
            page_table=page_table,
        )

        return logits, new_kv_caches

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            pixel_values: Optional image input
            pooled_patches_idx: Optional patch indices
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()

        # Initial forward pass (prefill)
        logits, kv_caches = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
        )

        # Get next token
        logits_torch = ttnn.to_torch(logits).squeeze()
        # Handle different output shapes
        if logits_torch.dim() == 2:
            # Shape: [seq_len, vocab_size] - take last position
            next_token_logits = logits_torch[-1:, :]
        else:
            # Shape: [batch, seq_len, vocab_size]
            next_token_logits = logits_torch[:, -1, :]

        for _ in range(max_new_tokens):
            # Sample or greedy decode
            if do_sample:
                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if (next_token == self.eos_token_id).all():
                break

            # Decode step (single token)
            logits, kv_caches = self.forward(
                input_ids=next_token,
                kv_caches=kv_caches,
                start_pos=generated_ids.shape[1] - 1,
            )

            logits_torch = ttnn.to_torch(logits).squeeze(0).squeeze(0)
            next_token_logits = logits_torch[:, -1, :]

        return generated_ids
