# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
vLLM integration for Molmo2-8B model.

This module provides the vLLM-compatible wrapper class for Molmo2-8B,
enabling integration with tt-inference-server via the vLLM plugin.
"""

from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch
from loguru import logger
from PIL.Image import Image
from tqdm import tqdm
from transformers import BatchFeature

import ttnn
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.molmo import MolmoProcessingInfo, get_patches_grid_size, select_tiling
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor, PromptReplacement, PromptUpdate
from vllm.multimodal.profiling import BaseDummyInputsBuilder

# Note: Model registration is handled by tt-vllm-plugin/__init__.py
# The plugin registers TTMolmo2ForConditionalGeneration with the module path
# models.demos.molmo2.tt.generator_vllm:Molmo2ForConditionalGeneration

# Module-level cache for image_token_pooling
# This bypasses vLLM's multimodal batching which can't handle the irregular shape
_image_token_pooling_cache = {"last": None}


def compute_image_token_pooling(
    image_grids: torch.Tensor,
    num_crops: int,
    crop_patch_size: int = 27,  # 378 / 14 = 27 patches per crop dimension
    pool_h: int = 2,
    pool_w: int = 2,
) -> torch.Tensor:
    """
    Compute image_token_pooling indices from image_grids.

    This replicates the logic from Molmo2ImageProcessor to compute pooling indices
    when they aren't available (e.g., cross-process caching issues).

    Args:
        image_grids: Grid info [resized_h, resized_w, h, w]
            - resized_h, resized_w: POOLED dimensions for low-res (global) image
            - h, w: POOLED dimensions for high-res image
        num_crops: Number of crops in pixel_values
        crop_patch_size: Patches per crop dimension (default 27 for 378/14)
        pool_h, pool_w: Pooling dimensions (default 2x2)

    Returns:
        image_token_pooling: Indices tensor [total_pooled_tokens, pool_h*pool_w]
    """
    import numpy as np
    from loguru import logger

    # Extract grid dimensions - these ARE the pooled dimensions
    if isinstance(image_grids, torch.Tensor):
        grid = image_grids.cpu().numpy().flatten()
    else:
        grid = np.array(image_grids).flatten()

    resized_h, resized_w, h, w = int(grid[0]), int(grid[1]), int(grid[2]), int(grid[3])
    logger.info(
        f"compute_image_token_pooling: resized_h={resized_h}, resized_w={resized_w}, h={h}, w={w}, num_crops={num_crops}"
    )

    def arange_for_pooling(idx_arr, pool_h, pool_w):
        """Pad and reshape index array for pooling."""
        h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
        w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
        idx_arr = np.pad(
            idx_arr,
            [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
            mode="constant",
            constant_values=-1,
        )
        # Reshape for pooling: (h dh) (w dw) -> h w (dh dw)
        new_h = idx_arr.shape[0] // pool_h
        new_w = idx_arr.shape[1] // pool_w
        idx_arr = idx_arr.reshape(new_h, pool_h, new_w, pool_w)
        idx_arr = idx_arr.transpose(0, 2, 1, 3)
        idx_arr = idx_arr.reshape(new_h, new_w, pool_h * pool_w)
        return idx_arr

    # Compute indices for low-res (global) image - first crop
    # The crop has crop_patch_size x crop_patch_size patches, pooled to resized_h x resized_w
    crop_patch_h = crop_patch_w = crop_patch_size
    resize_idx = np.arange(crop_patch_h * crop_patch_w).reshape(crop_patch_h, crop_patch_w)
    resize_idx = arange_for_pooling(resize_idx, pool_h, pool_w)
    # Truncate/select to match the target pooled size (resized_h x resized_w)
    resize_idx = resize_idx[:resized_h, :resized_w, :]
    resize_idx_flat = resize_idx.reshape(-1, pool_h * pool_w)
    logger.info(
        f"  Low-res: {crop_patch_h}x{crop_patch_w} -> {resized_h}x{resized_w} = {resize_idx_flat.shape[0]} tokens"
    )

    # Compute indices for high-res image - remaining crops
    if num_crops > 1:
        # High-res indices start after the first crop's patches
        offset = crop_patch_h * crop_patch_w

        # For high-res, we have (num_crops-1) crops arranged in a grid
        # Total high-res patches = (num_crops - 1) * crop_patch_size^2
        num_high_res_crops = num_crops - 1
        total_high_res_patches = num_high_res_crops * crop_patch_h * crop_patch_w

        # The high-res grid has h x w POOLED tokens
        # Create a grid of indices for the high-res patches
        # We need to map h*w pooled tokens to 4 patch indices each

        # Compute how the high-res crops are arranged
        # For a 2x2 crop grid: patches are arranged in scan order
        crops_per_row = int(np.ceil(np.sqrt(num_high_res_crops)))
        crops_per_col = int(np.ceil(num_high_res_crops / crops_per_row))

        # Pre-pooled grid size (before 2x2 pooling)
        pre_pool_h = h * pool_h
        pre_pool_w = w * pool_w

        # Create indices for the high-res patches
        # Map each position in the pre-pooled grid to a patch index
        high_res_idx = np.full((pre_pool_h, pre_pool_w), -1, dtype=np.int64)

        # Fill in patch indices based on crop tiling
        # This is a simplified mapping - assumes crops tile to cover h*pool_h x w*pool_w
        patch_idx = offset
        for cy in range(crops_per_col):
            for cx in range(crops_per_row):
                crop_num = cy * crops_per_row + cx
                if crop_num >= num_high_res_crops:
                    break
                # Each crop contributes crop_patch_size x crop_patch_size patches
                y_start = cy * crop_patch_h
                x_start = cx * crop_patch_w
                for py in range(crop_patch_h):
                    for px in range(crop_patch_w):
                        y = y_start + py
                        x = x_start + px
                        if y < pre_pool_h and x < pre_pool_w:
                            high_res_idx[y, x] = patch_idx
                        patch_idx += 1

        # Pool the high-res indices
        high_res_pooled = arange_for_pooling(high_res_idx, pool_h, pool_w)
        # Truncate to match target pooled size (h x w)
        high_res_pooled = high_res_pooled[:h, :w, :]
        high_res_flat = high_res_pooled.reshape(-1, pool_h * pool_w)
        logger.info(
            f"  High-res: {num_high_res_crops} crops, pre-pool {pre_pool_h}x{pre_pool_w} -> {h}x{w} = {high_res_flat.shape[0]} tokens"
        )

        pooling_idx = np.concatenate([resize_idx_flat, high_res_flat], axis=0)
    else:
        pooling_idx = resize_idx_flat

    logger.info(f"  Total pooling_idx shape: {pooling_idx.shape}")
    return torch.from_numpy(pooling_idx.astype(np.int64))


from models.demos.molmo2.demo.demo import (
    Molmo2Generator,
    create_model,
    load_model_weights,
    load_processor,
    preprocess_image_molmo2,
)
from models.demos.molmo2.tt.model_config import Molmo2ModelArgs


def allocate_molmo2_kv_cache(
    kv_cache_shape: Tuple[int, ...],
    dtype: torch.dtype,
    num_layers: int,
    mesh_device: ttnn.MeshDevice,
    tt_cache_path: str,
) -> List[List[ttnn.Tensor]]:
    """
    Allocate vLLM-style KV cache for Molmo2 text model.

    Args:
        kv_cache_shape: Shape of each KV cache tensor (num_blocks, num_kv_heads, block_size, head_size)
        dtype: Data type for KV cache
        num_layers: Number of transformer layers
        mesh_device: TT mesh device
        tt_cache_path: Path for caching TT tensors

    Returns:
        List of [K cache, V cache] pairs for each layer
    """
    kv_cache = []
    cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)

    for layer_num in tqdm(range(num_layers), desc="Allocating TT KV caches for Molmo2"):
        kv_tt_i = [
            ttnn.as_tensor(
                cache_kv,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,  # Molmo2 uses bfloat16 for KV cache
                cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
            )
            for kv in ["k", "v"]
        ]
        kv_cache.append(kv_tt_i)

    return [kv_cache]  # Wrap in list for data parallel compatibility


class Molmo2ProcessorWrapper:
    """
    Wrapper for Molmo2Processor that provides vLLM Molmo1-compatible interface.

    Molmo2 uses different attribute names than Molmo1. This wrapper maps
    Molmo2's API to the expected Molmo1 API that vLLM's MolmoProcessorWrapper expects.

    vLLM's MolmoProcessorWrapper accesses these as cached_properties on the wrapper:
    - max_crops, base_image_input_size, image_patch_size, overlap_margins
    - image_token_length_w, image_token_length_h, vocab
    """

    def __init__(self, processor):
        self.processor = processor
        # Forward key attributes
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

        # Add Molmo1-compatible attributes to the image_processor
        ip = self.image_processor

        # base_image_input_size: Molmo2 uses size dict, Molmo1 uses tuple
        if hasattr(ip, "size"):
            ip.base_image_input_size = (ip.size["height"], ip.size["width"])
        else:
            ip.base_image_input_size = (378, 378)  # Molmo2 default

        # image_patch_size: Molmo2 uses patch_size
        if hasattr(ip, "patch_size"):
            ip.image_patch_size = ip.patch_size
        else:
            ip.image_patch_size = 14  # Molmo2 default

        # image_token_length_w/h: computed from image size, patch size, and pooling
        pooling_size = getattr(ip, "pooling_size", [2, 2])
        if isinstance(pooling_size, (list, tuple)):
            pool_h, pool_w = pooling_size
        else:
            pool_h = pool_w = pooling_size

        patches_h = ip.base_image_input_size[0] // ip.image_patch_size  # 27
        patches_w = ip.base_image_input_size[1] // ip.image_patch_size  # 27

        # Token length after pooling (ceiling division)
        ip.image_token_length_h = (patches_h + pool_h - 1) // pool_h  # 14
        ip.image_token_length_w = (patches_w + pool_w - 1) // pool_w  # 14

    # Properties that vLLM's MolmoProcessorWrapper expects on the wrapper itself
    @property
    def vocab(self) -> dict:
        return self.tokenizer.vocab

    @property
    def image_patch_id(self) -> int:
        """Token ID for <im_patch>."""
        return self.vocab.get("<im_patch>", 151938)  # Default Molmo2 token ID

    @property
    def im_col_id(self) -> int:
        """Token ID for <im_col>."""
        return self.vocab.get("<im_col>", 151939)

    @property
    def im_start_id(self) -> int:
        """Token ID for <im_start>."""
        return self.vocab.get("<im_start>", 151940)

    @property
    def im_end_id(self) -> int:
        """Token ID for <im_end>."""
        return self.vocab.get("<im_end>", 151941)

    @property
    def message_format(self) -> str:
        return "role"

    @property
    def always_start_with_space(self) -> bool:
        return True

    @property
    def max_crops(self) -> int:
        return getattr(self.image_processor, "max_crops", 8)

    @property
    def base_image_input_size(self) -> tuple:
        return self.image_processor.base_image_input_size

    @property
    def image_patch_size(self) -> int:
        return self.image_processor.image_patch_size

    @property
    def overlap_margins(self) -> tuple:
        margins = getattr(self.image_processor, "overlap_margins", [4, 4])
        return tuple(margins)

    @property
    def image_token_length_w(self) -> int:
        return self.image_processor.image_token_length_w

    @property
    def image_token_length_h(self) -> int:
        return self.image_processor.image_token_length_h

    @property
    def pooling_size(self) -> int:
        return 2  # Molmo2 default pooling size

    def select_tiling(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple:
        """Select tiling for an image."""
        max_crops = self.max_crops
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d

        tiling_h, tiling_w = select_tiling(
            height=image_height - total_margin_pixels,
            width=image_width - total_margin_pixels,
            patch_size=crop_window_size,
            max_num_patches=max_crops,
        )

        return tiling_w, tiling_h

    def get_patches_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple:
        """Get patches grid size for an image."""
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size
        pooling_size = self.pooling_size

        crop_patches = base_image_input_size[0] // base_image_input_d
        tiling_w, tiling_h = self.select_tiling(
            image_height=image_height,
            image_width=image_width,
        )

        nrows, ncols = get_patches_grid_size(
            tiling_h=tiling_h,
            tiling_w=tiling_w,
            crop_patches=crop_patches,
            left_margin=left_margin,
            right_margin=right_margin,
            pooling_size=pooling_size,
        )

        return ncols, nrows

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        """
        Call the underlying Molmo2Processor.

        For vLLM compatibility, we need to:
        1. Process images/videos to get pixel_values, image_grids, etc.
        2. Tokenize text WITHOUT replacing <|image|> placeholder
        3. Let vLLM's _get_prompt_updates handle the token replacement

        For videos, we extract frames and process them as images.
        """
        import numpy as np
        from loguru import logger

        logger.info(
            f"Molmo2ProcessorWrapper.__call__ invoked: text={text[:50] if text else None}..., images={type(images)}, videos={type(videos)}"
        )

        # Handle video input by extracting frames
        # vLLM passes videos as numpy array of shape [num_videos, num_frames, H, W, C]
        self._is_video_input = False
        if videos is not None and images is None:
            logger.info(f"  Processing video input: type={type(videos)}")
            self._is_video_input = True
            if isinstance(videos, list) and len(videos) > 0:
                video_frames = videos[0]  # Take first video
                if isinstance(video_frames, np.ndarray):
                    # video_frames shape: [num_frames, H, W, C]
                    logger.info(f"    Video frames shape: {video_frames.shape}")
                    # Sample 8 frames evenly if more than 8
                    num_frames = video_frames.shape[0]
                    if num_frames > 8:
                        indices = np.linspace(0, num_frames - 1, 8, dtype=int)
                        video_frames = video_frames[indices]
                    # Treat video frames as a batch of images
                    images = [video_frames[i] for i in range(video_frames.shape[0])]
                    logger.info(f"    Extracted {len(images)} frames for processing")

        # Process images to get all outputs including image_grids
        if images is not None:
            logger.info(f"  Processing images: type={type(images)}")
            image_outputs = self.image_processor(images, return_tensors="np")
            # Ensure image_grids is present (needed for _get_prompt_updates)
            if "image_grids" not in image_outputs:
                # Compute default image_grids if not present
                # Format: [resized_h, resized_w, height, width] per image
                num_images = 1 if not isinstance(images, list) else len(images)
                # Default to 14x14 grid (378/27 patches, pooled by 2)
                image_outputs["image_grids"] = np.array([[14, 14, 14, 14]] * num_images)
            # Ensure image_num_crops is present
            if "image_num_crops" not in image_outputs:
                num_images = 1 if not isinstance(images, list) else len(images)
                image_outputs["image_num_crops"] = np.array([1] * num_images)
        else:
            image_outputs = {}

        # Tokenize text keeping <|image|> or <|video|> placeholder intact
        # Don't call the full processor - just tokenize and process images separately
        if text is not None:
            num_images = 0
            is_video_input = False
            if images is not None:
                num_images = len(images) if isinstance(images, list) else 1
                # If we have more than 1 image and they came from video processing, use <|video|>
                if num_images > 1 and hasattr(self, "_is_video_input") and self._is_video_input:
                    is_video_input = True

            # Check for existing placeholders
            existing_video = text.count("<|video|>")
            existing_image = text.count("<|image|>")

            if is_video_input:
                # For video input, use <|video|> token (not multiple <|image|>)
                if existing_video == 0 and existing_image == 0:
                    text = "<|video|>" + text
                    logger.info(f"  Added <|video|> placeholder for video frames ({num_images} frames)")
            elif num_images > 0 and existing_image == 0 and existing_video == 0:
                # For regular images, add <|image|> for each
                image_placeholders = "<|image|>" * num_images
                text = image_placeholders + text
                logger.info(f"  Added {num_images} <|image|> placeholders")
            elif num_images > existing_image and existing_video == 0:
                # Not enough placeholders, add more
                additional = num_images - existing_image
                text = "<|image|>" * additional + text
                logger.info(f"  Added {additional} additional <|image|> placeholders")

            # Tokenize without image token expansion
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        else:
            text_inputs = {}

        # Merge outputs
        outputs = BatchFeature({**text_inputs, **image_outputs}, tensor_type=return_tensors)
        return outputs


class Molmo2MultiModalProcessor(BaseMultiModalProcessor["TT_MolmoProcessingInfo"]):
    """
    Custom multimodal processor for Molmo2.

    Molmo2 uses different field names than Molmo1:
    - `pixel_values` instead of `images`
    - `image_num_crops` instead of `num_crops`
    - `image_token_pooling` instead of `image_input_idx`

    This class overrides the necessary methods to handle Molmo2's output format.
    """

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,  # MultiModalDataItems
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """
        Return False so vLLM applies prompt updates via _get_prompt_updates.

        For Molmo2, we want vLLM to handle the placeholder replacement rather
        than having the HF processor do it. This is because Molmo2's processor
        replaces <|image|> at the string level, but vLLM expects to handle
        the replacement at the token level via _get_prompt_updates.
        """
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """
        Configure multimodal fields for Molmo2.

        Molmo2 returns:
        - pixel_values: image crops
        - image_token_pooling: pooling indices
        - image_grids: grid information
        - image_num_crops: number of crops per image
        """
        import numpy as np
        from loguru import logger

        # Debug: log what we received from the HF processor
        logger.info(f"_get_mm_fields_config called: hf_inputs keys = {list(hf_inputs.keys())}")
        for key, value in hf_inputs.items():
            if hasattr(value, "shape"):
                logger.info(f"  {key}: shape={value.shape}, dtype={getattr(value, 'dtype', 'N/A')}")
            elif isinstance(value, (list, tuple)):
                logger.info(f"  {key}: list/tuple len={len(value)}")
            else:
                logger.info(f"  {key}: type={type(value)}")

        # Molmo2 uses image_num_crops instead of num_crops
        num_crops_raw = hf_inputs.get("image_num_crops", None)

        # Convert to numpy array for flat_from_sizes
        if num_crops_raw is None:
            num_crops = np.array([], dtype=np.int64)
        elif isinstance(num_crops_raw, torch.Tensor):
            num_crops = num_crops_raw.numpy() if num_crops_raw.numel() > 0 else np.array([], dtype=np.int64)
        elif isinstance(num_crops_raw, np.ndarray):
            num_crops = num_crops_raw
        elif isinstance(num_crops_raw, (list, tuple)):
            num_crops = np.array(num_crops_raw, dtype=np.int64)
        else:
            num_crops = np.array([], dtype=np.int64)

        num_images = len(num_crops)

        if num_images == 0:
            # NOTE: image_token_pooling is NOT included - it can't be batched
            return dict(
                pixel_values=MultiModalFieldConfig.batched("image"),
                image_grids=MultiModalFieldConfig.batched("image"),
                image_num_crops=MultiModalFieldConfig.batched("image"),
            )

        logger.info(f"  num_crops = {num_crops}, num_images = {num_images}")

        # Log image_token_pooling info for debugging
        image_token_pooling = hf_inputs.get("image_token_pooling", None)
        if image_token_pooling is not None:
            logger.info(f"  image_token_pooling shape: {image_token_pooling.shape}")

        # Cache image_token_pooling for retrieval in prefill_forward
        # We can't pass it through vLLM's batching because its first dimension
        # is total_pooled_tokens (not batch size)
        image_token_pooling = hf_inputs.get("image_token_pooling", None)
        if image_token_pooling is not None:
            _image_token_pooling_cache["last"] = image_token_pooling
            logger.info(f"  Cached image_token_pooling: shape={image_token_pooling.shape}")

        return dict(
            # Map pixel_values to "image" modality - indexed by num_crops
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_crops),
            # image_grids and image_num_crops are per-image (1 per image)
            image_grids=MultiModalFieldConfig.batched("image"),
            image_num_crops=MultiModalFieldConfig.batched("image"),
            # NOTE: image_token_pooling is NOT included here - it can't be batched
            # because its shape is (total_pooled_tokens, 4), not (num_images, ...)
            # We cache it above and retrieve in prefill_forward
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """
        Generate prompt updates for image tokens.

        For Molmo2, we need to replace the <|image|> placeholder with the
        actual image tokens that depend on the image grid size.
        """
        from loguru import logger

        logger.info(f"_get_prompt_updates called: out_mm_kwargs keys = {list(out_mm_kwargs.keys())}")
        if "image" in out_mm_kwargs:
            logger.info(f"  out_mm_kwargs['image'] len = {len(out_mm_kwargs['image'])}")
            for i, item in enumerate(out_mm_kwargs["image"]):
                logger.info(
                    f"  out_mm_kwargs['image'][{i}] keys = {list(item.keys()) if hasattr(item, 'keys') else type(item)}"
                )

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        # Get the <|image|> placeholder token ID
        # Molmo2 uses "<|image|>" as the placeholder
        image_placeholder = "<|image|>"
        image_placeholder_id = tokenizer.convert_tokens_to_ids(image_placeholder)

        def get_replacement_molmo2(item_idx: int) -> list:
            """Generate the replacement tokens for image at item_idx.

            IMPORTANT: We compute from image_grids only (not from cache) because:
            1. vLLM runs multimodal processor and model in different processes
            2. Module-level cache doesn't work across processes
            3. image_grids contains the correct dimensions for THIS specific image
            """
            import numpy as np

            out_item = out_mm_kwargs["image"][item_idx]
            image_grids = out_item.get("image_grids")

            if image_grids is None:
                logger.warning(f"  get_replacement_molmo2[{item_idx}]: no image_grids, using default 392 tokens")
                return [hf_processor.image_patch_id] * 392

            # Extract grid dimensions
            # image_grids format: [resized_h, resized_w, h, w]
            # These are the POOLED dimensions (after 2x2 pooling)
            if hasattr(image_grids, "data"):
                grid = image_grids.data
            else:
                grid = image_grids

            if isinstance(grid, torch.Tensor):
                grid = grid.numpy()

            grid = np.array(grid).flatten()
            logger.info(f"  get_replacement_molmo2[{item_idx}]: raw grid={grid}")

            if len(grid) >= 4:
                resized_h, resized_w, h, w = int(grid[0]), int(grid[1]), int(grid[2]), int(grid[3])
            else:
                logger.warning(f"  get_replacement_molmo2[{item_idx}]: grid too short ({len(grid)}), using defaults")
                resized_h, resized_w, h, w = 14, 14, 14, 14

            # Total tokens = global/low-res pooled + high-res pooled
            total_tokens = resized_h * resized_w + h * w
            logger.info(
                f"  get_replacement_molmo2[{item_idx}]: grid=[{resized_h},{resized_w},{h},{w}], total_tokens={total_tokens}"
            )
            return [hf_processor.image_patch_id] * total_tokens

        # Get the <|video|> placeholder token ID
        video_placeholder = "<|video|>"
        video_placeholder_id = tokenizer.convert_tokens_to_ids(video_placeholder)

        def get_replacement_video(item_idx: int) -> list:
            """Generate the replacement tokens for video (all frames combined).

            For video, we replace a single <|video|> token with all frame tokens.
            """
            import numpy as np

            # Collect tokens for ALL frames in the video
            all_tokens = []
            num_frames = len(out_mm_kwargs["image"])
            logger.info(f"  get_replacement_video: generating tokens for {num_frames} frames")

            for frame_idx in range(num_frames):
                out_item = out_mm_kwargs["image"][frame_idx]
                image_grids = out_item.get("image_grids")

                if image_grids is None:
                    # Default tokens for this frame
                    frame_tokens = 392
                else:
                    if hasattr(image_grids, "data"):
                        grid = image_grids.data
                    else:
                        grid = image_grids

                    if isinstance(grid, torch.Tensor):
                        grid = grid.numpy()

                    grid = np.array(grid).flatten()
                    if len(grid) >= 4:
                        resized_h, resized_w, h, w = int(grid[0]), int(grid[1]), int(grid[2]), int(grid[3])
                        frame_tokens = resized_h * resized_w + h * w
                    else:
                        frame_tokens = 392

                all_tokens.extend([hf_processor.image_patch_id] * frame_tokens)
                logger.info(f"  get_replacement_video: frame {frame_idx} = {frame_tokens} tokens")

            logger.info(f"  get_replacement_video: total = {len(all_tokens)} tokens")
            return all_tokens

        # Return prompt replacements for both image and video
        prompt_replacements = [
            PromptReplacement(
                modality="image",
                target=[image_placeholder_id],
                replacement=get_replacement_molmo2,
            )
        ]

        # Add video replacement if <|video|> is a valid token
        if video_placeholder_id != tokenizer.unk_token_id:
            prompt_replacements.append(
                PromptReplacement(
                    modality="image",  # Video uses image modality internally
                    target=[video_placeholder_id],
                    replacement=get_replacement_video,
                )
            )
            logger.info(f"  Added video replacement: placeholder_id={video_placeholder_id}")

        return prompt_replacements


class Molmo2DummyInputsBuilder(BaseDummyInputsBuilder["TT_MolmoProcessingInfo"]):
    """Dummy inputs builder for Molmo2 memory profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """
        Build the text input corresponding to mm_counts.

        For Molmo2, we need a simple prompt with image placeholders.
        """
        num_images = mm_counts.get("image", 0)
        # Use Molmo2's <|image|> placeholder for images
        image_placeholders = "<|image|>" * num_images
        return f"{image_placeholders}Describe this image."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: object,
    ) -> Mapping[str, object]:
        """
        Create dummy image data for memory profiling.
        """
        num_images = mm_counts.get("image", 0)
        if num_images == 0:
            return {}

        # Create a dummy PIL image for profiling
        from PIL import Image

        target_width, target_height = self.info.get_image_size_with_most_features()
        dummy_image = Image.new("RGB", (target_width, target_height), color=(128, 128, 128))

        return {"image": [dummy_image] * num_images}


class TT_MolmoProcessingInfo(MolmoProcessingInfo):
    """
    TT-specific processing info for Molmo2.

    Subclasses vLLM's MolmoProcessingInfo and overrides get_hf_processor()
    to return our Molmo2ProcessorWrapper which adds Molmo1-compatible attributes
    to the Molmo2 image processor.

    Following the same pattern as TT_Qwen2_5_VLProcessingInfo.
    """

    def get_hf_processor(self, **kwargs) -> Molmo2ProcessorWrapper:
        """Return Molmo2-specific processor wrapper with Molmo1-compatible attributes."""
        from loguru import logger

        logger.info(f"TT_MolmoProcessingInfo.get_hf_processor called with kwargs={list(kwargs.keys())}")
        processor = self.ctx.get_hf_processor(**kwargs)
        logger.info(f"  Wrapping processor type: {type(processor)}")
        wrapped = Molmo2ProcessorWrapper(processor)
        logger.info(f"  Returning Molmo2ProcessorWrapper")
        return wrapped

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1, "video": 1}


@MULTIMODAL_REGISTRY.register_processor(
    Molmo2MultiModalProcessor,
    info=TT_MolmoProcessingInfo,
    dummy_inputs=Molmo2DummyInputsBuilder,
)
class Molmo2ForConditionalGeneration(SupportsMultiModal):
    """
    vLLM-compatible wrapper for Molmo2-8B vision-language model.

    This class provides the interface expected by vLLM's TT plugin for:
    - Model initialization via `initialize_vllm_model`
    - KV cache allocation via `allocate_kv_cache`
    - Prefill and decode forward passes
    """

    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,  # Vision models typically don't support prefix caching
    }

    # Molmo2-specific constants
    MOLMO2_IMAGE_TOKEN_ID = 151938  # <im_patch> token

    def __init__(
        self,
        model: "Molmo2Model",
        model_args: Molmo2ModelArgs,
        mesh_device: ttnn.MeshDevice,
        tokenizer,
        generator: Molmo2Generator,
    ):
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.tokenizer = tokenizer
        self.generator = generator
        self.max_gen_len = model_args.max_seq_len - 1

    def __del__(self):
        """Release traces and cleanup resources on destruction."""
        try:
            if hasattr(self, "generator") and self.generator is not None:
                # Release prefill traces
                if hasattr(self.generator, "prefill_traces") and self.generator.prefill_traces:
                    for seq_len, (trace_id, _, _) in self.generator.prefill_traces.items():
                        try:
                            ttnn.release_trace(self.mesh_device, trace_id)
                            logger.debug(f"Released prefill trace for seq_len={seq_len}")
                        except Exception as e:
                            logger.warning(f"Failed to release prefill trace: {e}")
                    self.generator.prefill_traces.clear()

                # Release decode trace
                if hasattr(self.generator, "decode_trace_id") and self.generator.decode_trace_id is not None:
                    try:
                        ttnn.release_trace(self.mesh_device, self.generator.decode_trace_id)
                        logger.debug("Released decode trace")
                    except Exception as e:
                        logger.warning(f"Failed to release decode trace: {e}")
                    self.generator.decode_trace_id = None

                # Release vision trace
                if hasattr(self.generator, "vision_trace_id") and self.generator.vision_trace_id is not None:
                    try:
                        ttnn.release_trace(self.mesh_device, self.generator.vision_trace_id)
                        logger.debug("Released vision trace")
                    except Exception as e:
                        logger.warning(f"Failed to release vision trace: {e}")
                    self.generator.vision_trace_id = None
        except Exception as e:
            logger.warning(f"Error in Molmo2ForConditionalGeneration.__del__: {e}")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int,
        max_seq_len: int,
        tt_data_parallel: int = 1,
        optimizations: Optional[str] = None,
    ):
        """
        Initialize Molmo2-8B for vLLM inference.

        Args:
            hf_config: HuggingFace model config
            mesh_device: TT mesh device
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            tt_data_parallel: Data parallel factor
            optimizations: Optimization mode (not used for Molmo2)

        Returns:
            Initialized Molmo2ForConditionalGeneration instance
        """
        logger.info(f"Initializing Molmo2-8B for vLLM with max_batch_size={max_batch_size}, max_seq_len={max_seq_len}")

        # Set HF_MODEL env var for tt_transformers model_config.py compatibility
        # This is needed because vLLM spawns subprocesses that may not inherit the env var
        import os

        os.environ["HF_MODEL"] = "allenai/Molmo2-8B"

        # Load tokenizer
        tokenizer = load_processor()

        # Load model weights
        state_dict = load_model_weights()

        # Create model
        model = create_model(mesh_device, state_dict, num_layers=None)

        # Create model args
        model_args = Molmo2ModelArgs(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # Create generator
        generator = Molmo2Generator(
            mesh_device=mesh_device,
            model=model,
            tokenizer=tokenizer,
            num_layers=36,
            batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # Warmup: Initialize KV cache and capture traces during initialization
        # This ensures all inference requests have good performance
        logger.info("Running warmup to capture traces...")
        cls._warmup_traces(generator, mesh_device)

        logger.info("Molmo2-8B initialized successfully for vLLM")

        return cls(
            model=model,
            model_args=model_args,
            mesh_device=mesh_device,
            tokenizer=tokenizer,
            generator=generator,
        )

    @classmethod
    def _warmup_traces(cls, generator: Molmo2Generator, mesh_device: ttnn.MeshDevice):
        """
        Warmup traces during initialization for optimal inference performance.

        Captures vision, prefill, and decode traces so all inference requests
        execute traced code paths without compilation overhead.
        """
        import tempfile
        import time

        from PIL import Image as PILImage

        # 1. Initialize KV cache
        logger.info("Warmup: Initializing KV cache...")
        generator.init_kv_cache()

        # 2. Create dummy image for vision warmup (378x378 is Molmo2's image size)
        logger.info("Warmup: Creating dummy image for vision trace...")
        dummy_image = PILImage.new("RGB", (378, 378), color=(128, 128, 128))

        # Save to temp file since preprocess_image_molmo2 expects a file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dummy_image.save(tmp.name)
            image_inputs = preprocess_image_molmo2(tmp.name)

        # 3. Create dummy input_ids with image token
        # Molmo2 uses token ID 151938 for <im_patch>
        # Get actual number of visual tokens from preprocessed image
        num_visual_tokens = image_inputs["image_token_pooling"].shape[0]
        logger.info(f"Warmup: Image produces {num_visual_tokens} visual tokens")
        # Create a simple prompt: [BOS] + image tokens + text tokens
        dummy_input_ids = torch.tensor([[1] + [151938] * num_visual_tokens + [2]], dtype=torch.long)

        # 4. Run prefill with vision trace to capture vision + prefill traces
        logger.info("Warmup: Running prefill with vision trace (this captures traces)...")
        warmup_start = time.perf_counter()

        try:
            # Use unified trace for best performance (Vision + Prefill in single trace)
            logits, timing = generator.run_prefill(
                input_ids=dummy_input_ids,
                pixel_values=image_inputs["pixel_values"],
                pooled_patches_idx=image_inputs["image_token_pooling"].unsqueeze(0),
                use_trace=True,
                use_vision_trace=True,
                use_unified_trace=True,  # Combines Vision + Prefill for ~85ms total
            )

            # 5. Run decode warmup to capture decode trace
            logger.info("Warmup: Running decode trace capture...")
            # Get first token from prefill output
            mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
            logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0]
            next_token = torch.argmax(logits_torch[:, -1, :], dim=-1, keepdim=True)

            # Convert to TT tensor for decode
            token_id_ttnn = ttnn.from_torch(
                next_token,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

            # Run decode step with tracing to capture decode trace
            _, _ = generator.run_decode_step(
                token_id_ttnn=token_id_ttnn,
                use_trace=True,
                is_first=True,
            )

            warmup_time = (time.perf_counter() - warmup_start) * 1000
            logger.info(f"Warmup: All traces captured in {warmup_time:.2f}ms")
            has_unified = hasattr(generator, "unified_traces") and len(generator.unified_traces) > 0
            logger.info(f"  - Unified trace (Vision+Prefill): {'captured' if has_unified else 'not captured'}")
            logger.info(f"  - Decode trace: {'captured' if generator.decode_trace_id else 'not captured'}")

        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")
            logger.warning("Traces will be captured on first inference request")

    @property
    def cache_path(self):
        """Path for TT tensor caching."""
        return self.model_args.get_text_args().model_cache_path

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        images: Union[List[Image], List[List[Image]], None] = None,
        page_table: Optional[torch.Tensor] = None,
        kv_cache=None,
        prompt_lens=None,
        cross_page_table: Optional[torch.Tensor] = None,
        enable_trace: bool = False,  # Disabled for vLLM: trace causes hang after multiple runs
        sampling_params=None,
        pixel_values: Optional[List] = None,
        image_token_pooling: Optional[List] = None,
        image_grids: Optional[List] = None,
        image_num_crops: Optional[List] = None,
        **kwargs,
    ):
        """
        Run prefill forward pass for Molmo2.

        Args:
            tokens: Input token IDs [batch, seq_len]
            images: List of PIL images (for standalone demo, not vLLM)
            page_table: Page table for paged attention
            kv_cache: KV cache tensors
            prompt_lens: Length of each prompt in the batch
            cross_page_table: Cross-attention page table (not used for Molmo2)
            enable_trace: Whether to use tracing
            sampling_params: Sampling parameters (not used)
            pixel_values: Pre-processed pixel values from vLLM multimodal processor
            image_token_pooling: Token pooling indices from vLLM multimodal processor
            image_grids: Grid info from vLLM multimodal processor
            image_num_crops: Number of crops per image
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            Logits tensor
        """
        batch_size = tokens.shape[0]

        # Debug logging for multimodal kwargs
        logger.info(f"prefill_forward called: batch_size={batch_size}, tokens.shape={tokens.shape}")
        logger.info(f"  pixel_values type: {type(pixel_values)}, len: {len(pixel_values) if pixel_values else 0}")
        logger.info(
            f"  image_token_pooling type: {type(image_token_pooling)}, len: {len(image_token_pooling) if image_token_pooling else 0}"
        )
        logger.info(f"  image_grids type: {type(image_grids)}, len: {len(image_grids) if image_grids else 0}")
        logger.info(f"  other kwargs: {list(kwargs.keys())}")
        if "image_grid_thw" in kwargs:
            logger.info(f"  image_grid_thw: {kwargs['image_grid_thw']}")
        if pixel_values and len(pixel_values) > 0:
            pv0 = pixel_values[0]
            logger.info(f"  pixel_values[0] type: {type(pv0)}, shape: {pv0.shape if hasattr(pv0, 'shape') else 'N/A'}")

        # vLLM might pass image_grids as None but provide data in kwargs
        # Try to extract from kwargs if not provided directly
        if image_grids is None and "image_grid_thw" in kwargs:
            image_grid_thw = kwargs.get("image_grid_thw")
            logger.info(f"  Attempting to use image_grid_thw: {image_grid_thw}")
            # image_grid_thw might be nested lists - flatten to get actual grid data
            # But note: image_grid_thw format may differ from image_grids

        # Handle prompt_lens default
        if prompt_lens is None:
            prompt_lens = torch.tensor([tokens.shape[1]] * batch_size)

        # Check if we have pre-processed pixel_values from vLLM
        has_vllm_images = pixel_values is not None and len(pixel_values) > 0

        # Handle images default for standalone mode
        if images is None:
            images = [None] * batch_size

        # Collect last token logits for each user
        output_logits = []

        for user_id in range(batch_size):
            # Check for vLLM-style pre-processed images first
            if has_vllm_images and user_id < len(pixel_values) and pixel_values[user_id] is not None:
                # vLLM provides pre-processed pixel_values
                pv = pixel_values[user_id]
                # Handle nested list structure from vLLM
                if isinstance(pv, list) and len(pv) > 0:
                    pv = pv[0]  # Take first image's pixel values
                if isinstance(pv, torch.Tensor):
                    pv_tensor = pv
                else:
                    pv_tensor = torch.from_numpy(pv) if hasattr(pv, "__array__") else torch.tensor(pv)

                # Get image_token_pooling - compute from image_grids
                # NOTE: We always compute from image_grids (not cache) because:
                # 1. Cache doesn't work across vLLM's separate processes
                # 2. Cache can have stale data from warmup images
                pooling = None

                # Compute from image_grids - this is the reliable source
                if image_grids is not None and len(image_grids) > user_id:
                    grid_data = image_grids[user_id]
                    logger.info(f"  prefill_forward: image_grids[{user_id}] raw = {grid_data}, type={type(grid_data)}")
                    if isinstance(grid_data, list) and len(grid_data) > 0:
                        grid_data = grid_data[0]
                    if grid_data is not None:
                        import numpy as np

                        if hasattr(grid_data, "numpy"):
                            grid_arr = grid_data.numpy().flatten()
                        else:
                            grid_arr = np.array(grid_data).flatten()
                        logger.info(f"  prefill_forward: Computing pooling from grid_data={grid_arr}")
                        # Get num_crops from pixel_values shape
                        num_crops = pv_tensor.shape[0] if pv_tensor.dim() >= 2 else 1
                        logger.info(f"  prefill_forward: pv_tensor.shape={pv_tensor.shape}, num_crops={num_crops}")
                        computed_pooling = compute_image_token_pooling(grid_data, num_crops)
                        pooling = computed_pooling.unsqueeze(0)
                        logger.info(f"  prefill_forward: Computed pooling shape={pooling.shape}")

                if pooling is None:
                    logger.warning(f"  No image_token_pooling available - vision features may not work correctly")

                # Detect video input (multiple frames) vs image (crops)
                # Video: pv_tensor.shape[0] == 8 frames
                # Image: pv_tensor.shape[0] == 1-9 crops
                is_video = pv_tensor.shape[0] == 8 and pooling is not None and pooling.shape[1] > 1000

                # Run prefill with pre-processed image/video
                # NOTE: Vision trace disabled for vLLM mode because vLLM uses multi-crop
                # images with variable patch counts, but trace requires fixed tensor sizes.
                # The pre-allocated vision trace tensors assume single-crop (729 patches),
                # but vLLM may send 5+ crops (3645+ patches).
                logits_ttnn, prefill_timing = self.generator.run_prefill(
                    input_ids=tokens[user_id : user_id + 1, : prompt_lens[user_id]],
                    pixel_values=pv_tensor,
                    pooled_patches_idx=pooling,
                    use_trace=enable_trace,
                    use_vision_trace=False,  # Disabled for vLLM: variable multi-crop sizes
                    use_unified_trace=False,
                )
                original_seq_len = prefill_timing.get("original_seq_len", prompt_lens[user_id])
            else:
                # Standalone mode: check for PIL images
                image = images[user_id] if user_id < len(images) else None
                if isinstance(image, list):
                    assert len(image) == 1, "Only one image per prompt is supported"
                    image = image[0]

                if image is not None:
                    # Preprocess PIL image
                    image_inputs = preprocess_image_molmo2(image)

                    # Run prefill with image
                    logits_ttnn, prefill_timing = self.generator.run_prefill(
                        input_ids=tokens[user_id : user_id + 1, : prompt_lens[user_id]],
                        pixel_values=image_inputs["pixel_values"],
                        pooled_patches_idx=image_inputs["image_token_pooling"].unsqueeze(0),
                        use_trace=enable_trace,
                        use_vision_trace=True,  # Vision trace accuracy bug fixed
                        use_unified_trace=False,
                    )
                    original_seq_len = prefill_timing.get("original_seq_len", prompt_lens[user_id])
                else:
                    # Run prefill without image (text-only)
                    logits_ttnn, prefill_timing = self.generator.run_prefill(
                        input_ids=tokens[user_id : user_id + 1, : prompt_lens[user_id]],
                        pixel_values=None,
                        pooled_patches_idx=None,
                        use_trace=enable_trace,
                        use_vision_trace=False,
                    )
                    original_seq_len = prefill_timing.get("original_seq_len", prompt_lens[user_id])

            # Synchronize device before reading - trace execution is async
            ttnn.synchronize_device(self.mesh_device)

            # Convert ttnn tensor to torch tensor
            # With T3K (8 devices), mesh_composer concatenates along dim=0 giving [8, seq_len, vocab_size]
            # Take [0] to get single device output, squeeze to remove batch dim if present
            mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            logits_torch = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0].squeeze()

            # Deallocate logits_ttnn only when trace is disabled.
            # When trace IS enabled, logits_ttnn is a trace output tensor that's reused
            # across calls - deallocating would cause "Buffer must be allocated" errors.
            if not enable_trace:
                ttnn.deallocate(logits_ttnn)

            # Extract last token's logits - shape: [vocab_size]
            # Use original_seq_len - 1 to index correctly (accounting for padding)
            # logits_torch is [seq_len, vocab_size] or [vocab_size]
            if logits_torch.dim() == 2:
                last_token_logits = logits_torch[original_seq_len - 1, :]  # [vocab_size]
            else:
                last_token_logits = logits_torch  # Already [vocab_size]

            output_logits.append(last_token_logits.unsqueeze(0).unsqueeze(1))  # [1, 1, vocab_size]

        # Stack all users' logits: [batch_size, 1, vocab_size]
        # vLLM expects 3D: [batch, seq_len, vocab] and indexes with [:batch, -1, :]
        logits = torch.cat(output_logits, dim=0)

        # Ensure 3D shape [batch, seq, vocab] - vLLM requires this
        if logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(1)  # [vocab] -> [1, 1, vocab]
        elif logits.dim() == 2:
            logits = logits.unsqueeze(1)  # [batch, vocab] -> [batch, 1, vocab]

        logger.info(f"prefill_forward returning: shape={logits.shape}")
        return logits

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: Optional[torch.Tensor] = None,
        kv_cache=None,
        enable_trace: bool = False,  # Disabled for vLLM: trace causes hang after multiple runs
        read_from_device: bool = True,
        sampling_params=None,
    ):
        """
        Run decode forward pass for Molmo2.

        Args:
            tokens: Current token IDs [batch, 1]
            start_pos: Current position in sequence for each batch item
            page_table: Page table for paged attention
            kv_cache: KV cache tensors
            enable_trace: Whether to use tracing
            read_from_device: Whether to read output from device
            sampling_params: Sampling parameters (not used)

        Returns:
            Logits tensor
        """
        # Check if generator is properly initialized (has run prefill)
        # rot_mat_idxs is set during prefill - if not set, return dummy output
        if not hasattr(self.generator, "rot_mat_idxs") or self.generator.rot_mat_idxs is None:
            logger.warning("decode_forward called before prefill - returning dummy output")
            batch_size = tokens.shape[0]
            vocab_size = 152064  # Molmo2 vocab size
            return torch.zeros(batch_size, 1, vocab_size)

        # Convert tokens to device
        token_id_ttnn = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Embed tokens
        hidden_states = self.generator.model.text_model.embed_tokens(token_id_ttnn)

        # Deallocate input tensor after embedding
        ttnn.deallocate(token_id_ttnn)

        # Run forward_decode directly to get logits (not argmax like run_decode_step)
        logits_ttnn = self.generator.model.text_model.forward_decode(
            hidden_states=hidden_states,
            kv_caches=self.generator.kv_caches,
            current_pos=self.generator.current_pos,
            rot_mat_idxs=self.generator.rot_mat_idxs,
        )

        # Increment position counters
        ttnn.plus_one(self.generator.current_pos)
        ttnn.plus_one(self.generator.rot_mat_idxs)
        self.generator.decode_position += 1

        # Deallocate intermediate tensor
        ttnn.deallocate(hidden_states)

        # During trace capture, we cannot read from device
        # Return dummy output if read_from_device is False
        if not read_from_device:
            batch_size = tokens.shape[0]
            vocab_size = 152064  # Molmo2 vocab size
            return torch.zeros(batch_size, 1, vocab_size)

        # Synchronize device before reading
        ttnn.synchronize_device(self.mesh_device)

        # Convert logits to torch - mesh device returns tensor for each device
        # Use ConcatMeshToTensor and take first device since all devices have same logits
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        logits = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0]

        # Deallocate logits_ttnn after conversion (safe when trace is disabled)
        ttnn.deallocate(logits_ttnn)

        # Logits shape from text_model.forward_decode: [1, 1, padded_seq, vocab_size]
        # Slice to actual sequence length and reshape for vLLM
        if logits.dim() == 4:
            logits = logits[:, :, :1, :]  # [1, 1, 1, vocab] -> [1, 1, 1, vocab]
            logits = logits.squeeze(1)  # [1, 1, vocab]
        elif logits.dim() == 3:
            logits = logits[:, :1, :]  # [1, seq, vocab] -> [1, 1, vocab]
        elif logits.dim() == 2:
            logits = logits.unsqueeze(1)  # [batch, vocab] -> [batch, 1, vocab]
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(1)  # [vocab] -> [1, 1, vocab]

        logger.info(f"decode_forward returning: shape={logits.shape}")
        return logits

    def allocate_kv_cache(
        self,
        kv_cache_shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_layers: int,
    ) -> List[List[ttnn.Tensor]]:
        """
        Allocate KV cache for Molmo2 text model.

        Args:
            kv_cache_shape: Shape of KV cache tensors
            dtype: Data type for KV cache
            num_layers: Number of transformer layers

        Returns:
            List of KV cache tensor pairs per layer
        """
        return allocate_molmo2_kv_cache(
            kv_cache_shape=kv_cache_shape,
            dtype=dtype,
            num_layers=num_layers,
            mesh_device=self.mesh_device,
            tt_cache_path=self.cache_path,
        )

    def warmup_model_prefill(self, *args, **kwargs) -> None:
        """
        Warmup prefill path for Molmo2.

        Note: Tracing in prefill mode is not yet supported for Molmo2.
        """
        logger.warning("Warmup model prefill not implemented for Molmo2 - tracing in prefill mode is not supported")

    def warmup_model_decode(self, *args, **kwargs) -> None:
        """
        Warmup decode path for Molmo2.

        Note: Tracing in decode mode is not yet supported for Molmo2.
        """
        logger.warning("Warmup model decode not implemented for Molmo2 - tracing in decode mode is not supported")
