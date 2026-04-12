# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
vLLM integration for Molmo2-8B model.

This module provides the vLLM-compatible wrapper class for Molmo2-8B,
enabling integration with tt-inference-server via the vLLM plugin.
"""

import time
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch
from loguru import logger
from PIL.Image import Image
from tqdm import tqdm
from transformers import BatchFeature

import ttnn
from models.demos.molmo2.tt.hf_processor import hf_patches_to_images
from models.demos.molmo2.tt.prefill_attention_mask import build_molmo2_prefill_attention_bias
from models.demos.molmo2.tt.trace_capture_utils import trace_capture_run_begin, trace_capture_run_end
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.molmo import MolmoProcessingInfo, get_patches_grid_size, select_tiling
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import BaseMultiModalProcessor, PromptReplacement, PromptUpdate

# BaseDummyInputsBuilder location varies between vLLM versions
try:
    from vllm.multimodal.profiling import BaseDummyInputsBuilder
except ImportError:
    from vllm.multimodal.processing import BaseDummyInputsBuilder

# Note: Model registration is handled by tt-vllm-plugin/__init__.py
# The plugin registers TTMolmo2ForConditionalGeneration with the module path
# models.demos.molmo2.tt.generator_vllm:Molmo2ForConditionalGeneration

# Module-level cache for image_token_pooling
# This bypasses vLLM's multimodal batching which can't handle the irregular shape
_image_token_pooling_cache = {"last": None}

# Module-level cache for video timestamps
# The processor computes actual timestamps from video metadata but get_replacement_video
# runs in a different context (Molmo2MultiModalProcessor). Cache passes timestamps through.
_video_timestamps_cache = {"last": None}


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


from models.demos.molmo2.tt.model_config import Molmo2ModelArgs
from models.demos.molmo2.tt.model_loader import create_model, load_model_weights, load_processor
from models.demos.molmo2.tt.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    arange_for_pooling,
    get_padded_prefill_len,
    normalize_image,
    preprocess_image_molmo2,
    resize_image,
)


def allocate_molmo2_kv_cache(
    kv_cache_shape: Tuple[int, ...],
    dtype: torch.dtype,
    num_layers: int,
    mesh_device: ttnn.MeshDevice = None,
    tt_cache_path: str = None,
    submesh_devices: List[ttnn.MeshDevice] = None,
) -> List[List[ttnn.Tensor]]:
    """
    Allocate vLLM-style KV cache for Molmo2 text model.

    Supports both DP=1 (single mesh_device) and DP>1 (list of submesh_devices).

    Args:
        kv_cache_shape: Shape of each KV cache tensor (num_blocks, num_kv_heads, block_size, head_size)
        dtype: Data type for KV cache
        num_layers: Number of transformer layers
        mesh_device: Single TT mesh device (DP=1, backward compat)
        tt_cache_path: Path for caching TT tensors
        submesh_devices: List of per-replica submesh devices (DP>1)

    Returns:
        List of per-DP-replica KV caches: [dp_idx][layer][k_or_v]
    """
    if submesh_devices is None:
        submesh_devices = [mesh_device]

    all_kv_caches = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        kv_cache = []
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        for layer_num in tqdm(
            range(num_layers),
            desc=f"Allocating TT KV caches for Molmo2 (submesh {mesh_idx + 1}/{len(submesh_devices)})",
        ):
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,  # Molmo2 uses bfloat16 for KV cache
                    cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
                )
                for kv in ["k", "v"]
            ]
            kv_cache.append(kv_tt_i)
        all_kv_caches.append(kv_cache)

    return all_kv_caches  # [dp_idx][layer][k_or_v]


class Molmo2ProcessorWrapper:
    """
    Wrapper for Molmo2Processor that provides vLLM Molmo1-compatible interface.

    Molmo2 uses different attribute names than Molmo1. This wrapper maps
    Molmo2's API to the expected Molmo1 API that vLLM's MolmoProcessorWrapper expects.

    vLLM's MolmoProcessorWrapper accesses these as cached_properties on the wrapper:
    - max_crops, base_image_input_size, image_patch_size, overlap_margins
    - image_token_length_w, image_token_length_h, vocab
    """

    # Molmo2 native video sampling parameters (from video_processing_molmo2.py)
    _MAX_FPS = 2
    _MAX_FRAMES = 384

    def __init__(self, processor):
        self.processor = processor
        # Forward key attributes
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.video_processor = getattr(processor, "video_processor", None)

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

    def get_video_string(self, video_grid, timestamps):
        """Delegate to underlying processor's get_video_string method."""
        return self.processor.get_video_string(video_grid, timestamps)

    def _adaptive_sample_frames(self, frames: "np.ndarray", metadata: "dict | None") -> "tuple[np.ndarray, np.ndarray]":
        """
        Replicate Molmo2's native video sampling inside the wrapper.

        Molmo2's HF ``video_processing_molmo2.Molmo2VideoProcessor`` uses:
          - ``frame_sample_mode = "uniform_last_frame"``
          - ``max_fps = 2``
          - ``num_frames = 384``  (hard cap)

        HF has two sampling paths based on video duration:
        1. If duration > (num_frames - 1) / max_fps: UNIFORM sampling via linspace
        2. Otherwise: FPS-based sampling via arange at max_fps intervals

        Returns:
            (sampled_frames, timestamps) — timestamps in seconds.
        """
        import numpy as np

        total_decoded = frames.shape[0]  # frames vLLM actually decoded
        max_fps = self._MAX_FPS  # 2
        num_frames_cap = self._MAX_FRAMES  # 384

        # --- path 1: exact HF sampling when we have full metadata ---
        if metadata is not None and isinstance(metadata, dict) and "fps" in metadata and "total_num_frames" in metadata:
            fps = float(metadata["fps"])
            total_num_frames = int(metadata["total_num_frames"])
            duration = total_num_frames / fps

            # HF's branching logic for uniform_last_frame mode
            threshold = (num_frames_cap - 1) / max_fps  # e.g., 191.5s for cap=384, max_fps=2

            if duration > threshold:
                # UNIFORM sampling: linspace from 0 to duration
                sample_times = np.linspace(0, duration, num=num_frames_cap, endpoint=True)
            else:
                # FPS-based sampling: HF's sample_times formula
                # arange in time space, then append duration
                sample_times = np.arange(0.0, duration, 1.0 / max_fps)
                sample_times = np.concatenate([sample_times, [duration]])  # include last frame
                sample_times = sample_times[:num_frames_cap]  # cap at max frames

            # Convert times to frame indices
            frame_indices = np.clip(np.floor(sample_times * fps), 0, total_num_frames - 1).astype(int)

            # Map original indices → decoded frame positions
            # When num_frames=-1, vLLM decodes all frames so decoded_raw = [0,1,2,...]
            decoded_raw = np.array(metadata.get("frames_indices", list(range(total_decoded))))

            # Map target indices to decoded frames (handles case when vLLM subsampled)
            mapped = np.searchsorted(decoded_raw, frame_indices)
            mapped = np.clip(mapped, 0, total_decoded - 1)

            sampled = frames[mapped]
            return sampled, sample_times

        # --- path 2: heuristic when metadata is absent (vLLM stripped it) ---
        # vLLM default decodes 32 frames; with media_io_kwargs fps=2 it already
        # gives the right count.  Just cap at num_frames_cap and compute timestamps.
        if total_decoded > num_frames_cap:
            indices = np.linspace(0, total_decoded - 1, num_frames_cap, dtype=int)
            frames = frames[indices]
        timestamps = np.arange(frames.shape[0], dtype=float) / max_fps
        return frames, timestamps

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
        2. For VIDEOS: Follow demo flow - expand tokens at STRING level before tokenization
        3. For IMAGES: Keep placeholder for vLLM's _get_prompt_updates to handle

        For videos, we use the demo's approach:
        - Preprocess frames into combined tensor [n_frames, 3, H, W]
        - Replace <|video|> with demo-style token string including frame markers & timestamps
        - Tokenize the expanded string
        """
        import numpy as np
        from loguru import logger

        logger.info(
            f"Molmo2ProcessorWrapper.__call__ invoked: text={text[:50] if text else None}..., images={type(images)}, videos={type(videos)}"
        )

        # Handle video input using demo flow (string-level token expansion)
        self._is_video_input = False
        self._video_data = None
        if videos is not None and images is None:
            logger.info(f"  Processing video input using DEMO FLOW: type={type(videos)}")
            self._is_video_input = True

            if isinstance(videos, list) and len(videos) > 0:
                video_item = videos[0]  # Take first video

                # vLLM may pass (frames, metadata) tuple when video_needs_metadata=True,
                # or a plain ndarray when metadata is stripped (default).
                if isinstance(video_item, tuple) and len(video_item) == 2:
                    video_frames, vllm_metadata = video_item
                else:
                    video_frames, vllm_metadata = video_item, None

                if isinstance(video_frames, np.ndarray):
                    # video_frames shape: [raw_frames, H, W, C] — may be many frames
                    logger.info(f"    Video frames shape: {video_frames.shape}, metadata={vllm_metadata is not None}")
                    if vllm_metadata:
                        logger.info(
                            f"    vLLM metadata: fps={vllm_metadata.get('fps')}, total_frames={vllm_metadata.get('total_num_frames')}, decoded_indices={vllm_metadata.get('frames_indices', [])[:10]}..."
                        )

                    # Use _adaptive_sample_frames to match HF's frame sampling exactly
                    # This maps vLLM's decoded frame indices to HF's expected samples
                    sampled_frames, timestamps = self._adaptive_sample_frames(video_frames, vllm_metadata)
                    n_frames = sampled_frames.shape[0]
                    logger.info(f"    After adaptive sampling: {n_frames} frames, timestamps={timestamps[:5]}...")

                    # Use HF video processor's _preprocess method directly on frames
                    # This avoids temp video compression and ensures exact parity with demo
                    from transformers.image_utils import SizeDict

                    vp = self.video_processor

                    # _preprocess expects list of numpy arrays
                    frames_list = [sampled_frames]  # Wrap as single video (list of frames is [N,H,W,C])

                    # Call _preprocess to get pre-embedded patches
                    # Must provide size as SizeDict
                    hf_result = vp._preprocess(
                        frames_list,
                        size=SizeDict(height=378, width=378),
                        return_tensors="pt",
                    )

                    # Extract results - pixel_values_videos is in HF patch format [n_frames, 729, 588]
                    # Convert to image format [n_frames, 3, H, W] to match demo path
                    pixel_values_patches = hf_result["pixel_values_videos"]
                    pixel_values = hf_patches_to_images(pixel_values_patches.numpy())  # [n_frames, 3, 378, 378]
                    video_grids = hf_result["video_grids"]  # tensor [[n_frames, pooled_h, pooled_w]]
                    n_frames = int(video_grids[0, 0].item())
                    pooled_h_out = int(video_grids[0, 1].item())
                    pooled_w_out = int(video_grids[0, 2].item())
                    k_pool = 9  # 3x3 pooling for video

                    # Get pooling indices: [n_tokens, k_pool] where n_tokens = n_frames * pooled_h * pooled_w
                    pooling_idx = hf_result["video_token_pooling"]  # [n_tokens, k_pool]
                    n_out = pooled_h_out * pooled_w_out
                    image_token_pooling = pooling_idx.view(n_frames, n_out, k_pool)

                    logger.info(
                        f"    HF _preprocess output (patches): {pixel_values_patches.shape}, converted to images: {pixel_values.shape}, pooling={image_token_pooling.shape}"
                    )
                    logger.info(
                        f"    n_frames={n_frames}, pooled_h={pooled_h_out}, pooled_w={pooled_w_out}, k_pool={k_pool}"
                    )

                    # Store video data for later use
                    self._video_data = {
                        "pixel_values": pixel_values,
                        "image_token_pooling": image_token_pooling,
                        "n_frames": n_frames,
                        "timestamps": timestamps,
                        "pooled_h": pooled_h_out,
                        "pooled_w": pooled_w_out,
                    }
                    # Cache timestamps for get_replacement_video (runs in different context)
                    _video_timestamps_cache["last"] = timestamps
                    logger.info(f"    Cached video timestamps: {timestamps[:5]}... (len={len(timestamps)})")

        # Process images using raw pixels (same as video frame processing)
        # Our TTNN model expects [n_crops, 3, H, W] raw pixels, not HF's pre-embedded format
        if images is not None and not self._is_video_input:
            logger.info(f"  Processing images using RAW PIXEL format: type={type(images)}")

            base_size = 378
            patch_size = 14
            pool_h, pool_w = 2, 2
            crop_patches = base_size // patch_size  # 27

            # Handle single image or list of images
            if not isinstance(images, list):
                images_list = [images]
            else:
                images_list = images

            all_crops = []
            all_pooling_idx = []
            patches_per_crop = crop_patches * crop_patches  # 729

            # Pre-compute pooling indices template
            resize_idx_template = np.arange(patches_per_crop).reshape(crop_patches, crop_patches)
            resize_idx_template = arange_for_pooling(resize_idx_template, pool_h, pool_w)
            pooled_h, pooled_w = resize_idx_template.shape[0], resize_idx_template.shape[1]
            resize_idx_flat = resize_idx_template.reshape(-1, pool_h * pool_w)

            crop_idx = 0
            for img in images_list:
                # Convert PIL Image to numpy array
                if hasattr(img, "convert"):
                    img_np = np.array(img.convert("RGB"))
                elif isinstance(img, np.ndarray):
                    img_np = img
                else:
                    img_np = np.array(img)

                # Resize and normalize (same as video frame processing)
                img_resized = resize_image(img_np, [base_size, base_size])
                img_normalized = normalize_image(img_resized, IMAGENET_MEAN, IMAGENET_STD)
                # [H, W, C] -> [C, H, W]
                crop = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
                all_crops.append(crop)

                # Pooling indices with offset for this crop
                offset = crop_idx * patches_per_crop
                crop_idx_with_offset = np.where(
                    resize_idx_flat >= 0,
                    resize_idx_flat + offset,
                    resize_idx_flat,
                )
                all_pooling_idx.append(crop_idx_with_offset)
                crop_idx += 1

            # Stack into combined tensors
            pixel_values = torch.stack(all_crops, dim=0)  # [n_crops, 3, H, W]
            image_token_pooling = torch.from_numpy(np.concatenate(all_pooling_idx, axis=0)).long()

            num_crops = len(all_crops)
            logger.info(
                f"    Processed {num_crops} image(s): pixel_values={pixel_values.shape}, pooling={image_token_pooling.shape}"
            )

            image_outputs = {
                "pixel_values": pixel_values.numpy(),  # [n_crops, 3, H, W] - raw pixels for TTNN
                "image_token_pooling": image_token_pooling.numpy(),  # [total_tokens, K_pool]
                "image_grids": np.array([[pooled_h, pooled_w, 0, 0]] * len(images_list)),
                "image_num_crops": np.array([1] * len(images_list)),  # 1 crop per image for simple mode
            }
        elif self._video_data is not None:
            # For video: use video-specific field names
            # vLLM will handle token replacement via _get_prompt_updates
            n_frames = self._video_data["n_frames"]
            pooled_h = self._video_data["pooled_h"]
            pooled_w = self._video_data["pooled_w"]
            image_outputs = {
                # Video-specific fields for vLLM multimodal pipeline
                "pixel_values_videos": self._video_data["pixel_values"].numpy(),  # [n_frames, 3, H, W]
                "video_grid_thw": np.array([[n_frames, pooled_h, pooled_w]]),  # [1, 3] for 1 video
                "video_token_pooling": self._video_data["image_token_pooling"].numpy(),  # [n_frames, N_out, K_pool]
            }
            logger.info(
                f"    Video outputs: pixel_values_videos={image_outputs['pixel_values_videos'].shape}, "
                f"video_grid_thw={image_outputs['video_grid_thw']}"
            )
        else:
            image_outputs = {}

        # Handle text tokenization
        # NOTE: For VIDEO, do NOT expand video tokens here. vLLM's cached path ignores our
        # input_ids and tokenizes text separately. Let vLLM's PromptReplacement handle
        # video token expansion to avoid double expansion.
        if text is not None:
            if self._is_video_input and self._video_data is not None:
                # VIDEO FLOW: Keep <|video|> placeholder, let vLLM's PromptReplacement expand it.
                # The demo uses preprocess_video() directly which expands tokens itself.
                # vLLM's wrapper here should NOT expand tokens since vLLM handles it.
                logger.info(f"    VIDEO FLOW (vLLM): Keeping <|video|> placeholder for vLLM to expand")

                # Ensure chat template is applied with <|video|> placeholder preserved
                has_chat_template = "<|im_start|>" in text

                if not has_chat_template:
                    # Apply chat template while preserving <|video|> placeholder
                    if "<|video|>" in text:
                        # The placeholder is in text, wrap in chat template
                        messages = [{"role": "user", "content": text}]
                        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    elif "<|image|>" in text:
                        # Convert <|image|> to <|video|> for vLLM's video PromptReplacement
                        text = text.replace("<|image|>", "<|video|>", 1)
                        messages = [{"role": "user", "content": text}]
                        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        # No placeholder - add <|video|> at start of user content
                        messages = [{"role": "user", "content": f"<|video|>{text}"}]
                        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    logger.info(f"    VIDEO FLOW (vLLM): Applied chat template with <|video|> placeholder")

                text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
                logger.info(
                    f"    VIDEO FLOW (vLLM): Tokenized with PLACEHOLDER: "
                    f"input_ids len={text_inputs['input_ids'].shape if hasattr(text_inputs.get('input_ids', None), 'shape') else 'N/A'}"
                )
            else:
                # IMAGE FLOW: Apply chat template in processor to ensure <|image|> placeholder
                # is at the correct position INSIDE the user message, not at the start
                num_images = 0
                if images is not None:
                    num_images = len(images) if isinstance(images, list) else 1

                existing_image = text.count("<|image|>")
                existing_video = text.count("<|video|>")
                logger.info(f"  IMAGE FLOW: text (first 100 chars) = {repr(text[:100])}")
                logger.info(
                    f"  IMAGE FLOW: num_images={num_images}, existing_image={existing_image}, existing_video={existing_video}"
                )

                # Check if chat template is already applied
                has_chat_template = "<|im_start|>" in text

                # If no chat template, apply it manually to ensure <|image|> is inside user message
                if num_images > 0 and not has_chat_template:
                    # Strip leading <|image|> placeholder if present (vLLM adds it at wrong position)
                    user_content = text
                    if user_content.startswith("<|image|>"):
                        user_content = user_content[len("<|image|>") :].lstrip()

                    # Build proper prompt with <|image|> inside user message
                    image_placeholders = "<|image|>" * num_images
                    user_content_with_image = f"{image_placeholders}{user_content}"

                    # Apply chat template
                    messages = [{"role": "user", "content": user_content_with_image}]
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    logger.info(f"  IMAGE FLOW: Applied chat template with image inside user message")
                elif num_images > 0 and existing_image == 0 and existing_video == 0:
                    # Has chat template but no image placeholder - add it
                    image_placeholders = "<|image|>" * num_images
                    if "<|im_start|>user\n" in text:
                        text = text.replace("<|im_start|>user\n", f"<|im_start|>user\n{image_placeholders}", 1)
                        logger.info(f"  IMAGE FLOW: Inserted {num_images} <|image|> placeholders inside user message")

                logger.info(f"  IMAGE FLOW: final text (first 200 chars) = {repr(text[:200])}")
                text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
                if "input_ids" in text_inputs:
                    ids = text_inputs["input_ids"]
                    if hasattr(ids, "tolist"):
                        ids_list = ids.flatten().tolist()
                    else:
                        ids_list = list(ids[0]) if len(ids) > 0 else []
                    logger.info(f"  IMAGE FLOW: tokenized input_ids (first 20) = {ids_list[:20]}")
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

    def _get_data_parser(self) -> MultiModalDataParser:
        # video_needs_metadata=True keeps the (frames, metadata) tuple that
        # vLLM's OpenCV backend attaches (fps, total_num_frames, frames_indices).
        # Molmo2ProcessorWrapper._adaptive_sample_frames uses this to call
        # the real HF sample_frames() for exact Molmo2-native sampling.
        return MultiModalDataParser(video_needs_metadata=True)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,  # MultiModalDataItems
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """
        Check if HF processor already applied token expansion.

        For IMAGES: Returns False - vLLM handles token replacement via _get_prompt_updates
        For VIDEOS: Returns True - processor already expanded video tokens directly
                    to avoid vLLM's buggy string fallback that corrupts chat template tokens
        """
        from loguru import logger

        # Check if this is a video request by looking for video in mm_items
        try:
            mm_counts = mm_items.get_all_counts()
            has_video = mm_counts.get("video", 0) > 0
            if has_video:
                # VIDEO: Return False so vLLM applies PromptReplacement.
                # Our processor keeps <|video|> placeholder and vLLM expands it.
                logger.info("  _hf_processor_applies_updates: False (VIDEO - vLLM will expand placeholder)")
                return False
        except Exception as e:
            logger.warning(f"  _hf_processor_applies_updates: error checking mm_items: {e}")

        logger.info("  _hf_processor_applies_updates: False (IMAGE - vLLM will apply prompt updates)")
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """
        Configure multimodal fields for Molmo2.

        Molmo2 returns:
        - pixel_values: image crops or video frames
        - image_token_pooling: pooling indices
        - image_grids: grid information
        - image_num_crops: number of crops per image

        For VIDEO: Tokens are already expanded at string level (demo flow).
                   pixel_values is [n_frames, 3, H, W] - treat as single "video" item.
        For IMAGE: pixel_values is per-image crops, use flat_from_sizes.
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

        # Check if this is video data by looking for video-specific fields in hf_inputs
        # Video uses pixel_values_videos or video_grid_thw fields
        is_video = "pixel_values_videos" in hf_inputs or "video_grid_thw" in hf_inputs

        # IMPORTANT: Handle video FIRST before image logic
        # Video doesn't have image_num_crops, so we need to check before early return
        if is_video:
            # VIDEO: Entire video is ONE item (not one item per frame!)
            # pixel_values_videos: [n_frames, 3, H, W] -> 1 video item
            # video_grid_thw: [[n_frames, pooled_h, pooled_w]] -> 1 video item
            # video_token_pooling: [n_frames, N_out, K_pool] -> 1 video item
            #
            # Use shared("video", 1) to treat entire tensor as single item
            # batched("video") would incorrectly split by first dimension (frames)
            pixel_values_videos = hf_inputs.get("pixel_values_videos")
            n_frames = pixel_values_videos.shape[0] if pixel_values_videos is not None else 8

            # Cache video_token_pooling for prefill_forward
            video_token_pooling = hf_inputs.get("video_token_pooling", None)
            if video_token_pooling is not None:
                _image_token_pooling_cache["last"] = video_token_pooling
                logger.info(f"  Cached video_token_pooling: shape={video_token_pooling.shape}")

            logger.info(f"  VIDEO mode: shared config (n_frames={n_frames}) - entire video is 1 item")
            return dict(
                pixel_values_videos=MultiModalFieldConfig.shared("video", 1),
                video_grid_thw=MultiModalFieldConfig.shared("video", 1),
                video_token_pooling=MultiModalFieldConfig.shared("video", 1),
            )

        # Molmo2 uses image_num_crops instead of num_crops
        num_crops_raw = hf_inputs.get("image_num_crops", None)

        # Convert to numpy array
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
            return dict(
                pixel_values=MultiModalFieldConfig.batched("image"),
                image_grids=MultiModalFieldConfig.batched("image"),
                image_num_crops=MultiModalFieldConfig.batched("image"),
            )

        logger.info(f"  num_crops = {num_crops}, num_images = {num_images}")

        # Cache image_token_pooling for prefill_forward (can't be batched by vLLM)
        image_token_pooling = hf_inputs.get("image_token_pooling", None)
        if image_token_pooling is not None:
            _image_token_pooling_cache["last"] = image_token_pooling
            logger.info(f"  Cached image_token_pooling: shape={image_token_pooling.shape}")

        # IMAGE: Use flat_from_sizes for multi-crop images
        logger.info(f"  IMAGE mode: flat_from_sizes config (num_crops={num_crops})")
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_crops),
            image_grids=MultiModalFieldConfig.batched("image"),
            image_num_crops=MultiModalFieldConfig.batched("image"),
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
        # Log mm_items counts (critical for understanding modality mismatch)
        try:
            mm_counts = mm_items.get_all_counts()
            logger.info(f"  mm_items.get_all_counts() = {dict(mm_counts)}")
        except Exception as e:
            logger.error(f"  Error getting mm_items counts: {e}")
        if "image" in out_mm_kwargs:
            logger.info(f"  out_mm_kwargs['image'] len = {len(out_mm_kwargs['image'])}")
            for i, item in enumerate(out_mm_kwargs["image"]):
                logger.info(
                    f"  out_mm_kwargs['image'][{i}] keys = {list(item.keys()) if hasattr(item, 'keys') else type(item)}"
                )

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        # Get placeholder token IDs
        # Molmo2 uses "<|image|>" for images and "<|video|>" for videos
        image_placeholder_id = tokenizer.convert_tokens_to_ids("<|image|>")
        video_placeholder_id = tokenizer.convert_tokens_to_ids("<|video|>")

        def get_replacement_molmo2(item_idx: int) -> list:
            """Generate the replacement tokens for image at item_idx.

            IMPORTANT: We compute from image_grids only (not from cache) because:
            1. vLLM runs multimodal processor and model in different processes
            2. Module-level cache doesn't work across processes
            3. image_grids contains the correct dimensions for THIS specific image
            """
            import numpy as np

            try:
                num_images = len(out_mm_kwargs["image"])
                logger.info(f"  get_replacement_molmo2[{item_idx}]: num_images={num_images}")
                if item_idx >= num_images:
                    logger.error(
                        f"  get_replacement_molmo2[{item_idx}]: item_idx out of range (num_images={num_images})"
                    )
                    return [hf_processor.image_patch_id] * 392  # Default fallback
                out_item = out_mm_kwargs["image"][item_idx]
                image_grids = out_item.get("image_grids")
            except Exception as e:
                logger.error(f"  get_replacement_molmo2[{item_idx}]: Error accessing out_mm_kwargs: {e}")
                return [hf_processor.image_patch_id] * 392

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

        # Return prompt replacements for both image and video modalities
        # vLLM replaces <|image|> placeholder tokens with actual image patch tokens
        prompt_replacements = [
            PromptReplacement(
                modality="image",
                target=[image_placeholder_id],
                replacement=get_replacement_molmo2,
            )
        ]

        # NOTE: Video modality replacement is DISABLED because the processor now expands
        # video tokens directly (no <|video|> placeholder). The processor outputs fully
        # expanded tokens with chat template + video frames + prompt.
        # Keeping this code commented for reference.
        mm_counts = mm_items.get_all_counts()
        if mm_counts.get("video", 0) > 0:
            logger.info(
                f"  Video detected (count={mm_counts['video']}) - SKIPPING PromptReplacement (processor expands tokens)"
            )

            # Try to get video grid info from mm_items directly
            # mm_items contains the raw video data before batching
            video_grid_info = None
            try:
                video_items = mm_items.get_items("video")
                logger.info(f"  video_items from mm_items: {len(video_items)} items")
                if len(video_items) > 0:
                    first_video = video_items[0]
                    logger.info(f"  first_video type: {type(first_video)}")
                    # Try to extract grid info
                    if hasattr(first_video, "video_grid_thw"):
                        video_grid_info = first_video.video_grid_thw
                    elif hasattr(first_video, "__getitem__"):
                        video_grid_info = first_video.get("video_grid_thw")
                    logger.info(f"  video_grid_info: {video_grid_info}")
            except Exception as e:
                logger.warning(f"  Could not get video items from mm_items: {e}")

            def get_replacement_video(item_idx: int) -> list:
                """Generate replacement tokens for video at item_idx.

                CRITICAL: Must use HF processor's get_video_string() to ensure EXACT token match
                with what the processor expansion produces. Any mismatch causes _find_mm_placeholders
                to fail when is_update_applied=True.
                """
                import numpy as np

                nonlocal video_grid_info

                def generate_video_tokens(n_frames: int, pooled_h: int, pooled_w: int) -> list:
                    """Generate video tokens using HF processor's get_video_string method.

                    Uses the same method as processor expansion to ensure exact token match.
                    CRITICAL: Uses cached timestamps from _adaptive_sample_frames for parity with demo.
                    """
                    # Use cached timestamps from processor (matches demo's HF-derived timestamps)
                    cached_timestamps = _video_timestamps_cache.get("last")
                    if cached_timestamps is not None and len(cached_timestamps) == n_frames:
                        timestamps = cached_timestamps
                        logger.info(f"  get_replacement_video: Using CACHED timestamps: {timestamps[:5]}...")
                    else:
                        # Fallback: generate evenly spaced timestamps at 0.5 sec intervals
                        timestamps = np.arange(n_frames, dtype=float) * 0.5
                        logger.warning(
                            f"  get_replacement_video: No cached timestamps (cached={cached_timestamps is not None}, "
                            f"len={len(cached_timestamps) if cached_timestamps is not None else 0}, need={n_frames}), "
                            f"using fallback: {timestamps[:5]}..."
                        )

                    # Use HF processor's get_video_string to ensure exact match
                    video_grid = [n_frames, pooled_h, pooled_w]
                    video_string = hf_processor.get_video_string(video_grid, timestamps)

                    # Tokenize the video string to get proper token IDs
                    token_ids = tokenizer.encode(video_string, add_special_tokens=False)

                    logger.info(
                        f"  get_replacement_video[{item_idx}]: Generated {len(token_ids)} tokens "
                        f"(n_frames={n_frames}, pooled={pooled_h}x{pooled_w}, using HF get_video_string)"
                    )
                    return token_ids

                # Try to get video data from out_mm_kwargs first
                video_data = out_mm_kwargs.get("video", [])
                logger.info(f"  get_replacement_video[{item_idx}]: video_data len={len(video_data)}")

                if len(video_data) > 0 and item_idx < len(video_data):
                    video_item = video_data[item_idx]
                    try:
                        video_grid = None
                        if hasattr(video_item, "get_data"):
                            item_data = video_item.get_data()
                            video_grid = item_data.get("video_grid_thw")
                        elif hasattr(video_item, "__getitem__"):
                            video_grid_elem = video_item.get("video_grid_thw")
                            if video_grid_elem is not None:
                                video_grid = getattr(video_grid_elem, "data", video_grid_elem)

                        if video_grid is not None:
                            logger.info(f"  get_replacement_video[{item_idx}]: video_grid={video_grid}")
                            if hasattr(video_grid, "tolist"):
                                grid = video_grid.tolist()
                            else:
                                grid = list(video_grid)
                            if len(grid) > 0 and isinstance(grid[0], (list, tuple)):
                                grid = grid[0]
                            if len(grid) >= 3:
                                n_frames, pooled_h, pooled_w = int(grid[0]), int(grid[1]), int(grid[2])
                                return generate_video_tokens(n_frames, pooled_h, pooled_w)
                    except Exception as e:
                        logger.warning(f"  get_replacement_video[{item_idx}]: Error accessing video_item: {e}")

                # Fallback: use video_grid_info captured earlier
                if video_grid_info is not None:
                    try:
                        if hasattr(video_grid_info, "tolist"):
                            grid = video_grid_info.tolist()
                        else:
                            grid = list(video_grid_info)
                        if isinstance(grid[0], (list, tuple)):
                            grid = grid[0]
                        if len(grid) >= 3:
                            n_frames, pooled_h, pooled_w = int(grid[0]), int(grid[1]), int(grid[2])
                            return generate_video_tokens(n_frames, pooled_h, pooled_w)
                    except Exception as e:
                        logger.warning(f"  Error parsing video_grid_info: {e}")

                # Default: 8 frames × 14×14 pooled
                logger.info(f"  get_replacement_video[{item_idx}]: using default 8 frames × 14×14")
                return generate_video_tokens(8, 14, 14)

            # Register video PromptReplacement - vLLM needs this for tracking even when tokens
            # are pre-expanded. When _hf_processor_applies_updates returns True, vLLM will
            # use this info for merging but NOT re-apply the replacement.
            prompt_replacements.append(
                PromptReplacement(
                    modality="video",
                    target=[video_placeholder_id],
                    replacement=get_replacement_video,
                )
            )
            logger.info(f"  Registered video PromptReplacement: target=[{video_placeholder_id}]")

        return prompt_replacements


class Molmo2DummyInputsBuilder(BaseDummyInputsBuilder["TT_MolmoProcessingInfo"]):
    """Dummy inputs builder for Molmo2 memory profiling."""

    def __init__(self, info: "TT_MolmoProcessingInfo") -> None:
        super().__init__(info)

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
        mm_options: Optional[Mapping[str, object]] = None,
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
        model: "Molmo2Model" = None,
        model_args: Molmo2ModelArgs = None,
        mesh_device: ttnn.MeshDevice = None,
        tokenizer=None,
        # DP-mode parameters (used when data_parallel > 1)
        models: List["Molmo2Model"] = None,
        model_args_list: List[Molmo2ModelArgs] = None,
        data_parallel: int = 1,
        batch_per_dp: int = None,
    ):
        # Normalize single-model (backward compat) vs list (DP mode)
        if models is None:
            assert model is not None, "Either 'model' or 'models' must be provided"
            models = [model]
        if model_args_list is None:
            assert model_args is not None, "Either 'model_args' or 'model_args_list' must be provided"
            model_args_list = [model_args]

        assert (
            len(models) == len(model_args_list) == data_parallel
        ), f"models/model_args_list length {len(models)} must equal data_parallel={data_parallel}"

        self.data_parallel = data_parallel
        self.models = models
        self.model_args_list = model_args_list
        # _full_mesh_device: original full mesh (Galaxy 32-chip or T3K 8-chip)
        self._full_mesh_device = mesh_device if mesh_device is not None else models[0].mesh_device
        self.tokenizer = tokenizer
        self.max_gen_len = model_args_list[0].max_seq_len - 1
        self.batch_per_dp = batch_per_dp if batch_per_dp is not None else model_args_list[0].max_batch_size

        # Per-replica state lists (indexed by dp_idx)
        self.kv_caches_per_dp = [None] * data_parallel
        self.current_pos_per_dp = [None] * data_parallel
        self.rot_mat_idxs_per_dp = [None] * data_parallel
        self.decode_position_per_dp = [0] * data_parallel
        self.prefill_traces_per_dp = [{} for _ in range(data_parallel)]
        self.decode_trace_id_per_dp = [None] * data_parallel
        self.decode_trace_tensors_per_dp = [None] * data_parallel
        self.decode_trace_output_per_dp = [None] * data_parallel
        self.vision_trace_id_per_dp = [None] * data_parallel
        self.vision_trace_tensors_per_dp = [None] * data_parallel
        self.vision_trace_outputs_per_dp = [None] * data_parallel
        self.decode_trace_needs_reset_per_dp = [True] * data_parallel
        self.prev_page_table_per_dp = [None] * data_parallel
        self._prefill_compiled_buckets_per_dp = [set() for _ in range(data_parallel)]
        self.mesh_mapper_per_dp = [ttnn.ReplicateTensorToMesh(m.mesh_device) for m in models]

        # sentinel so _set_active_replica skips save on first call
        self._active_dp_idx = -1

        # Initialize active replica to 0 (sets self.model, self.mesh_device, etc.)
        self._set_active_replica(0)

    def __del__(self):
        """Release traces and cleanup resources on destruction."""
        try:
            # Save current active replica state before iterating
            if hasattr(self, "_active_dp_idx") and self._active_dp_idx >= 0:
                self._save_active_replica(self._active_dp_idx)

            for dp_idx in range(getattr(self, "data_parallel", 1)):
                mesh_dev = None
                if hasattr(self, "models") and dp_idx < len(self.models):
                    mesh_dev = self.models[dp_idx].mesh_device
                elif hasattr(self, "mesh_device"):
                    mesh_dev = self.mesh_device

                # Release prefill traces for this replica
                prefill_traces = {}
                if hasattr(self, "prefill_traces_per_dp") and dp_idx < len(self.prefill_traces_per_dp):
                    prefill_traces = self.prefill_traces_per_dp[dp_idx] or {}
                if prefill_traces and mesh_dev:
                    for seq_len, (trace_id, _, _) in prefill_traces.items():
                        try:
                            ttnn.release_trace(mesh_dev, trace_id)
                        except Exception as e:
                            logger.warning(f"Failed to release prefill trace dp={dp_idx} seq={seq_len}: {e}")
                    prefill_traces.clear()

                # Release decode trace for this replica
                decode_trace_id = None
                if hasattr(self, "decode_trace_id_per_dp") and dp_idx < len(self.decode_trace_id_per_dp):
                    decode_trace_id = self.decode_trace_id_per_dp[dp_idx]
                if decode_trace_id is not None and mesh_dev:
                    try:
                        ttnn.release_trace(mesh_dev, decode_trace_id)
                    except Exception as e:
                        logger.warning(f"Failed to release decode trace dp={dp_idx}: {e}")

                # Release vision trace for this replica
                vision_trace_id = None
                if hasattr(self, "vision_trace_id_per_dp") and dp_idx < len(self.vision_trace_id_per_dp):
                    vision_trace_id = self.vision_trace_id_per_dp[dp_idx]
                if vision_trace_id is not None and mesh_dev:
                    try:
                        ttnn.release_trace(mesh_dev, vision_trace_id)
                    except Exception as e:
                        logger.warning(f"Failed to release vision trace dp={dp_idx}: {e}")
        except Exception as e:
            logger.warning(f"Error in Molmo2ForConditionalGeneration.__del__: {e}")

    def _set_active_replica(self, dp_idx: int) -> None:
        """
        Switch instance state to the specified DP replica.

        Saves the current replica's state first (if any), then loads the target
        replica's state into the instance attributes that the rest of the code uses.
        For DP=1 this is called once (dp_idx=0) and is effectively a no-op after init.
        """
        if self._active_dp_idx >= 0:
            self._save_active_replica(self._active_dp_idx)

        self._active_dp_idx = dp_idx
        model = self.models[dp_idx]
        self.model = model
        self.model_args = self.model_args_list[dp_idx]
        self.mesh_device = model.mesh_device
        self.mesh_mapper = self.mesh_mapper_per_dp[dp_idx]
        self.kv_caches = self.kv_caches_per_dp[dp_idx]
        self.current_pos = self.current_pos_per_dp[dp_idx]
        self.rot_mat_idxs = self.rot_mat_idxs_per_dp[dp_idx]
        self.decode_position = self.decode_position_per_dp[dp_idx]
        self.prefill_traces = self.prefill_traces_per_dp[dp_idx]
        self.decode_trace_id = self.decode_trace_id_per_dp[dp_idx]
        self.decode_trace_tensors = self.decode_trace_tensors_per_dp[dp_idx]
        self.decode_trace_output = self.decode_trace_output_per_dp[dp_idx]
        self.decode_trace_needs_reset = self.decode_trace_needs_reset_per_dp[dp_idx]
        self.prev_page_table = self.prev_page_table_per_dp[dp_idx]
        self.vision_trace_id = self.vision_trace_id_per_dp[dp_idx]
        self.vision_trace_tensors = self.vision_trace_tensors_per_dp[dp_idx]
        self.vision_trace_outputs = self.vision_trace_outputs_per_dp[dp_idx]
        self._prefill_compiled_buckets = self._prefill_compiled_buckets_per_dp[dp_idx]

    def _save_active_replica(self, dp_idx: int) -> None:
        """Save current instance state back to DP replica storage."""
        self.kv_caches_per_dp[dp_idx] = self.kv_caches
        self.current_pos_per_dp[dp_idx] = self.current_pos
        self.rot_mat_idxs_per_dp[dp_idx] = self.rot_mat_idxs
        self.decode_position_per_dp[dp_idx] = self.decode_position
        self.prefill_traces_per_dp[dp_idx] = self.prefill_traces
        self.decode_trace_id_per_dp[dp_idx] = self.decode_trace_id
        self.decode_trace_tensors_per_dp[dp_idx] = self.decode_trace_tensors
        self.decode_trace_output_per_dp[dp_idx] = self.decode_trace_output
        self.decode_trace_needs_reset_per_dp[dp_idx] = self.decode_trace_needs_reset
        self.prev_page_table_per_dp[dp_idx] = self.prev_page_table
        self.vision_trace_id_per_dp[dp_idx] = self.vision_trace_id
        self.vision_trace_tensors_per_dp[dp_idx] = self.vision_trace_tensors
        self.vision_trace_outputs_per_dp[dp_idx] = self.vision_trace_outputs
        if hasattr(self, "_prefill_compiled_buckets"):
            self._prefill_compiled_buckets_per_dp[dp_idx] = self._prefill_compiled_buckets

    def init_kv_cache(self) -> None:
        """
        Initialize internal KV cache.

        For vLLM mode, the KV cache is provided externally via allocate_kv_cache().
        This method is only used as a fallback for non-paged attention mode.
        """
        # For vLLM, KV cache is allocated externally - this is a no-op
        # The KV cache is set in warmup_model_prefill/warmup_model_decode
        logger.info("init_kv_cache called - KV cache will be provided by vLLM")

    def _prepare_text_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        pooled_patches_idx: Optional[torch.Tensor],
    ) -> "ttnn.Tensor":
        """
        Prepare text inputs with optional vision embedding fusion.

        Args:
            input_ids: Token IDs [batch, seq_len]
            pixel_values: Pixel values for images [n_crops, 3, H, W] or video [n_frames, 3, H, W]
            pooled_patches_idx: Pooling indices for vision tokens

        Returns:
            Fused hidden states tensor on device
        """
        if pixel_values is not None and pooled_patches_idx is not None:
            # Vision + text fusion
            # Track request number for debugging SDPA issues
            from models.demos.molmo2.tt.vision_attention import VisionAttention

            VisionAttention._current_request += 1
            request_num = VisionAttention._current_request
            sdpa_calls_before = VisionAttention._sdpa_call_count

            logger.info(f"_prepare_text_inputs: Starting vision+text fusion (REQUEST #{request_num})")
            logger.info(
                f"  pixel_values.shape={pixel_values.shape}, pooled_patches_idx.shape={pooled_patches_idx.shape}"
            )
            logger.info(f"  input_ids.shape={input_ids.shape}")
            logger.info(f"  SDPA calls so far: {sdpa_calls_before}")

            # CRITICAL: Synchronize ALL devices before starting vision processing
            # This ensures clean state from any previous requests
            logger.info(f"_prepare_text_inputs: Syncing all devices before vision processing...")
            try:
                ttnn.synchronize_device(self.mesh_device)
                logger.info(f"_prepare_text_inputs: All devices synced successfully")
            except Exception as e:
                logger.error(f"_prepare_text_inputs: Device sync FAILED: {e}")
                raise

            # Detect raw image format [C, H, W] vs pre-unfolded [num_crops, num_patches, 588]
            # and normalize to [B, C, H, W] for embed_image
            patch_features = 14 * 14 * 3  # 588
            if pixel_values.dim() == 3 and pixel_values.shape[-1] != patch_features:
                # Raw image [C, H, W] -> [1, C, H, W]
                pixel_values = pixel_values.unsqueeze(0)

            logger.info(f"_prepare_text_inputs: Calling embed_image...")
            visual_embeddings_ttnn, valid_token = self.model.embed_image(pixel_values, pooled_patches_idx)
            # CRITICAL: Synchronize after vision backbone to ensure operations complete
            # Without this, async TTNN operations may not finish, causing garbage output or hangs
            ttnn.synchronize_device(self.mesh_device)

            sdpa_calls_after = VisionAttention._sdpa_call_count
            logger.info(f"_prepare_text_inputs: embed_image completed (REQUEST #{request_num})")
            logger.info(f"  valid_token.shape={valid_token.shape}")
            logger.info(
                f"  SDPA calls this request: {sdpa_calls_after - sdpa_calls_before} (total: {sdpa_calls_after})"
            )

            logger.info(f"_prepare_text_inputs: Calling prepare_inputs_for_multimodal...")
            fused_ttnn = self.model.prepare_inputs_for_multimodal(input_ids, visual_embeddings_ttnn, valid_token)
            # CRITICAL: Synchronize after multimodal fusion to ensure operations complete
            ttnn.synchronize_device(self.mesh_device)
            logger.info(f"_prepare_text_inputs: prepare_inputs_for_multimodal completed")

            ttnn.deallocate(visual_embeddings_ttnn)
            return fused_ttnn
        else:
            # Text-only: just embed the tokens
            input_ids_ttnn = ttnn.from_torch(
                input_ids,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            fused_ttnn = self.model.text_model.embed_tokens(input_ids_ttnn)
            ttnn.deallocate(input_ids_ttnn)
            return fused_ttnn

    def _build_mm_prefill_attn_mask(
        self,
        token_type_ids: Optional[torch.Tensor],
        hf_attention_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> Optional["ttnn.Tensor"]:
        """HF-style multimodal prefill additive mask; ``None`` if not applicable."""
        logger.info(f"_build_mm_prefill_attn_mask: token_type_ids={token_type_ids is not None}, seq_len={seq_len}")
        if token_type_ids is not None:
            logger.info(f"  token_type_ids shape: {token_type_ids.shape}, sum={token_type_ids.sum().item()}")
        if token_type_ids is None or seq_len <= 1:
            logger.info(f"  Returning None (token_type_ids={token_type_ids is not None}, seq_len={seq_len})")
            return None
        if token_type_ids.shape[1] != seq_len:
            raise ValueError(f"token_type_ids length {token_type_ids.shape[1]} != hidden seq_len {seq_len}")
        if hf_attention_mask is not None and hf_attention_mask.shape[1] != seq_len:
            raise ValueError(f"hf_attention_mask length {hf_attention_mask.shape[1]} != seq_len {seq_len}")
        bias = build_molmo2_prefill_attention_bias(token_type_ids, attention_mask=hf_attention_mask).to(torch.bfloat16)
        is_mesh = self.mesh_device.__class__.__name__ == "MeshDevice"
        mm = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None
        return ttnn.from_torch(
            bias,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )

    # Maximum tokens processed in a single prefill forward pass.
    # Sequences longer than this are processed via chunked prefill.
    # 4096 is the largest trace-captured bucket size and stays within T3K DRAM limits.
    _MAX_PREFILL_CHUNK_SIZE = 4096

    # KV cache block size (must match model_spec.py "block_size": "64")
    _BLOCK_SIZE = 64

    def _run_chunked_prefill_vllm(
        self,
        hidden_states_ttnn: "ttnn.Tensor",
        actual_seq_len: int,
        page_table_torch: torch.Tensor,
        page_table_tt: Optional["ttnn.Tensor"],
        user_id: int,
    ) -> "ttnn.Tensor":
        """
        Run prefill in chunks to avoid OOM for long sequences on T3K.

        Splits hidden states into _MAX_PREFILL_CHUNK_SIZE-token chunks and
        processes each with chunked_scaled_dot_product_attention (reads
        previous KV from paged cache).  Only logits from the last token
        of the last chunk are retained.

        Mirrors demo.py Molmo2Generator._run_chunked_prefill().
        """
        chunk_size = self._MAX_PREFILL_CHUNK_SIZE
        block_size = self._BLOCK_SIZE
        mm = ttnn.ReplicateTensorToMesh(self.mesh_device)

        # Clamp to KV cache capacity (max_gen_len = max_seq_len - 1) so we
        # never try to write KV blocks beyond what was allocated.
        max_kv_tokens = self.max_gen_len  # e.g. 8191 for max_seq_len=8192
        if actual_seq_len > max_kv_tokens:
            logger.warning(
                f"Chunked prefill: seq_len={actual_seq_len} > max_kv_tokens={max_kv_tokens};"
                " truncating to KV cache capacity"
            )
            actual_seq_len = max_kv_tokens

        num_chunks = (actual_seq_len + chunk_size - 1) // chunk_size
        logger.info(f"Chunked prefill: seq_len={actual_seq_len}, {num_chunks} chunks of {chunk_size}")

        logits = None
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, actual_seq_len)
            actual_chunk_size = chunk_end - chunk_start

            logger.info(
                f"  Chunk {chunk_idx + 1}/{num_chunks}: positions {chunk_start}-{chunk_end}, size={actual_chunk_size}"
            )

            # Slice hidden states for this chunk [1, 1, chunk_size, hidden_dim]
            chunk_hidden = ttnn.slice(
                hidden_states_ttnn,
                (0, 0, chunk_start, 0),
                (1, 1, chunk_end, 4096),
            )

            # Rotation matrices for this chunk's positions
            rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(actual_chunk_size, start_pos=chunk_start)

            # Chunk page table (blocks covered by this chunk)
            chunk_start_block = chunk_start // block_size
            chunk_end_block = (chunk_end + block_size - 1) // block_size
            chunk_pt_torch = page_table_torch[:1, chunk_start_block:chunk_end_block]
            chunk_pt_tt = ttnn.from_torch(
                chunk_pt_torch,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mm,
            )

            chunk_logits, _ = self.model.text_model.forward(
                hidden_states=chunk_hidden,
                start_pos=chunk_start,
                attn_mask=None,
                kv_caches=self.kv_caches,
                rot_mats=rot_mats,
                page_table=page_table_tt,  # full table for reading previous KV
                user_id=user_id,
                chunk_page_table=chunk_pt_tt,  # chunk table for writing new KV
                chunk_start_idx=chunk_start,  # enables chunked SDPA
            )

            ttnn.deallocate(chunk_hidden)
            ttnn.deallocate(chunk_pt_tt)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])

            if chunk_idx == num_chunks - 1:
                logits = chunk_logits
            else:
                ttnn.deallocate(chunk_logits)

        return logits

    def _run_prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        use_trace: bool = True,
        use_vision_trace: bool = False,
        use_unified_trace: bool = False,
        page_table: Optional["ttnn.Tensor"] = None,
        page_table_torch: Optional[torch.Tensor] = None,
        user_id: int = 0,
        attn_mask: Optional["ttnn.Tensor"] = None,
    ) -> Tuple["ttnn.Tensor", dict]:
        """
        Run prefill forward pass directly on the model.

        This replaces Molmo2Generator.run_prefill() with direct model calls.

        Args:
            input_ids: Token IDs [batch, seq_len]
            pixel_values: Pixel values for images/video (optional)
            pooled_patches_idx: Pooling indices for vision tokens (optional)
            use_trace: Whether to use tracing (check for pre-captured trace)
            use_vision_trace: Whether to use vision tracing (ignored for now)
            use_unified_trace: Whether to use unified trace (ignored for now)
            page_table: Page table tensor for paged attention
            user_id: Batch index for multi-user batching (determines KV cache slot)
            attn_mask: Optional additive attention mask (for multimodal cross-attention)

        Returns:
            Tuple of (logits_ttnn, timing_dict)
        """
        original_seq_len = input_ids.shape[1]
        has_vision = pixel_values is not None

        # Step 1: Prepare hidden states (vision + text fusion or text-only)
        # Do this FIRST to get the actual seq_len after vision fusion
        hidden_states_ttnn = self._prepare_text_inputs(input_ids, pixel_values, pooled_patches_idx)

        # Step 2: Compute padded_seq_len from the ACTUAL hidden states
        # Vision fusion changes seq_len (replaces <|image|> with ~196 vision tokens)
        # Tensor format is [1, 1, seq_len, hidden_dim] (4D) - seq_len is at index 2
        tensor_shape = hidden_states_ttnn.shape
        if len(tensor_shape) == 4:
            actual_seq_len = tensor_shape[2]  # [batch, 1, seq_len, hidden_dim]
        elif len(tensor_shape) == 3:
            actual_seq_len = tensor_shape[1]  # [batch, seq_len, hidden_dim]
        else:
            actual_seq_len = tensor_shape[1]  # fallback
        logger.info(f"_run_prefill: tensor_shape={list(tensor_shape)}, actual_seq_len={actual_seq_len}")
        padded_seq_len = get_padded_prefill_len(actual_seq_len)

        # Step 3: Pad hidden states to bucket size
        if actual_seq_len != padded_seq_len:
            # Padding tuple must match tensor dimensions
            if len(tensor_shape) == 4:
                # [batch, 1, seq_len, hidden_dim] -> pad seq_len dimension (index 2)
                pad_tuple = ((0, 0), (0, 0), (0, padded_seq_len - actual_seq_len), (0, 0))
            else:
                # [batch, seq_len, hidden_dim] -> pad seq_len dimension (index 1)
                pad_tuple = ((0, 0), (0, padded_seq_len - actual_seq_len), (0, 0))
            hidden_states_ttnn = ttnn.pad(
                hidden_states_ttnn,
                padding=pad_tuple,
                value=0.0,
            )

        logger.info(f"_run_prefill: seq_len {actual_seq_len} -> padded {padded_seq_len} (has_vision={has_vision})")

        # Use vLLM's paged KV cache if available
        prefill_kv_cache = self.kv_caches

        # --- Chunked prefill for sequences > _MAX_PREFILL_CHUNK_SIZE ---
        # Avoids full [heads, seq_len, seq_len] attention matrix which OOMs on T3K
        # for sequences > 4096 tokens (e.g. 30-frame video → 6570 tokens).
        # Uses chunked_scaled_dot_product_attention (reads from paged KV cache).
        # Requires page_table_torch for per-chunk block-table slicing.
        # NOTE: Only use chunked prefill when there's NO attention mask.
        # If there IS a multimodal attention mask, use full prefill to preserve it
        # (matches demo.py behavior which prioritizes mask over chunking).
        if (
            actual_seq_len > self._MAX_PREFILL_CHUNK_SIZE
            and page_table_torch is not None
            and page_table is not None
            and attn_mask is None
        ):
            # No attention mask - safe to use chunked prefill
            logger.info(f"_run_prefill: seq_len={actual_seq_len} > {self._MAX_PREFILL_CHUNK_SIZE} → chunked prefill")
            ttnn.deallocate(hidden_states_ttnn)  # will be re-sliced inside chunked path

            # Re-build hidden states without seq padding (chunked path handles its own sizing)
            hidden_states_ttnn = self._prepare_text_inputs(input_ids, pixel_values, pooled_patches_idx)

            # effective_seq_len may be clamped inside _run_chunked_prefill_vllm
            effective_seq_len = min(actual_seq_len, self.max_gen_len)
            logits_ttnn = self._run_chunked_prefill_vllm(
                hidden_states_ttnn=hidden_states_ttnn,
                actual_seq_len=actual_seq_len,
                page_table_torch=page_table_torch,
                page_table_tt=page_table,
                user_id=user_id,
            )
            ttnn.deallocate(hidden_states_ttnn)

            logger.info("_run_prefill: Syncing device after chunked prefill...")
            ttnn.synchronize_device(self.mesh_device)

            self._reset_kv_cache(effective_seq_len)
            chunk_size = self._MAX_PREFILL_CHUNK_SIZE
            last_chunk_start = ((effective_seq_len - 1) // chunk_size) * chunk_size
            return logits_ttnn, {
                "original_seq_len": effective_seq_len,
                "padded_seq_len": effective_seq_len,
                "chunked_prefill": True,
                "last_chunk_start": last_chunk_start,
            }

        # --- Standard path ---
        # Log if we're using full prefill for long sequence with attention mask
        if actual_seq_len > self._MAX_PREFILL_CHUNK_SIZE and attn_mask is not None:
            logger.info(
                f"_run_prefill: seq_len={actual_seq_len} > {self._MAX_PREFILL_CHUNK_SIZE} with attn_mask: "
                "using FULL prefill to preserve multimodal attention (matches demo behavior)"
            )

        # Step 4: Get rotation matrices for the correct bucket size
        rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(padded_seq_len, start_pos=0)

        # Check for pre-captured trace (text-only path)
        if use_trace and padded_seq_len in self.prefill_traces:
            logger.info(f"_run_prefill: Executing prefill trace for seq_len={padded_seq_len}")
            trace_id, trace_tensors, trace_output = self.prefill_traces[padded_seq_len]

            # Copy inputs into trace tensors
            ttnn.copy(hidden_states_ttnn, trace_tensors["hidden_states"])
            ttnn.deallocate(hidden_states_ttnn)

            # Copy rot_mats
            ttnn.copy(rot_mats[0], trace_tensors["cos"])
            ttnn.copy(rot_mats[1], trace_tensors["sin"])
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])

            # Copy page_table if provided
            if page_table is not None and "page_table" in trace_tensors:
                ttnn.copy(page_table, trace_tensors["page_table"])

            # Execute trace
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
            logits_ttnn = trace_output
        else:
            # Direct model forward (no trace) - used for vision input
            logger.info(f"_run_prefill: Non-traced forward for seq_len={padded_seq_len}")

            logits_ttnn, _ = self.model.text_model.forward(
                hidden_states=hidden_states_ttnn,
                start_pos=0,
                attn_mask=attn_mask,
                kv_caches=prefill_kv_cache,
                rot_mats=rot_mats,
                page_table=page_table,
                user_id=user_id,
            )

            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])
            ttnn.deallocate(hidden_states_ttnn)

            logger.info("_run_prefill: Syncing device after non-traced forward...")
            ttnn.synchronize_device(self.mesh_device)
            logger.info("_run_prefill: Device sync complete")

        # Initialize position tensors for decode phase
        self._reset_kv_cache(original_seq_len)

        return logits_ttnn, {"original_seq_len": original_seq_len, "padded_seq_len": padded_seq_len}

    def _reset_kv_cache(self, start_pos: int) -> None:
        """
        Reset KV cache position state for decode phase.

        Args:
            start_pos: Starting position for decode (= prefill sequence length)
        """
        # Initialize or update current_pos tensor
        if self.current_pos is None:
            current_pos_torch = torch.tensor([start_pos], dtype=torch.int32)
            self.current_pos = ttnn.from_torch(
                current_pos_torch,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
        else:
            # Update existing tensor
            current_pos_torch = torch.tensor([start_pos], dtype=torch.int32)
            new_pos = ttnn.from_torch(
                current_pos_torch,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(new_pos, self.current_pos)
            ttnn.deallocate(new_pos)

        # Initialize or update rot_mat_idxs tensor
        if self.rot_mat_idxs is None:
            self.rot_mat_idxs = self.model.text_model.rotary_setup.allocate_decode_rot_idxs(initial_pos=start_pos)
        else:
            # Update existing tensor with new position
            batch_size = 1  # vLLM processes one request at a time in prefill
            pad_size = ((batch_size + 31) // 32) * 32 - batch_size
            position_idxs = torch.full((1, batch_size + pad_size), start_pos, dtype=torch.int32)
            new_rot_idxs = ttnn.from_torch(
                position_idxs,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(new_rot_idxs, self.rot_mat_idxs)
            ttnn.deallocate(new_rot_idxs)

        self.decode_position = start_pos

    def _get_user_page_table_tt(
        self, page_table: torch.Tensor, user_id: int, trace_num_blocks: Optional[int]
    ) -> Optional["ttnn.Tensor"]:
        """
        Create a per-user page_table TTNN tensor, padded to match trace tensor shape.

        Args:
            page_table: Full batched page_table from vLLM [batch, num_blocks]
            user_id: Index of the user in the batch
            trace_num_blocks: Number of blocks in trace tensor (for padding)

        Returns:
            Per-user page_table TTNN tensor [1, padded_num_blocks], or None if page_table is None
        """
        if page_table is None:
            return None

        # Slice page_table for this user: [batch, num_blocks] -> [1, num_blocks]
        user_page_table = page_table[user_id : user_id + 1]

        # Pad to match trace tensor shape if needed
        if trace_num_blocks is not None:
            actual_num_blocks = user_page_table.shape[-1]
            if actual_num_blocks < trace_num_blocks:
                pad_size = trace_num_blocks - actual_num_blocks
                user_page_table = torch.nn.functional.pad(user_page_table, (0, pad_size), value=0)
                logger.debug(
                    f"Padded user {user_id} page_table from {page_table.shape[-1]} to {trace_num_blocks} blocks"
                )

        # Convert to TTNN tensor
        page_table_tt = ttnn.from_torch(
            user_page_table,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return page_table_tt

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

        For tt_data_parallel > 1 (e.g. Galaxy DP=4), the mesh_device is split
        into tt_data_parallel submeshes and one model replica is created per
        submesh, following the same pattern as tt_transformers (Llama/Qwen VL).

        Args:
            hf_config: HuggingFace model config
            mesh_device: TT mesh device (full Galaxy mesh for DP>1)
            max_batch_size: Maximum batch size (total across all DP replicas)
            max_seq_len: Maximum sequence length
            tt_data_parallel: Data parallel factor (e.g. 4 for Galaxy)
            optimizations: Optimization mode (not used for Molmo2)

        Returns:
            Initialized Molmo2ForConditionalGeneration instance
        """
        logger.info(
            f"Initializing Molmo2-8B for vLLM: max_batch_size={max_batch_size}, "
            f"max_seq_len={max_seq_len}, data_parallel={tt_data_parallel}"
        )

        import os

        os.environ["HF_MODEL"] = "allenai/Molmo2-8B"

        # Load tokenizer and weights (shared across all DP replicas)
        tokenizer = load_processor()
        state_dict = load_model_weights()

        # Split mesh into submeshes for data parallelism.
        # For DP=1 (T3K), create_submeshes returns [mesh_device] unchanged.
        # For DP=4 (Galaxy 8x4), returns 4 submeshes of shape (1, 8).
        from models.tt_transformers.tt.generator import create_submeshes

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
        batch_per_dp = max_batch_size // tt_data_parallel
        logger.info(f"Created {len(submesh_devices)} submesh(es), batch_per_dp={batch_per_dp}")

        # Create one model replica per submesh (weights shared via state_dict)
        models_list = []
        model_args_list = []
        for dp_idx, submesh in enumerate(submesh_devices):
            logger.info(f"Creating Molmo2 model replica {dp_idx} on {submesh.shape} submesh...")
            model_i = create_model(
                submesh,
                state_dict,
                num_layers=None,
                max_batch_size=batch_per_dp,
                max_seq_len=max_seq_len,
            )
            model_args_i = Molmo2ModelArgs(
                mesh_device=submesh,
                max_batch_size=batch_per_dp,
                max_seq_len=max_seq_len,
            )
            models_list.append(model_i)
            model_args_list.append(model_args_i)

        # Create instance with all replicas
        instance = cls(
            models=models_list,
            model_args_list=model_args_list,
            mesh_device=mesh_device,
            tokenizer=tokenizer,
            data_parallel=tt_data_parallel,
            batch_per_dp=batch_per_dp,
        )

        # Note: Trace warmup is handled by TTWorker.compile_or_warm_up_model()
        # which calls warmup_model_decode and warmup_model_prefill.
        # Do NOT call _warmup_traces here - it would create duplicate traces.

        logger.info("Molmo2-8B initialized successfully for vLLM")
        return instance

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
        enable_trace: bool = True,  # Enable traces by default
        sampling_params=None,
        # Image kwargs
        pixel_values: Optional[List] = None,
        image_token_pooling: Optional[List] = None,
        image_grids: Optional[List] = None,
        image_num_crops: Optional[List] = None,
        # Video kwargs (NEW)
        pixel_values_videos: Optional[List] = None,
        video_grid_thw: Optional[List] = None,
        video_token_pooling: Optional[List] = None,
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
            pixel_values: Pre-processed pixel values from vLLM (images)
            image_token_pooling: Token pooling indices for images
            image_grids: Grid info for images
            image_num_crops: Number of crops per image
            pixel_values_videos: Pre-processed pixel values from vLLM (videos) [n_frames, 3, H, W]
            video_grid_thw: Video grid info [[n_frames, pooled_h, pooled_w]]
            video_token_pooling: Token pooling indices for video frames
            **kwargs: Additional keyword arguments

        Returns:
            Logits tensor
        """
        batch_size = tokens.shape[0]

        # NOTE: DO NOT reset KV cache here! vLLM may call prefill_forward for
        # a new request while a previous request is still decoding. Resetting
        # would corrupt the in-progress decode state.
        #
        # The demo.py's run_prefill already handles position reset internally:
        # - Prefill always starts at position 0 (writes to KV cache 0..seq_len-1)
        # - run_prefill calls reset_kv_cache(original_seq_len) after completing
        # - Decode continues from position original_seq_len
        #
        # For proper multi-request support, Molmo2 would need paged attention.

        # Debug logging for multimodal kwargs
        logger.info(f"prefill_forward called: batch_size={batch_size}, tokens.shape={tokens.shape}")
        # Log full token sequence to trace chat template issue
        if tokens.shape[1] <= 50:
            logger.info(f"  FULL tokens: {tokens[0].tolist()}")
        else:
            logger.info(f"  tokens first 20: {tokens[0, :20].tolist()}")
            logger.info(f"  tokens last 20: {tokens[0, -20:].tolist()}")
        logger.info(f"  pixel_values type: {type(pixel_values)}, len: {len(pixel_values) if pixel_values else 0}")
        logger.info(
            f"  pixel_values_videos type: {type(pixel_values_videos)}, len: {len(pixel_values_videos) if pixel_values_videos else 0}"
        )
        logger.info(f"  video_grid_thw: {video_grid_thw}")
        logger.info(f"  image_grids type: {type(image_grids)}, len: {len(image_grids) if image_grids else 0}")
        logger.info(f"  other kwargs: {list(kwargs.keys())}")
        if pixel_values and len(pixel_values) > 0:
            pv0 = pixel_values[0]
            logger.info(f"  pixel_values[0] type: {type(pv0)}, shape: {pv0.shape if hasattr(pv0, 'shape') else 'N/A'}")
        if pixel_values_videos and len(pixel_values_videos) > 0:
            pvv0 = pixel_values_videos[0]
            logger.info(
                f"  pixel_values_videos[0] type: {type(pvv0)}, shape: {pvv0.shape if hasattr(pvv0, 'shape') else 'N/A'}"
            )

        # vLLM might pass image_grids as None but provide data in kwargs
        # Try to extract from kwargs if not provided directly
        if image_grids is None and "image_grid_thw" in kwargs:
            image_grid_thw = kwargs.get("image_grid_thw")
            logger.info(f"  Attempting to use image_grid_thw: {image_grid_thw}")
            # image_grid_thw might be nested lists - flatten to get actual grid data
            # But note: image_grid_thw format may differ from image_grids

        # Extract multimodal attention mask inputs from kwargs (vLLM may pass these)
        # vLLM passes these with 'mm_' prefix for multimodal
        token_type_ids = kwargs.get("token_type_ids", None) or kwargs.get("mm_token_type_ids", None)
        hf_attention_mask = kwargs.get("attention_mask", None) or kwargs.get("mm_attention_mask", None)
        if token_type_ids is not None:
            logger.info(
                f"  Found token_type_ids from kwargs: shape={token_type_ids.shape if hasattr(token_type_ids, 'shape') else 'N/A'}"
            )
        if hf_attention_mask is not None:
            logger.info(
                f"  Found hf_attention_mask from kwargs: shape={hf_attention_mask.shape if hasattr(hf_attention_mask, 'shape') else 'N/A'}"
            )

        # NOTE: token_type_ids generation is done INSIDE the per-user loop after we know the
        # actual sequence length post-vision-fusion. The raw input tokens have placeholder tokens
        # that get replaced with expanded vision tokens, changing the sequence length.
        # Generating token_type_ids here would create a shape mismatch.

        # Handle prompt_lens default
        if prompt_lens is None:
            prompt_lens = torch.tensor([tokens.shape[1]] * batch_size)

        # Get trace num_blocks for page_table padding (used per-user below)
        trace_num_blocks = None
        if page_table is not None and hasattr(self, "prefill_traces") and self.prefill_traces:
            for seq_len, (trace_id, trace_tensors, trace_output) in self.prefill_traces.items():
                if "page_table" in trace_tensors:
                    trace_page_table_shape = list(trace_tensors["page_table"].shape)
                    trace_num_blocks = trace_page_table_shape[-1]
                    logger.info(f"Trace page_table num_blocks={trace_num_blocks}")
                    break

        # Check if we have pre-processed data from vLLM
        has_vllm_images = pixel_values is not None and len(pixel_values) > 0
        has_vllm_videos = pixel_values_videos is not None and len(pixel_values_videos) > 0

        # Handle images default for standalone mode
        if images is None:
            images = [None] * batch_size

        # Collect last token logits for each user
        output_logits = []

        for user_id in range(batch_size):
            # Route this user to the correct DP replica.
            # Users [0 .. batch_per_dp-1]  -> replica 0
            # Users [batch_per_dp .. 2*batch_per_dp-1] -> replica 1, etc.
            model_id = min(user_id // self.batch_per_dp, self.data_parallel - 1)
            if model_id != self._active_dp_idx:
                self._set_active_replica(model_id)
                # Refresh trace_num_blocks from the newly-active replica's traces
                trace_num_blocks = None
                if page_table is not None and self.prefill_traces:
                    for _sl, (_tid, _tt, _to) in self.prefill_traces.items():
                        if "page_table" in _tt:
                            trace_num_blocks = list(_tt["page_table"].shape)[-1]
                        break

            # Create per-user page_table_tt with proper padding for trace tensors
            # This is critical for batched requests - trace expects [1, num_blocks]
            page_table_tt = self._get_user_page_table_tt(page_table, user_id, trace_num_blocks)

            # Build multimodal prefill attention mask for this user (if token_type_ids provided)
            user_token_type_ids = None
            user_hf_attention_mask = None
            if token_type_ids is not None:
                # vLLM may pass as list of lists - extract user's data
                if isinstance(token_type_ids, list) and len(token_type_ids) > user_id:
                    user_token_type_ids = token_type_ids[user_id]
                elif isinstance(token_type_ids, torch.Tensor):
                    user_token_type_ids = token_type_ids[user_id : user_id + 1]
                else:
                    logger.warning(f"Unexpected token_type_ids type: {type(token_type_ids)}")
            if hf_attention_mask is not None:
                if isinstance(hf_attention_mask, list) and len(hf_attention_mask) > user_id:
                    user_hf_attention_mask = hf_attention_mask[user_id]
                elif isinstance(hf_attention_mask, torch.Tensor):
                    user_hf_attention_mask = hf_attention_mask[user_id : user_id + 1]

            # Check for VIDEO first (new vLLM multimodal flow)
            # NOTE: pixel_values_videos may be [[None]] for image requests (filled with None in tt_model_runner)
            # Must check the actual data after unwrapping nested lists
            pvv_data = None
            if has_vllm_videos and user_id < len(pixel_values_videos) and pixel_values_videos[user_id] is not None:
                pvv = pixel_values_videos[user_id]
                if isinstance(pvv, list) and len(pvv) > 0:
                    pvv = pvv[0]
                # After unwrapping, check if we have actual data
                if pvv is not None:
                    if isinstance(pvv, torch.Tensor):
                        pvv_data = pvv
                    elif hasattr(pvv, "__array__"):
                        pvv_data = torch.from_numpy(pvv)

            if pvv_data is not None:
                # VIDEO PATH: vLLM provides pixel_values_videos [n_frames, 3, H, W]
                pv_tensor = pvv_data

                n_frames = pv_tensor.shape[0]
                logger.info(f"  prefill_forward[user={user_id}]: VIDEO with {n_frames} frames, shape={pv_tensor.shape}")

                # Get video_token_pooling - shape [n_frames, N_out, K_pool]
                pooling = None
                if video_token_pooling is not None and len(video_token_pooling) > user_id:
                    vtp = video_token_pooling[user_id]
                    if isinstance(vtp, list) and len(vtp) > 0:
                        vtp = vtp[0]
                    # After unwrapping, check if we have actual data
                    if vtp is not None:
                        if isinstance(vtp, torch.Tensor):
                            pooling = vtp
                        elif hasattr(vtp, "__array__"):
                            pooling = torch.from_numpy(vtp)
                    if pooling is not None:
                        logger.info(f"    video_token_pooling shape: {pooling.shape}")

                    # Keep pooling as [n_frames, N_out, K_pool] for multi-frame videos.
                    # embed_image routes batch_size>1 to _embed_image_data_parallel which
                    # processes frames in chunks (max_frames_per_pool_chunk=1) to avoid OOM.
                    # Do NOT flatten to [1, n_frames*N_out, K_pool]: that causes batch_size=1
                    # routing to forward_ttnn which processes all pooling positions at once,
                    # requiring huge (447MB+) allocations for long videos.
                    #
                    # vLLM multimodal pipeline may flatten 3D to 2D [n_tokens, k_pool].
                    # If so, reshape back to [n_frames, n_out, k_pool].
                    if pooling is not None and pooling.dim() == 2:
                        n_tokens, k_pool = pooling.shape
                        n_out = n_tokens // n_frames
                        pooling = pooling.view(n_frames, n_out, k_pool)
                        logger.info(f"    Reshaped 2D pooling to 3D: {pooling.shape}")
                    logger.info(f"    Pooling shape for prefill: {pooling.shape}")

                if pooling is None:
                    # Generate default pooling for video frames
                    # Video uses 3×3 pooling: 27/3 = 9 → 9×9 = 81 pooled tokens per frame
                    import numpy as np

                    pooled_h, pooled_w = 9, 9  # Video: 27/3 = 9
                    pool_h, pool_w = 3, 3  # Video uses 3×3 pooling
                    k_pool = pool_h * pool_w  # 9
                    patches_per_frame = 27 * 27  # 729

                    all_pooling_idx = []
                    for frame_idx in range(n_frames):
                        resize_idx = np.arange(patches_per_frame).reshape(27, 27)
                        resize_idx = arange_for_pooling(resize_idx, pool_h, pool_w)
                        resize_idx_flat = resize_idx.reshape(-1, pool_h * pool_w)[: pooled_h * pooled_w, :]
                        offset = frame_idx * patches_per_frame
                        frame_pooling = np.where(resize_idx_flat >= 0, resize_idx_flat + offset, resize_idx_flat)
                        all_pooling_idx.append(frame_pooling)

                    # Stack as [n_frames, n_out, k_pool] for data-parallel processing
                    n_out = pooled_h * pooled_w  # 81 for video
                    pooling = torch.from_numpy(np.stack(all_pooling_idx, axis=0)).long()
                    logger.info(
                        f"    Generated default video pooling: {pooling.shape} (n_frames={n_frames}, n_out={n_out}, k_pool={k_pool})"
                    )

                # Run prefill with video data
                user_tokens = tokens[user_id : user_id + 1]
                user_prompt_len = (
                    prompt_lens[user_id].item() if hasattr(prompt_lens[user_id], "item") else prompt_lens[user_id]
                )

                # Run vision + prefill using generator's run_prefill
                # CRITICAL: Disable traces for video - traces captured during warmup (text-only)
                # don't work with vision input due to device write restrictions during trace execution.
                # Demo.py confirms video works without traces.
                logger.info(f"  VIDEO: Running prefill WITHOUT traces (traces incompatible with vision)")
                logger.info(
                    f"  VIDEO DEBUG: user_tokens shape={user_tokens.shape}, first 30={user_tokens[0, :30].tolist()}"
                )

                # FIX: vLLM may strip chat template tokens. Demo expects [151645, 151644, 872, 198, ...]
                # (BOS + <|im_start|> + user + \n). Check what's missing and prepend it.
                first_tok = user_tokens[0, 0].item()
                if first_tok == 151645:  # BOS present - chat template complete
                    pass
                elif first_tok == 151644:  # <|im_start|> without BOS
                    prefix = torch.tensor([[151645]], dtype=user_tokens.dtype)
                    user_tokens = torch.cat([prefix, user_tokens], dim=1)
                    logger.info(f"  VIDEO FIX: Prepended BOS, new shape={user_tokens.shape}")
                else:  # Entire chat template missing
                    prefix = torch.tensor([[151645, 151644, 872, 198]], dtype=user_tokens.dtype)
                    user_tokens = torch.cat([prefix, user_tokens], dim=1)
                    logger.info(f"  VIDEO FIX: Prepended full chat template, new shape={user_tokens.shape}")

                # Generate/validate token_type_ids for the POST-FUSION sequence length.
                # vLLM may provide mm_token_type_ids, but we need to verify it matches post-fusion length.
                # Compute: post_fusion_len = original_len - num_placeholders + num_visual_tokens
                image_patch_id = 151938  # <im_patch> token
                num_placeholders = (user_tokens == image_patch_id).sum().item()
                # pooling shape is [n_frames, n_out, k_pool] - total visual tokens = n_frames * n_out
                num_visual_tokens = pooling.shape[0] * pooling.shape[1]  # n_frames * n_out
                original_seq_len = user_tokens.shape[1]
                post_fusion_seq_len = original_seq_len - num_placeholders + num_visual_tokens

                # Compute padded sequence length (hidden states get padded to bucket size in _run_prefill)
                padded_seq_len = get_padded_prefill_len(post_fusion_seq_len)

                # Check if vLLM provided token_type_ids with correct length
                # Note: vLLM may expand tokens before calling us, so lengths might match
                final_token_type_ids = None
                if user_token_type_ids is not None:
                    # Convert to tensor if needed (vLLM may pass as list)
                    # Skip if list contains None values
                    if isinstance(user_token_type_ids, list):
                        if any(x is None for x in user_token_type_ids):
                            logger.info(f"    user_token_type_ids list contains None, will generate our own")
                            user_token_type_ids = None
                        else:
                            try:
                                user_token_type_ids = torch.tensor(user_token_type_ids)
                            except Exception as e:
                                logger.warning(f"    Failed to convert user_token_type_ids to tensor: {e}")
                                user_token_type_ids = None

                if user_token_type_ids is not None:
                    if not isinstance(user_token_type_ids, torch.Tensor):
                        logger.warning(f"    user_token_type_ids is unexpected type: {type(user_token_type_ids)}")
                        user_token_type_ids = None
                    elif user_token_type_ids.dim() < 2:
                        user_token_type_ids = user_token_type_ids.unsqueeze(0)

                    if isinstance(user_token_type_ids, torch.Tensor) and user_token_type_ids.dim() >= 2:
                        vllm_len = user_token_type_ids.shape[1]
                        logger.info(
                            f"    vLLM provided token_type_ids: len={vllm_len}, "
                            f"post_fusion_len={post_fusion_seq_len}, padded_len={padded_seq_len}"
                        )
                        # If vLLM's token_type_ids matches post-fusion length (or is close), use it
                        if vllm_len == post_fusion_seq_len or vllm_len == original_seq_len:
                            # Pad to bucket size
                            if vllm_len < padded_seq_len:
                                final_token_type_ids = torch.zeros(1, padded_seq_len, dtype=torch.long)
                                final_token_type_ids[0, :vllm_len] = user_token_type_ids[0, :vllm_len]
                            else:
                                final_token_type_ids = user_token_type_ids
                            logger.info(f"    Using vLLM-provided token_type_ids (padded to {padded_seq_len})")

                # If no valid token_type_ids from vLLM, generate our own
                if final_token_type_ids is None:
                    # HF marks ONLY these specific token types (not ranges):
                    # - <im_start> (151936)
                    # - <im_end> (151937)
                    # - <im_patch> (151938)
                    # Timestamps, frame markers, etc. between visual regions are NOT marked.
                    im_start_id = 151936
                    im_end_id = 151937
                    # im_patch_id = 151938 already defined above as image_patch_id

                    tokens_flat = user_tokens[0]

                    # Find all positions of visual tokens to mark
                    visual_mask = (
                        (tokens_flat == im_start_id) | (tokens_flat == im_end_id) | (tokens_flat == image_patch_id)
                    )

                    # Create token_type_ids marking only visual token positions
                    final_token_type_ids = torch.zeros(1, padded_seq_len, dtype=torch.long)

                    # Mark visual token positions (positions stay same after fusion
                    # since num_visual_tokens == num_placeholders for standard video)
                    visual_positions = visual_mask.nonzero(as_tuple=True)[0]
                    for pos in visual_positions:
                        if pos < padded_seq_len:
                            final_token_type_ids[0, pos] = 1

                    n_im_start = (tokens_flat == im_start_id).sum().item()
                    n_im_end = (tokens_flat == im_end_id).sum().item()
                    n_im_patch = (tokens_flat == image_patch_id).sum().item()
                    total_marked = visual_mask.sum().item()

                    logger.info(
                        f"    Generated token_type_ids: marked {total_marked} visual tokens "
                        f"(<im_start>={n_im_start}, <im_end>={n_im_end}, <im_patch>={n_im_patch}), "
                        f"padded_len={padded_seq_len}"
                    )

                # Convert user_hf_attention_mask if needed, or set to None
                # We generally don't need the HF attention mask when we have token_type_ids
                if user_hf_attention_mask is not None:
                    if isinstance(user_hf_attention_mask, list):
                        if any(x is None for x in user_hf_attention_mask):
                            user_hf_attention_mask = None
                        else:
                            try:
                                user_hf_attention_mask = torch.tensor(user_hf_attention_mask)
                                if user_hf_attention_mask.dim() < 2:
                                    user_hf_attention_mask = user_hf_attention_mask.unsqueeze(0)
                            except Exception:
                                user_hf_attention_mask = None
                    elif not isinstance(user_hf_attention_mask, torch.Tensor):
                        user_hf_attention_mask = None

                user_attn_mask = self._build_mm_prefill_attn_mask(
                    final_token_type_ids, user_hf_attention_mask, padded_seq_len
                )
                logits_ttnn, prefill_timing = self._run_prefill(
                    input_ids=user_tokens,
                    pixel_values=pv_tensor,
                    pooled_patches_idx=pooling,
                    use_trace=False,  # DISABLED for video
                    use_vision_trace=False,
                    page_table=page_table_tt,
                    page_table_torch=page_table,  # CPU tensor for chunked prefill block slicing
                    user_id=user_id,
                    attn_mask=user_attn_mask,
                )

                # Convert ttnn tensor to torch tensor (same as image/text path)
                ttnn.synchronize_device(self.mesh_device)
                mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                logits_torch = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0].squeeze()

                # Deallocate if trace is disabled
                if not enable_trace:
                    ttnn.deallocate(logits_ttnn)

                # Extract last token's logits
                original_seq_len = prefill_timing.get("original_seq_len", user_prompt_len)
                last_chunk_start = prefill_timing.get("last_chunk_start", 0)
                if logits_torch.dim() == 2:
                    # For chunked prefill, logits_torch is [last_chunk_size, vocab]; index within chunk.
                    # For standard prefill, logits_torch is [seq_len, vocab]; index is original_seq_len-1.
                    last_token_idx = original_seq_len - 1 - last_chunk_start
                    last_token_logits = logits_torch[last_token_idx, :]
                else:
                    last_token_logits = logits_torch  # Already [vocab_size]

                # DEBUG: Log top-5 predictions like demo does
                logger.info(f"=== VIDEO PREFILL DIAGNOSTICS (SERVER) ===")
                logger.info(f"  original_seq_len={original_seq_len}, last_chunk_start={last_chunk_start}")
                logger.info(f"  logits_torch shape: {logits_torch.shape}")
                logger.info(f"  last_token_idx: {last_token_idx if logits_torch.dim() == 2 else 'N/A (1D)'}")
                logger.info(f"  last_token_logits shape: {last_token_logits.shape}")
                logger.info(
                    f"  last_token_logits stats: mean={last_token_logits.mean().item():.4f}, "
                    f"std={last_token_logits.std().item():.4f}, "
                    f"min={last_token_logits.min().item():.4f}, max={last_token_logits.max().item():.4f}"
                )
                # Top 5 predictions
                top5_values, top5_indices = torch.topk(last_token_logits, 5)
                logger.info(f"  Top 5 predictions:")
                for i, (val, idx) in enumerate(zip(top5_values.tolist(), top5_indices.tolist())):
                    decoded = self.tokenizer.decode([idx])
                    logger.info(f"    {i+1}. token={idx}, logit={val:.2f}, decoded='{decoded}'")
                logger.info(f"=== END VIDEO PREFILL DIAGNOSTICS (SERVER) ===")

                output_logits.append(last_token_logits.unsqueeze(0).unsqueeze(1))  # [1, 1, vocab_size]

                # Deallocate per-user page_table_tt before continue
                if page_table_tt is not None:
                    ttnn.deallocate(page_table_tt)

                # DEBUG: Force synchronization and cleanup after video request
                logger.info("  VIDEO DEBUG: Syncing device after video request...")
                try:
                    ttnn.synchronize_device(self.mesh_device)
                    logger.info("  VIDEO DEBUG: Device sync complete")
                except Exception as e:
                    logger.error(f"  VIDEO DEBUG: Device sync FAILED: {e}")
                    # Don't raise - continue to next request
                continue

            # Check for vLLM-style pre-processed images
            elif has_vllm_images and user_id < len(pixel_values) and pixel_values[user_id] is not None:
                # vLLM provides pre-processed pixel_values
                # Two formats:
                # 1. Demo-style VIDEO: pixel_values[0] is combined tensor [n_frames, 3, H, W]
                # 2. Old-style or IMAGE: pixel_values[user_id] is per-image tensor [n_crops, 3, H, W]

                pv = pixel_values[user_id]
                # Handle nested list structure from vLLM
                if isinstance(pv, list) and len(pv) > 0:
                    pv = pv[0]
                if isinstance(pv, torch.Tensor):
                    pv_tensor = pv
                elif hasattr(pv, "__array__"):
                    pv_tensor = torch.from_numpy(pv)
                else:
                    pv_tensor = torch.tensor(pv)

                # Detect video: Check multiple sources for video indicator
                # 1. Cached pooling with 3D shape [n_frames, N_out, K_pool] indicates video
                # 2. image_token_pooling parameter with 3D shape indicates video
                # 3. pixel_values shape [n_frames, 3, 378, 378] with n_frames >= 2
                cached_pooling = _image_token_pooling_cache.get("last")

                # Check image_token_pooling from parameter (vLLM path)
                param_pooling = None
                if image_token_pooling is not None and len(image_token_pooling) > user_id:
                    itp = image_token_pooling[user_id]
                    if isinstance(itp, list) and len(itp) > 0:
                        itp = itp[0]
                    if itp is not None:
                        if isinstance(itp, torch.Tensor):
                            param_pooling = itp
                        elif hasattr(itp, "__array__"):
                            import numpy as np

                            param_pooling = torch.from_numpy(np.array(itp))

                # Detect video from pooling shape (most reliable)
                has_video_pooling = (
                    cached_pooling is not None and cached_pooling.dim() == 3
                ) or (  # [n_frames, N_out, K_pool]
                    param_pooling is not None and param_pooling.dim() == 3
                )

                # Also check pixel_values shape as backup
                has_video_shape = (
                    pv_tensor.dim() == 4
                    and pv_tensor.shape[0] >= 2  # Multiple frames
                    and pv_tensor.shape[1] == 3  # RGB
                    and pv_tensor.shape[2] == 378
                    and pv_tensor.shape[3] == 378
                )

                is_video_input = has_video_pooling and has_video_shape

                logger.info(
                    f"  prefill_forward: has_video_pooling={has_video_pooling}, has_video_shape={has_video_shape}"
                )

                if is_video_input:
                    # VIDEO detected: pixel_values is [n_frames, 3, H, W]
                    n_frames = pv_tensor.shape[0]
                    logger.info(f"  prefill_forward: Detected VIDEO with {n_frames} frames")
                    logger.info(f"    pixel_values shape: {pv_tensor.shape}")

                    # Get pooling from param (vLLM path) or cache (demo path)
                    # Expected shape: [n_frames, N_out, K_pool]
                    # vLLM may flatten to 2D [n_tokens, k_pool], so reshape if needed
                    pooling = None
                    if param_pooling is not None:
                        pooling = param_pooling
                        logger.info(f"    Using param pooling shape: {pooling.shape}")
                    elif cached_pooling is not None:
                        pooling = cached_pooling
                        logger.info(f"    Using cached pooling shape: {pooling.shape}")

                    # Reshape 2D [n_tokens, k_pool] -> 3D [n_frames, n_out, k_pool]
                    if pooling is not None and pooling.dim() == 2:
                        n_tokens, k_pool = pooling.shape
                        n_out = n_tokens // n_frames
                        pooling = pooling.view(n_frames, n_out, k_pool)
                        logger.info(f"    Reshaped 2D pooling to 3D: {pooling.shape}")
                    elif pooling is None:
                        logger.warning(f"    No pooling found for video!")

                    # Keep pooling as [n_frames, N_out, K_pool] for data-parallel processing.
                    # DO NOT flatten to [1, n_frames*N_out, K_pool] - that routes to single-batch path!
                    # embed_image uses batch_size (dim 0) to decide between DP and single-batch paths.
                    if pooling is not None:
                        logger.info(f"    Pooling for prefill: {pooling.shape} (keeping 3D for DP routing)")
                else:
                    # Single image or old-style format
                    logger.info(f"  prefill_forward: Processing as IMAGE, pixel_values shape={pv_tensor.shape}")

                    # Get image_token_pooling - compute from image_grids
                    pooling = None

                    # Compute from image_grids - this is the reliable source
                    if image_grids is not None and len(image_grids) > user_id:
                        grid_data = image_grids[user_id]
                        logger.info(
                            f"  prefill_forward: image_grids[{user_id}] raw = {grid_data}, type={type(grid_data)}"
                        )
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

                # Run prefill with pre-processed image/video
                # CRITICAL: Disable traces for vision input - traces captured during warmup (text-only)
                # don't work correctly with vision-fused embeddings, producing garbled output.
                # Video path already uses use_trace=False and produces coherent output.
                logger.info(f"  IMAGE: Running prefill WITHOUT traces (vision incompatible with text traces)")
                user_tokens_img = tokens[user_id : user_id + 1, : prompt_lens[user_id]]

                # FIX: vLLM may strip chat template tokens
                first_tok = user_tokens_img[0, 0].item()
                if first_tok == 151645:
                    pass
                elif first_tok == 151644:
                    prefix = torch.tensor([[151645]], dtype=user_tokens_img.dtype)
                    user_tokens_img = torch.cat([prefix, user_tokens_img], dim=1)
                    logger.info(f"  IMAGE FIX: Prepended BOS, new shape={user_tokens_img.shape}")
                else:
                    prefix = torch.tensor([[151645, 151644, 872, 198]], dtype=user_tokens_img.dtype)
                    user_tokens_img = torch.cat([prefix, user_tokens_img], dim=1)
                    logger.info(f"  IMAGE FIX: Prepended full chat template, new shape={user_tokens_img.shape}")

                user_attn_mask = self._build_mm_prefill_attn_mask(
                    user_token_type_ids, user_hf_attention_mask, user_tokens_img.shape[1]
                )
                logits_ttnn, prefill_timing = self._run_prefill(
                    input_ids=user_tokens_img,
                    pixel_values=pv_tensor,
                    pooled_patches_idx=pooling,
                    use_trace=False,  # DISABLED for images - same as video path
                    use_vision_trace=False,  # Disabled for vLLM: variable multi-crop sizes
                    use_unified_trace=False,
                    page_table=page_table_tt,
                    user_id=user_id,
                    attn_mask=user_attn_mask,
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
                    user_tokens_pil = tokens[user_id : user_id + 1, : prompt_lens[user_id]]

                    # FIX: vLLM may strip chat template tokens
                    first_tok = user_tokens_pil[0, 0].item()
                    if first_tok == 151645:
                        pass
                    elif first_tok == 151644:
                        prefix = torch.tensor([[151645]], dtype=user_tokens_pil.dtype)
                        user_tokens_pil = torch.cat([prefix, user_tokens_pil], dim=1)
                        logger.info(f"  PIL IMAGE FIX: Prepended BOS, new shape={user_tokens_pil.shape}")
                    else:
                        prefix = torch.tensor([[151645, 151644, 872, 198]], dtype=user_tokens_pil.dtype)
                        user_tokens_pil = torch.cat([prefix, user_tokens_pil], dim=1)
                        logger.info(f"  PIL IMAGE FIX: Prepended full chat template, new shape={user_tokens_pil.shape}")

                    user_attn_mask = self._build_mm_prefill_attn_mask(
                        user_token_type_ids, user_hf_attention_mask, user_tokens_pil.shape[1]
                    )
                    logits_ttnn, prefill_timing = self._run_prefill(
                        input_ids=user_tokens_pil,
                        pixel_values=image_inputs["pixel_values"],
                        pooled_patches_idx=image_inputs["image_token_pooling"].unsqueeze(0),
                        use_trace=enable_trace,
                        use_vision_trace=True,  # Vision trace accuracy bug fixed
                        use_unified_trace=False,
                        page_table=page_table_tt,
                        user_id=user_id,
                        attn_mask=user_attn_mask,
                    )
                    original_seq_len = prefill_timing.get("original_seq_len", prompt_lens[user_id])
                else:
                    # Run prefill without image (text-only)
                    user_tokens_txt = tokens[user_id : user_id + 1, : prompt_lens[user_id]]

                    # FIX: vLLM may strip chat template tokens
                    first_tok = user_tokens_txt[0, 0].item()
                    if first_tok == 151645:
                        pass
                    elif first_tok == 151644:
                        prefix = torch.tensor([[151645]], dtype=user_tokens_txt.dtype)
                        user_tokens_txt = torch.cat([prefix, user_tokens_txt], dim=1)
                        logger.info(f"  TEXT FIX: Prepended BOS, new shape={user_tokens_txt.shape}")
                    else:
                        prefix = torch.tensor([[151645, 151644, 872, 198]], dtype=user_tokens_txt.dtype)
                        user_tokens_txt = torch.cat([prefix, user_tokens_txt], dim=1)
                        logger.info(f"  TEXT FIX: Prepended full chat template, new shape={user_tokens_txt.shape}")

                    user_attn_mask = self._build_mm_prefill_attn_mask(
                        user_token_type_ids, user_hf_attention_mask, user_tokens_txt.shape[1]
                    )
                    logits_ttnn, prefill_timing = self._run_prefill(
                        input_ids=user_tokens_txt,
                        pixel_values=None,
                        pooled_patches_idx=None,
                        use_trace=enable_trace,
                        use_vision_trace=False,
                        page_table=page_table_tt,
                        user_id=user_id,
                        attn_mask=user_attn_mask,
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
            # For chunked prefill, logits_torch is only the last chunk; use last_chunk_start offset.
            last_chunk_start = prefill_timing.get("last_chunk_start", 0)
            if logits_torch.dim() == 2:
                last_token_idx = original_seq_len - 1 - last_chunk_start
                last_token_logits = logits_torch[last_token_idx, :]
            else:
                last_token_logits = logits_torch  # Already [vocab_size]

            output_logits.append(last_token_logits.unsqueeze(0).unsqueeze(1))  # [1, 1, vocab_size]

            # Deallocate per-user page_table_tt at end of iteration
            if page_table_tt is not None:
                ttnn.deallocate(page_table_tt)

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
        Run decode forward pass for Molmo2 with true batched processing.

        Uses vLLM's start_pos (per-request positions) to create batch-sized
        current_pos and rot_mat_idxs tensors, enabling true parallel decode
        for multiple concurrent requests.

        Args:
            tokens: Current token IDs [batch_size, 1]
            start_pos: Per-request positions from vLLM [batch_size]
            page_table: Page table for paged attention [batch_size, num_blocks]
            kv_cache: KV cache tensors
            enable_trace: Whether to use tracing (disabled for batch>1)
            read_from_device: Whether to read output from device
            sampling_params: Sampling parameters (not used)

        Returns:
            Logits tensor [batch_size, 1, vocab_size]
        """
        batch_size = tokens.shape[0]

        logger.info(
            f"decode_forward called: batch_size={batch_size}, tokens.shape={tokens.shape}, "
            f"start_pos.shape={start_pos.shape}, page_table.shape={page_table.shape if page_table is not None else None}"
        )

        # For DP>1, delegate to the per-replica decode path
        if self.data_parallel > 1:
            return self._decode_forward_dp(tokens, start_pos, page_table, kv_cache, enable_trace, read_from_device)

        # --- Single-replica (DP=1) decode path ---

        # Prepare batch-sized decode inputs using vLLM's per-request positions
        token_id_ttnn, current_pos_tt, rot_mat_idxs_tt, page_table_tt = self.prepare_decode_inputs(
            tokens, start_pos, page_table
        )

        # Embed tokens
        hidden_states = self.model.text_model.embed_tokens(token_id_ttnn)
        ttnn.deallocate(token_id_ttnn)

        # Check if we have a captured decode trace and should use it
        # IMPORTANT: Tracing only works for batch_size=1 with scalar positions
        # For batched decode (batch_size>1), we must use non-traced path
        has_trace_id = hasattr(self, "decode_trace_id") and self.decode_trace_id is not None
        has_trace_tensors = hasattr(self, "decode_trace_tensors") and self.decode_trace_tensors is not None

        # Disable trace for batched decode - trace was captured with scalar positions
        use_traced_decode = enable_trace and has_trace_id and has_trace_tensors and batch_size == 1

        # Log only on first decode to avoid spam
        if not hasattr(self, "_logged_trace_status"):
            logger.info(
                f"decode_forward trace status: enable_trace={enable_trace}, has_trace_id={has_trace_id}, "
                f"has_trace_tensors={has_trace_tensors}, batch_size={batch_size}, use_traced_decode={use_traced_decode}"
            )
            self._logged_trace_status = True

        if use_traced_decode and batch_size == 1:
            # For batch_size=1, we can use traced decode with the existing scalar path
            # Update scalar position tensors from vLLM's start_pos
            new_pos = ttnn.from_torch(
                start_pos[:1].int(),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy(new_pos, self.current_pos)
            ttnn.copy(new_pos, self.rot_mat_idxs)
            ttnn.deallocate(new_pos)

            # Copy hidden states into trace input tensor
            ttnn.copy(hidden_states, self.decode_trace_tensors["hidden_states"])
            ttnn.deallocate(hidden_states)

            # Copy page_table to trace tensor if provided
            if page_table_tt is not None and "page_table" in self.decode_trace_tensors:
                ttnn.copy(page_table_tt, self.decode_trace_tensors["page_table"])
            if page_table_tt is not None:
                ttnn.deallocate(page_table_tt)

            # Deallocate batch tensors (not needed for traced path)
            ttnn.deallocate(current_pos_tt)
            ttnn.deallocate(rot_mat_idxs_tt)

            # Execute the captured trace (blocking so logits are complete before read)
            ttnn.execute_trace(self.mesh_device, self.decode_trace_id, cq_id=0, blocking=True)

            # Get output from trace output tensor
            logits_ttnn = self.decode_trace_output
        else:
            # Non-traced batched decode path - use batch-sized position tensors
            # Let forward_decode compute rot_mats internally based on batch size
            # (batch>1 uses get_rot_mats_decode_batched, batch=1 uses traced)

            # Use vLLM's KV cache if provided (paged), otherwise fall back to internal cache
            if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
                decode_kv_cache = kv_cache[0]  # Use first data parallel shard
            else:
                decode_kv_cache = self.kv_caches

            # Run forward_decode with batch-sized current_pos
            # Pass rot_mat_idxs so forward_decode picks correct rot_mats method
            logits_ttnn = self.model.text_model.forward_decode(
                hidden_states=hidden_states,
                kv_caches=decode_kv_cache,
                current_pos=current_pos_tt,  # Batch-sized from vLLM start_pos
                rot_mat_idxs=rot_mat_idxs_tt,  # Let forward_decode compute batched rot_mats
                page_table=page_table_tt,
            )

            # Deallocate intermediate tensors
            ttnn.deallocate(hidden_states)
            ttnn.deallocate(current_pos_tt)
            ttnn.deallocate(rot_mat_idxs_tt)
            if page_table_tt is not None:
                ttnn.deallocate(page_table_tt)

        # NOTE: No position increment needed - vLLM tracks positions externally

        # During trace capture, we cannot read from device
        if not read_from_device:
            vocab_size = 152064  # Molmo2 vocab size
            return torch.zeros(batch_size, 1, vocab_size)

        # Synchronize device before reading
        ttnn.synchronize_device(self.mesh_device)

        # Convert logits to torch
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        logits = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0]

        # Deallocate logits_ttnn (only safe when trace is disabled)
        if not use_traced_decode:
            ttnn.deallocate(logits_ttnn)

        # Reshape logits for vLLM: [batch_size, 1, vocab_size]
        # TTNN output: [1, 1, padded_batch, vocab] but to_torch may drop dims
        # Actual observed shapes after to_torch:
        #   - 4D: [1, 1, padded_batch, vocab] (rare)
        #   - 3D: [1, padded_batch, vocab] (common - to_torch drops a dim)
        #   - 2D: [padded_batch, vocab] (possible)
        if logits.dim() == 4:
            # [1, 1, padded_batch, vocab] -> [batch, 1, vocab]
            logits = logits[0, 0, :batch_size, :]
            logits = logits.unsqueeze(1)
        elif logits.dim() == 3:
            # [1, padded_batch, vocab] -> [batch, 1, vocab]
            # Batch is in position 1 (the "seq_len" position), NOT position 0
            logits = logits[0, :batch_size, :]  # [batch, vocab]
            logits = logits.unsqueeze(1)  # [batch, 1, vocab]
        elif logits.dim() == 2:
            logits = logits[:batch_size, :].unsqueeze(1)
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(1)

        return logits

    def _decode_forward_dp(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: Optional[torch.Tensor],
        kv_cache,
        enable_trace: bool,
        read_from_device: bool,
    ) -> torch.Tensor:
        """
        Decode forward pass for DP>1: chunk the batch across DP replicas and
        run each chunk on its corresponding model replica.

        Tokens [B, 1] are split into data_parallel chunks along dim=0.
        Each chunk [B//DP, 1] runs on replica dp_idx with kv_cache[dp_idx].
        Logits from all replicas are concatenated and returned.
        """
        batch_size = tokens.shape[0]
        vocab_size = 152064  # Molmo2 vocab size

        # Chunk inputs along batch dimension across DP replicas
        tokens_chunks = tokens.chunk(self.data_parallel, dim=0)
        pos_chunks = start_pos.chunk(self.data_parallel, dim=0)
        pt_chunks = (
            page_table.chunk(self.data_parallel, dim=0) if page_table is not None else [None] * self.data_parallel
        )

        all_logits = []

        for dp_idx, (tok_chunk, pos_chunk, pt_chunk) in enumerate(zip(tokens_chunks, pos_chunks, pt_chunks)):
            chunk_batch = tok_chunk.shape[0]

            # Switch to this replica (saves previous replica state automatically)
            self._set_active_replica(dp_idx)

            # Use the per-replica KV cache
            if kv_cache is not None and dp_idx < len(kv_cache) and kv_cache[dp_idx] is not None:
                self.kv_caches = kv_cache[dp_idx]

            # Prepare decode inputs for this chunk on the replica's device
            token_id_ttnn, current_pos_tt, rot_mat_idxs_tt, page_table_tt = self.prepare_decode_inputs(
                tok_chunk, pos_chunk, pt_chunk
            )

            # Embed tokens on this replica
            hidden_states = self.model.text_model.embed_tokens(token_id_ttnn)
            ttnn.deallocate(token_id_ttnn)

            # Check per-replica trace availability
            has_trace_id = self.decode_trace_id is not None
            has_trace_tensors = self.decode_trace_tensors is not None
            use_traced_decode = enable_trace and has_trace_id and has_trace_tensors and chunk_batch == 1

            if use_traced_decode:
                new_pos = ttnn.from_torch(
                    pos_chunk[:1].int(),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                ttnn.copy(new_pos, self.current_pos)
                ttnn.copy(new_pos, self.rot_mat_idxs)
                ttnn.deallocate(new_pos)

                ttnn.copy(hidden_states, self.decode_trace_tensors["hidden_states"])
                ttnn.deallocate(hidden_states)

                if page_table_tt is not None and "page_table" in self.decode_trace_tensors:
                    ttnn.copy(page_table_tt, self.decode_trace_tensors["page_table"])
                if page_table_tt is not None:
                    ttnn.deallocate(page_table_tt)

                ttnn.deallocate(current_pos_tt)
                ttnn.deallocate(rot_mat_idxs_tt)

                ttnn.execute_trace(self.mesh_device, self.decode_trace_id, cq_id=0, blocking=True)
                logits_ttnn = self.decode_trace_output
            else:
                logits_ttnn = self.model.text_model.forward_decode(
                    hidden_states=hidden_states,
                    kv_caches=self.kv_caches,
                    current_pos=current_pos_tt,
                    rot_mat_idxs=rot_mat_idxs_tt,
                    page_table=page_table_tt,
                )
                ttnn.deallocate(hidden_states)
                ttnn.deallocate(current_pos_tt)
                ttnn.deallocate(rot_mat_idxs_tt)
                if page_table_tt is not None:
                    ttnn.deallocate(page_table_tt)

            # Save replica state (includes updated kv_caches reference)
            self._save_active_replica(dp_idx)

            if not read_from_device:
                all_logits.append(torch.zeros(chunk_batch, 1, vocab_size))
                if not use_traced_decode:
                    ttnn.deallocate(logits_ttnn)
                continue

            # Read logits from this replica
            ttnn.synchronize_device(self.mesh_device)
            mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            logits_i = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0]

            if not use_traced_decode:
                ttnn.deallocate(logits_ttnn)

            # Reshape to [chunk_batch, 1, vocab_size]
            if logits_i.dim() == 4:
                logits_i = logits_i[0, 0, :chunk_batch, :].unsqueeze(1)
            elif logits_i.dim() == 3:
                logits_i = logits_i[0, :chunk_batch, :].unsqueeze(1)
            elif logits_i.dim() == 2:
                logits_i = logits_i[:chunk_batch, :].unsqueeze(1)
            elif logits_i.dim() == 1:
                logits_i = logits_i.unsqueeze(0).unsqueeze(1)

            all_logits.append(logits_i)

        if not read_from_device:
            return torch.zeros(batch_size, 1, vocab_size)

        return torch.cat(all_logits, dim=0)

    def allocate_kv_cache(
        self,
        kv_cache_shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_layers: int,
    ) -> List[List[ttnn.Tensor]]:
        """
        Allocate KV cache for Molmo2 text model.

        For DP>1, allocates a separate KV cache per DP replica (submesh).
        Returns list indexed by [dp_idx][layer][k_or_v].

        Args:
            kv_cache_shape: Shape of KV cache tensors
            dtype: Data type for KV cache
            num_layers: Number of transformer layers

        Returns:
            List of per-DP-replica KV cache lists: [dp_idx][layer][k_or_v]
        """
        return allocate_molmo2_kv_cache(
            kv_cache_shape=kv_cache_shape,
            dtype=dtype,
            num_layers=num_layers,
            submesh_devices=[model.mesh_device for model in self.models],
            tt_cache_path=self.cache_path,
        )

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace: bool,
        can_sample_on_device: bool = False,
        non_greedy_decoding_on_device: bool = False,
        num_blocks: int = 64,
        max_seq_len: int = 8192,
        **kwargs,
    ) -> None:
        """
        Warmup prefill path for Molmo2.

        For DP>1, iterates over all DP replicas and warms up each one separately
        using its per-replica KV cache.

        Args:
            kv_cache: KV cache tensors from vLLM (paged format) [dp_idx][layer][k_or_v]
            enable_trace: Whether to capture prefill trace
            can_sample_on_device: Whether sampling can happen on device
            non_greedy_decoding_on_device: Whether non-greedy decoding is on device
            num_blocks: Number of KV cache blocks
            max_seq_len: Maximum sequence length (buckets exceeding this are skipped)
        """
        for dp_idx in range(self.data_parallel):
            logger.info(f"Warmup prefill: starting replica {dp_idx}/{self.data_parallel}")
            self._set_active_replica(dp_idx)

            # Select this replica's KV cache: kv_cache[dp_idx] is [layer][k_or_v]
            per_dp_kv = [kv_cache[dp_idx]] if (kv_cache and dp_idx < len(kv_cache)) else kv_cache

            self._warmup_prefill_single_replica(per_dp_kv, enable_trace, num_blocks, max_seq_len)
            self._save_active_replica(dp_idx)
            logger.info(f"Warmup prefill: replica {dp_idx} done")

    def _warmup_prefill_single_replica(
        self,
        kv_cache,
        enable_trace: bool,
        num_blocks: int = 64,
        max_seq_len: int = 8192,
    ) -> None:
        """
        Warmup prefill for the currently active DP replica.

        kv_cache must be [per_replica_kv_cache] (list with one element that is
        [layer][k_or_v]).  self.model / self.mesh_device / self.kv_caches must
        already be set to the target replica via _set_active_replica().
        """
        # Always run vision compile warmup, even when traces are disabled
        # This compiles the vision + prefill path during initialization, not during inference
        self._warmup_vision_compile(kv_cache, num_blocks, max_seq_len)

        if not enable_trace:
            logger.info("Prefill trace disabled - skipping trace capture (vision already compiled)")
            return

        logger.info("Warmup: Capturing prefill traces (two-phase approach)...")

        # Use vLLM's paged KV cache if provided, otherwise fall back to internal cache
        if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
            # kv_cache[0] is this replica's [layer][k_or_v] cache
            warmup_kv_cache = kv_cache[0]
            # CRITICAL: Set generator's kv_caches to vLLM's cache so run_prefill uses it
            self.kv_caches = warmup_kv_cache
            logger.info(f"Warmup: Using vLLM paged KV cache with {len(warmup_kv_cache)} layers")

            # CRITICAL: Infer actual num_blocks from KV cache shape
            # KV cache shape is [num_blocks, 1, num_kv_heads, head_dim]
            # The num_blocks parameter default (64) is often wrong - vLLM allocates more blocks
            if len(warmup_kv_cache) > 0 and warmup_kv_cache[0] is not None:
                k_cache = warmup_kv_cache[0][0]  # First layer's K cache
                actual_num_blocks = k_cache.shape[0]
                if actual_num_blocks != num_blocks:
                    logger.info(
                        f"Warmup: Overriding num_blocks from {num_blocks} to {actual_num_blocks} (from KV cache shape)"
                    )
                    num_blocks = actual_num_blocks
        else:
            # Fall back to internal generator cache (for non-paged mode)
            if not hasattr(self, "kv_caches") or self.kv_caches is None:
                logger.info("Warmup: Initializing internal KV cache...")
                self.init_kv_cache()
            warmup_kv_cache = self.kv_caches
            logger.info("Warmup: Using internal KV cache (non-paged mode)")

        # Create warmup page_table with sequential block mapping
        # This is used to initialize page_table trace tensors before capture
        warmup_page_table_torch = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
        warmup_page_table = ttnn.from_torch(
            warmup_page_table_torch,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Capture prefill traces for multiple bucket sizes
        # Include 2048 and 4096 for video requests (video with HF adaptive sampling can exceed 4000 tokens)
        warmup_bucket_sizes = [128, 256, 512, 1024, 2048, 4096]
        hidden_dim = 4096

        if not hasattr(self, "prefill_traces"):
            self.prefill_traces = {}

        # PHASE 1: Allocate ALL trace tensors BEFORE capturing any traces
        # This prevents memory corruption from interleaved allocation/capture
        all_trace_tensors = {}
        valid_buckets = []

        for warmup_seq_len in warmup_bucket_sizes:
            if warmup_seq_len > max_seq_len:
                logger.info(f"Warmup: Skipping bucket {warmup_seq_len} (exceeds max_seq_len {max_seq_len})")
                continue

            valid_buckets.append(warmup_seq_len)
            logger.info(
                f"Warmup: Phase 1 - Allocating prefill trace tensors for seq_len={warmup_seq_len}, num_blocks={num_blocks}"
            )
            prefill_trace_tensors = self._allocate_prefill_trace_tensors(
                warmup_seq_len, hidden_dim, max_num_blocks=num_blocks
            )

            # Initialize page_table with sequential block mapping
            # This is CRITICAL - uninitialized page_table causes garbage output
            if "page_table" in prefill_trace_tensors:
                ttnn.copy(warmup_page_table, prefill_trace_tensors["page_table"])

            all_trace_tensors[warmup_seq_len] = prefill_trace_tensors

        logger.info(f"Warmup: Phase 1 complete - allocated tensors for {len(valid_buckets)} buckets")

        # PHASE 2: Capture traces for each bucket (tensors already allocated)
        for warmup_seq_len in valid_buckets:
            prefill_trace_tensors = all_trace_tensors[warmup_seq_len]

            # Capture prefill trace with appropriate KV cache
            logger.info(f"Warmup: Phase 2 - Capturing prefill trace for seq_len={warmup_seq_len}...")
            trace_id, trace_output = self._capture_prefill_trace(prefill_trace_tensors, kv_cache=warmup_kv_cache)

            # Store trace for reuse - run_prefill will find this
            self.prefill_traces[warmup_seq_len] = (trace_id, prefill_trace_tensors, trace_output)
            logger.info(f"Warmup: Prefill trace captured for seq_len={warmup_seq_len}")

        # Clean up warmup page_table
        ttnn.deallocate(warmup_page_table)

        logger.info(f"Warmup: Prefill traces captured for buckets: {valid_buckets}")

    def _warmup_vision_compile(self, kv_cache, num_blocks: int = 64, max_seq_len: int = 4096) -> None:
        """
        Compile the vision encoder and text model prefill for ALL bucket sizes.

        This ensures:
        1. Vision encoder (ViT + pooling + projector) is compiled once
        2. Text model forward is compiled for ALL bucket sizes

        Args:
            kv_cache: KV cache tensors from vLLM (paged format)
            num_blocks: Number of KV cache blocks
            max_seq_len: Maximum sequence length (buckets exceeding this are skipped)
        """
        logger.info("Warmup: Starting vision compile warmup for ALL buckets...")

        # Use vLLM's paged KV cache
        if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
            warmup_kv_cache = kv_cache[0]
            self.kv_caches = warmup_kv_cache

            # Get actual num_blocks from KV cache shape
            if len(warmup_kv_cache) > 0 and warmup_kv_cache[0] is not None:
                k_cache = warmup_kv_cache[0][0]
                actual_num_blocks = k_cache.shape[0]
                if actual_num_blocks != num_blocks:
                    logger.info(f"Warmup vision: Overriding num_blocks from {num_blocks} to {actual_num_blocks}")
                    num_blocks = actual_num_blocks
        else:
            warmup_kv_cache = self.kv_caches
            logger.info("Warmup vision: Using internal KV cache")

        # Initialize compiled buckets tracker
        if not hasattr(self, "_prefill_compiled_buckets"):
            self._prefill_compiled_buckets = set()

        # Create page_table for warmup (use first blocks)
        warmup_page_table_torch = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
        warmup_page_table = ttnn.from_torch(
            warmup_page_table_torch,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # ========== PHASE 1: Compile vision encoder with dummy image ==========
        logger.info("Warmup vision: Phase 1 - Compiling vision encoder...")
        try:
            # Create dummy image (378x378 RGB)
            image_size = 378
            dummy_pixel_values = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)

            # Create dummy pooling indices for a single image
            # 27x27 = 729 patches -> 14x14 = 196 pooled tokens
            import numpy as np

            pooled_h, pooled_w = 14, 14
            pool_h, pool_w = 2, 2
            patches_per_side = 27
            resize_idx = np.arange(patches_per_side * patches_per_side).reshape(patches_per_side, patches_per_side)
            resize_idx = arange_for_pooling(resize_idx, pool_h, pool_w)
            pooling_idx = resize_idx.reshape(-1, pool_h * pool_w)[: pooled_h * pooled_w, :]
            dummy_pooling = torch.from_numpy(pooling_idx).long().unsqueeze(0)  # [1, 196, 4]

            # Create dummy tokens with 196 <im_patch> placeholders (token ID 151938)
            # This matches the 196 vision tokens from pooling (14x14 pooled patches)
            # Format: [BOS, im_patch*196, text_tokens...]
            image_patch_id = 151938  # <im_patch> token
            num_vision_tokens = 196  # 14x14 pooled patches
            dummy_tokens = torch.cat(
                [
                    torch.tensor([[1]], dtype=torch.long),  # BOS token
                    torch.full((1, num_vision_tokens), image_patch_id, dtype=torch.long),  # 196 image patches
                    torch.tensor([[2277, 374]], dtype=torch.long),  # Some text tokens
                ],
                dim=1,
            )  # [1, 199] tokens
            logger.info(
                f"Warmup vision: Created dummy tokens with {num_vision_tokens} image patches, shape={dummy_tokens.shape}"
            )

            # Run vision encoder to compile it
            vision_start = time.time()
            _ = self._prepare_text_inputs(
                input_ids=dummy_tokens,
                pixel_values=dummy_pixel_values,
                pooled_patches_idx=dummy_pooling,
            )
            ttnn.synchronize_device(self.mesh_device)
            vision_time = (time.time() - vision_start) * 1000
            logger.info(f"Warmup vision: Vision encoder compiled in {vision_time:.2f}ms")

        except Exception as e:
            logger.warning(f"Warmup vision: Vision encoder compile failed: {e}")

        # ========== PHASE 1b: Compile video ViT + pooling for all chunk sizes ==========
        # Two sources of cold compilation for video requests:
        #   1. ViT chunks: max_frames_per_chunk=8, so last chunk c = batch_size % 8 ∈ {1..8}
        #      Each unique c produces a unique TTNN input shape [c*729, 1152].
        #   2. Pooling chunks: max_frames_per_pool_chunk=16, so last pool chunk
        #      p = batch_size % 16 ∈ {1..16} produces unique output shapes.
        #
        # Running embed_image_chunked with batch_size = 1..16 covers every possible
        # (ViT last-chunk size, pool last-chunk size) combination, eliminating all
        # cold compiles during inference.
        logger.info("Warmup vision: Phase 1b - Compiling video ViT+pool for batch_sizes 1-16 (k_pool=9)...")
        max_vit_chunk = 8  # max_frames_per_chunk default in _embed_image_chunked
        max_pool_chunk = 16  # max_frames_per_pool_chunk default in pool_and_project_chunked_ttnn
        pool_h_vid, pool_w_vid = 3, 3  # video uses 3x3 pooling (k_pool=9)
        patches_per_side = 27
        n_out_vid = (patches_per_side // pool_h_vid) * (patches_per_side // pool_w_vid)  # 81
        k_pool_vid = pool_h_vid * pool_w_vid  # 9

        for b in range(1, max_pool_chunk + 1):
            try:
                compile_start = time.time()
                dummy_vid_frames = torch.zeros((b, 3, 378, 378), dtype=torch.float32)
                # Zero pooled_patches_idx - indices are clamped inside the model so zeros are safe
                dummy_vid_pooling = torch.zeros((b, n_out_vid, k_pool_vid), dtype=torch.long)

                vis_emb, _ = self.model._embed_image_chunked(
                    pixel_values=dummy_vid_frames,
                    pooled_patches_idx=dummy_vid_pooling,
                    max_frames_per_chunk=max_vit_chunk,
                )
                if vis_emb is not None:
                    ttnn.deallocate(vis_emb)
                ttnn.synchronize_device(self.mesh_device)
                compile_time = (time.time() - compile_start) * 1000
                logger.info(f"Warmup vision: Video batch_size={b} compiled in {compile_time:.2f}ms")
            except Exception as e:
                logger.warning(f"Warmup vision: Video batch_size={b} compile failed: {e}")

        # ========== PHASE 2: Compile text_model.forward for ALL bucket sizes ==========
        logger.info("Warmup vision: Phase 2 - Compiling text model for all buckets...")
        warmup_bucket_sizes = [128, 256, 512, 1024, 2048, 4096]
        hidden_dim = 4096

        for bucket_size in warmup_bucket_sizes:
            if bucket_size > max_seq_len:
                logger.info(f"Warmup vision: Skipping bucket {bucket_size} (exceeds max_seq_len {max_seq_len})")
                continue

            try:
                compile_start = time.time()

                # Create dummy hidden states for this bucket size
                dummy_hidden = torch.zeros((1, bucket_size, hidden_dim), dtype=torch.bfloat16)
                dummy_hidden_ttnn = ttnn.from_torch(
                    dummy_hidden,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )

                # Get rotation matrices for this bucket size
                rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(bucket_size, start_pos=0)

                # Run forward to compile
                _, _ = self.model.text_model.forward(
                    hidden_states=dummy_hidden_ttnn,
                    start_pos=0,
                    attn_mask=None,
                    kv_caches=warmup_kv_cache,
                    rot_mats=rot_mats,
                    page_table=warmup_page_table,
                )
                ttnn.synchronize_device(self.mesh_device)

                compile_time = (time.time() - compile_start) * 1000
                self._prefill_compiled_buckets.add(bucket_size)
                logger.info(f"Warmup vision: Compiled bucket {bucket_size} in {compile_time:.2f}ms")

                # Cleanup
                ttnn.deallocate(rot_mats[0])
                ttnn.deallocate(rot_mats[1])
                ttnn.deallocate(dummy_hidden_ttnn)

            except Exception as e:
                logger.warning(f"Warmup vision: Failed to compile bucket {bucket_size}: {e}")

        # ========== PHASE 3: Compile chunked prefill path (chunk_start_idx != None) ==========
        # This compiles chunked_scaled_dot_product_attention used for sequences > _MAX_PREFILL_CHUNK_SIZE.
        # Compile one pass: chunk_size = _MAX_PREFILL_CHUNK_SIZE tokens, chunk_start_idx=0.
        chunk_size = self._MAX_PREFILL_CHUNK_SIZE
        block_size = self._BLOCK_SIZE
        blocks_per_chunk = chunk_size // block_size  # 4096 // 64 = 64 blocks
        if chunk_size in warmup_bucket_sizes:
            logger.info("Warmup vision: Phase 3 - Compiling chunked prefill path (chunk_start_idx=0)...")
            try:
                compile_start = time.time()
                chunk_hidden = torch.zeros((1, chunk_size, 4096), dtype=torch.bfloat16)
                chunk_hidden_ttnn = ttnn.from_torch(
                    chunk_hidden,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                rot_mats_chunk = self.model.text_model.rotary_setup.get_rot_mats_prefill(chunk_size, start_pos=0)
                # Full page table (for reading prev KV) — reuse warmup_page_table shape
                full_pt = ttnn.from_torch(
                    torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                # Chunk page table (for writing new KV for this chunk)
                chunk_pt = ttnn.from_torch(
                    torch.arange(blocks_per_chunk, dtype=torch.int32).unsqueeze(0),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                _, _ = self.model.text_model.forward(
                    hidden_states=chunk_hidden_ttnn,
                    start_pos=0,
                    attn_mask=None,
                    kv_caches=warmup_kv_cache,
                    rot_mats=rot_mats_chunk,
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx=0,
                )
                ttnn.synchronize_device(self.mesh_device)
                ttnn.deallocate(rot_mats_chunk[0])
                ttnn.deallocate(rot_mats_chunk[1])
                ttnn.deallocate(chunk_hidden_ttnn)
                ttnn.deallocate(full_pt)
                ttnn.deallocate(chunk_pt)
                compile_time = (time.time() - compile_start) * 1000
                logger.info(f"Warmup vision: Chunked prefill path compiled in {compile_time:.2f}ms")
            except Exception as e:
                logger.warning(f"Warmup vision: Failed to compile chunked prefill path: {e}")

        # Cleanup
        ttnn.deallocate(warmup_page_table)
        logger.info(f"Warmup vision: Compiled buckets: {sorted(self._prefill_compiled_buckets)}")

    def warmup_model_decode(
        self,
        kv_cache,
        enable_trace: bool,
        max_batch_size: int = 1,
        num_blocks: int = 64,
        **kwargs,
    ) -> None:
        """
        Warmup decode path for Molmo2.

        For DP>1, iterates over all DP replicas and warms up each one separately
        using its per-replica KV cache.

        Args:
            kv_cache: KV cache tensors from vLLM (paged format) [dp_idx][layer][k_or_v]
            enable_trace: Whether to capture decode trace
            max_batch_size: Maximum batch size (total across all DP replicas)
            num_blocks: Number of KV cache blocks
        """
        if not enable_trace:
            logger.info("Decode trace disabled - skipping warmup_model_decode")
            return

        for dp_idx in range(self.data_parallel):
            logger.info(f"Warmup decode: starting replica {dp_idx}/{self.data_parallel}")
            self._set_active_replica(dp_idx)

            # Select this replica's KV cache
            per_dp_kv = [kv_cache[dp_idx]] if (kv_cache and dp_idx < len(kv_cache)) else kv_cache

            self._warmup_decode_single_replica(per_dp_kv, num_blocks)
            self._save_active_replica(dp_idx)
            logger.info(f"Warmup decode: replica {dp_idx} done")

    def _warmup_decode_single_replica(self, kv_cache, num_blocks: int = 64) -> None:
        """
        Capture decode trace for the currently active DP replica.

        kv_cache must be [per_replica_kv_cache] (list with one element that is
        [layer][k_or_v]).  self.model / self.mesh_device must already be set to
        the target replica via _set_active_replica().
        """
        logger.info("Warmup: Capturing decode trace...")

        # Use vLLM's paged KV cache if provided, otherwise fall back to internal cache
        if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
            warmup_kv_cache = kv_cache[0]  # This replica's [layer][k_or_v] cache
            self.kv_caches = warmup_kv_cache
            logger.info(f"Warmup: Using vLLM paged KV cache with {len(warmup_kv_cache)} layers")

            if len(warmup_kv_cache) > 0 and warmup_kv_cache[0] is not None:
                k_cache = warmup_kv_cache[0][0]
                actual_num_blocks = k_cache.shape[0]
                if actual_num_blocks != num_blocks:
                    logger.info(f"Warmup decode: Overriding num_blocks from {num_blocks} to {actual_num_blocks}")
                    num_blocks = actual_num_blocks
        else:
            if not hasattr(self, "kv_caches") or self.kv_caches is None:
                logger.info("Warmup: Initializing internal KV cache...")
                self.init_kv_cache()
            warmup_kv_cache = self.kv_caches
            logger.info("Warmup: Using internal KV cache (non-paged mode)")

        # Initialize position tensors if not present
        if not hasattr(self, "current_pos") or self.current_pos is None:
            logger.info("Warmup: Initializing position tensors...")
            self._init_decode_position_tensors()

        # Create warmup page_table with sequential block mapping
        warmup_page_table_torch = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
        warmup_page_table = ttnn.from_torch(
            warmup_page_table_torch,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Allocate decode trace tensors with num_blocks matching the KV cache
        hidden_dim = 4096
        logger.info(f"Warmup: Allocating decode trace tensors (num_blocks={num_blocks})...")
        decode_trace_tensors = self._allocate_decode_trace_tensors(hidden_dim, max_num_blocks=num_blocks)

        # Initialize page_table with sequential block mapping
        # This is CRITICAL - uninitialized page_table causes garbage output
        if "page_table" in decode_trace_tensors:
            ttnn.copy(warmup_page_table, decode_trace_tensors["page_table"])

        # Create dummy hidden states for trace capture
        dummy_hidden = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(dummy_hidden, decode_trace_tensors["hidden_states"])
        ttnn.deallocate(dummy_hidden)

        # Capture decode trace with appropriate KV cache
        logger.info("Warmup: Capturing decode trace...")
        trace_id, trace_output = self._capture_decode_trace(decode_trace_tensors, kv_cache=warmup_kv_cache)

        # Store trace for reuse in this replica's state
        self.decode_trace_id = trace_id
        self.decode_trace_tensors = decode_trace_tensors
        self.decode_trace_output = trace_output

        # Clean up warmup page_table
        ttnn.deallocate(warmup_page_table)

        logger.info("Warmup: Decode trace captured")

    def _init_decode_position_tensors(self) -> None:
        """Initialize position tensors for decode tracing."""
        # Initialize current_pos tensor on device
        current_pos_torch = torch.tensor([0], dtype=torch.int32)
        self.current_pos = ttnn.from_torch(
            current_pos_torch,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Initialize rot_mat_idxs tensor on device using proper allocation
        # Must match the shape expected by reset_kv_cache: [1, batch_size padded to 32]
        self.rot_mat_idxs = self.model.text_model.rotary_setup.allocate_decode_rot_idxs(initial_pos=0)

        self.decode_position = 0

    def prepare_decode_inputs(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: Optional[torch.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        """
        Prepare decode inputs for batched vLLM inference.

        Creates batch-sized current_pos and rot_mat_idxs tensors from vLLM's
        start_pos (per-request positions), similar to tt_transformers model.py:403.

        Args:
            tokens: Current token IDs [batch_size, 1]
            start_pos: Per-request positions from vLLM [batch_size]
            page_table: Page table for paged attention [batch_size, num_blocks]

        Returns:
            Tuple of (tokens_tt, current_pos_tt, rot_mat_idxs_tt, page_table_tt)
        """
        batch_size = tokens.shape[0]

        # Reshape tokens for embed_tokens which expects [1, seq_len]
        # vLLM sends [batch_size, 1], we need [1, padded_batch]
        # Following tt_transformers model.py:426 approach
        tokens_flat = tokens.view(-1)  # [batch_size, 1] -> [batch_size]

        # Pad to multiple of 32 for tile alignment
        pad_size = ((batch_size + 31) // 32) * 32 - batch_size
        if pad_size > 0:
            tokens_padded = torch.nn.functional.pad(tokens_flat, (0, pad_size), value=0)
        else:
            tokens_padded = tokens_flat

        # Reshape to [1, padded_batch] for embed_tokens
        tokens_reshaped = tokens_padded.unsqueeze(0)  # [padded_batch] -> [1, padded_batch]

        # Convert tokens to device
        tokens_tt = ttnn.from_torch(
            tokens_reshaped,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Create batch-sized current_pos from vLLM's start_pos
        # Ensure positions are non-negative (vLLM can pass -1 for invalid)
        current_pos_clipped = torch.maximum(start_pos, torch.tensor(0, dtype=start_pos.dtype))

        # Pad to multiple of 32 for tile alignment
        pad_size = ((batch_size + 31) // 32) * 32 - batch_size
        if pad_size > 0:
            current_pos_padded = torch.nn.functional.pad(current_pos_clipped, (0, pad_size), value=0)
        else:
            current_pos_padded = current_pos_clipped

        # CRITICAL: current_pos must be 1D [padded_batch] for paged_update_cache
        # (paged_update_cache expects update_idxs with batch elements, not [1, batch])
        current_pos_tt = ttnn.from_torch(
            current_pos_padded,  # Shape: [padded_batch] - 1D tensor
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Create batch-sized rot_mat_idxs from positions (same values as current_pos)
        rot_mat_idxs_tt = ttnn.from_torch(
            current_pos_padded.reshape(1, -1),  # Shape: [1, padded_batch]
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Convert page_table to device if provided
        page_table_tt = None
        if page_table is not None:
            # Slice to actual batch size
            page_table_sliced = page_table[:batch_size]
            page_table_tt = ttnn.from_torch(
                page_table_sliced,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return tokens_tt, current_pos_tt, rot_mat_idxs_tt, page_table_tt

    def _allocate_prefill_trace_tensors(self, seq_len: int, hidden_dim: int = 4096, max_num_blocks: int = 64) -> dict:
        """
        Pre-allocate all tensors needed for traced prefill.

        Args:
            seq_len: Sequence length for this trace bucket
            hidden_dim: Hidden dimension (default 4096 for Molmo2-8B)
            max_num_blocks: Maximum number of blocks per sequence for page_table

        Returns:
            Dict with allocated trace tensors
        """
        # Allocate hidden states input tensor
        hidden_states_shape = [1, 1, seq_len, hidden_dim]
        trace_hidden_states = ttnn.allocate_tensor_on_device(
            ttnn.Shape(hidden_states_shape),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-compute rotation matrices (these will be used during trace)
        rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)

        # Allocate rot_mats tensors (we'll copy into these)
        trace_cos = ttnn.allocate_tensor_on_device(
            rot_mats[0].shape,
            rot_mats[0].dtype,
            rot_mats[0].layout,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_sin = ttnn.allocate_tensor_on_device(
            rot_mats[1].shape,
            rot_mats[1].dtype,
            rot_mats[1].layout,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Copy initial values
        ttnn.copy(rot_mats[0], trace_cos)
        ttnn.copy(rot_mats[1], trace_sin)

        # Clean up temporary rot_mats
        ttnn.deallocate(rot_mats[0])
        ttnn.deallocate(rot_mats[1])

        # Allocate page_table trace tensor for paged attention
        # Shape: [batch_size=1, max_num_blocks]
        trace_page_table = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, max_num_blocks]),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return {
            "hidden_states": trace_hidden_states,
            "cos": trace_cos,
            "sin": trace_sin,
            "seq_len": seq_len,
            "page_table": trace_page_table,
        }

    def _capture_prefill_trace(self, trace_tensors: dict, kv_cache=None) -> Tuple[int, ttnn.Tensor]:
        """
        Capture trace for text model prefill phase.

        Args:
            trace_tensors: Dict with pre-allocated trace tensors
            kv_cache: KV cache tensors to use (paged vLLM cache or internal cache)

        Returns:
            Tuple of (trace_id, logits_trace_output)
        """
        logger.info("Capturing text model prefill trace...")
        rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]
        page_table = trace_tensors.get("page_table")  # Get page_table from trace_tensors

        # Use provided kv_cache or fall back to internal cache
        if kv_cache is None:
            kv_cache = self.kv_caches

        # CRITICAL: Run a warmup forward pass BEFORE trace capture
        # This ensures all lazy tensor allocations and weight transfers happen
        # before we start tracing (writes are not allowed during trace capture)
        logger.info("Running warmup forward pass before trace capture...")
        warmup_logits, _ = self.model.text_model.forward(
            hidden_states=trace_tensors["hidden_states"],
            start_pos=0,
            attn_mask=None,
            kv_caches=kv_cache,
            rot_mats=rot_mats,
            page_table=page_table,  # Compile paged attention ops
        )
        ttnn.deallocate(warmup_logits)
        logger.info("Warmup forward pass complete")

        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

            logits_trace, _ = self.model.text_model.forward(
                hidden_states=trace_tensors["hidden_states"],
                start_pos=0,
                attn_mask=None,
                kv_caches=kv_cache,
                rot_mats=rot_mats,
                page_table=page_table,  # Capture with paged attention ops
            )

            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        logger.info("Text model prefill trace captured with paged attention")

        return trace_id, logits_trace

    def _allocate_decode_trace_tensors(self, hidden_dim: int = 4096, max_num_blocks: int = 64) -> dict:
        """
        Allocate tensors needed for traced decode.

        Args:
            hidden_dim: Hidden dimension (default 4096 for Molmo2-8B)
            max_num_blocks: Maximum number of blocks per sequence for page_table

        Returns:
            Dict with allocated trace tensors
        """
        trace_hidden_states = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Allocate page_table trace tensor for paged attention
        # Shape: [batch_size=1, max_num_blocks]
        trace_page_table = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, max_num_blocks]),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return {
            "hidden_states": trace_hidden_states,
            "page_table": trace_page_table,
        }

    def _copy_decode_trace_inputs(
        self,
        trace_tensors: dict,
        hidden_states: "ttnn.Tensor",
        page_table: Optional["ttnn.Tensor"] = None,
    ) -> None:
        """Copy runtime decode inputs into persistent trace input buffers."""
        ttnn.copy(hidden_states, trace_tensors["hidden_states"])

        if page_table is not None and "page_table" in trace_tensors:
            trace_page_table_shape = list(trace_tensors["page_table"].shape)
            page_table_shape = list(page_table.shape)
            if page_table_shape[-1] < trace_page_table_shape[-1]:
                pad_size = trace_page_table_shape[-1] - page_table_shape[-1]
                page_table_torch = ttnn.to_torch(
                    page_table, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                )[0]
                page_table_padded = torch.nn.functional.pad(page_table_torch, (0, pad_size), value=0)
                page_table_tt = ttnn.from_torch(
                    page_table_padded.unsqueeze(0) if page_table_padded.dim() == 1 else page_table_padded,
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                ttnn.copy(page_table_tt, trace_tensors["page_table"])
                ttnn.deallocate(page_table_tt)
            else:
                ttnn.copy(page_table, trace_tensors["page_table"])

    def _execute_decode_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: "ttnn.Tensor",
        hidden_states: "ttnn.Tensor",
        page_table: Optional["ttnn.Tensor"] = None,
    ) -> "ttnn.Tensor":
        """Execute captured decode trace with new inputs.

        Tracks state changes and forces full synchronization when switching between
        requests. This prevents stale state from causing incorrect behavior across
        multiple sequential inferences.

        Position tensors (current_pos, rot_mat_idxs) are kept on device and
        incremented via ttnn.plus_one after trace execution.
        The trace reads their current values for RoPE and KV cache updates.
        """
        reset_inputs = self.decode_trace_needs_reset

        if page_table is not None:
            if self.prev_page_table is None:
                reset_inputs = True
            else:
                try:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                    curr_pt = ttnn.to_torch(page_table, mesh_composer=mesh_composer)[0]
                    prev_pt = ttnn.to_torch(self.prev_page_table, mesh_composer=mesh_composer)[0]
                    if not torch.equal(curr_pt, prev_pt):
                        reset_inputs = True
                except Exception:
                    reset_inputs = True

        if reset_inputs:
            ttnn.synchronize_device(self.mesh_device)
            logger.debug("Decode trace: full input reset (new request or page_table changed)")
            if page_table is not None:
                self.prev_page_table = page_table
            self.decode_trace_needs_reset = False

        self._copy_decode_trace_inputs(trace_tensors, hidden_states, page_table)
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        return trace_output

    def _capture_decode_trace(self, trace_tensors: dict, kv_cache=None) -> Tuple[int, ttnn.Tensor]:
        """
        Capture trace for decode phase (single token generation).

        The RoPE embedding lookup reads from self.rot_mat_idxs (managed via
        ttnn.plus_one outside the trace). KV cache position reads from
        self.current_pos (also managed via ttnn.plus_one outside the trace).

        The trace includes paged attention operations with page_table as an input.
        Before each trace execution, the actual page_table values are copied to
        the trace_tensors["page_table"] tensor.

        Args:
            trace_tensors: Dict with pre-allocated trace tensors
            kv_cache: KV cache tensors to use (paged vLLM cache or internal cache)

        Returns:
            Tuple of (trace_id, logits_trace_output)
        """
        logger.info("Capturing decode trace...")
        page_table = trace_tensors.get("page_table")  # Get page_table from trace_tensors

        # Use provided kv_cache or fall back to internal cache
        if kv_cache is None:
            kv_cache = self.kv_caches

        # Compile the same graph as capture (rot_mats path) on a scratch buffer. rot_mat_idxs
        # path is a different graph and does not JIT the traced path. forward_decode deallocates
        # inputs — never run compile on trace_tensors["hidden_states"]. Avoid allocations/writes
        # inside begin/end_trace_capture (TT_FATAL: writes not supported during trace capture).
        rot_mats_warmup = self.model.text_model.rotary_setup.get_rot_mats_decode_traced(self.rot_mat_idxs)
        logger.info("Compile warmup: decode forward (rot_mats path, scratch buffer)...")
        compile_hidden = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(trace_tensors["hidden_states"].shape)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(trace_tensors["hidden_states"], compile_hidden)
        compile_logits = self.model.text_model.forward_decode(
            hidden_states=compile_hidden,
            kv_caches=kv_cache,
            current_pos=self.current_pos,
            rot_mats=rot_mats_warmup,
            page_table=page_table,
        )
        ttnn.deallocate(compile_hidden)
        ttnn.deallocate(compile_logits)
        logger.info("Decode compile warmup complete")

        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

            # RoPE embedding lookup INSIDE trace capture so it uses updated rot_mat_idxs
            # on each trace replay. This is critical for correct position encoding.
            rot_mats = self.model.text_model.rotary_setup.get_rot_mats_decode_traced(self.rot_mat_idxs)

            logits_trace = self.model.text_model.forward_decode(
                hidden_states=trace_tensors["hidden_states"],
                kv_caches=kv_cache,
                current_pos=self.current_pos,
                rot_mats=rot_mats,
                page_table=page_table,
            )

            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)
        # See Molmo2Generator._capture_decode_trace: post-capture allocator hint is OK.
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Decode trace captured with paged attention support")

        return trace_id, logits_trace

    def warmup_model_vision(self) -> None:
        """
        Warmup vision path for Molmo2.

        Captures vision trace for the ViT encoder + pooling + projection.
        This is called separately from prefill/decode warmup.
        """
        logger.info("Warmup: Capturing vision trace...")

        # Vision trace parameters for standard 378x378 image with 9 crops
        num_patches = 729  # 27x27 patches per crop
        n_out = 169  # 13x13 pooled output per crop
        k_pool = 4  # Pooling factor
        batch_size = 1

        # Vision trace I/O buffers are pre-allocated on VisionBackbone at model init
        vision_trace_tensors = self._allocate_vision_trace_tensors(
            batch_size=batch_size,
            n_out=n_out,
            k_pool=k_pool,
            num_patches=num_patches,
        )

        # Capture vision trace
        trace_id, trace_output = self._capture_vision_trace(vision_trace_tensors)

        # Store trace for reuse
        self.vision_trace_id = trace_id
        self.vision_trace_tensors = vision_trace_tensors
        self.vision_trace_outputs = trace_output

        logger.info("Warmup: Vision trace captured")

    def _allocate_vision_trace_tensors(
        self,
        batch_size: int,
        n_out: int,
        k_pool: int,
        num_patches: int,
    ) -> dict:
        """Return slices into vision trace buffers pre-allocated on VisionBackbone at model init."""
        return self.model.vision_backbone.get_vision_trace_tensors(
            batch_size=batch_size,
            n_out=n_out,
            k_pool=k_pool,
            num_patches=num_patches,
        )

    def _capture_vision_trace(self, trace_tensors: dict) -> Tuple[int, ttnn.Tensor]:
        """
        Capture vision trace for ViT + pooling + projection.

        Args:
            trace_tensors: Dict with pre-allocated trace tensors

        Returns:
            Tuple of (trace_id, visual_embeddings_trace_output)
        """
        logger.info("Capturing vision trace...")

        # CRITICAL: Run a warmup forward pass BEFORE trace capture
        # This ensures all lazy tensor allocations and weight transfers happen
        # before we start tracing (writes are not allowed during trace capture)
        logger.info("Running warmup vision forward pass before trace capture...")
        warmup_embeddings = self.model.vision_backbone.forward_ttnn(
            images_embedded=trace_tensors["embedded"],
            pooled_patches_idx_ttnn=trace_tensors["idx"],
            valid_mask_ttnn=trace_tensors["valid_mask"],
            valid_token_ttnn=trace_tensors["valid_token"],
            n_out=trace_tensors["n_out"],
            k_pool=trace_tensors["k_pool"],
            batch_size=trace_tensors["batch_size"],
        )
        ttnn.deallocate(warmup_embeddings)
        logger.info("Warmup vision forward pass complete")

        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

            visual_embeddings = self.model.vision_backbone.forward_ttnn(
                images_embedded=trace_tensors["embedded"],
                pooled_patches_idx_ttnn=trace_tensors["idx"],
                valid_mask_ttnn=trace_tensors["valid_mask"],
                valid_token_ttnn=trace_tensors["valid_token"],
                n_out=trace_tensors["n_out"],
                k_pool=trace_tensors["k_pool"],
                batch_size=trace_tensors["batch_size"],
                trace_capture=True,
            )

            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        logger.info("Vision trace captured")

        return trace_id, visual_embeddings
