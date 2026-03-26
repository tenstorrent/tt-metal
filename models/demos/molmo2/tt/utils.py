# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for Molmo2 model.

Contains image/video preprocessing, constants, and helper functions
used by both the standalone demo and vLLM integration.
"""

from typing import Tuple

import einops
import numpy as np
import torch
import torchvision.transforms
from PIL import Image

# =============================================================================
# Constants
# =============================================================================

# Molmo2 image tokens
IMAGE_PATCH_TOKEN = "<im_patch>"
IM_START_TOKEN = "<im_start>"
IM_END_TOKEN = "<im_end>"
IM_COL_TOKEN = "<im_col>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
IMAGE_PROMPT = "<|image|>"
FRAME_START_TOKEN = "<frame_start>"
FRAME_END_TOKEN = "<frame_end>"
VIDEO_PROMPT = "<|video|>"

# Default video parameters matching HF Molmo2 video processor defaults
VIDEO_MAX_FRAMES = 8
VIDEO_MAX_FPS = 2.0

# Molmo2 normalization constants (from HuggingFace Molmo2ImageProcessor)
# These differ from standard ImageNet normalization
IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]

# Prefill sequence length buckets for trace reuse
PREFILL_SEQ_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]


# =============================================================================
# Sequence Length Utilities
# =============================================================================


def get_padded_prefill_len(seq_len: int) -> int:
    """
    Get the padded sequence length for prefill trace reuse.

    Pads to the next bucket size to allow trace reuse across similar sequence lengths.
    Uses buckets: 128, 256, 512, 1024, 2048, 4096, 8192, 16384, then powers of 2.
    """
    for bucket in PREFILL_SEQ_BUCKETS:
        if seq_len <= bucket:
            return bucket
    # For very long sequences, use next power of 2
    return 2 ** (seq_len - 1).bit_length()


def pad_input_ids(input_ids: torch.Tensor, pad_token_id: int = 0) -> Tuple[torch.Tensor, int, int]:
    """
    Pad input_ids to the next bucket size for trace reuse.

    Args:
        input_ids: Token IDs tensor [batch, seq_len]
        pad_token_id: Token ID to use for padding (default 0)

    Returns:
        Tuple of (padded_input_ids, padded_len, original_len)
    """
    original_len = input_ids.shape[1]
    padded_len = get_padded_prefill_len(original_len)
    if padded_len > original_len:
        pad_amount = padded_len - original_len
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_amount), value=pad_token_id)
    return input_ids, padded_len, original_len


# =============================================================================
# Image Preprocessing Utilities
# =============================================================================


def resize_image(image: np.ndarray, target_size: list, resample=None) -> np.ndarray:
    """Resize image using PIL to match HuggingFace preprocessing."""
    # Convert numpy to PIL
    pil_image = Image.fromarray(image.astype(np.uint8))
    # Resize using PIL BILINEAR (same as HuggingFace)
    resized_pil = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    # Convert back to numpy and normalize to [0, 1]
    resized = np.array(resized_pil, dtype=np.float32) / 255.0
    return resized


def normalize_image(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    """Normalize image with ImageNet mean/std."""
    image = image - np.array(mean, dtype=np.float32)[None, None, :]
    image = image / np.array(std, dtype=np.float32)[None, None, :]
    return image


def select_tiling(h: int, w: int, patch_size: int, max_crops: int) -> tuple:
    """Select optimal tiling for image crops."""
    tilings = []
    for i in range(1, max_crops + 1):
        for j in range(1, max_crops + 1):
            if i * j <= max_crops:
                tilings.append((i, j))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))

    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size
    original_size = np.array([h, w], dtype=np.float32)

    with np.errstate(divide="ignore"):
        required_scale = np.min(candidate_resolutions.astype(np.float32) / original_size, axis=-1, keepdims=True)

    if np.all(required_scale < 1):
        ix = np.argmax(required_scale)
    else:
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)

    return tuple(candidate_tilings[ix])


def build_overlapping_crops(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: list,
    base_size: int,
    patch_size: int,
) -> tuple:
    """Build overlapping crops from image."""
    left_margin, right_margin = overlap_margins
    total_margin = patch_size * (left_margin + right_margin)
    crop_patches = base_size // patch_size
    crop_window_patches = crop_patches - (left_margin + right_margin)
    crop_window_size = crop_window_patches * patch_size

    h, w = image.shape[:2]
    tiling = select_tiling(h - total_margin, w - total_margin, crop_window_size, max_crops)

    # Resize to fit tiling
    target_h = tiling[0] * crop_window_size + total_margin
    target_w = tiling[1] * crop_window_size + total_margin
    src = resize_image(image, [target_h, target_w], torchvision.transforms.InterpolationMode.BILINEAR)
    src = normalize_image(src, IMAGENET_MEAN, IMAGENET_STD)

    # Extract crops
    n_crops = tiling[0] * tiling[1]
    crop_arr = np.zeros([n_crops, base_size, base_size, 3], dtype=src.dtype)
    patch_idx_arr = np.zeros([n_crops, crop_patches, crop_patches], dtype=np.int32)

    on_crop = 0
    for i in range(tiling[0]):
        y0 = i * crop_window_size
        for j in range(tiling[1]):
            x0 = j * crop_window_size
            crop_arr[on_crop] = src[y0 : y0 + base_size, x0 : x0 + base_size]

            patch_idx = np.arange(crop_patches * crop_patches).reshape(crop_patches, crop_patches)
            patch_idx = patch_idx + on_crop * crop_patches * crop_patches

            # Mask overlap regions
            if i != 0:
                patch_idx[:left_margin, :] = -1
            if j != 0:
                patch_idx[:, :left_margin] = -1
            if i != tiling[0] - 1:
                patch_idx[-right_margin:, :] = -1
            if j != tiling[1] - 1:
                patch_idx[:, -right_margin:] = -1

            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    # Reorder patch indices
    patch_idx_arr = patch_idx_arr.reshape(tiling[0], tiling[1], crop_patches, crop_patches)
    patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3]).reshape(-1)
    patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(src.shape[0] // patch_size, src.shape[1] // patch_size)

    return crop_arr, patch_idx_arr, tiling


def arange_for_pooling(idx_arr: np.ndarray, pool_h: int, pool_w: int) -> np.ndarray:
    """Arrange indices for pooling."""
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]], constant_values=-1)
    return einops.rearrange(idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


# =============================================================================
# Image Preprocessing Functions
# =============================================================================


def preprocess_image_molmo2_simple(
    image_path: str,
    base_size: int = 378,
    patch_size: int = 14,
    pooling_size: list = None,
) -> dict:
    """
    Simple image preprocessing for Molmo2 - just resize to base_size.

    This is a simplified version that skips multi-crop processing.
    Just resizes the image to 378x378 and computes pooling indices.

    Returns:
        Dict with pixel_values, image_token_pooling, image_grids, and image_num_crops
    """
    if pooling_size is None:
        pooling_size = [2, 2]

    pool_h, pool_w = pooling_size
    crop_patches = base_size // patch_size  # 27

    # Load and convert image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Resize to base_size
    resized = resize_image(image_np, [base_size, base_size], torchvision.transforms.InterpolationMode.BILINEAR)
    resized = normalize_image(resized, IMAGENET_MEAN, IMAGENET_STD)

    # Create pooling indices for 2x2 pooling
    # Original patch grid: 27x27 = 729 patches
    # After 2x2 pooling: 14x14 = 196 output positions (with some padding)
    resize_idx = np.arange(crop_patches * crop_patches).reshape(crop_patches, crop_patches)
    resize_idx = arange_for_pooling(resize_idx, pool_h, pool_w)
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape(-1, pool_h * pool_w)

    # Convert to tensors
    # pixel_values: [1, C, H, W] for single image
    pixel_values = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()

    image_token_pooling = torch.from_numpy(resize_idx).long()
    # For simple single-image mode, high-res grid is same as low-res (no extra crops)
    image_grids = torch.tensor([[resized_h, resized_w, 0, 0]])
    image_num_crops = torch.tensor([1])

    return {
        "pixel_values": pixel_values,  # [1, 3, H, W] for ViT
        "image_token_pooling": image_token_pooling,
        "image_grids": image_grids,
        "image_num_crops": image_num_crops,
    }


def preprocess_image_molmo2(
    image_path: str,
    base_size: int = 378,
    patch_size: int = 14,
    max_crops: int = 8,
    overlap_margins: list = None,
    pooling_size: list = None,
    use_simple: bool = True,  # Default to simple mode for now
) -> dict:
    """
    Preprocess image for Molmo2.

    Args:
        use_simple: If True, use simple resize-only preprocessing (faster, works with current implementation).
                   If False, use full multi-crop preprocessing (requires batch processing in ViT).

    Returns:
        Dict with pixel_values, image_token_pooling, image_grids, and image_num_crops
    """
    if use_simple:
        return preprocess_image_molmo2_simple(image_path, base_size, patch_size, pooling_size)

    # Full multi-crop preprocessing (for reference)
    if overlap_margins is None:
        overlap_margins = [4, 4]
    if pooling_size is None:
        pooling_size = [2, 2]

    pool_h, pool_w = pooling_size
    crop_patches = base_size // patch_size

    # Load and convert image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Build overlapping crops
    crop_arr, patch_idx_arr, tiling = build_overlapping_crops(
        image_np, max_crops, overlap_margins, base_size, patch_size
    )

    # Pooling indices for high-res crops
    pooling_idx = arange_for_pooling(patch_idx_arr, pool_h, pool_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape(-1, pool_h * pool_w)

    # Build low-res (global) image
    resized = resize_image(image_np, [base_size, base_size], torchvision.transforms.InterpolationMode.BILINEAR)
    resized = normalize_image(resized, IMAGENET_MEAN, IMAGENET_STD)
    resized = np.expand_dims(resized, 0)

    resize_idx = np.arange(crop_patches * crop_patches).reshape(crop_patches, crop_patches)
    resize_idx = arange_for_pooling(resize_idx, pool_h, pool_w)
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape(-1, pool_h * pool_w)

    # Combine: global image first, then crops
    # all_crops shape: [n_crops, H, W, C] (H, W = base_size)
    all_crops = np.concatenate([resized, crop_arr], axis=0)

    # Adjust pooling indices (global image patches come first)
    pooling_idx = np.where(pooling_idx >= 0, pooling_idx + crop_patches * crop_patches, -1)
    pooling_idx = np.concatenate([resize_idx, pooling_idx], axis=0)

    # Convert crops to [n_crops, C, H, W] format for ViT
    # all_crops is [n_crops, H, W, C] -> [n_crops, C, H, W]
    pixel_values = torch.from_numpy(all_crops).permute(0, 3, 1, 2).float()

    image_token_pooling = torch.from_numpy(pooling_idx).long()
    image_grids = torch.tensor([[resized_h, resized_w, h, w]])
    image_num_crops = torch.tensor([all_crops.shape[0]])

    return {
        "pixel_values": pixel_values,  # [n_crops, 3, H, W] for ViT
        "image_token_pooling": image_token_pooling,
        "image_grids": image_grids,
        "image_num_crops": image_num_crops,
    }


# =============================================================================
# Token String Generation
# =============================================================================


def get_image_tokens(image_grid: torch.Tensor, use_col_tokens: bool = True) -> str:
    """Generate image token string from grid dimensions."""
    resized_h, resized_w, h, w = image_grid.tolist()

    # Low-res tokens (always present)
    per_row_low = IMAGE_PATCH_TOKEN * resized_w
    if use_col_tokens:
        per_row_low += IM_COL_TOKEN
    low_res_tokens = LOW_RES_IMAGE_START_TOKEN + (per_row_low * resized_h) + IM_END_TOKEN

    # High-res tokens (only if present)
    if h > 0 and w > 0:
        per_row = IMAGE_PATCH_TOKEN * w
        if use_col_tokens:
            per_row += IM_COL_TOKEN
        high_res_tokens = IM_START_TOKEN + (per_row * h) + IM_END_TOKEN
        return low_res_tokens + high_res_tokens
    else:
        return low_res_tokens


def get_video_tokens(
    num_frames: int,
    pooled_h: int,
    pooled_w: int,
    timestamps: np.ndarray,
    use_col_tokens: bool = True,
) -> str:
    """
    Generate video token string for all frames.

    Each frame produces: "{timestamp} <frame_start><im_patch>*N<frame_end>"
    where N = pooled_h * pooled_w (14*14=196 for 378x378 with patch_size=14, pool=[2,2]).

    This matches the HuggingFace Molmo2Processor.get_video_string() output
    (video_grid=[n_frames, h, w] where h,w are pooled dimensions).
    """
    video_string = ""
    for frame_idx, frame_time in enumerate(timestamps):
        prev_space = " " if frame_idx > 0 else ""
        frame_prefix = prev_space + f"{frame_time:.1f} "
        video_string += frame_prefix

        per_row = IMAGE_PATCH_TOKEN * pooled_w
        if use_col_tokens:
            per_row += IM_COL_TOKEN
        video_string += FRAME_START_TOKEN + (per_row * pooled_h) + FRAME_END_TOKEN

    return video_string


# =============================================================================
# Video Preprocessing
# =============================================================================


def preprocess_video_molmo2(
    video_path: str,
    base_size: int = 378,
    patch_size: int = 14,
    pooling_size: list = None,
    max_frames: int = VIDEO_MAX_FRAMES,
    max_fps: float = VIDEO_MAX_FPS,
) -> dict:
    """
    Preprocess a video file for Molmo2.

    Uses molmo-utils to extract frames, then applies the same per-frame
    preprocessing as preprocess_image_molmo2_simple (resize + normalize).
    Each frame is treated as a single image crop.

    Args:
        video_path: Path or URL to video file
        base_size: Target frame size (378 for Molmo2)
        patch_size: ViT patch size (14)
        pooling_size: [pool_h, pool_w] for cross-attention pooling (default [2, 2])
        max_frames: Maximum frames to extract
        max_fps: Maximum frames per second to sample

    Returns:
        Dict with:
          - pixel_values: [n_frames, 3, H, W] tensor
          - image_token_pooling: [n_frames * N_out, K_pool] pooling indices
          - n_frames: int
          - timestamps: np.ndarray of frame timestamps
    """
    from molmo_utils import process_vision_info

    if pooling_size is None:
        pooling_size = [2, 2]
    pool_h, pool_w = pooling_size
    crop_patches = base_size // patch_size  # 27

    # Convert URL/path to molmo-utils message format
    if video_path.startswith("http://") or video_path.startswith("https://"):
        video_src = video_path
    else:
        video_src = f"file://{video_path}" if not video_path.startswith("file://") else video_path

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_src,
                    "num_frames": max_frames,
                    "max_fps": max_fps,
                    "frame_sampling_mode": "uniform_last_frame",
                },
                {"type": "text", "text": ""},
            ],
        }
    ]

    _, videos, _ = process_vision_info(messages)
    if not videos:
        raise ValueError(f"Could not extract frames from video: {video_path}")

    frames_array, metadata = videos[0]
    # frames_array: (T, H, W, 3) numpy uint8

    n_frames = len(frames_array)

    # Build per-frame pooling indices (with global offsets across frames)
    resize_idx_per_frame = np.arange(crop_patches * crop_patches).reshape(crop_patches, crop_patches)
    resize_idx_per_frame = arange_for_pooling(resize_idx_per_frame, pool_h, pool_w)
    n_out = resize_idx_per_frame.shape[0] * resize_idx_per_frame.shape[1]
    resize_idx_per_frame = resize_idx_per_frame.reshape(-1, pool_h * pool_w)  # [N_out, K_pool]

    all_crops = []
    all_pooling_idx = []
    patches_per_frame = crop_patches * crop_patches  # 729

    for frame_idx, frame in enumerate(frames_array):
        # Resize and normalize frame (same as preprocess_image_molmo2_simple)
        frame_resized = resize_image(frame, [base_size, base_size])
        frame_normalized = normalize_image(frame_resized, IMAGENET_MEAN, IMAGENET_STD)
        # [H, W, C] -> [C, H, W]
        crop = torch.from_numpy(frame_normalized).permute(2, 0, 1).float()
        all_crops.append(crop)

        # Pooling indices with offset for this frame's patch range
        offset = frame_idx * patches_per_frame
        frame_idx_with_offset = np.where(
            resize_idx_per_frame >= 0,
            resize_idx_per_frame + offset,
            resize_idx_per_frame,
        )
        all_pooling_idx.append(frame_idx_with_offset)

    pixel_values = torch.stack(all_crops, dim=0)  # [n_frames, 3, H, W]
    image_token_pooling = torch.from_numpy(np.stack(all_pooling_idx, axis=0)).long()  # [n_frames, N_out, K_pool]

    # Compute timestamps from metadata
    if hasattr(metadata, "get"):
        frames_indices = metadata.get("frames_indices", np.arange(n_frames))
        fps = metadata.get("fps", max_fps)
    else:
        frames_indices = getattr(metadata, "frames_indices", np.arange(n_frames))
        fps = getattr(metadata, "fps", max_fps)

    timestamps = np.array(frames_indices, dtype=float) / fps

    # Pooled grid dimensions for get_video_tokens()
    # resize_idx_per_frame was [14, 14, 4] before reshape -> pooled_h=14, pooled_w=14
    resize_idx_before_reshape = np.arange(crop_patches * crop_patches).reshape(crop_patches, crop_patches)
    resize_idx_before_reshape = arange_for_pooling(resize_idx_before_reshape, pool_h, pool_w)
    pooled_h, pooled_w = resize_idx_before_reshape.shape[0], resize_idx_before_reshape.shape[1]

    return {
        "pixel_values": pixel_values,  # [n_frames, 3, H, W]
        "image_token_pooling": image_token_pooling,  # [n_frames, N_out, K_pool]
        "n_frames": n_frames,
        "timestamps": timestamps,
        "patches_per_frame": patches_per_frame,
        "pooled_h": pooled_h,
        "pooled_w": pooled_w,
    }
