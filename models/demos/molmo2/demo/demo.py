# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B Demo for Tenstorrent Hardware.

This demo showcases multimodal visual question answering using Molmo2-8B
running on Tenstorrent devices (N150/N300/T3K).

Features:
- Vision-language multimodal inference
- KV cache for efficient autoregressive generation
- Optional tracing for improved performance
- Proper warm-up and timing (TTFT, decode throughput)

Usage:
    # Run with default image and prompt
    python -m models.demos.molmo2.demo.demo

    # Run with custom image
    python -m models.demos.molmo2.demo.demo --image path/to/image.jpg

    # Run with tracing enabled
    python -m models.demos.molmo2.demo.demo --use-trace

    # Video on PyTorch/HuggingFace only (no Tenstorrent device)
    python -m models.demos.molmo2.demo.demo --video path/to/video.mp4 --torch --max-tokens 64

    # Video: TTNN first, then HuggingFace PyTorch reference (both paths)
    python -m models.demos.molmo2.demo.demo --video path/to/video.mp4 --also-torch --max-tokens 64
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import einops
import numpy as np
import torch
import torchvision.transforms
from loguru import logger
from PIL import Image

import ttnn

# Default paths
DEMO_DIR = Path(__file__).parent
DEFAULT_IMAGE = DEMO_DIR / "dog.jpg"
MODEL_ID = "allenai/Molmo2-8B"
# Pin Hub revision for reproducible PyTorch/HF runs. Set MOLMO2_HF_REVISION="" to use the default branch.
DEFAULT_MOLMO2_HF_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
MOLMO2_HF_REVISION = os.environ.get("MOLMO2_HF_REVISION", DEFAULT_MOLMO2_HF_REVISION).strip()


def _molmo2_hf_from_pretrained_kwargs(local_files_only: bool) -> dict:
    kwargs = {"trust_remote_code": True, "local_files_only": local_files_only}
    if MOLMO2_HF_REVISION:
        kwargs["revision"] = MOLMO2_HF_REVISION
    return kwargs


def check_transformers_supports_molmo2_hub() -> None:
    """
    allenai/Molmo2-8B hub video code imports ``video_utils.make_batched_metadata`` (added in v4.56).
    """
    try:
        from transformers.video_utils import make_batched_metadata  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Molmo2 Hugging Face hub code requires transformers>=4.56 "
            "(module transformers.video_utils.make_batched_metadata). "
            "Install e.g. pip install -U 'transformers>=4.56' or "
            "pip install -r models/demos/molmo2/requirements.txt"
        ) from e


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


def load_processor():
    """Load the Molmo2 tokenizer from HuggingFace."""
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    return tokenizer


def load_model_weights():
    """Load all model weights from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    logger.info(f"Loading model weights from {MODEL_ID}")
    state_dict = load_state_dict_from_safetensors(MODEL_ID)
    logger.info(f"Loaded {len(state_dict)} weight tensors")
    return state_dict


# =============================================================================
# Molmo2 Image Preprocessing (standalone, no video dependencies)
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
    # resize_idx_per_frame was [14, 14, 4] before reshape → pooled_h=14, pooled_w=14
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


def get_hf_video_payload(
    video_path: str,
    max_frames: int = VIDEO_MAX_FRAMES,
    max_fps: float = VIDEO_MAX_FPS,
) -> dict:
    """
    Build the videos= entry for HuggingFace Molmo2Processor.

    Uses the same molmo_utils.process_vision_info path as preprocess_video_molmo2.
    """
    from molmo_utils import process_vision_info

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
    n_frames = len(frames_array)
    if hasattr(metadata, "get"):
        frames_indices = metadata.get("frames_indices", np.arange(n_frames))
        fps = float(metadata.get("fps", max_fps))
        total_num_frames = metadata.get("total_num_frames", None)
    else:
        frames_indices = getattr(metadata, "frames_indices", np.arange(n_frames))
        fps = float(getattr(metadata, "fps", max_fps))
        total_num_frames = getattr(metadata, "total_num_frames", None)

    frames_indices = np.asarray(frames_indices, dtype=np.int64).reshape(-1)
    if total_num_frames is None:
        total_num_frames = int(frames_indices.max()) + 1 if len(frames_indices) else n_frames

    timestamps = frames_indices.astype(np.float64) / max(fps, 1e-6)

    return {
        "frames": frames_array,
        "timestamps": timestamps,
        "sampled_fps": fps,
        "frames_indices": frames_indices,
        "total_num_frames": int(total_num_frames),
    }


def run_video_demo_torch(
    video_path: str,
    prompt: str = "<|video|> Describe what happens in this video.",
    max_new_tokens: int = 64,
    max_frames: int = VIDEO_MAX_FRAMES,
    max_fps: float = VIDEO_MAX_FPS,
) -> Tuple[str, dict]:
    """
    Run video QA on PyTorch + HuggingFace Molmo2 (reference path; no TTNN).
    """
    check_transformers_supports_molmo2_hub()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("=" * 60)
    logger.info("Molmo2-8B Video Demo (PyTorch / HuggingFace)")
    logger.info("=" * 60)

    if VIDEO_PROMPT not in prompt:
        logger.warning(f"Prompt missing {VIDEO_PROMPT!r}; prepending it.")
        prompt = f"{VIDEO_PROMPT} {prompt}"

    logger.info(f"Loading video payload from: {video_path}")
    video_payload = get_hf_video_payload(video_path, max_frames=max_frames, max_fps=max_fps)

    hf_kw = _molmo2_hf_from_pretrained_kwargs(os.getenv("CI") == "true")
    rev_note = f" @{hf_kw['revision']}" if "revision" in hf_kw else ""
    logger.info(f"Loading HF processor and model: {MODEL_ID}{rev_note}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, **hf_kw)

    if torch.cuda.is_available():
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **hf_kw,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            **hf_kw,
        ).to("cpu")
    model.eval()

    # Molmo2Processor forwards `videos` to video_utils.make_batched_videos: pass a T,H,W,C array per
    # batch item (not a custom dict). Pre-sampled frames from molmo_utils need do_sample_frames=False
    # and batched VideoMetadata so placeholder expansion gets correct timestamps.
    frames = video_payload["frames"]
    fi = video_payload["frames_indices"]
    video_metadata = [
        {
            "total_num_frames": video_payload["total_num_frames"],
            "fps": float(video_payload["sampled_fps"]),
            "frames_indices": [int(x) for x in np.asarray(fi).reshape(-1)],
        }
    ]
    inputs = processor(
        text=[prompt],
        videos=[frames],
        videos_kwargs={"do_sample_frames": False, "video_metadata": video_metadata},
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    logger.info(f"Generating up to {max_new_tokens} tokens...")
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    decoded_prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, decoded_prompt_len:].tolist()
    response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    perf = {
        "backend": "torch",
        "video_path": video_path,
        "generate_ms": elapsed_ms,
        "max_new_tokens": max_new_tokens,
        "n_prompt_tokens": int(decoded_prompt_len),
        "n_output_tokens": int(output_ids.shape[1]),
        "n_new_tokens": len(new_tokens),
        "device": str(device),
    }
    logger.info(f"Generation finished in {elapsed_ms:.2f} ms ({perf['n_new_tokens']} new tokens)")
    logger.info(f"Response: {response}")
    return response, perf


def create_model(
    mesh_device,
    state_dict,
    num_layers: Optional[int] = None,
    max_seq_len: int = 8192,
):
    """
    Create the Molmo2 TTNN model.

    Args:
        mesh_device: TTNN device or mesh device
        state_dict: Model state dict
        num_layers: Optional number of text layers (default: 36)
        max_seq_len: RoPE table length; use >= KV cache max (e.g. 16384 for video)

    Returns:
        Molmo2Model instance
    """
    from models.demos.molmo2.tt.molmo2_model import Molmo2Model

    logger.info("Creating Molmo2 TTNN model")

    text_num_layers = num_layers if num_layers is not None else 36

    model = Molmo2Model(
        mesh_device=mesh_device,
        state_dict=state_dict,
        # Vision config
        vit_num_layers=25,
        vit_hidden_dim=1152,
        vit_intermediate_dim=4304,
        vit_num_heads=16,
        vit_head_dim=72,
        patch_size=14,
        image_size=378,
        feature_layers=(24, 18),  # HF order: vit_layers=[-3, -9] -> layers [24, 18]
        # Adapter config
        adapter_hidden_dim=1152,
        adapter_intermediate_dim=12288,
        adapter_num_heads=16,
        adapter_head_dim=72,
        # Text config
        text_num_layers=text_num_layers,
        text_hidden_dim=4096,
        text_intermediate_dim=12288,
        text_num_heads=32,
        text_num_kv_heads=8,
        text_head_dim=128,
        vocab_size=152064,
        max_seq_len=max_seq_len,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
        dtype=ttnn.bfloat8_b,
    )

    logger.info("Model created successfully")
    return model


class Molmo2Generator:
    """
    Molmo2 generator with separate tracing for prefill and decode.

    Tracing captures the computation graph and replays it for improved performance.
    - Prefill trace: processes the full input sequence
    - Decode trace: processes one token at a time with KV cache

    Timing follows simple_text_demo.py pattern:
    - compile_prefill: First prefill run (warm-up)
    - inference_prefill: Actual prefill (TTFT)
    - compile_decode: First decode run (warm-up)
    - inference_decode: Subsequent decode iterations
    """

    def __init__(
        self,
        mesh_device,
        model,
        tokenizer,
        num_layers: int,
        batch_size: int = 1,
        max_seq_len: int = 2048,
    ):
        self.mesh_device = mesh_device
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Separate trace state for prefill and decode
        self.prefill_traces = {}  # {seq_len: (trace_id, trace_inputs, trace_output)}
        self.decode_trace_id = None
        self.decode_trace_tensors = None
        self.decode_trace_output = None

        # Vision trace state (ViT encoder)
        self.vision_trace_id = None
        self.vision_trace_tensors = None
        self.vision_trace_outputs = None  # [feature_layer_18, feature_layer_24]

        # KV cache (initialized on first run)
        self.kv_caches = None
        self.current_pos = None
        self.decode_position = 0  # Track position on host for trace updates

        # Mesh mapper
        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None

    def init_kv_cache(self):
        """Initialize KV cache, position tensors, and RoPE index tensors."""
        from models.demos.molmo2.tt.text_model import init_decode_position, init_kv_cache

        if self.kv_caches is None:
            self.kv_caches = init_kv_cache(
                mesh_device=self.mesh_device,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                num_kv_heads=8,
                max_seq_len=self.max_seq_len,
                head_dim=128,
                dtype=ttnn.bfloat16,
            )
            self.current_pos = init_decode_position(
                mesh_device=self.mesh_device,
                batch_size=self.batch_size,
                initial_pos=0,
            )
            self.rot_mat_idxs = self.model.text_model.rotary_setup.allocate_decode_rot_idxs(initial_pos=0)

    def reset_kv_cache(self, start_pos: int = 0):
        """Reset KV cache position and RoPE indices for new generation."""
        self.decode_position = start_pos

        if self.current_pos is not None:
            pos_tensor = torch.full((self.batch_size,), start_pos, dtype=torch.int32)
            pos_ttnn = ttnn.from_torch(
                pos_tensor,
                dtype=ttnn.int32,
                device=self.mesh_device,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(pos_ttnn, self.current_pos)
            ttnn.deallocate(pos_ttnn)

        if self.rot_mat_idxs is not None:
            batch = self.batch_size
            pad_size = ((batch + 31) // 32) * 32 - batch
            rot_idxs_tensor = torch.full((1, batch + pad_size), start_pos, dtype=torch.int32)
            rot_idxs_ttnn = ttnn.from_torch(
                rot_idxs_tensor,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(rot_idxs_ttnn, self.rot_mat_idxs)
            ttnn.deallocate(rot_idxs_ttnn)

    def _prepare_text_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Prepare text model inputs by processing vision and fusing embeddings on device.

        No CPU roundtrip: vision runs on TTNN, fusion is done via selector matmul.
        Must be called BEFORE trace capture since it computes new tensors.

        Args:
            input_ids: Input token IDs
            pixel_values: Preprocessed image tensor
            pooled_patches_idx: Indices for vision pooling

        Returns:
            Fused hidden states [1, 1, seq_len, hidden_dim] on device
        """
        if pixel_values is not None and pooled_patches_idx is not None:
            visual_embeddings_ttnn, valid_token = self.model.embed_image(pixel_values, pooled_patches_idx)
            fused_ttnn = self.model.prepare_inputs_for_multimodal(input_ids, visual_embeddings_ttnn, valid_token)
            ttnn.deallocate(visual_embeddings_ttnn)
        else:
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

    def _prepare_vision_inputs_for_trace(
        self,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> dict:
        """
        Prepare vision inputs for traced execution.

        Converts all inputs to TTNN tensors so the forward can be fully traced.

        Args:
            pixel_values: Raw pixel values [B, C, H, W]
            pooled_patches_idx: Patch indices [B, N_out, K_pool]

        Returns:
            Dict with TTNN tensors and metadata for traced forward
        """
        batch_size = pooled_patches_idx.shape[0]
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        # 1. Patch embedding on TTNN (unfold on CPU, linear+pos_embed on device)
        vit = self.model.vision_backbone.image_vit
        embedded_ttnn = vit.patch_embed_ttnn(pixel_values)  # [1, 1, B*N, hidden_dim] on device

        # 2. Prepare indices for TTNN gather
        # Identify valid indices (>= 0) and clip negative to 0
        valid = pooled_patches_idx >= 0  # [B, N_out, K_pool]
        valid_token = torch.any(valid, dim=-1)  # [B, N_out]
        clipped_idx = torch.clip(pooled_patches_idx, min=0)

        # Flatten indices for embedding lookup: [B, N_out, K_pool] -> [1, B*N_out*K_pool]
        flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)

        # Create valid mask: [1, 1, B*N_out*K_pool, 1]
        valid_mask = valid.reshape(1, 1, -1, 1).float()

        # 3. Convert remaining tensors to TTNN (embedded_ttnn already on device)

        idx_ttnn = ttnn.from_torch(
            flat_idx,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        valid_mask_ttnn = ttnn.from_torch(
            valid_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        valid_token_ttnn = ttnn.from_torch(
            valid_token.flatten(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        return {
            "embedded": embedded_ttnn,
            "idx": idx_ttnn,
            "valid_mask": valid_mask_ttnn,
            "valid_token": valid_token_ttnn,
            "valid_token_torch": valid_token,  # Keep torch version for final filtering
            "n_out": n_out,
            "k_pool": k_pool,
            "batch_size": batch_size,
        }

    def _allocate_vision_trace_tensors(
        self,
        num_patches: int = 729,
        hidden_dim: int = 1152,
        n_out: int = 169,
        k_pool: int = 4,
        pool_dim: int = 2304,
        batch_size: int = 1,
    ) -> dict:
        """Allocate tensors for vision trace."""
        # Input: embedded patches
        trace_embedded = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, batch_size * num_patches, hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Indices for gathering
        trace_idx = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, batch_size * n_out * k_pool]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Valid mask
        trace_valid_mask = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, batch_size * n_out * k_pool, 1]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Valid token mask
        trace_valid_token = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch_size * n_out]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return {
            "embedded": trace_embedded,
            "idx": trace_idx,
            "valid_mask": trace_valid_mask,
            "valid_token": trace_valid_token,
            "n_out": n_out,
            "k_pool": k_pool,
            "batch_size": batch_size,
        }

    def _capture_vision_trace(self, trace_tensors: dict) -> Tuple[int, ttnn.Tensor]:
        """Capture vision trace for ViT + pooling + projection."""
        logger.info("Capturing vision trace...")

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        visual_embeddings = self.model.vision_backbone.forward_ttnn(
            images_embedded=trace_tensors["embedded"],
            pooled_patches_idx_ttnn=trace_tensors["idx"],
            valid_mask_ttnn=trace_tensors["valid_mask"],
            valid_token_ttnn=trace_tensors["valid_token"],
            n_out=trace_tensors["n_out"],
            k_pool=trace_tensors["k_pool"],
            batch_size=trace_tensors["batch_size"],
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Vision trace captured")

        return trace_id, visual_embeddings

    def _execute_vision_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        vision_inputs: dict,
    ) -> ttnn.Tensor:
        """Execute vision trace with new inputs."""
        # Copy new inputs to trace tensors
        ttnn.copy(vision_inputs["embedded"], trace_tensors["embedded"])
        ttnn.copy(vision_inputs["idx"], trace_tensors["idx"])
        ttnn.copy(vision_inputs["valid_mask"], trace_tensors["valid_mask"])
        ttnn.copy(vision_inputs["valid_token"], trace_tensors["valid_token"])

        # Execute trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_output

    def _prepare_text_inputs_traced(
        self,
        input_ids: torch.Tensor,
        visual_embeddings_ttnn: ttnn.Tensor,
        valid_token_torch: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Prepare text inputs using traced vision output.

        Fuses visual embeddings with text embeddings entirely on device.
        Uses matmul with selector matrix to avoid CPU roundtrip.
        Returns fused hidden states [1, 1, seq_len, hidden_dim] on device.
        """
        batch_size, seq_len = input_ids.shape
        hidden_dim = 4096
        image_patch_id = 151938  # Molmo2 image patch token ID

        # 1. Get text embeddings on device
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        text_embeddings_ttnn = self.model.text_model.embed_tokens(input_ids_ttnn)
        ttnn.deallocate(input_ids_ttnn)

        # 2. Filter visual embeddings by valid tokens (on device using ttnn.embedding as gather)
        # valid_token_torch is [n_out] boolean, visual_embeddings_ttnn is [1, 1, n_out, hidden_dim]
        valid_indices = valid_token_torch.flatten().nonzero(as_tuple=True)[0].to(torch.int32)
        num_valid = len(valid_indices)

        if num_valid > 0:
            # Use ttnn.embedding as gather to select valid visual embeddings
            valid_indices_ttnn = ttnn.from_torch(
                valid_indices.unsqueeze(0),  # [1, num_valid]
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )

            # Reshape for gather using logical dims (last dim may be TILE-padded; [1,-1,4096] breaks reshape).
            n_vis_tokens = visual_embeddings_ttnn.shape[2]
            embed_width = visual_embeddings_ttnn.shape[3]
            visual_for_gather = ttnn.reshape(
                visual_embeddings_ttnn,
                [1, n_vis_tokens, embed_width],
            )

            # Gather valid embeddings: [1, num_valid, embed_width]
            valid_visual_ttnn = ttnn.embedding(valid_indices_ttnn, visual_for_gather)
            ttnn.deallocate(valid_indices_ttnn)

            valid_visual_ttnn = ttnn.reshape(valid_visual_ttnn, [1, 1, num_valid, embed_width])
            if embed_width != hidden_dim:
                valid_visual_ttnn = ttnn.slice(
                    valid_visual_ttnn,
                    (0, 0, 0, 0),
                    (1, 1, num_valid, hidden_dim),
                )

            # 3. Create selector matrix for fusion (CPU - just positions, very fast)
            # S[seq_pos, visual_idx] = 1.0 where seq_pos should get visual_idx embedding
            image_positions = (input_ids[0] == image_patch_id).nonzero(as_tuple=True)[0]

            if len(image_positions) == num_valid:
                # Create sparse selector matrix
                selector = torch.zeros(seq_len, num_valid, dtype=torch.bfloat16)
                for i, pos in enumerate(image_positions):
                    selector[pos, i] = 1.0

                # Transfer selector to device
                selector_ttnn = ttnn.from_torch(
                    selector.unsqueeze(0).unsqueeze(0),  # [1, 1, seq_len, num_valid]
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )

                # 4. Compute visual contribution: selector @ visual_embeddings
                # [1, 1, seq_len, num_valid] @ [1, 1, num_valid, hidden_dim] -> [1, 1, seq_len, hidden_dim]
                visual_contribution = ttnn.matmul(selector_ttnn, valid_visual_ttnn)
                ttnn.deallocate(selector_ttnn)
                ttnn.deallocate(valid_visual_ttnn)

                # 5. Fuse: text_embeddings + visual_contribution (on device)
                fused_ttnn = ttnn.add(text_embeddings_ttnn, visual_contribution)
                ttnn.deallocate(text_embeddings_ttnn)
                ttnn.deallocate(visual_contribution)
            else:
                logger.warning(
                    f"Position mismatch: {len(image_positions)} placeholders vs {num_valid} visual tokens. "
                    "Falling back to text-only."
                )
                fused_ttnn = text_embeddings_ttnn
        else:
            # No visual tokens - just use text embeddings
            fused_ttnn = text_embeddings_ttnn

        return fused_ttnn

    def _allocate_prefill_trace_tensors(
        self,
        seq_len: int,
        hidden_dim: int = 4096,
    ) -> dict:
        """Pre-allocate all tensors needed for traced prefill."""
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

        return {
            "hidden_states": trace_hidden_states,
            "cos": trace_cos,
            "sin": trace_sin,
            "seq_len": seq_len,
        }

    def _capture_prefill_trace(
        self,
        trace_tensors: dict,
    ) -> Tuple[int, ttnn.Tensor]:
        """Capture trace for text model prefill phase."""
        logger.info("Capturing text model prefill trace...")
        rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        logits_trace, _ = self.model.text_model.forward(
            hidden_states=trace_tensors["hidden_states"],
            start_pos=0,
            attn_mask=None,
            kv_caches=self.kv_caches,  # Pass KV cache to fill during prefill
            rot_mats=rot_mats,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Text model prefill trace captured")

        return trace_id, logits_trace

    def _execute_prefill_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        hidden_states_ttnn: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Execute captured prefill trace with new inputs."""
        # Device-to-device copy: no host roundtrip
        ttnn.copy(hidden_states_ttnn, trace_tensors["hidden_states"])

        # Execute trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_output

    # =========================================================================
    # UNIFIED VISION + PREFILL TRACE (on-device fusion)
    # =========================================================================

    def _allocate_unified_trace_tensors(
        self,
        seq_len: int,
        num_visual_tokens: int,
        num_patches: int = 729,
        vit_hidden_dim: int = 1152,
        hidden_dim: int = 4096,
        n_out: int = 169,
        k_pool: int = 4,
        batch_size: int = 1,
    ) -> dict:
        """Allocate all tensors needed for unified vision + prefill trace.

        This includes input_ids so embed_tokens can be called inside the trace.
        """
        # Vision inputs
        trace_embedded = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, batch_size * num_patches, vit_hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_idx = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, batch_size * n_out * k_pool]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_valid_mask = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, batch_size * n_out * k_pool, 1]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_valid_token = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch_size * n_out]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Input IDs for embed_tokens (called INSIDE trace)
        trace_input_ids = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, seq_len]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Selector matrix for matmul-based fusion [1, 1, seq_len, num_visual_tokens]
        # Each row has a 1 at the column corresponding to which visual token goes there
        trace_selector_matrix = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, seq_len, num_visual_tokens]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Rotation matrices for prefill
        rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)
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
        ttnn.copy(rot_mats[0], trace_cos)
        ttnn.copy(rot_mats[1], trace_sin)
        ttnn.deallocate(rot_mats[0])
        ttnn.deallocate(rot_mats[1])

        return {
            # Vision inputs
            "embedded": trace_embedded,
            "idx": trace_idx,
            "valid_mask": trace_valid_mask,
            "valid_token": trace_valid_token,
            "n_out": n_out,
            "k_pool": k_pool,
            "batch_size": batch_size,
            # Text inputs (for embed_tokens inside trace)
            "input_ids": trace_input_ids,
            "selector_matrix": trace_selector_matrix,
            "num_visual_tokens": num_visual_tokens,
            # Prefill inputs
            "cos": trace_cos,
            "sin": trace_sin,
            "seq_len": seq_len,
        }

    def _capture_unified_trace(self, trace_tensors: dict) -> Tuple[int, ttnn.Tensor]:
        """
        Capture unified trace for Vision + embed_tokens + Fusion + Text Prefill.

        This eliminates the CPU roundtrip between vision and prefill by keeping
        everything on device:
        1. Vision backbone (ViT + pooling + projection)
        2. Text embeddings via embed_tokens (INSIDE trace)
        3. Fusion via matmul with selector matrix
        4. Text model forward (transformer layers + lm_head)
        """
        logger.info("Capturing unified vision + prefill trace...")

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        # Step 1: Vision backbone (ViT + pooling + projection)
        visual_embeddings = self.model.vision_backbone.forward_ttnn(
            images_embedded=trace_tensors["embedded"],
            pooled_patches_idx_ttnn=trace_tensors["idx"],
            valid_mask_ttnn=trace_tensors["valid_mask"],
            valid_token_ttnn=trace_tensors["valid_token"],
            n_out=trace_tensors["n_out"],
            k_pool=trace_tensors["k_pool"],
            batch_size=trace_tensors["batch_size"],
        )
        # visual_embeddings shape: [1, 1, num_visual_tokens, 4096]

        # Step 2: Text embeddings (INSIDE trace - this is key for performance)
        text_embeddings = self.model.text_model.embed_tokens(trace_tensors["input_ids"])
        # text_embeddings shape: [1, 1, seq_len, 4096]

        # Step 3: On-device fusion using matmul with selector matrix
        # selector_matrix: [1, 1, seq_len, num_visual_tokens] - one-hot rows indicating placement
        # visual_part = selector_matrix @ visual_embeddings => [1, 1, seq_len, 4096]
        # fused = text_embeddings + visual_part
        visual_part = ttnn.matmul(
            trace_tensors["selector_matrix"],
            visual_embeddings,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # visual_part: [1, 1, seq_len, 4096]

        # Add visual part to text embeddings
        fused_embed = ttnn.add(
            text_embeddings,
            visual_part,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Step 3: Text model prefill
        rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]
        logits, _ = self.model.text_model.forward(
            hidden_states=fused_embed,
            start_pos=0,
            attn_mask=None,
            kv_caches=self.kv_caches,
            rot_mats=rot_mats,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Unified vision + prefill trace captured")

        return trace_id, logits

    def _prepare_unified_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> dict:
        """
        Prepare all inputs for the unified trace.

        Note: embed_tokens is called INSIDE the trace, not here.
        This just prepares input_ids and other tensors for copying to trace tensors.
        """
        batch_size = 1
        seq_len = input_ids.shape[1]

        # Prepare vision inputs -- patch embedding on TTNN (no CPU matmul)
        vit = self.model.vision_backbone.image_vit
        embedded_ttnn = vit.patch_embed_ttnn(pixel_values)  # [1, 1, B*N, hidden_dim] on device

        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)
        clipped_idx = torch.clip(pooled_patches_idx, min=0)
        flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)
        valid_mask = valid.reshape(1, 1, -1, 1).float()

        # Get number of valid visual tokens
        num_visual_tokens = valid_token.flatten().sum().item()

        # Find image token positions in input_ids (CPU - fast)
        image_patch_id = self.model.image_patch_id
        image_positions = (input_ids[0] == image_patch_id).nonzero(as_tuple=True)[0]

        # Verify counts match
        assert (
            len(image_positions) == num_visual_tokens
        ), f"Mismatch: {len(image_positions)} placeholders vs {num_visual_tokens} visual tokens"

        # Create selector matrix for matmul-based fusion: [seq_len, num_visual_tokens]
        # Row i has 1 at column j if position i should receive visual embedding j
        selector_matrix = torch.zeros(seq_len, num_visual_tokens, dtype=torch.float32)
        for j, pos in enumerate(image_positions):
            selector_matrix[pos, j] = 1.0
        # Reshape to [1, 1, seq_len, num_visual_tokens] for TTNN
        selector_matrix = selector_matrix.unsqueeze(0).unsqueeze(0)

        # Convert input_ids to TTNN (embed_tokens called inside trace)
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Convert other tensors to TTNN
        # embedded_ttnn already on device from patch_embed_ttnn above
        idx_ttnn = ttnn.from_torch(
            flat_idx,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        valid_mask_ttnn = ttnn.from_torch(
            valid_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        valid_token_ttnn = ttnn.from_torch(
            valid_token.flatten().float(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        selector_ttnn = ttnn.from_torch(
            selector_matrix,  # [1, 1, seq_len, num_visual_tokens]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        return {
            # Vision inputs
            "embedded": embedded_ttnn,
            "idx": idx_ttnn,
            "valid_mask": valid_mask_ttnn,
            "valid_token": valid_token_ttnn,
            "n_out": n_out,
            "k_pool": k_pool,
            "batch_size": batch_size,
            # Text inputs (embed_tokens called inside trace)
            "input_ids": input_ids_ttnn,
            "selector_matrix": selector_ttnn,
            "num_visual_tokens": num_visual_tokens,
            # Metadata
            "seq_len": seq_len,
        }

    def _execute_unified_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        inputs: dict,
    ) -> ttnn.Tensor:
        """Execute unified trace with new inputs."""
        # Copy vision inputs to trace tensors
        ttnn.copy(inputs["embedded"], trace_tensors["embedded"])
        ttnn.copy(inputs["idx"], trace_tensors["idx"])
        ttnn.copy(inputs["valid_mask"], trace_tensors["valid_mask"])
        ttnn.copy(inputs["valid_token"], trace_tensors["valid_token"])

        # Copy text/fusion inputs
        ttnn.copy(inputs["input_ids"], trace_tensors["input_ids"])
        ttnn.copy(inputs["selector_matrix"], trace_tensors["selector_matrix"])

        # Execute trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_output

    def _run_unified_prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        timing: dict,
    ) -> Tuple[ttnn.Tensor, dict]:
        """
        Run unified Vision + Fusion + Prefill in single trace.

        This eliminates the CPU roundtrip between vision and prefill,
        keeping everything on device using scatter_add for fusion.
        """
        seq_len = input_ids.shape[1]
        logger.info("Running unified Vision + Prefill trace...")

        # Prepare all inputs (vision + text embeddings + fusion indices)
        prep_start = time.perf_counter()
        inputs = self._prepare_unified_inputs(input_ids, pixel_values, pooled_patches_idx)
        timing["prep_ms"] = (time.perf_counter() - prep_start) * 1000
        logger.info(f"Input preparation: {timing['prep_ms']:.2f}ms")

        # Check if we have a cached trace for this configuration
        trace_key = (seq_len, inputs["num_visual_tokens"], inputs["n_out"], inputs["k_pool"])

        if not hasattr(self, "unified_traces"):
            self.unified_traces = {}

        if trace_key not in self.unified_traces:
            # First run: warmup + capture trace
            logger.info("Running unified warmup (compile)...")
            warmup_start = time.perf_counter()

            # Run full pipeline once for compilation (same ops as trace)
            # Step 1: Vision
            visual_embeddings = self.model.vision_backbone.forward_ttnn(
                images_embedded=inputs["embedded"],
                pooled_patches_idx_ttnn=inputs["idx"],
                valid_mask_ttnn=inputs["valid_mask"],
                valid_token_ttnn=inputs["valid_token"],
                n_out=inputs["n_out"],
                k_pool=inputs["k_pool"],
                batch_size=inputs["batch_size"],
            )

            # Step 2: Text embeddings (compile embed_tokens)
            text_embeddings = self.model.text_model.embed_tokens(inputs["input_ids"])

            # Step 3: Fusion via matmul with selector matrix
            visual_part = ttnn.matmul(
                inputs["selector_matrix"],
                visual_embeddings,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            fused_embed = ttnn.add(
                text_embeddings,
                visual_part,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Step 4: Text prefill
            rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)
            logits, _ = self.model.text_model.forward(
                hidden_states=fused_embed,
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,
                rot_mats=rot_mats,
            )
            ttnn.synchronize_device(self.mesh_device)
            timing["compile_ms"] = (time.perf_counter() - warmup_start) * 1000
            logger.info(f"Unified compile completed in {timing['compile_ms']:.2f}ms")

            # Deallocate warmup outputs
            ttnn.deallocate(visual_embeddings)
            ttnn.deallocate(text_embeddings)
            ttnn.deallocate(visual_part)
            ttnn.deallocate(fused_embed)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])
            ttnn.deallocate(logits)

            # Allocate trace tensors
            logger.info("Allocating unified trace tensors...")
            trace_tensors = self._allocate_unified_trace_tensors(
                seq_len=seq_len,
                num_visual_tokens=inputs["num_visual_tokens"],
                n_out=inputs["n_out"],
                k_pool=inputs["k_pool"],
                batch_size=inputs["batch_size"],
            )

            # Copy initial data to trace tensors
            ttnn.copy(inputs["embedded"], trace_tensors["embedded"])
            ttnn.copy(inputs["idx"], trace_tensors["idx"])
            ttnn.copy(inputs["valid_mask"], trace_tensors["valid_mask"])
            ttnn.copy(inputs["valid_token"], trace_tensors["valid_token"])
            ttnn.copy(inputs["input_ids"], trace_tensors["input_ids"])
            ttnn.copy(inputs["selector_matrix"], trace_tensors["selector_matrix"])

            # Synchronize before trace capture to ensure all copies are complete
            ttnn.synchronize_device(self.mesh_device)

            # Capture trace
            trace_id, trace_output = self._capture_unified_trace(trace_tensors)
            self.unified_traces[trace_key] = (trace_id, trace_tensors, trace_output)

        trace_id, trace_tensors, trace_output = self.unified_traces[trace_key]

        # Execute trace (actual timing measurement)
        ttft_start = time.perf_counter()
        logits = self._execute_unified_trace(trace_id, trace_tensors, trace_output, inputs)
        ttnn.synchronize_device(self.mesh_device)
        timing["ttft_ms"] = (time.perf_counter() - ttft_start) * 1000
        timing["vision_ms"] = 0  # Included in ttft_ms for unified trace

        logger.info(f"Unified TTFT: {timing['ttft_ms']:.2f}ms")

        # Cleanup temporary input tensors (trace tensors are reused)
        ttnn.deallocate(inputs["embedded"])
        ttnn.deallocate(inputs["idx"])
        ttnn.deallocate(inputs["valid_mask"])
        ttnn.deallocate(inputs["valid_token"])
        ttnn.deallocate(inputs["input_ids"])
        ttnn.deallocate(inputs["selector_matrix"])

        # Update position for decode
        self.reset_kv_cache(seq_len)

        return logits, timing

    def _allocate_decode_trace_tensors(self, hidden_dim: int = 4096) -> dict:
        """Allocate tensors needed for traced decode."""
        trace_hidden_states = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return {
            "hidden_states": trace_hidden_states,
        }

    def _capture_decode_trace(self, trace_tensors: dict) -> Tuple[int, ttnn.Tensor]:
        """Capture trace for decode phase (single token generation).

        The RoPE embedding lookup reads from self.rot_mat_idxs (managed via
        ttnn.plus_one outside the trace). KV cache position reads from
        self.current_pos (also managed via ttnn.plus_one outside the trace).
        """
        logger.info("Capturing decode trace...")

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        rot_mats = self.model.text_model.rotary_setup.get_rot_mats_decode_traced(self.rot_mat_idxs)

        logits_trace = self.model.text_model.forward_decode(
            hidden_states=trace_tensors["hidden_states"],
            kv_caches=self.kv_caches,
            current_pos=self.current_pos,
            rot_mats=rot_mats,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Decode trace captured")

        return trace_id, logits_trace

    def _execute_decode_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        hidden_states: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Execute captured decode trace with new inputs.

        Position tensors (current_pos, rot_mat_idxs) are kept on device and
        incremented via ttnn.plus_one in run_decode_step after trace execution.
        The trace reads their current values for RoPE and KV cache updates.
        """
        ttnn.copy(hidden_states, trace_tensors["hidden_states"])
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        return trace_output

    def warmup_prefill(
        self,
        hidden_states_ttnn: ttnn.Tensor,
        trace_tensors: dict,
        use_trace: bool,
    ):
        """Run prefill warm-up (compile) pass."""
        logger.info("Running prefill warm-up (compile)...")
        start = time.perf_counter()

        if use_trace:
            # Copy hidden states to trace tensor
            ttnn.copy(hidden_states_ttnn, trace_tensors["hidden_states"])

            # Run forward to compile - MUST pass kv_caches to compile fill_cache ops
            # This matches the tt_transformers pattern where compilation happens with kv_cache
            rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]
            logits, _ = self.model.text_model.forward(
                hidden_states=trace_tensors["hidden_states"],
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,  # Pass KV cache to compile fill_cache
                rot_mats=rot_mats,
            )
        else:
            logits, _ = self.model.text_model.forward(
                hidden_states=hidden_states_ttnn,
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,  # Also pass KV cache for non-traced warmup
            )

        compile_time = (time.perf_counter() - start) * 1000
        logger.info(f"Prefill compile completed in {compile_time:.2f}ms")
        return compile_time

    def warmup_decode(
        self,
        hidden_states: ttnn.Tensor,
        trace_tensors: dict,
        use_trace: bool,
    ):
        """Run decode warm-up (compile) pass."""
        logger.info("Running decode warm-up (compile)...")
        start = time.perf_counter()

        if use_trace:
            ttnn.copy(hidden_states, trace_tensors["hidden_states"])

            rot_mats = self.model.text_model.rotary_setup.get_rot_mats_decode_traced(self.rot_mat_idxs)
            logits = self.model.text_model.forward_decode(
                hidden_states=trace_tensors["hidden_states"],
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
                rot_mats=rot_mats,
            )
        else:
            logits = self.model.text_model.forward_decode(
                hidden_states=hidden_states,
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
                rot_mat_idxs=self.rot_mat_idxs,
            )

        compile_time = (time.perf_counter() - start) * 1000
        logger.info(f"Decode compile completed in {compile_time:.2f}ms")
        return compile_time

    def run_prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        use_trace: bool = False,
        use_vision_trace: bool = False,
        use_unified_trace: bool = False,
    ) -> Tuple[ttnn.Tensor, dict]:
        """
        Run prefill phase (process prompt + image).

        Args:
            input_ids: Token IDs
            pixel_values: Image tensor
            pooled_patches_idx: Patch pooling indices
            use_trace: Whether to trace text model prefill
            use_vision_trace: Whether to trace vision backbone (ViT + pooling)
            use_unified_trace: Whether to use unified Vision+Prefill trace (eliminates CPU roundtrip)

        Returns:
            Tuple of (logits, timing_dict)
        """
        # Initialize KV cache if needed
        self.init_kv_cache()

        seq_len = input_ids.shape[1]
        timing = {}

        # Unified trace path: Vision + embed_tokens + Fusion + Prefill in single trace
        # This eliminates CPU roundtrip between vision and text prefill
        if use_unified_trace and pixel_values is not None:
            return self._run_unified_prefill(input_ids, pixel_values, pooled_patches_idx, timing)

        # Start end-to-end TTFT timer (vision + fusion + prefill)
        e2e_ttft_start = None

        if use_vision_trace and pixel_values is not None:
            # Vision tracing path - uses forward_ttnn with TTNN-native gather
            logger.info("Preparing vision inputs for tracing...")
            vision_prep_start = time.perf_counter()
            vision_inputs = self._prepare_vision_inputs_for_trace(pixel_values, pooled_patches_idx)
            timing["vision_prep_ms"] = (time.perf_counter() - vision_prep_start) * 1000

            # Check if we need to capture a new trace
            if self.vision_trace_id is None:
                # First run: warmup + capture trace
                logger.info("Running vision warmup (compile)...")
                warmup_start = time.perf_counter()
                warmup_output = self.model.vision_backbone.forward_ttnn(
                    images_embedded=vision_inputs["embedded"],
                    pooled_patches_idx_ttnn=vision_inputs["idx"],
                    valid_mask_ttnn=vision_inputs["valid_mask"],
                    valid_token_ttnn=vision_inputs["valid_token"],
                    n_out=vision_inputs["n_out"],
                    k_pool=vision_inputs["k_pool"],
                    batch_size=vision_inputs["batch_size"],
                )
                ttnn.synchronize_device(self.mesh_device)
                timing["vision_compile_ms"] = (time.perf_counter() - warmup_start) * 1000
                logger.info(f"Vision compile completed in {timing['vision_compile_ms']:.2f}ms")
                ttnn.deallocate(warmup_output)

                # Allocate trace tensors
                logger.info("Allocating vision trace tensors...")
                self.vision_trace_tensors = self._allocate_vision_trace_tensors(
                    n_out=vision_inputs["n_out"],
                    k_pool=vision_inputs["k_pool"],
                    batch_size=vision_inputs["batch_size"],
                )

                # Copy initial data to trace tensors
                ttnn.copy(vision_inputs["embedded"], self.vision_trace_tensors["embedded"])
                ttnn.copy(vision_inputs["idx"], self.vision_trace_tensors["idx"])
                ttnn.copy(vision_inputs["valid_mask"], self.vision_trace_tensors["valid_mask"])
                ttnn.copy(vision_inputs["valid_token"], self.vision_trace_tensors["valid_token"])

                # Capture trace
                self.vision_trace_id, self.vision_trace_outputs = self._capture_vision_trace(self.vision_trace_tensors)

            # Execute vision trace - START of end-to-end TTFT measurement
            e2e_ttft_start = time.perf_counter()
            vision_trace_start = time.perf_counter()
            # Copy new inputs to trace tensors
            ttnn.copy(vision_inputs["embedded"], self.vision_trace_tensors["embedded"])
            ttnn.copy(vision_inputs["idx"], self.vision_trace_tensors["idx"])
            ttnn.copy(vision_inputs["valid_mask"], self.vision_trace_tensors["valid_mask"])
            ttnn.copy(vision_inputs["valid_token"], self.vision_trace_tensors["valid_token"])
            # Execute trace
            ttnn.execute_trace(self.mesh_device, self.vision_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            timing["vision_trace_ms"] = (time.perf_counter() - vision_trace_start) * 1000
            logger.info(f"Vision trace executed in {timing['vision_trace_ms']:.2f}ms")

            # Get visual embeddings from trace output
            visual_embeddings_ttnn = self.vision_trace_outputs

            # Fuse visual embeddings with text embeddings
            logger.info("Fusing visual and text embeddings...")
            fuse_start = time.perf_counter()
            hidden_states_ttnn = self._prepare_text_inputs_traced(
                input_ids, visual_embeddings_ttnn, vision_inputs["valid_token_torch"]
            )
            timing["fuse_ms"] = (time.perf_counter() - fuse_start) * 1000

            # Cleanup temporary vision input tensors (trace tensors are reused)
            ttnn.deallocate(vision_inputs["embedded"])
            ttnn.deallocate(vision_inputs["idx"])
            ttnn.deallocate(vision_inputs["valid_mask"])
            ttnn.deallocate(vision_inputs["valid_token"])

            # Total vision time
            timing["vision_ms"] = (
                timing.get("vision_prep_ms", 0) + timing.get("vision_trace_ms", 0) + timing.get("fuse_ms", 0)
            )
            logger.info(f"Vision processing completed in {timing['vision_ms']:.2f}ms (traced)")
        else:
            # Original path - no vision tracing
            # START of end-to-end TTFT measurement
            e2e_ttft_start = time.perf_counter()
            logger.info("Preparing inputs (vision processing)...")
            vision_start = time.perf_counter()
            hidden_states_ttnn = self._prepare_text_inputs(input_ids, pixel_values, pooled_patches_idx)
            timing["vision_ms"] = (time.perf_counter() - vision_start) * 1000
            logger.info(f"Vision processing completed in {timing['vision_ms']:.2f}ms")

        if use_trace:
            # Allocate trace tensors if needed
            if seq_len not in self.prefill_traces:
                logger.info("Allocating prefill trace tensors...")
                trace_tensors = self._allocate_prefill_trace_tensors(seq_len, hidden_dim=4096)

                # Warm-up (compile)
                timing["compile_prefill_ms"] = self.warmup_prefill(hidden_states_ttnn, trace_tensors, use_trace=True)

                # Capture trace
                trace_id, trace_output = self._capture_prefill_trace(trace_tensors)
                self.prefill_traces[seq_len] = (trace_id, trace_tensors, trace_output)

            trace_id, trace_tensors, trace_output = self.prefill_traces[seq_len]

            # Execute trace (actual TTFT measurement)
            ttft_start = time.perf_counter()
            logits = self._execute_prefill_trace(trace_id, trace_tensors, trace_output, hidden_states_ttnn)
            ttnn.synchronize_device(self.mesh_device)
            timing["ttft_ms"] = (time.perf_counter() - ttft_start) * 1000

            ttnn.deallocate(hidden_states_ttnn)
        else:
            # Warm-up (compile)
            timing["compile_prefill_ms"] = self.warmup_prefill(hidden_states_ttnn, None, use_trace=False)

            # Actual prefill (TTFT) - pass KV cache to fill during forward
            ttft_start = time.perf_counter()
            logits, _ = self.model.text_model.forward(
                hidden_states=hidden_states_ttnn,
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,  # Pass pre-allocated cache to fill
            )
            ttnn.synchronize_device(self.mesh_device)
            timing["ttft_ms"] = (time.perf_counter() - ttft_start) * 1000

        # Calculate end-to-end TTFT (vision start to prefill end)
        if e2e_ttft_start is not None:
            timing["e2e_ttft_ms"] = (time.perf_counter() - e2e_ttft_start) * 1000
            logger.info(f"End-to-end TTFT (vision + fusion + prefill): {timing['e2e_ttft_ms']:.2f}ms")

        logger.info(f"Prefill-only TTFT: {timing['ttft_ms']:.2f}ms")

        # Update position for decode
        self.reset_kv_cache(seq_len)

        return logits, timing

    def run_decode_step(
        self,
        token_id_ttnn: ttnn.Tensor,
        use_trace: bool = False,
        is_first: bool = False,
    ) -> Tuple[ttnn.Tensor, float]:
        """
        Run single decode step with CPU-free forward pass.

        After input prep, the entire forward pass (embed -> transformer -> logits ->
        argmax -> position increment) runs on device with no CPU roundtrips.

        Args:
            token_id_ttnn: Token ID on device [1, 1] uint32
            use_trace: Whether to use tracing
            is_first: Whether this is the first decode step (for warm-up)

        Returns:
            Tuple of (tt_next_token on device [1, 1] uint32, decode_time_ms)
        """
        hidden_states = self.model.text_model.embed_tokens(token_id_ttnn)

        if use_trace:
            if self.decode_trace_id is None:
                logger.info("Allocating decode trace tensors...")
                self.decode_trace_tensors = self._allocate_decode_trace_tensors(hidden_dim=4096)
                compile_time = self.warmup_decode(hidden_states, self.decode_trace_tensors, use_trace=True)
                trace_id, trace_output = self._capture_decode_trace(self.decode_trace_tensors)
                self.decode_trace_id = trace_id
                self.decode_trace_output = trace_output

            start_time = time.perf_counter()
            logits = self._execute_decode_trace(
                self.decode_trace_id,
                self.decode_trace_tensors,
                self.decode_trace_output,
                hidden_states,
            )
            ttnn.synchronize_device(self.mesh_device)
            decode_time = (time.perf_counter() - start_time) * 1000
        else:
            if is_first:
                compile_time = self.warmup_decode(hidden_states, None, use_trace=False)

            start_time = time.perf_counter()
            logits = self.model.text_model.forward_decode(
                hidden_states=hidden_states,
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
                rot_mat_idxs=self.rot_mat_idxs,
            )
            ttnn.synchronize_device(self.mesh_device)
            decode_time = (time.perf_counter() - start_time) * 1000

        ttnn.deallocate(hidden_states)

        # On-device argmax (no CPU logits transfer)
        tt_next_token = self._argmax_on_device(logits)

        # On-device position increment (no CPU tensor creation)
        ttnn.plus_one(self.current_pos)
        ttnn.plus_one(self.rot_mat_idxs)
        self.decode_position += 1

        return tt_next_token, decode_time

    def _argmax_on_device(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """
        Run argmax on device and return token as [1, 1] uint32 tensor.

        Note: ttnn.argmax requires TILE_LAYOUT input -- do NOT untilize first.
        """
        tt_token = ttnn.argmax(logits, dim=-1, keepdim=False)
        tt_token = ttnn.reshape(tt_token, [1, 1])
        if tt_token.dtype != ttnn.uint32:
            tt_token = ttnn.typecast(tt_token, dtype=ttnn.uint32)
        return tt_token

    def _read_token_from_device(self, tt_token: ttnn.Tensor) -> int:
        """Read a single token value from device (tiny transfer for EOS check)."""
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        return ttnn.to_torch(tt_token, mesh_composer=mesh_composer)[0].item()

    def run_inference(
        self,
        image_inputs: dict,
        prompt: str,
        max_new_tokens: int = 100,
        use_trace: bool = False,
        use_decode_trace: bool = False,
        use_vision_trace: bool = False,
        use_unified_trace: bool = False,
    ) -> Tuple[str, dict]:
        """
        Run full inference with autoregressive generation.

        Args:
            image_inputs: Dict from preprocess_image_molmo2
            prompt: Text prompt (should include <|image|> token)
            max_new_tokens: Maximum tokens to generate
            use_trace: Whether to use tracing for prefill
            use_decode_trace: Whether to use tracing for decode (disabled by default
                              to avoid memory corruption from tensor allocation during trace)
            use_vision_trace: Whether to use tracing for vision backbone (ViT + pooling)
            use_unified_trace: Whether to use unified Vision+Prefill trace (eliminates CPU roundtrip)

        Returns:
            Tuple of (output_text, perf_metrics)
        """
        # Build prompt with image tokens
        image_grid = image_inputs["image_grids"][0]
        image_tokens_str = get_image_tokens(image_grid)
        full_prompt = prompt.replace(IMAGE_PROMPT, image_tokens_str)

        # Tokenize input
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")

        # Get pooling indices
        pooled_patches_idx = image_inputs["image_token_pooling"].unsqueeze(0)

        # Get pixel values (need to reshape for our model)
        # pixel_values shape: [n_crops, n_patches, patch_pixels]
        # We need: [B, C, H, W] for vision encoder
        pixel_values = image_inputs["pixel_values"]

        # Run prefill
        logits, prefill_timing = self.run_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            use_trace=use_trace,
            use_vision_trace=use_vision_trace,
            use_unified_trace=use_unified_trace,
        )

        # Get first prediction from prefill (one-time CPU argmax is acceptable)
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0].squeeze()
        if logits_torch.dim() == 2:
            next_token_logits = logits_torch[-1, :]
        else:
            next_token_logits = logits_torch

        next_token = torch.argmax(next_token_logits).item()
        generated_tokens = [next_token]

        # Put first token on device for CPU-free decode loop
        tt_next_token = ttnn.from_torch(
            torch.tensor([[next_token]], dtype=torch.long),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Autoregressive generation -- forward pass is fully on device
        decode_times = []
        eos_token_id = self.tokenizer.eos_token_id

        for i in range(max_new_tokens - 1):
            if next_token == eos_token_id:
                break

            # CPU-free decode: embed -> forward -> argmax -> plus_one all on device
            tt_next_token, decode_time = self.run_decode_step(
                tt_next_token,
                use_trace=use_decode_trace,
                is_first=(i == 0),
            )
            decode_times.append(decode_time)

            # Read back single token int for EOS check and logging
            next_token = self._read_token_from_device(tt_next_token)
            generated_tokens.append(next_token)

            # Log progress
            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i + 1} tokens, last decode: {decode_time:.2f}ms")

        # Decode generated tokens
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Calculate metrics
        total_decode_time = sum(decode_times) if decode_times else 0.0
        avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0.0
        tokens_per_sec = len(decode_times) / (total_decode_time / 1000) if total_decode_time > 0 else 0

        perf_metrics = {
            "vision_ms": prefill_timing.get("vision_ms", 0),
            "vision_trace_ms": prefill_timing.get("vision_trace_ms", 0),
            "compile_vision_ms": prefill_timing.get("compile_vision_ms", 0),
            "compile_prefill_ms": prefill_timing.get("compile_prefill_ms", 0),
            "ttft_ms": prefill_timing.get("ttft_ms", 0),
            "e2e_ttft_ms": prefill_timing.get("e2e_ttft_ms", 0),  # End-to-end TTFT (vision + fusion + prefill)
            # Unified trace specific metrics
            "prep_ms": prefill_timing.get("prep_ms", 0),
            "compile_ms": prefill_timing.get("compile_ms", 0),
            # Decode metrics
            "avg_decode_ms": avg_decode_time,
            "total_decode_ms": total_decode_time,
            "input_tokens": input_ids.shape[1],
            "generated_tokens": len(generated_tokens),
            "num_generated_tokens": len(generated_tokens),  # Alias for compatibility
            "tokens_per_sec": tokens_per_sec,
            "decode_throughput": tokens_per_sec,  # Alias for compatibility
            "output_text": output_text,
        }

        logger.info(f"Input tokens: {input_ids.shape[1]}")
        logger.info(f"Generated {len(generated_tokens)} tokens")
        logger.info(f"Output: '{output_text[:100]}...' " if len(output_text) > 100 else f"Output: '{output_text}'")

        return output_text, perf_metrics

    def run_video_inference(
        self,
        video_inputs: dict,
        prompt: str,
        max_new_tokens: int = 200,
        use_trace: bool = False,
        use_decode_trace: bool = False,
        use_vision_trace: bool = False,
        use_unified_trace: bool = False,
    ) -> Tuple[str, dict]:
        """
        Run full inference on a video input with autoregressive generation.

        Args:
            video_inputs: Dict from preprocess_video_molmo2
            prompt: Text prompt (should include <|video|> token)
            max_new_tokens: Maximum tokens to generate
            use_trace: Whether to use tracing for prefill
            use_decode_trace: Whether to use tracing for decode
            use_vision_trace: Whether to use tracing for vision backbone
            use_unified_trace: Whether to use unified Vision+Prefill trace

        Returns:
            Tuple of (output_text, perf_metrics)
        """
        n_frames = video_inputs["n_frames"]
        timestamps = video_inputs["timestamps"]
        pooled_h = video_inputs["pooled_h"]
        pooled_w = video_inputs["pooled_w"]

        # Build prompt with video tokens (replaces <|video|> with per-frame patch tokens)
        video_tokens_str = get_video_tokens(n_frames, pooled_h, pooled_w, timestamps)
        full_prompt = prompt.replace(VIDEO_PROMPT, video_tokens_str)

        # Tokenize input
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")

        # pooled_patches_idx shape: [n_frames, N_out, K_pool] -- no unsqueeze needed
        pooled_patches_idx = video_inputs["image_token_pooling"]  # [n_frames, N_out, K_pool]
        pixel_values = video_inputs["pixel_values"]  # [n_frames, 3, H, W]

        num_visual_tokens = n_frames * pooled_h * pooled_w
        logger.info(
            f"Video: {n_frames} frames, {num_visual_tokens} visual tokens, {input_ids.shape[1]} total input tokens"
        )

        # Vision timing starts here
        vision_start = time.perf_counter()

        # Run prefill (vision + text model)
        logits, prefill_timing = self.run_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            use_trace=use_trace,
            use_vision_trace=use_vision_trace,
            use_unified_trace=use_unified_trace,
        )

        vision_total_ms = (time.perf_counter() - vision_start) * 1000

        # First token from prefill logits (one-time CPU argmax)
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0].squeeze()
        if logits_torch.dim() == 2:
            next_token_logits = logits_torch[-1, :]
        else:
            next_token_logits = logits_torch

        next_token = torch.argmax(next_token_logits).item()
        generated_tokens = [next_token]

        # Put first token on device for CPU-free decode loop
        tt_next_token = ttnn.from_torch(
            torch.tensor([[next_token]], dtype=torch.long),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Autoregressive generation
        decode_times = []
        eos_token_id = self.tokenizer.eos_token_id

        for i in range(max_new_tokens - 1):
            if next_token == eos_token_id:
                break

            tt_next_token, decode_time = self.run_decode_step(
                tt_next_token,
                use_trace=use_decode_trace,
                is_first=(i == 0),
            )
            decode_times.append(decode_time)

            next_token = self._read_token_from_device(tt_next_token)
            generated_tokens.append(next_token)

            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i + 1} tokens, last decode: {decode_time:.2f}ms")

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        total_decode_time = sum(decode_times) if decode_times else 0.0
        avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0.0
        tokens_per_sec = len(decode_times) / (total_decode_time / 1000) if total_decode_time > 0 else 0
        vision_ms = prefill_timing.get("vision_ms", vision_total_ms)
        frames_per_sec = n_frames / (vision_ms / 1000) if vision_ms > 0 else 0

        perf_metrics = {
            "n_frames": n_frames,
            "frames_per_sec": frames_per_sec,
            "vision_ms": vision_ms,
            "vision_trace_ms": prefill_timing.get("vision_trace_ms", 0),
            "compile_vision_ms": prefill_timing.get("compile_vision_ms", 0),
            "compile_prefill_ms": prefill_timing.get("compile_prefill_ms", 0),
            "ttft_ms": prefill_timing.get("ttft_ms", 0),
            "e2e_ttft_ms": prefill_timing.get("e2e_ttft_ms", 0),
            "prep_ms": prefill_timing.get("prep_ms", 0),
            "compile_ms": prefill_timing.get("compile_ms", 0),
            "avg_decode_ms": avg_decode_time,
            "total_decode_ms": total_decode_time,
            "input_tokens": input_ids.shape[1],
            "generated_tokens": len(generated_tokens),
            "tokens_per_sec": tokens_per_sec,
            "output_text": output_text,
        }

        logger.info(f"Video: {n_frames} frames processed")
        logger.info(f"Vision throughput: {frames_per_sec:.2f} frames/sec")
        logger.info(f"TTFT: {prefill_timing.get('ttft_ms', 0):.2f}ms")
        logger.info(f"Decode: {tokens_per_sec:.2f} tok/s ({len(generated_tokens)} tokens)")
        logger.info(f"Output: '{output_text[:100]}...' " if len(output_text) > 100 else f"Output: '{output_text}'")

        return output_text, perf_metrics


def run_video_demo(
    video_path: str,
    prompt: str = "<|video|> Describe what happens in this video.",
    max_new_tokens: int = 200,
    device_id: int = 0,
    num_layers: Optional[int] = None,
    max_seq_len: int = 16384,
    max_frames: int = VIDEO_MAX_FRAMES,
    max_fps: float = VIDEO_MAX_FPS,
    use_trace: bool = False,
    use_decode_trace: bool = False,
    use_vision_trace: bool = False,
    use_unified_trace: bool = False,
):
    """
    Run the Molmo2 demo with video input.

    Args:
        video_path: Path or URL to video file (.mp4, .webm)
        prompt: Text prompt (must include <|video|>)
        max_new_tokens: Maximum tokens to generate
        device_id: TTNN device ID
        num_layers: Number of text layers (default: 36)
        max_seq_len: Maximum sequence length for KV cache (default: 16384 for video)
        max_frames: Maximum frames to sample from video (default: 8)
        max_fps: Maximum frames per second to sample (default: 2.0)
        use_trace: Whether to use tracing for prefill
        use_decode_trace: Whether to use tracing for decode
        use_vision_trace: Whether to use tracing for vision backbone
        use_unified_trace: Whether to use unified Vision+Prefill trace
    """
    logger.info("=" * 60)
    logger.info("Molmo2-8B Video Demo")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = load_processor()

    # Preprocess video
    logger.info(f"Preprocessing video: {video_path}")
    video_extraction_start = time.perf_counter()
    video_inputs = preprocess_video_molmo2(
        video_path,
        max_frames=max_frames,
        max_fps=max_fps,
    )
    video_extraction_ms = (time.perf_counter() - video_extraction_start) * 1000
    n_frames = video_inputs["n_frames"]
    logger.info(f"Extracted {n_frames} frames in {video_extraction_ms:.2f}ms")
    logger.info(f"Frame extraction: {n_frames / (video_extraction_ms / 1000):.2f} frames/sec (CPU)")

    # Load weights
    state_dict = load_model_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    logger.info(f"Opening TTNN mesh device with shape {mesh_shape}")
    device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Opened mesh device with {device.get_num_devices()} devices")

    try:
        model = create_model(device, state_dict, num_layers, max_seq_len=max_seq_len)
        text_num_layers = num_layers if num_layers is not None else 36

        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=max_seq_len,
        )

        logger.info("\n" + "=" * 60)
        logger.info(f"Prompt: {prompt}")
        logger.info("=" * 60)

        response, perf_metrics = generator.run_video_inference(
            video_inputs=video_inputs,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_trace=use_trace,
            use_decode_trace=use_decode_trace,
            use_vision_trace=use_vision_trace,
            use_unified_trace=use_unified_trace,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Video Performance Metrics:")
        logger.info(f"  Frames processed:    {perf_metrics['n_frames']}")
        logger.info(
            f"  Frame extraction:    {video_extraction_ms:.2f}ms ({n_frames / (video_extraction_ms/1000):.2f} frames/sec CPU)"
        )
        logger.info(
            f"  Vision (TTNN):       {perf_metrics['vision_ms']:.2f}ms ({perf_metrics['frames_per_sec']:.2f} frames/sec)"
        )
        logger.info(f"  TTFT:                {perf_metrics['ttft_ms']:.2f}ms")
        if perf_metrics.get("e2e_ttft_ms", 0) > 0:
            logger.info(f"  E2E TTFT:            {perf_metrics['e2e_ttft_ms']:.2f}ms")
        logger.info(f"  Input tokens:        {perf_metrics['input_tokens']}")
        logger.info(f"  Generated tokens:    {perf_metrics['generated_tokens']}")
        logger.info(
            f"  Decode:              {perf_metrics['avg_decode_ms']:.2f}ms/token ({perf_metrics['tokens_per_sec']:.2f} tok/s)"
        )
        logger.info("=" * 60)
        logger.info(f"Response: {response}")
        logger.info("=" * 60)

        return perf_metrics

    finally:
        ttnn.close_mesh_device(device)
        logger.info("Device closed")


def run_demo(
    image_path: Optional[str] = None,
    prompt: str = "<|image|> Describe this image in detail.",
    max_new_tokens: int = 100,
    device_id: int = 0,
    num_layers: Optional[int] = None,
    max_seq_len: int = 2048,
    use_trace: bool = False,
    use_decode_trace: bool = False,
    use_vision_trace: bool = False,
    use_unified_trace: bool = False,
):
    """
    Run the Molmo2 demo.

    Args:
        image_path: Path to input image (uses default if None)
        prompt: Text prompt for the model (must include <|image|>)
        max_new_tokens: Maximum tokens to generate
        device_id: TTNN device ID
        num_layers: Number of text layers (default: 36)
        max_seq_len: Maximum sequence length for KV cache (default: 2048 for image)
        use_trace: Whether to use tracing for text prefill
        use_decode_trace: Whether to use tracing for decode
        use_vision_trace: Whether to use tracing for vision backbone
        use_unified_trace: Whether to use unified Vision+Prefill trace (eliminates CPU roundtrip)
    """
    if image_path is None:
        image_path = str(DEFAULT_IMAGE)

    logger.info("=" * 60)
    logger.info("Molmo2-8B Demo")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = load_processor()

    # Preprocess image using Molmo2 processor
    logger.info("Preprocessing image...")
    image_inputs = preprocess_image_molmo2(image_path)
    logger.info(f"Image preprocessed: {image_inputs['image_num_crops'].item()} crops")

    # Load weights
    state_dict = load_model_weights()

    # Open multi-device mesh for T3K (8 devices) to enable bfloat16 weight sharding
    # This prevents numerical overflow during decode by using higher precision weights
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    logger.info(f"Opening TTNN mesh device with shape {mesh_shape}")
    device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Opened mesh device with {device.get_num_devices()} devices")

    try:
        # Create model
        model = create_model(device, state_dict, num_layers, max_seq_len=max_seq_len)
        text_num_layers = num_layers if num_layers is not None else 36

        # Create generator
        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=max_seq_len,
        )

        # Run inference
        logger.info("\n" + "=" * 60)
        logger.info(f"Prompt: {prompt}")
        logger.info("=" * 60)

        response, perf_metrics = generator.run_inference(
            image_inputs=image_inputs,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_trace=use_trace,
            use_decode_trace=use_decode_trace,
            use_vision_trace=use_vision_trace,
            use_unified_trace=use_unified_trace,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Performance Metrics:")
        # Check if unified trace was actually used (indicated by prep_ms and compile_ms being set)
        unified_trace_used = use_unified_trace and perf_metrics.get("prep_ms", 0) > 0
        if unified_trace_used:
            logger.info("  [Unified Vision+Prefill Trace]")
            logger.info(f"    - Input preparation: {perf_metrics['prep_ms']:.2f}ms")
            if perf_metrics.get("compile_ms", 0) > 0:
                logger.info(f"    - Unified compile: {perf_metrics['compile_ms']:.2f}ms")
            logger.info(f"  TTFT (Vision+Prefill): {perf_metrics['ttft_ms']:.2f}ms")
        else:
            logger.info(f"  Vision processing: {perf_metrics['vision_ms']:.2f}ms")
            if perf_metrics.get("vision_trace_ms", 0) > 0:
                logger.info(f"    - Vision trace execution: {perf_metrics['vision_trace_ms']:.2f}ms")
            if perf_metrics.get("compile_vision_ms", 0) > 0:
                logger.info(f"    - Vision compile: {perf_metrics['compile_vision_ms']:.2f}ms")
            if perf_metrics.get("compile_prefill_ms", 0) > 0:
                logger.info(f"  Prefill compile: {perf_metrics['compile_prefill_ms']:.2f}ms")
            logger.info(f"  Prefill-only TTFT: {perf_metrics['ttft_ms']:.2f}ms")
            if perf_metrics.get("e2e_ttft_ms", 0) > 0:
                logger.info(f"  ** End-to-End TTFT (Vision+Fusion+Prefill): {perf_metrics['e2e_ttft_ms']:.2f}ms **")
        logger.info(f"  Avg decode time: {perf_metrics['avg_decode_ms']:.2f}ms")
        logger.info(f"  Total decode time: {perf_metrics['total_decode_ms']:.2f}ms")
        logger.info(f"  Input tokens: {perf_metrics['input_tokens']}")
        logger.info(f"  Generated tokens: {perf_metrics['generated_tokens']}")
        logger.info(f"  Decode throughput: {perf_metrics['tokens_per_sec']:.2f} tok/s")
        logger.info("=" * 60)
        logger.info(f"Output: {perf_metrics['output_text']}")
        logger.info("=" * 60)

        return perf_metrics

    finally:
        ttnn.close_mesh_device(device)
        logger.info("Device closed")


def main():
    parser = argparse.ArgumentParser(description="Molmo2-8B Demo")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (uses default dog.jpg if not specified)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path or URL to input video (.mp4, .webm) for video inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for the model (must include <|image|> or <|video|> token)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="TTNN device ID",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of text layers (default: 36, use fewer for faster testing)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length for KV cache (default: 2048 for image, 16384 for video)",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=VIDEO_MAX_FRAMES,
        help=f"Maximum video frames to sample (default: {VIDEO_MAX_FRAMES})",
    )
    parser.add_argument(
        "--max-video-fps",
        type=float,
        default=VIDEO_MAX_FPS,
        help=f"Maximum frames per second for video sampling (default: {VIDEO_MAX_FPS})",
    )
    parser.add_argument(
        "--use-trace",
        action="store_true",
        help="Enable tracing for prefill (improved compilation)",
    )
    parser.add_argument(
        "--use-decode-trace",
        action="store_true",
        help="Enable tracing for decode (experimental - may cause memory corruption)",
    )
    parser.add_argument(
        "--use-vision-trace",
        action="store_true",
        help="Enable tracing for vision backbone (ViT + pooling)",
    )
    parser.add_argument(
        "--use-unified-trace",
        action="store_true",
        help="Enable unified Vision+Prefill trace (eliminates CPU roundtrip, best TTFT)",
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="With --video: run HuggingFace PyTorch only (skip TTNN; no Tenstorrent device)",
    )
    parser.add_argument(
        "--also-torch",
        action="store_true",
        help="With --video: after the TTNN run, also run HuggingFace PyTorch for comparison "
        "(ignored if --torch is set; needs HF weights + GPU or CPU RAM)",
    )

    args = parser.parse_args()

    if args.video is not None:
        prompt = args.prompt if args.prompt is not None else f"{VIDEO_PROMPT} Describe what happens in this video."
        if args.torch and args.also_torch:
            logger.info("--also-torch ignored: --torch selects HuggingFace-only mode (no TTNN run).")
        # Separate paths: --torch = HF-only; default = TTNN; --also-torch = TTNN then HF
        if args.torch:
            run_video_demo_torch(
                video_path=args.video,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                max_frames=args.max_video_frames,
                max_fps=args.max_video_fps,
            )
        else:
            max_seq_len = args.max_seq_len if args.max_seq_len is not None else 16384
            run_video_demo(
                video_path=args.video,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                device_id=args.device,
                num_layers=args.num_layers,
                max_seq_len=max_seq_len,
                max_frames=args.max_video_frames,
                max_fps=args.max_video_fps,
                use_trace=args.use_trace,
                use_decode_trace=args.use_decode_trace,
                use_vision_trace=args.use_vision_trace,
                use_unified_trace=args.use_unified_trace,
            )
            if args.also_torch:
                logger.info("")
                logger.info("=" * 60)
                logger.info("Second pass: HuggingFace PyTorch reference (same prompt / video)")
                logger.info("=" * 60)
                run_video_demo_torch(
                    video_path=args.video,
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    max_frames=args.max_video_frames,
                    max_fps=args.max_video_fps,
                )
    else:
        prompt = args.prompt if args.prompt is not None else "<|image|> Describe this image in detail."
        max_seq_len = args.max_seq_len if args.max_seq_len is not None else 2048
        run_demo(
            image_path=args.image,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            device_id=args.device,
            num_layers=args.num_layers,
            max_seq_len=max_seq_len,
            use_trace=args.use_trace,
            use_decode_trace=args.use_decode_trace,
            use_vision_trace=args.use_vision_trace,
            use_unified_trace=args.use_unified_trace,
        )


if __name__ == "__main__":
    main()
