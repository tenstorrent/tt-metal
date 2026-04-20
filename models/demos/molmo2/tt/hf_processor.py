"""
HuggingFace processor wrapper for Molmo2.

Loads ``AutoProcessor.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)``.
Image and video preprocessing use the HF remote implementations and **their defaults**:

- Video: `video_processing_molmo2.Molmo2VideoProcessor`
  (see https://huggingface.co/allenai/Molmo2-8B/blob/main/video_processing_molmo2.py )
- Image: `image_processing_molmo2.Molmo2ImageProcessor`
  (see https://huggingface.co/allenai/Molmo2-8B/blob/main/image_processing_molmo2.py )

Calls use ``return_tensors`` plus ``text`` and ``images`` or ``videos``. Optional
video overrides: ``num_frames``, ``max_fps``, ``fps``, ``sampling_fps`` (see
``Molmo2VideoProcessor`` / HF config). Other kwargs use HF defaults unless set.

Converts HF outputs to tensors expected by our TTNN model.
"""

import math
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer

# Global processor instance (lazy loaded)
_processor = None
_tokenizer = None


def get_processor():
    """Get or create the HF processor."""
    global _processor
    if _processor is None:
        _processor = AutoProcessor.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)
    return _processor


def get_tokenizer():
    """Get or create the HF tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)
    return _tokenizer


def pooling_size_from_k_pool(k_pool: int) -> list[int]:
    """
    Map total pooling width ``k_pool = pool_h * pool_w`` to HF ``pooling_size`` ``[pool_h, pool_w]``.

    Uses a square grid when ``k_pool`` is a perfect square; otherwise the factorization with
    factors closest to ``sqrt(k_pool)`` (e.g. 6 -> ``[2, 3]``).
    """
    if k_pool < 1:
        raise ValueError(f"k_pool must be >= 1, got {k_pool}")
    r = int(math.isqrt(k_pool))
    if r * r == k_pool:
        return [r, r]
    for h in range(r, 0, -1):
        if k_pool % h == 0:
            w = k_pool // h
            return [h, w] if h <= w else [w, h]
    raise ValueError(f"Cannot factor k_pool={k_pool} into pooling_size")


def apply_chat_template(prompt: str) -> str:
    """
    Apply Molmo2 chat template to a prompt.

    Args:
        prompt: Raw prompt text (may contain <|image|> or <|video|>)

    Returns:
        Prompt with chat template applied
    """
    tokenizer = get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def hf_patches_to_images(pixel_values: np.ndarray, patch_size: int = 14) -> torch.Tensor:
    """
    Convert HF patch format to image format.

    HF format: [batch, n_patches, patch_pixels]
        where n_patches = h_patches * w_patches
        and patch_pixels = patch_h * patch_w * channels

    Output: [batch, channels, height, width]

    Args:
        pixel_values: HF format patches [batch, 729, 588]
        patch_size: Patch size (14 for Molmo2)

    Returns:
        Images in [batch, 3, H, W] format
    """
    batch_size = pixel_values.shape[0]
    n_patches = pixel_values.shape[1]  # 729 = 27*27
    patch_pixels = pixel_values.shape[2]  # 588 = 14*14*3

    # Calculate grid dimensions
    h_patches = int(np.sqrt(n_patches))  # 27
    w_patches = h_patches  # 27 (assuming square)
    channels = 3

    assert h_patches * w_patches == n_patches
    assert patch_size * patch_size * channels == patch_pixels

    height = h_patches * patch_size  # 378
    width = w_patches * patch_size  # 378

    # Reshape: [batch, h_patches * w_patches, patch_h * patch_w * channels]
    #       -> [batch, h_patches, w_patches, patch_h, patch_w, channels]
    patches = pixel_values.reshape(batch_size, h_patches, w_patches, patch_size, patch_size, channels)

    # Rearrange: [batch, h_patches, w_patches, patch_h, patch_w, channels]
    #         -> [batch, channels, h_patches, patch_h, w_patches, patch_w]
    #         -> [batch, channels, height, width]
    images = np.transpose(patches, (0, 5, 1, 3, 2, 4))
    images = images.reshape(batch_size, channels, height, width)

    return torch.from_numpy(images).float()


def preprocess_text(text: str, apply_template: bool = True) -> dict:
    """
    Preprocess text-only input using HF processor.

    Args:
        text: Input text prompt
        apply_template: Whether to apply chat template (default True)

    Returns:
        Dict with:
          - input_ids: [1, seq_len] tensor
          - attention_mask: [1, seq_len] tensor
    """
    processor = get_processor()

    # Apply chat template if requested
    prompt = apply_chat_template(text) if apply_template else text

    result = processor(text=prompt, return_tensors="pt")

    out = {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }
    if "token_type_ids" in result:
        out["token_type_ids"] = result["token_type_ids"].long()
    return out


def preprocess_image(
    image_path: str,
    prompt: str,
    apply_template: bool = True,
    pooling_size: Optional[list[int]] = None,
) -> dict:
    """
    Preprocess image + text using HF processor.

    Args:
        image_path: Path to image file
        prompt: Text prompt (should contain <|image|>)
        apply_template: Whether to apply chat template (default True)
        pooling_size: Optional ``[pool_h, pool_w]`` for the vision adapter (default: HF ``[2, 2]``).

    Returns:
        Dict with:
          - input_ids: [1, seq_len] tensor (with image tokens already inserted)
          - attention_mask: [1, seq_len] tensor
          - pixel_values: [n_crops, 3, H, W] tensor
          - image_token_pooling: [n_tokens, k_pool] tensor
          - pooled_patches_idx_flat: [1, n_tokens * k_pool] tensor for TTNN embedding
          - valid_mask_flat: [1, 1, n_tokens * k_pool, 1] mask tensor
          - image_grids: [[resized_h, resized_w, height, width]]
          - n_crops: int
          - n_tokens: int (number of visual tokens)
          - k_pool: int (4 for images)
    """
    from PIL import Image

    processor = get_processor()
    image = Image.open(image_path).convert("RGB")

    # Apply chat template if requested
    text = apply_chat_template(prompt) if apply_template else prompt

    call_kwargs = dict(text=text, images=image, return_tensors="np")
    if pooling_size is not None:
        call_kwargs["pooling_size"] = list(pooling_size)
    result = processor(**call_kwargs)

    # Convert pixel_values from HF patch format to image format
    pixel_values = hf_patches_to_images(result["pixel_values"])

    # Pooling indices: [n_tokens, k_pool]
    pooling_idx = result["image_token_pooling"]
    n_tokens = pooling_idx.shape[0]
    k_pool = pooling_idx.shape[1]

    # Create flat version for TTNN: [1, n_tokens * k_pool]
    # Replace -1 (invalid) with 0 for embedding lookup (will be masked)
    pooling_idx_flat = np.clip(pooling_idx.flatten(), 0, None)

    # Create valid mask: 1 for valid, 0 for invalid
    valid_mask = (result["image_token_pooling"] >= 0).astype(np.float32)
    valid_mask_flat = valid_mask.flatten()

    out = {
        "input_ids": torch.from_numpy(result["input_ids"]),
        "attention_mask": torch.from_numpy(result["attention_mask"]),
        "pixel_values": pixel_values,
        "image_token_pooling": torch.from_numpy(result["image_token_pooling"]).long(),
        "pooled_patches_idx_flat": torch.from_numpy(pooling_idx_flat.reshape(1, -1)).long(),
        "valid_mask_flat": torch.from_numpy(valid_mask_flat.reshape(1, 1, -1, 1)).float(),
        "image_grids": result["image_grids"],
        "n_crops": pixel_values.shape[0],
        "n_tokens": n_tokens,
        "k_pool": k_pool,
    }
    if "token_type_ids" in result:
        out["token_type_ids"] = torch.from_numpy(np.array(result["token_type_ids"])).long()
    return out


def preprocess_video(
    video_path: str,
    prompt: str,
    num_frames: Optional[int] = None,
    apply_template: bool = True,
    max_fps: Optional[float] = None,
    fps: Optional[float] = None,
    sampling_fps: Optional[float] = None,
    pooling_size: Optional[list[int]] = None,
) -> dict:
    """
    Preprocess video + text using HF processor.

    Args:
        video_path: Path to video file
        prompt: Text prompt (should contain <|video|>)
        num_frames: Passed to HF as ``num_frames`` when set; otherwise HF default.
        apply_template: Whether to apply chat template (default True)
        max_fps: Cap on frames per second during sampling (HF ``max_fps``). ``None`` = omit
            (HF processor default from config, typically 2.0).
        fps: Optional container / target frame rate hint (HF ``fps``; semantics match HF video processor).
        sampling_fps: Optional sampling rate override (HF ``sampling_fps``).
        pooling_size: Optional ``[pool_h, pool_w]`` for the vision adapter (default: HF ``[3, 3]`` for video).

    Returns:
        Dict with:
          - input_ids: [1, seq_len] tensor (with video tokens already inserted)
          - attention_mask: [1, seq_len] tensor
          - pixel_values: [n_frames, 729, 588] tensor (HF patch format, already unfolded)
          - image_token_pooling: [n_frames * h * w, k_pool] tensor
          - pooled_patches_idx_flat: [1, n_tokens * k_pool] tensor for TTNN embedding
          - valid_mask_flat: [1, 1, n_tokens * k_pool, 1] mask tensor
          - video_grids: [[n_frames, h, w]]
          - n_frames: int
          - pooled_h: int
          - pooled_w: int
          - n_tokens: int (n_frames * pooled_h * pooled_w)
          - k_pool: int (9 for videos)
          - timestamps: np.ndarray
    """
    processor = get_processor()

    # Apply chat template if requested
    text = apply_chat_template(prompt) if apply_template else prompt

    call_kwargs = dict(text=text, videos=video_path, return_tensors="np")
    if num_frames is not None:
        call_kwargs["num_frames"] = num_frames
    if max_fps is not None:
        call_kwargs["max_fps"] = max_fps
    if fps is not None:
        call_kwargs["fps"] = fps
    if sampling_fps is not None:
        call_kwargs["sampling_fps"] = sampling_fps
    if pooling_size is not None:
        call_kwargs["pooling_size"] = list(pooling_size)

    result = processor(**call_kwargs)

    # Convert pixel_values from HF patch format to image format
    pixel_values = hf_patches_to_images(result["pixel_values_videos"])

    # Extract video grid info
    n_frames, pooled_h, pooled_w = result["video_grids"][0]

    # Pooling indices: [n_tokens, k_pool]
    pooling_idx = result["video_token_pooling"]
    n_tokens = pooling_idx.shape[0]
    k_pool = pooling_idx.shape[1]

    # Create flat version for TTNN: [1, n_tokens * k_pool]
    # Replace -1 (invalid) with 0 for embedding lookup (will be masked)
    pooling_idx_flat = np.clip(pooling_idx.flatten(), 0, None)

    # Create valid mask: 1 for valid, 0 for invalid
    valid_mask = (pooling_idx >= 0).astype(np.float32)
    valid_mask_flat = valid_mask.flatten()

    # Get timestamps from metadata (prefer decoded fps; else kwargs / defaults)
    metadata = result.get("video_metadata", [None])[0]
    fallback_fps = 2.0
    if max_fps is not None:
        fallback_fps = float(max_fps)
    elif fps is not None:
        fallback_fps = float(fps)
    elif sampling_fps is not None:
        fallback_fps = float(sampling_fps)

    if metadata is not None:
        meta_fps = getattr(metadata, "fps", None)
        fps_for_ts = float(meta_fps) if meta_fps is not None else fallback_fps
        frames_indices = getattr(metadata, "frames_indices", np.arange(n_frames))
        timestamps = np.array(frames_indices, dtype=float) / fps_for_ts
    else:
        timestamps = np.arange(n_frames, dtype=float) / fallback_fps

    out = {
        "input_ids": torch.from_numpy(result["input_ids"]),
        "attention_mask": torch.from_numpy(result["attention_mask"]),
        "pixel_values": pixel_values,
        "image_token_pooling": torch.from_numpy(pooling_idx).long(),
        "pooled_patches_idx_flat": torch.from_numpy(pooling_idx_flat.reshape(1, -1)).long(),
        "valid_mask_flat": torch.from_numpy(valid_mask_flat.reshape(1, 1, -1, 1)).float(),
        "video_grids": result["video_grids"],
        "n_frames": int(n_frames),
        "pooled_h": int(pooled_h),
        "pooled_w": int(pooled_w),
        "n_tokens": n_tokens,
        "k_pool": k_pool,
        "timestamps": timestamps,
    }
    if "token_type_ids" in result:
        out["token_type_ids"] = torch.from_numpy(np.array(result["token_type_ids"])).long()
    return out
