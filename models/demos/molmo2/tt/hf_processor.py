"""
HuggingFace processor wrapper for Molmo2.

Uses the official Molmo2Processor from HuggingFace for correct preprocessing:
- Image: pooling_size [2,2] -> k_pool=4
- Video: pooling_size [3,3] -> k_pool=9

Converts HF outputs to format expected by our TTNN model.
"""

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

    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }


def preprocess_image(image_path: str, prompt: str, apply_template: bool = True) -> dict:
    """
    Preprocess image + text using HF processor.

    Args:
        image_path: Path to image file
        prompt: Text prompt (should contain <|image|>)
        apply_template: Whether to apply chat template (default True)

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

    result = processor(text=text, images=image, return_tensors="np")

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

    return {
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


def preprocess_video(
    video_path: str,
    prompt: str,
    num_frames: Optional[int] = None,
    apply_template: bool = True,
) -> dict:
    """
    Preprocess video + text using HF processor.

    Args:
        video_path: Path to video file
        prompt: Text prompt (should contain <|video|>)
        num_frames: Maximum frames to extract (default: HF default 384)
        apply_template: Whether to apply chat template (default True)

    Returns:
        Dict with:
          - input_ids: [1, seq_len] tensor (with video tokens already inserted)
          - attention_mask: [1, seq_len] tensor
          - pixel_values: [n_frames, 3, H, W] tensor
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

    # Build kwargs
    kwargs = {"return_tensors": "np", "return_metadata": True}
    if num_frames is not None:
        kwargs["num_frames"] = num_frames

    result = processor(text=text, videos=video_path, **kwargs)

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

    # Get timestamps from metadata
    metadata = result.get("video_metadata", [None])[0]
    if metadata is not None:
        fps = getattr(metadata, "fps", 2.0)
        frames_indices = getattr(metadata, "frames_indices", np.arange(n_frames))
        timestamps = np.array(frames_indices) / fps
    else:
        timestamps = np.arange(n_frames) / 2.0  # Default 2 fps

    return {
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
