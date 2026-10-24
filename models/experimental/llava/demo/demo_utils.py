# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Demo utilities for LLaVA inference

Provides helper functions for:
- Image preprocessing (CLIP normalization)
- Prompt formatting (LLaVA conversation templates)
- Sample image downloading

Reference:
    - LLaVA/llava/mm_utils.py
    - LLaVA/llava/conversation.py
"""

from pathlib import Path

import torchvision.transforms as transforms
from loguru import logger
from PIL import Image


def preprocess_image(image, resolution=336):
    """
    Preprocess image for CLIP vision encoder

    Critical: Uses CLIP normalization, NOT ImageNet

    Args:
        image: PIL.Image or path to image
        resolution: Target resolution (336 for CLIP, 384 for SigLIP)

    Returns:
        torch.Tensor: Preprocessed image [1, 3, H, W]

    Reference:
        - openai/clip-vit-large-patch14-336 preprocessing
        - LLaVA/llava/mm_utils.py
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # CLIP normalization (NOT ImageNet!)
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]  # CLIP stats
    )

    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transform(image).unsqueeze(0)


def create_conversation_prompt(question, system_message=None, image_token="<image>"):
    """
    Create LLaVA conversation prompt

    Format follows LLaVA conversation template:
        USER: <image>\n{question} ASSISTANT:

    Args:
        question: User question about the image
        system_message: Optional system instruction (default: None)
        image_token: Token representing image position (default: "<image>")

    Returns:
        str: Formatted prompt

    Reference:
        - LLaVA/llava/conversation.py
        - LLaVA conversation templates (v1, vicuna_v1, llama_3, etc.)
    """
    if system_message:
        prompt = f"{system_message}\n\nUSER: {image_token}\n{question} ASSISTANT:"
    else:
        prompt = f"USER: {image_token}\n{question} ASSISTANT:"

    return prompt


def create_multi_turn_prompt(turns, image_token="<image>"):
    """
    Create multi-turn conversation prompt

    Args:
        turns: List of (question, answer) tuples
        image_token: Token representing image position

    Returns:
        str: Multi-turn conversation prompt

    Example:
        turns = [
            ("What is in this image?", "A cat sitting on a laptop."),
            ("What color is the cat?", "The cat is orange and white."),
        ]
        prompt = create_multi_turn_prompt(turns)
    """
    # First turn includes image
    prompt_parts = [f"USER: {image_token}\n{turns[0][0]} ASSISTANT: {turns[0][1]}"]

    # Subsequent turns (no image token)
    for question, answer in turns[1:]:
        prompt_parts.append(f"USER: {question} ASSISTANT: {answer}")

    # Final turn (no answer yet)
    prompt = " ".join(prompt_parts)
    return prompt


def download_sample_image(cache_dir=None):
    """
    Download or use cached sample image for testing

    Tries a real photo first (llava.cc COCO image used by upstream LLaVA demos);
    falls back to a generated test image when offline.

    Args:
        cache_dir: Directory to cache images (default: ~/.cache/llava_demo)

    Returns:
        Path: Path to sample image
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "llava_demo"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sample_path = cache_dir / "sample_image.jpg"
    if sample_path.exists() and sample_path.stat().st_size > 10_000:
        return sample_path

    # Real photo from the upstream LLaVA demo set (llava_v1_5_radar.jpg)
    url = "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_v1_5_radar.jpg"
    try:
        import urllib.request

        logger.info(f"Downloading sample image from {url}")
        with urllib.request.urlopen(url, timeout=30) as resp, open(sample_path, "wb") as f:
            f.write(resp.read())
        if sample_path.stat().st_size > 10_000:
            return sample_path
    except Exception as e:
        logger.warning(f"Sample image download failed ({e}); generating a test image")

    # Fallback: create a simple RGB test image
    # TODO: Bundle deterministic sample images from the LLaVA demo set
    logger.info(f"Creating test image at {sample_path}")
    import numpy as np

    test_img = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
    Image.fromarray(test_img).save(sample_path)

    return sample_path


def format_llama3_prompt(question, image_token="<image>"):
    """
    Format prompt for LLaMA 3.1 conversation template

    LLaMA 3.1 uses specific tokens:
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Args:
        question: User question
        image_token: Token representing image

    Returns:
        str: LLaMA 3.1 formatted prompt

    Reference:
        - meta-llama/Llama-3.1-8B-Instruct tokenizer
    """
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{image_token}\n{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def get_clip_normalization():
    """
    Get CLIP normalization transform

    Returns:
        torchvision.transforms.Normalize: CLIP normalization
    """
    return transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


def get_siglip_normalization():
    """
    Get SigLIP normalization transform

    Returns:
        torchvision.transforms.Normalize: SigLIP normalization

    Note: SigLIP uses different normalization than CLIP
    """
    # TODO: Verify SigLIP normalization stats
    # For now, using CLIP stats (to be updated in Week 5)
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # SigLIP uses simpler normalization


def prepare_image_for_encoder(image, encoder_type="clip", resolution=336):
    """
    Prepare image for specific vision encoder

    Args:
        image: PIL.Image or path
        encoder_type: "clip" or "siglip"
        resolution: Target resolution

    Returns:
        torch.Tensor: Preprocessed image
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Select normalization based on encoder
    if encoder_type.lower() == "clip":
        normalize = get_clip_normalization()
        resolution = resolution or 336
    elif encoder_type.lower() == "siglip":
        normalize = get_siglip_normalization()
        resolution = resolution or 384
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transform(image).unsqueeze(0)
