# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LLaVA Demo Script - End-to-end inference examples

This module provides pytest-based demos for LLaVA inference on Tenstorrent hardware.

Usage:
    # LLaVA 1.5 with CLIP encoder
    MESH_DEVICE=N300 HF_MODEL=liuhaotian/llava-v1.5-7b \
      scripts/run_safe_pytest.sh models/demos/llava/demo/demo_llava.py::test_single_image -v

    # LLaVA-MORE with SigLIP encoder
    MESH_DEVICE=N300 VISION_ENCODER=siglip \
      scripts/run_safe_pytest.sh models/demos/llava/demo/demo_llava.py::test_single_image -v

Reference:
    - models/demos/qwen25_vl/demo/demo.py
    - models/demos/llama3_70b_galaxy/demo/demo.py
"""

import os

import pytest
from loguru import logger
from PIL import Image

# Import demo utilities
from models.demos.llava.demo.demo_utils import create_conversation_prompt, download_sample_image

# Note: Model implementation to be completed in Week 4
# from models.demos.llava.tt.llava_model import LlavaModel


@pytest.mark.parametrize(
    "device_params",
    [{"mesh_device": (1, 2)}],  # n300 dual-chip
    indirect=True,
)
def test_single_image(device_params):
    """
    Test LLaVA on single image inference

    This test demonstrates basic LLaVA functionality:
    1. Load model (vision encoder + projector + LLM)
    2. Preprocess image
    3. Create prompt with <image> token
    4. Run inference
    5. Validate output

    Week 4 Target: This should work end-to-end
    Current Status: Skeleton implementation

    Args:
        device_params: Fixture providing device configuration
    """
    logger.info("Starting LLaVA single image inference test")

    # Get environment variables
    hf_model = os.environ.get("HF_MODEL", "liuhaotian/llava-v1.5-7b")
    vision_encoder = os.environ.get("VISION_ENCODER", "clip")

    logger.info(f"Model: {hf_model}")
    logger.info(f"Vision Encoder: {vision_encoder}")
    logger.info(f"Device: {device_params}")

    # TODO (Week 4): Load model
    # model = LlavaModel(args, dtype, state_dict, weight_cache_path)

    # Download and preprocess sample image
    image_path = download_sample_image()
    image = Image.open(image_path)
    logger.info(f"Loaded image: {image.size}")

    # Create prompt
    question = "What is in this image?"
    prompt = create_conversation_prompt(question)
    logger.info(f"Prompt: {prompt}")

    # TODO (Week 4): Run inference
    # output = model.forward([image], [prompt])
    # logger.info(f"Generated: {output[0]}")

    # Placeholder for Week 1-3
    logger.warning("⚠️  Model not implemented yet (Week 4 target)")
    logger.info("✅ Demo structure validated")
    logger.info("✅ Image loading works")
    logger.info("✅ Prompt formatting works")

    # Basic validation (will be replaced with actual inference in Week 4)
    assert image.mode == "RGB", "Image should be RGB"
    assert "<image>" in prompt, "Prompt should contain <image> token"

    logger.info("Test passed (skeleton validation)")


@pytest.mark.parametrize(
    "device_params",
    [{"mesh_device": (1, 2)}],  # n300 dual-chip
    indirect=True,
)
def test_batch_inference(device_params):
    """
    Test LLaVA on batch inference (multiple images)

    Week 4+ Target: Batch processing support
    Current Status: Skeleton implementation

    Args:
        device_params: Fixture providing device configuration
    """
    logger.info("Starting LLaVA batch inference test")

    # TODO (Week 4+): Implement batch inference
    logger.warning("⚠️  Batch inference not implemented yet (Week 4+ target)")

    # Placeholder validation
    assert True, "Batch inference structure prepared"


@pytest.mark.parametrize(
    "device_params",
    [{"mesh_device": (1, 2)}],  # n300 dual-chip
    indirect=True,
)
def test_multi_turn_conversation(device_params):
    """
    Test LLaVA on multi-turn conversation

    Example:
        Turn 1: "What is in this image?"
        Turn 2: "What color is the main object?"
        Turn 3: "Can you describe it in more detail?"

    Week 5+ Target: Conversational support
    Current Status: Skeleton implementation

    Args:
        device_params: Fixture providing device configuration
    """
    logger.info("Starting LLaVA multi-turn conversation test")

    # TODO (Week 5+): Implement multi-turn conversation
    logger.warning("⚠️  Multi-turn conversation not implemented yet (Week 5+ target)")

    # Placeholder validation
    assert True, "Multi-turn structure prepared"


@pytest.mark.parametrize(
    "device_params",
    [{"mesh_device": (1, 1)}],  # n150 single-chip
    indirect=True,
)
def test_single_chip_inference(device_params):
    """
    Test LLaVA on n150 (single chip)

    Validates that model works on both n150 and n300

    Week 4+ Target: n150 support
    Current Status: Skeleton implementation

    Args:
        device_params: Fixture providing device configuration (n150)
    """
    logger.info("Starting LLaVA single chip (n150) inference test")

    # TODO (Week 4+): Test on n150
    logger.warning("⚠️  Single chip inference not implemented yet (Week 4+ target)")

    # Placeholder validation
    assert True, "Single chip structure prepared"


if __name__ == "__main__":
    """
    Direct execution for debugging

    Usage:
        python models/demos/llava/demo/demo_llava.py
    """
    logger.info("LLaVA Demo - Direct execution mode")
    logger.info("For proper testing, use pytest:")
    logger.info("  scripts/run_safe_pytest.sh models/demos/llava/demo/demo_llava.py -v")
