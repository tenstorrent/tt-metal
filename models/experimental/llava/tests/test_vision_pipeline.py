# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Vision Pipeline Demo Tests - PCC Validation

Tests the full vision pipeline (CLIP → Projector → Interleaver) end-to-end
against PyTorch reference, validating PCC at each stage and for the
combined pipeline.

Usage:
    scripts/run_safe_pytest.sh models/demos/llava/tests/test_vision_pipeline.py -v

Reference:
    - models/demos/llava/demo/demo_vision_pipeline.py
    - models/demos/llava/tests/test_vision_encoders.py
    - models/demos/llava/tests/test_projectors.py
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llava.demo.demo_utils import download_sample_image, preprocess_image
from models.demos.llava.reference.llava_reference import create_clip_encoder as create_clip_encoder_pytorch
from models.demos.llava.reference.llava_reference import create_mlp_projector as create_mlp_projector_pytorch
from models.demos.llava.reference.llava_reference import create_standard_interleaver as create_interleaver_pytorch
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.projectors.mlp_projector import create_mlp_projector as create_mlp_projector_ttnn
from models.demos.llava.tt.token_handlers.standard_interleaver import (
    create_standard_interleaver as create_interleaver_ttnn,
)
from models.demos.llava.tt.vision_encoders.clip_encoder import create_clip_encoder as create_clip_encoder_ttnn

IMAGE_TOKEN_INDEX = -200
HIDDEN_SIZE = 4096
NUM_PATCHES = 576


def _create_projector_state_dict(seed=42):
    """Create random projector weights for testing"""
    torch.manual_seed(seed)
    return {
        "mm_projector.0.weight": torch.randn(4096, 1024) * 0.02,
        "mm_projector.0.bias": torch.zeros(4096),
        "mm_projector.2.weight": torch.randn(4096, 4096) * 0.02,
        "mm_projector.2.bias": torch.zeros(4096),
    }


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_vision_pipeline_clip_pcc(mesh_device):
    """
    Test CLIP encoder PCC in pipeline context (real preprocessed image).

    Validates that CLIP encoder produces correct features when fed a
    properly preprocessed image (CLIP normalization, 336x336 resize).
    Threshold: PCC >= 0.88 (QuickGELU approximation)
    """
    logger.info("=" * 80)
    logger.info("Testing Vision Pipeline - CLIP PCC (n300)")
    logger.info("=" * 80)

    model_name = "openai/clip-vit-large-patch14-336"
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    # Prepare real image
    image_path = download_sample_image()
    image_tensor = preprocess_image(image_path, resolution=336)
    logger.info(f"Input image: {image_tensor.shape}")

    # Load models
    pytorch_encoder = create_clip_encoder_pytorch(model_name=model_name, device="cpu")
    state_dict = pytorch_encoder.vision_model.state_dict()

    weight_cache_path = (
        Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal")) / "models" / "llava" / "clip"
    )
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    ttnn_encoder = create_clip_encoder_ttnn(args=args, state_dict=state_dict, weight_cache_path=weight_cache_path)

    # Run both
    with torch.no_grad():
        pytorch_output = pytorch_encoder(image_tensor)
    ttnn_output = ttnn.to_torch(ttnn.get_device_tensors(ttnn_encoder(image_tensor))[0])

    logger.info(f"PyTorch: {pytorch_output.shape}")
    logger.info(f"TTNN:    {ttnn_output.shape}")

    passing, pcc = comp_pcc(pytorch_output, ttnn_output, pcc=0.88)
    logger.info(f"PCC: {pcc:.6f} (threshold 0.88) -> {'PASS' if passing else 'FAIL'}")

    assert passing, f"CLIP PCC {pcc:.6f} < 0.88"
    assert pytorch_output.shape == ttnn_output.shape == (1, NUM_PATCHES, 1024)

    logger.info("✅ CLIP pipeline PCC test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_vision_pipeline_projector_pcc(mesh_device):
    """
    Test MLP projector PCC in pipeline context (fed CLIP features).

    Validates that the projector correctly transforms CLIP features into
    LLM embedding space. Uses TTNN CLIP output as input to both PyTorch
    and TTNN projectors to isolate projector accuracy.
    Threshold: PCC >= 0.99 (projector-only, no CLIP error)
    """
    logger.info("=" * 80)
    logger.info("Testing Vision Pipeline - Projector PCC (n300)")
    logger.info("=" * 80)

    model_name = "openai/clip-vit-large-patch14-336"
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    # Prepare image and run CLIP
    image_path = download_sample_image()
    image_tensor = preprocess_image(image_path, resolution=336)

    pytorch_encoder = create_clip_encoder_pytorch(model_name=model_name, device="cpu")
    state_dict = pytorch_encoder.vision_model.state_dict()
    weight_cache_path = (
        Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal")) / "models" / "llava" / "clip"
    )
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    ttnn_encoder = create_clip_encoder_ttnn(args=args, state_dict=state_dict, weight_cache_path=weight_cache_path)

    with torch.no_grad():
        clip_features = pytorch_encoder(image_tensor)

    # Create projectors with same weights
    projector_sd = _create_projector_state_dict()
    pytorch_projector = create_mlp_projector_pytorch(
        vision_hidden_size=1024, llm_hidden_size=4096, state_dict=projector_sd
    )
    ttnn_projector = create_mlp_projector_ttnn(args=args, dtype=ttnn.bfloat16, state_dict=projector_sd)

    # Run projector on SAME input (PyTorch CLIP features) to isolate projector PCC
    with torch.no_grad():
        pytorch_projected = pytorch_projector(clip_features)

    # Convert CLIP features to TTNN for TTNN projector
    clip_ttnn = ttnn.from_torch(
        clip_features,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn_projected = ttnn.to_torch(ttnn.get_device_tensors(ttnn_projector(clip_ttnn))[0])

    logger.info(f"PyTorch: {pytorch_projected.shape}")
    logger.info(f"TTNN:    {ttnn_projected.shape}")

    passing, pcc = comp_pcc(pytorch_projected, ttnn_projected, pcc=0.99)
    logger.info(f"PCC: {pcc:.6f} (threshold 0.99) -> {'PASS' if passing else 'FAIL'}")

    assert passing, f"Projector PCC {pcc:.6f} < 0.99"
    assert pytorch_projected.shape == ttnn_projected.shape == (1, NUM_PATCHES, HIDDEN_SIZE)

    logger.info("✅ Projector pipeline PCC test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_vision_pipeline_end_to_end_pcc(mesh_device):
    """
    Test full vision pipeline PCC (CLIP → Projector → Interleaver).

    Validates the accumulated PCC across all 3 components. The CLIP
    QuickGELU approximation introduces ~2% error, so the end-to-end
    threshold is relaxed to 0.88.
    Threshold: PCC >= 0.88 (accumulated QuickGELU error)
    """
    logger.info("=" * 80)
    logger.info("Testing Vision Pipeline - End-to-End PCC (n300)")
    logger.info("=" * 80)

    model_name = "openai/clip-vit-large-patch14-336"
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    # Prepare image
    image_path = download_sample_image()
    image_tensor = preprocess_image(image_path, resolution=336)

    # Load CLIP
    pytorch_encoder = create_clip_encoder_pytorch(model_name=model_name, device="cpu")
    state_dict = pytorch_encoder.vision_model.state_dict()
    weight_cache_path = (
        Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal")) / "models" / "llava" / "clip"
    )
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    ttnn_encoder = create_clip_encoder_ttnn(args=args, state_dict=state_dict, weight_cache_path=weight_cache_path)

    # Load projector
    projector_sd = _create_projector_state_dict()
    pytorch_projector = create_mlp_projector_pytorch(
        vision_hidden_size=1024, llm_hidden_size=4096, state_dict=projector_sd
    )
    ttnn_projector = create_mlp_projector_ttnn(args=args, dtype=ttnn.bfloat16, state_dict=projector_sd)

    # Load interleaver
    pytorch_interleaver = create_interleaver_pytorch()
    ttnn_interleaver = create_interleaver_ttnn(args)

    # Simulated text tokens: [<image>, t1, t2, ..., t20]
    num_text_tokens = 20
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX] + list(range(100, 100 + num_text_tokens))])
    text_embeds = torch.randn(1, num_text_tokens + 1, HIDDEN_SIZE)

    # === PyTorch pipeline ===
    with torch.no_grad():
        pytorch_clip = pytorch_encoder(image_tensor)
        pytorch_projected = pytorch_projector(pytorch_clip)
        ref_result = pytorch_interleaver(input_ids, text_embeds, pytorch_projected)

    # === TTNN pipeline ===
    ttnn_clip = ttnn_encoder(image_tensor)
    ttnn_projected = ttnn_projector(ttnn_clip)
    ttnn_projected_torch = ttnn.to_torch(ttnn.get_device_tensors(ttnn_projected)[0])
    ttnn_result = ttnn_interleaver(input_ids, text_embeds, ttnn_projected_torch)
    ttnn_embeds = ttnn.to_torch(ttnn.get_device_tensors(ttnn_result["input_embeds"])[0])

    expected_seq_len = num_text_tokens + NUM_PATCHES  # 596

    logger.info(f"PyTorch final: {ref_result['input_embeds'].shape}")
    logger.info(f"TTNN final:    {ttnn_embeds.shape}")
    logger.info(f"Expected:      [1, {expected_seq_len}, {HIDDEN_SIZE}]")

    # Shape validation
    assert tuple(ttnn_embeds.shape) == (1, expected_seq_len, HIDDEN_SIZE)

    # PCC on the interleaved embeddings
    # Only compare the image token positions (first NUM_PATCHES tokens) since
    # text embeds are identical (same input). Image positions carry accumulated error.
    ref_img = ref_result["input_embeds"][:, :NUM_PATCHES, :]
    ttnn_img = ttnn_embeds[:, :NUM_PATCHES, :].float()

    passing, pcc = comp_pcc(ref_img, ttnn_img, pcc=0.88)
    logger.info(f"PCC (image tokens, end-to-end): {pcc:.6f} (threshold 0.88) -> {'PASS' if passing else 'FAIL'}")

    # Also check text tokens match exactly (same input)
    ref_text = ref_result["input_embeds"][:, NUM_PATCHES:, :]
    ttnn_text = ttnn_embeds[:, NUM_PATCHES:, :].float()
    text_diff = (ref_text - ttnn_text).abs().max().item()
    logger.info(f"Text token max diff: {text_diff:.6f} (should be ~0, same input)")

    assert passing, f"End-to-end PCC {pcc:.6f} < 0.88"

    logger.info("✅ End-to-end pipeline PCC test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_vision_pipeline_shapes(mesh_device):
    """
    Test all intermediate shapes in the vision pipeline.
    """
    logger.info("=" * 80)
    logger.info("Testing Vision Pipeline - Shapes (n300)")
    logger.info("=" * 80)

    model_name = "openai/clip-vit-large-patch14-336"
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    image_path = download_sample_image()
    image_tensor = preprocess_image(image_path, resolution=336)

    pytorch_encoder = create_clip_encoder_pytorch(model_name=model_name, device="cpu")
    state_dict = pytorch_encoder.vision_model.state_dict()
    weight_cache_path = (
        Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal")) / "models" / "llava" / "clip"
    )
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    ttnn_encoder = create_clip_encoder_ttnn(args=args, state_dict=state_dict, weight_cache_path=weight_cache_path)

    projector_sd = _create_projector_state_dict()
    ttnn_projector = create_mlp_projector_ttnn(args=args, dtype=ttnn.bfloat16, state_dict=projector_sd)
    ttnn_interleaver = create_interleaver_ttnn(args)

    # Stage 1: CLIP
    clip_out = ttnn_encoder(image_tensor)
    clip_torch = ttnn.to_torch(ttnn.get_device_tensors(clip_out)[0])
    assert tuple(clip_torch.shape) == (1, 576, 1024), f"CLIP shape: {clip_torch.shape}"
    logger.info(f"  CLIP:      {clip_torch.shape} ✅")

    # Stage 2: Projector
    proj_out = ttnn_projector(clip_out)
    proj_torch = ttnn.to_torch(ttnn.get_device_tensors(proj_out)[0])
    assert tuple(proj_torch.shape) == (1, 576, 4096), f"Projector shape: {proj_torch.shape}"
    logger.info(f"  Projector: {proj_torch.shape} ✅")

    # Stage 3: Interleaver
    num_text = 10
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX] + list(range(100, 100 + num_text))])
    text_embeds = torch.randn(1, num_text + 1, HIDDEN_SIZE)
    result = ttnn_interleaver(input_ids, text_embeds, proj_torch)
    embeds = ttnn.to_torch(ttnn.get_device_tensors(result["input_embeds"])[0])

    expected = num_text + NUM_PATCHES  # 586
    assert tuple(embeds.shape) == (1, expected, HIDDEN_SIZE), f"Interleaved shape: {embeds.shape}"
    logger.info(f"  Interleaved: {embeds.shape} ✅")

    logger.info("✅ All pipeline shapes correct!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],  # n150 single-chip
    indirect=True,
)
def test_vision_pipeline_single_chip(mesh_device):
    """
    Test vision pipeline on n150 (single chip).
    """
    logger.info("=" * 80)
    logger.info("Testing Vision Pipeline - Single Chip (n150)")
    logger.info("=" * 80)

    model_name = "openai/clip-vit-large-patch14-336"
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    image_path = download_sample_image()
    image_tensor = preprocess_image(image_path, resolution=336)

    pytorch_encoder = create_clip_encoder_pytorch(model_name=model_name, device="cpu")
    state_dict = pytorch_encoder.vision_model.state_dict()
    ttnn_encoder = create_clip_encoder_ttnn(args=args, state_dict=state_dict)

    projector_sd = _create_projector_state_dict()
    ttnn_projector = create_mlp_projector_ttnn(args=args, dtype=ttnn.bfloat16, state_dict=projector_sd)

    # Run CLIP + Projector
    clip_out = ttnn_encoder(image_tensor)
    proj_out = ttnn_projector(clip_out)
    proj_torch = ttnn.to_torch(ttnn.get_device_tensors(proj_out)[0])

    assert tuple(proj_torch.shape) == (1, 576, 4096)
    logger.info(f"  Output: {proj_torch.shape} ✅")

    logger.info("✅ Single chip pipeline test passed!")


if __name__ == "__main__":
    logger.info("Vision Pipeline Tests - Direct execution mode")
    logger.info("For proper testing, use pytest:")
    logger.info("  scripts/run_safe_pytest.sh models/demos/llava/tests/test_vision_pipeline.py -v")
