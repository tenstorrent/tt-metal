# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Token Handler Tests - Shape and Correctness Validation

Tests the StandardInterleaver that interleaves text embeddings with
image features at <image> placeholder positions.

Since token interleaving is a deterministic rearrangement (no floating-point
arithmetic), tests focus on:
    1. Output shape correctness (sequence length = text_len - num_images + num_patches)
    2. Exact correctness (image features at right positions, text preserved)
    3. Determinism (same input -> same output)
    4. Multi-image interleaving
    5. Single-chip (n150) and dual-chip (n300) compatibility

Usage:
    scripts/run_safe_pytest.sh models/demos/llava/tests/test_token_handlers.py -v

Reference:
    - models/demos/llava/tests/test_vision_encoders.py (test patterns)
    - LLaVA/llava/model/llava_arch.py:prepare_inputs_labels_for_multimodal
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llava.reference.llava_reference import StandardInterleaver as PyTorchInterleaver
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.token_handlers.standard_interleaver import StandardInterleaver as TTNNInterleaver

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
NUM_PATCHES = 576  # CLIP ViT-L/14-336
HIDDEN_SIZE = 4096  # LLaMA hidden size


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_interleaver_shape_single_image(mesh_device):
    """
    Test output shape with a single image token.

    Input:  [<image>, t1, t2, t3, t4]  (5 tokens, 1 image placeholder)
    Output: [img_0, ..., img_575, t1, t2, t3, t4]  (576 + 4 = 580 tokens)
    """
    logger.info("=" * 80)
    logger.info("Testing Token Interleaver Shape (Single Image, n300)")
    logger.info("=" * 80)

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    # Input: [<image>, "USER", ":", "What", "?"]  (5 tokens)
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 100, 101, 102, 103]])
    text_embeds = torch.randn(1, 5, HIDDEN_SIZE)
    image_features = torch.randn(1, NUM_PATCHES, HIDDEN_SIZE)

    # Run TTNN interleaver
    ttnn_interleaver = TTNNInterleaver(args)
    result = ttnn_interleaver(input_ids, text_embeds, image_features)

    # Convert output to torch for validation
    output_embeds = ttnn.to_torch(ttnn.get_device_tensors(result["input_embeds"])[0])

    expected_seq_len = 5 - 1 + NUM_PATCHES  # 4 text + 576 image = 580
    expected_shape = (1, expected_seq_len, HIDDEN_SIZE)

    logger.info(f"Input seq len: {input_ids.shape[1]}")
    logger.info(f"Output seq len: {output_embeds.shape[1]}")
    logger.info(f"Expected seq len: {expected_seq_len}")

    assert tuple(output_embeds.shape) == expected_shape, f"Shape mismatch: {output_embeds.shape} != {expected_shape}"

    # Validate attention mask
    assert result["attention_mask"].shape == (1, expected_seq_len)
    assert result["attention_mask"][0].all(), "All tokens should be unmasked (no padding)"

    # Validate position_ids
    assert result["position_ids"].shape == (1, expected_seq_len)
    expected_pos = torch.arange(0, expected_seq_len).unsqueeze(0)
    assert torch.equal(result["position_ids"], expected_pos), "Position IDs should be 0..N-1"

    logger.info("✅ Shape test (single image) passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_interleaver_correctness(mesh_device):
    """
    Test interleaving correctness: image features at right positions, text preserved.

    Verifies that:
    1. First NUM_PATCHES tokens are image features
    2. Remaining tokens are text embeddings (in order)
    """
    logger.info("=" * 80)
    logger.info("Testing Token Interleaver Correctness (n300)")
    logger.info("=" * 80)

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    # Input: [<image>, t1, t2, t3]  (4 tokens, 1 image)
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 100, 101, 102]])
    text_embeds = torch.randn(1, 4, HIDDEN_SIZE)
    image_features = torch.randn(1, NUM_PATCHES, HIDDEN_SIZE)

    ttnn_interleaver = TTNNInterleaver(args)
    result = ttnn_interleaver(input_ids, text_embeds, image_features)
    output = ttnn.to_torch(ttnn.get_device_tensors(result["input_embeds"])[0])

    # Check image features are at positions 0..575
    img_feat = image_features[0]  # [576, 4096]
    output_img = output[0, :NUM_PATCHES, :].float()  # [576, 4096] — cast from bfloat16
    assert torch.allclose(output_img, img_feat, atol=0.05), "Image features mismatch at start"

    # Check text embeds are at positions 576..579 (tokens t1, t2, t3)
    # text_embeds[0, 0] is the dummy for <image>, skip it
    expected_text = text_embeds[0, 1:, :]  # [3, 4096]
    output_text = output[0, NUM_PATCHES : NUM_PATCHES + 3, :].float()  # [3, 4096]
    assert torch.allclose(output_text, expected_text, atol=0.05), "Text embeddings mismatch after image"

    logger.info("✅ Correctness test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_interleaver_matches_pytorch(mesh_device):
    """
    Test TTNN interleaver matches PyTorch reference exactly.

    Since interleaving is deterministic (no arithmetic), outputs should be identical.
    """
    logger.info("=" * 80)
    logger.info("Testing TTNN vs PyTorch Reference (n300)")
    logger.info("=" * 80)

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    torch.manual_seed(42)
    # Input: [<image>, t1, t2, t3, t4, t5]  (6 tokens, 1 image)
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 10, 20, 30, 40, 50]])
    text_embeds = torch.randn(1, 6, HIDDEN_SIZE)
    image_features = torch.randn(1, NUM_PATCHES, HIDDEN_SIZE)

    # PyTorch reference
    pytorch_interleaver = PyTorchInterleaver()
    ref_result = pytorch_interleaver(input_ids, text_embeds, image_features)

    # TTNN
    ttnn_interleaver = TTNNInterleaver(args)
    ttnn_result = ttnn_interleaver(input_ids, text_embeds, image_features)
    ttnn_output = ttnn.to_torch(ttnn.get_device_tensors(ttnn_result["input_embeds"])[0])

    # Compare (cast TTNN output to float32 for comparison)
    ref_output = ref_result["input_embeds"]
    ttnn_output_f32 = ttnn_output.float()

    logger.info(f"PyTorch output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {ttnn_output.shape}")

    assert ref_output.shape == ttnn_output.shape, "Shape mismatch between PyTorch and TTNN"

    max_diff = (ref_output - ttnn_output_f32).abs().max().item()
    logger.info(f"Max difference: {max_diff:.8f}")

    assert max_diff < 0.05, f"Outputs differ (bfloat16 precision): max_diff={max_diff}"

    logger.info("✅ TTNN matches PyTorch reference!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_interleaver_deterministic(mesh_device):
    """
    Test that interleaving is deterministic (same input -> same output).
    """
    logger.info("=" * 80)
    logger.info("Testing Token Interleaver Determinism (n300)")
    logger.info("=" * 80)

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    torch.manual_seed(42)
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 100, 101, 102]])
    text_embeds = torch.randn(1, 4, HIDDEN_SIZE)
    image_features = torch.randn(1, NUM_PATCHES, HIDDEN_SIZE)

    ttnn_interleaver = TTNNInterleaver(args)

    result1 = ttnn.to_torch(
        ttnn.get_device_tensors(ttnn_interleaver(input_ids, text_embeds, image_features)["input_embeds"])[0]
    )
    result2 = ttnn.to_torch(
        ttnn.get_device_tensors(ttnn_interleaver(input_ids, text_embeds, image_features)["input_embeds"])[0]
    )

    max_diff = (result1 - result2).abs().max().item()
    logger.info(f"Max difference between runs: {max_diff:.10f}")

    assert max_diff < 1e-6, f"Outputs differ: max_diff={max_diff}"

    logger.info("✅ Determinism test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_interleaver_multi_image(mesh_device):
    """
    Test interleaving with multiple images in a single sequence.

    Input:  [t1, <image>, t2, <image>, t3]  (5 tokens, 2 image placeholders)
    Output: [t1, img0_0..img0_575, t2, img1_0..img1_575, t3]
            (1 + 576 + 1 + 576 + 1 = 1155 tokens)
    """
    logger.info("=" * 80)
    logger.info("Testing Token Interleaver Multi-Image (n300)")
    logger.info("=" * 80)

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    # Input: [t1, <image>, t2, <image>, t3]
    input_ids = torch.tensor([[100, IMAGE_TOKEN_INDEX, 101, IMAGE_TOKEN_INDEX, 102]])
    text_embeds = torch.randn(1, 5, HIDDEN_SIZE)
    image_features = torch.randn(1, NUM_PATCHES, HIDDEN_SIZE)  # Same features for both

    ttnn_interleaver = TTNNInterleaver(args)
    result = ttnn_interleaver(input_ids, text_embeds, image_features)
    output = ttnn.to_torch(ttnn.get_device_tensors(result["input_embeds"])[0])

    expected_seq_len = 5 - 2 + 2 * NUM_PATCHES  # 3 text + 2*576 image = 1155
    expected_shape = (1, expected_seq_len, HIDDEN_SIZE)

    logger.info(f"Input seq len: {input_ids.shape[1]}")
    logger.info(f"Output seq len: {output.shape[1]}")
    logger.info(f"Expected seq len: {expected_seq_len}")

    assert tuple(output.shape) == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"

    # Verify structure: t1 at pos 0, img at 1..576, t2 at 577, img at 578..1153, t3 at 1154
    assert torch.allclose(output[0, 0, :].float(), text_embeds[0, 0, :], atol=0.05), "t1 mismatch"
    assert torch.allclose(output[0, 577, :].float(), text_embeds[0, 2, :], atol=0.05), "t2 mismatch"
    assert torch.allclose(output[0, 1154, :].float(), text_embeds[0, 4, :], atol=0.05), "t3 mismatch"

    logger.info("✅ Multi-image test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_interleaver_batch_padding(mesh_device):
    """
    Test batch padding: sequences of different lengths are right-padded.
    """
    logger.info("=" * 80)
    logger.info("Testing Token Interleaver Batch Padding (n300)")
    logger.info("=" * 80)

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=2)

    # Batch of 2:
    #   Seq 0: [<image>, t1, t2]  -> 576 + 2 = 578 tokens (padded with 0s in input)
    #   Seq 1: [<image>, t1, t2, t3, t4]  -> 576 + 4 = 580 tokens
    input_ids = torch.tensor(
        [
            [IMAGE_TOKEN_INDEX, 100, 101, 0, 0],  # padded with 0
            [IMAGE_TOKEN_INDEX, 200, 201, 202, 203],
        ]
    )
    text_embeds = torch.randn(2, 5, HIDDEN_SIZE)
    image_features = torch.randn(2, NUM_PATCHES, HIDDEN_SIZE)

    # Attention mask: 1 for real tokens, 0 for padding
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],  # Seq 0: 3 real tokens
            [1, 1, 1, 1, 1],  # Seq 1: 5 real tokens
        ]
    )

    ttnn_interleaver = TTNNInterleaver(args)
    result = ttnn_interleaver(input_ids, text_embeds, image_features, attention_mask=attention_mask)
    output = ttnn.to_torch(ttnn.get_device_tensors(result["input_embeds"])[0])

    # Both sequences padded to max(578, 580) = 580
    expected_seq_len = 580
    expected_shape = (2, expected_seq_len, HIDDEN_SIZE)

    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Expected shape: {expected_shape}")

    assert tuple(output.shape) == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"

    # Check attention mask
    attn_mask = result["attention_mask"]
    logger.info(f"Attention mask sums: {attn_mask.sum(dim=1)}")
    assert attn_mask[0].sum().item() == 578, f"Seq 0 should have 578 real tokens, got {attn_mask[0].sum()}"
    assert attn_mask[1].sum().item() == 580, f"Seq 1 should have 580 real tokens, got {attn_mask[1].sum()}"

    logger.info("✅ Batch padding test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],  # n150 single-chip
    indirect=True,
)
def test_interleaver_single_chip(mesh_device):
    """
    Test interleaver on n150 (single chip).
    """
    logger.info("=" * 80)
    logger.info("Testing Token Interleaver on Single Chip (n150)")
    logger.info("=" * 80)

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 100, 101, 102]])
    text_embeds = torch.randn(1, 4, HIDDEN_SIZE)
    image_features = torch.randn(1, NUM_PATCHES, HIDDEN_SIZE)

    ttnn_interleaver = TTNNInterleaver(args)
    result = ttnn_interleaver(input_ids, text_embeds, image_features)
    output = ttnn.to_torch(ttnn.get_device_tensors(result["input_embeds"])[0])

    expected_seq_len = 4 - 1 + NUM_PATCHES  # 579
    expected_shape = (1, expected_seq_len, HIDDEN_SIZE)

    assert tuple(output.shape) == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"

    logger.info("✅ Single chip test passed!")


if __name__ == "__main__":
    logger.info("Token Handler Tests - Direct execution mode")
    logger.info("For proper testing, use pytest:")
    logger.info("  scripts/run_safe_pytest.sh models/demos/llava/tests/test_token_handlers.py -v")
