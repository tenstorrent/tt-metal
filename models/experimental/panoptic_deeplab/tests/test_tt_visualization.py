# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for TTNN-accelerated visualization module.

Tests the TtDeeplabV3PlusVisualization module which provides on-device
acceleration for semantic segmentation visualization.
"""

import pytest
import torch
import numpy as np
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_visualization import TtDeeplabV3PlusVisualization


@pytest.fixture
def device():
    """Create and return a TTNN device for testing."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def viz_module(device):
    """Create and return a TtDeeplabV3PlusVisualization module."""
    return TtDeeplabV3PlusVisualization(
        device=device,
        num_classes=19,
        alpha=0.6,
        dtype=ttnn.bfloat16,
    )


@pytest.fixture
def semantic_pred_ttnn(device):
    """Create a semantic prediction TTNN tensor with real input size [1, 512, 1024, 19]."""
    batch_size, height, width, num_classes = 1, 512, 1024, 19

    # Create random logits with one class more likely per pixel (for realistic argmax)
    semantic_pred_np = np.random.randn(batch_size, height, width, num_classes).astype(np.float32)
    for i in range(height):
        for j in range(width):
            class_id = (i + j) % num_classes
            semantic_pred_np[0, i, j, class_id] += 2.0

    semantic_pred_torch = torch.from_numpy(semantic_pred_np).float()
    semantic_pred_ttnn = ttnn.from_torch(
        semantic_pred_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return semantic_pred_ttnn


@pytest.fixture
def original_image_ttnn(device):
    """Create an original image TTNN tensor with real input size [1, 512, 1024, 3]."""
    batch_size, height, width, channels = 1, 512, 1024, 3
    original_image_np = np.random.rand(batch_size, height, width, channels).astype(np.float32)

    original_image_torch = torch.from_numpy(original_image_np).float()
    original_image_ttnn = ttnn.from_torch(
        original_image_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if original_image_ttnn.layout != ttnn.TILE_LAYOUT:
        original_image_ttnn = ttnn.to_layout(original_image_ttnn, ttnn.TILE_LAYOUT)

    return original_image_ttnn


def test_forward_full_pipeline(viz_module, semantic_pred_ttnn, original_image_ttnn):
    """
    Test full forward pass of the visualization module with real input size.

    This test covers the complete pipeline:
    1. Argmax to get class IDs
    2. Color mapping
    3. Image blending
    """
    # Verify input shapes
    assert semantic_pred_ttnn.shape == ttnn.Shape(
        [1, 512, 1024, 19]
    ), f"Expected semantic_pred shape [1, 512, 1024, 19], got {semantic_pred_ttnn.shape}"
    assert original_image_ttnn.shape == ttnn.Shape(
        [1, 512, 1024, 3]
    ), f"Expected original_image shape [1, 512, 1024, 3], got {original_image_ttnn.shape}"

    # Run forward pass
    vis_image, panoptic_info = viz_module.forward(
        semantic_pred=semantic_pred_ttnn,
        original_image=original_image_ttnn,
    )

    # Check output shape
    assert vis_image.shape == ttnn.Shape(
        [1, 512, 1024, 3]
    ), f"Expected vis_image shape [1, 512, 1024, 3], got {vis_image.shape}"

    # Check that colors are in [0, 1] range (normalized)
    vis_image_torch = ttnn.to_torch(vis_image).float().cpu().numpy()
    assert np.all(vis_image_torch >= 0.0), "All pixel values should be >= 0"
    assert np.all(vis_image_torch <= 1.0), "All pixel values should be <= 1"

    # Check panoptic info structure
    assert "mode" in panoptic_info, "panoptic_info should contain 'mode'"
    assert panoptic_info["mode"] == "DEEPLAB_V3_PLUS", f"Expected mode 'DEEPLAB_V3_PLUS', got {panoptic_info['mode']}"
    assert "num_classes" in panoptic_info, "panoptic_info should contain 'num_classes'"
    assert "class_distribution" in panoptic_info, "panoptic_info should contain 'class_distribution'"

    # Check that num_classes is reasonable (should be <= 19)
    assert panoptic_info["num_classes"] > 0, "Should have at least one class"
    assert panoptic_info["num_classes"] <= 19, "Should not exceed 19 classes"

    # Check that class_distribution is a dictionary
    assert isinstance(panoptic_info["class_distribution"], dict), "class_distribution should be a dictionary"

    logger.info(f"Forward pass completed successfully. Output shape: {vis_image.shape}")
    logger.info(f"Panoptic info: {panoptic_info}")
