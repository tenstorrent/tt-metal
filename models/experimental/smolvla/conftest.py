# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def reset_seeds():
    """Reset random seeds for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield


@pytest.fixture
def test_image():
    """Create a deterministic test image for SmolVLA testing."""
    np.random.seed(42)
    # Create a gradient image that's more realistic than random noise
    h, w = 384, 384
    img_array = np.zeros((h, w, 3), dtype=np.uint8)

    # Create gradient background
    for i in range(h):
        for j in range(w):
            img_array[i, j, 0] = int(255 * i / h)  # Red gradient
            img_array[i, j, 1] = int(255 * j / w)  # Green gradient
            img_array[i, j, 2] = 128  # Blue constant

    return Image.fromarray(img_array)


@pytest.fixture
def test_instruction():
    """Default test instruction for SmolVLA."""
    return "pick up the red block"
