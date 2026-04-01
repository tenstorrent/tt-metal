# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for RF-DETR Medium PCC tests.
"""

import pytest
import torch

from models.experimental.rfdetr_medium.common import RESOLUTION


@pytest.fixture(scope="session")
def torch_model():
    """Load the RF-DETR Medium PyTorch model (cached per session)."""
    from models.experimental.rfdetr_medium.common import load_torch_model

    model = load_torch_model()
    model.eval()
    return model


@pytest.fixture(scope="session")
def sample_image():
    """Generate a random sample image for testing."""
    torch.manual_seed(42)
    return torch.randn(1, 3, RESOLUTION, RESOLUTION)


@pytest.fixture(scope="session")
def reference_outputs(torch_model, sample_image):
    """Run full PyTorch reference forward and cache results."""
    from models.experimental.rfdetr_medium.reference.rfdetr_medium import full_reference_forward

    with torch.no_grad():
        return full_reference_forward(torch_model, sample_image)
