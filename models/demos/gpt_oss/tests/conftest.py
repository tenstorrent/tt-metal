"""
Minimal pytest configuration - eliminates duplicate fixtures
"""

import pytest

import ttnn


@pytest.fixture(scope="session")
def reset_seeds():
    """Reset seeds for reproducible tests"""
    import random

    import numpy as np
    import torch

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def mesh_device(request):
    """Universal mesh device fixture - no more per-file duplication"""
    mesh_shape = request.param

    # Open devices with correct parameters
    devices = ttnn.open_mesh_device(ttnn.MeshShape(mesh_shape[0], mesh_shape[1]))

    yield devices

    # Clean up
    ttnn.close_mesh_device(devices)


@pytest.fixture
def device_params(request):
    """Universal device parameters fixture"""
    return request.param if hasattr(request, "param") else {}
