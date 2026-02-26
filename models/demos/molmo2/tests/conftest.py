# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
import os

import pytest
import torch

import ttnn


@pytest.fixture(autouse=True)
def ensure_gc():
    """Ensure garbage collection runs after each test."""
    yield
    gc.collect()


@pytest.fixture(autouse=True)
def reset_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def model_location():
    """Get the model checkpoint location from environment."""
    model_path = os.environ.get("HF_MODEL", "allenai/Molmo2-8B")
    return model_path


@pytest.fixture
def molmo2_reference(model_location):
    """Load the Molmo2 reference model for testing."""
    from models.demos.molmo2.reference.model import Molmo2Reference

    return Molmo2Reference(model_location, torch_dtype=torch.float32)


@pytest.fixture
def device_params(request, galaxy_type):
    """Device parameters fixture from tt_transformers."""
    # Get param dict passed in from test parametrize (or default to empty dict)
    params = getattr(request, "param", {}).copy()

    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
        os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    )
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] is True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )

    return params


@pytest.fixture
def galaxy_type():
    """Determine galaxy type for multi-device setups."""
    return os.environ.get("GALAXY_TYPE", "6U")
