# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared pytest fixtures for CosyVoice TTNN tests.
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default paths — override via environment variables
COSYVOICE_WEIGHTS_DIR = os.environ.get(
    "COSYVOICE_WEIGHTS_DIR",
    "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B",
)

COSYVOICE_REF_DIR = os.environ.get(
    "COSYVOICE_REF_DIR",
    "models/demos/wormhole/cosy_voice/ref/CosyVoice",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_pcc(golden, calculated, threshold=0.99, msg=""):
    """Assert that the Pearson Correlation Coefficient meets the threshold."""
    passing, pcc_value = comp_pcc(golden, calculated, pcc=threshold)
    assert passing, f"PCC {pcc_value:.6f} < {threshold} {msg}"
    logger.info(f"PCC: {pcc_value:.6f} (threshold={threshold}) {msg}")
    return pcc_value


# ---------------------------------------------------------------------------
# Device fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device():
    """Open and close a single Wormhole device for the test session."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="session")
def mesh_device():
    """Open and close a mesh device (N300 = 2 chips) for the test session.

    Fabric must be enabled for all_gather_async used by DistributedNorm in the
    transformer layers. Without it the Ethernet semaphores are never initialized
    and the gather call deadlocks.
    """
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Weight fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cosyvoice_weights_dir():
    """Return the path to CosyVoice pretrained weights."""
    path = Path(COSYVOICE_WEIGHTS_DIR)
    if not path.exists():
        pytest.skip(
            f"CosyVoice weights not found at {path}. "
            f"Run: python models/demos/wormhole/cosy_voice/download_weights.py"
        )
    return path


@pytest.fixture(scope="session")
def llm_state_dict(cosyvoice_weights_dir):
    """Load the LLM state dict from CosyVoice checkpoint."""
    llm_path = cosyvoice_weights_dir / "llm.pt"
    if not llm_path.exists():
        pytest.skip(f"LLM weights not found at {llm_path}")
    logger.info(f"Loading LLM state dict from {llm_path}")
    state_dict = torch.load(llm_path, map_location="cpu", weights_only=True)
    return state_dict


@pytest.fixture(scope="session")
def flow_state_dict(cosyvoice_weights_dir):
    """Load the Flow decoder state dict from CosyVoice checkpoint."""
    flow_path = cosyvoice_weights_dir / "flow.pt"
    if not flow_path.exists():
        pytest.skip(f"Flow weights not found at {flow_path}")
    logger.info(f"Loading Flow state dict from {flow_path}")
    state_dict = torch.load(flow_path, map_location="cpu", weights_only=True)
    return state_dict


@pytest.fixture(scope="session")
def hift_state_dict(cosyvoice_weights_dir):
    """Load the HiFi-GAN state dict from CosyVoice checkpoint."""
    hift_path = cosyvoice_weights_dir / "hift.pt"
    if not hift_path.exists():
        pytest.skip(f"HiFi-GAN weights not found at {hift_path}")
    logger.info(f"Loading HiFi-GAN state dict from {hift_path}")
    state_dict = torch.load(hift_path, map_location="cpu", weights_only=True)
    # CosyVoice stores hifigan weights with 'generator.' prefix
    state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items()}
    return state_dict
