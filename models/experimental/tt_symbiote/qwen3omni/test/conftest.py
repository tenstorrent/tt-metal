"""Set MESH_DEVICE for Qwen3-Omni tests if not already set.

TTNNDistributedRMSNorm (used inside TTNNQwen3OmniAttention) requires MESH_DEVICE
to determine device architecture. When running via pytest, set a default so
tests work without requiring the caller to export MESH_DEVICE.
"""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _set_mesh_device():
    if "MESH_DEVICE" not in os.environ:
        os.environ["MESH_DEVICE"] = os.environ.get("TTNN_MESH_DEVICE", "N300")
