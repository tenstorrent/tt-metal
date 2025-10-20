#!/usr/bin/env python3
"""
Tests for automatic composition of multi-device sharded tensors using TensorTopology.

This test module validates that the auto-composition logic correctly infers
MeshToTensor composers from a sharded ttnn.Tensor's topology metadata.

It validates both host-sharded and device-sharded cases.
"""

import os

import pytest
import torch
import ttnn

from tt_transformers_v2.src.testing.auto_compose import to_torch_auto_compose

# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture(scope="module")
def mesh_shape():
    """Get mesh shape from environment variable."""
    return {
        "N150": [1, 1],
        "N300": [1, 2],
        "N150x4": [1, 4],
        "T3K": [1, 8],
        "TG": [8, 4],
        "P150": [1, 1],
        "P300": [1, 2],
        "P150x4": [1, 4],
        "P150x8": [1, 8],
    }.get(os.environ.get("MESH_DEVICE"), [1, 1])


@pytest.fixture(scope="module")
def mesh_device(mesh_shape):
    """Create and yield a mesh device, cleanup on teardown."""
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_shape))
    yield device
    ttnn.close_mesh_device(device)


@pytest.fixture(scope="module")
def num_devices(mesh_device):
    """Get number of devices in the mesh."""
    return mesh_device.get_num_devices()


# ======================================================================================
# Helper Functions
# ======================================================================================


def _make_known_pattern(num_chunks: int) -> torch.Tensor:
    """
    Produces shape [num_chunks, 1, 3, 1] with per-chunk distinct values.
    Chunk i contains [i*1, i*2, i*3].
    """
    rows = []
    for i in range(num_chunks):
        rows.append(torch.tensor([[[i * 1.0], [i * 2.0], [i * 3.0]]]).transpose(0, 1))  # [1,3,1]
    data = torch.stack(rows, dim=0)  # [num_chunks,1,3,1]
    return data.to(torch.bfloat16)


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.parametrize("min_devices", [2])
def test_host_sharded_1d(mesh_device: ttnn.MeshDevice, num_devices: int, min_devices: int) -> None:
    """Test automatic composition of host-sharded 1D tensors."""
    if num_devices < min_devices:
        pytest.skip(f"Test requires at least {min_devices} devices, found {num_devices}")

    # Input tensor of shape [num_devices, 1, 3, 1]
    torch_in = _make_known_pattern(num_devices)

    # Create a host-sharded tensor along dim=0
    tt_host_sharded = ttnn.from_torch(
        torch_in, device=None, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    # Validate that auto-composition returns original torch input
    torch_auto = to_torch_auto_compose(tt_host_sharded, device=mesh_device)

    # Reference using explicit composer
    torch_ref = ttnn.to_torch(tt_host_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert torch.equal(torch_ref, torch_in), "Explicit composer mismatch"
    assert torch.equal(torch_auto, torch_in), "Auto-composer mismatch on host-sharded tensor"


@pytest.mark.parametrize("min_devices", [2])
def test_device_sharded_1d(mesh_device: ttnn.MeshDevice, num_devices: int, min_devices: int) -> None:
    """Test automatic composition of device-sharded 1D tensors."""
    if num_devices < min_devices:
        pytest.skip(f"Test requires at least {min_devices} devices, found {num_devices}")

    # Input tensor of shape [num_devices, 1, 3, 1]
    torch_in = _make_known_pattern(num_devices)

    # Distribute to mesh device directly (device storage)
    tt_dev_sharded = ttnn.from_torch(
        torch_in, device=mesh_device, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )
    # [INFO]{ equivalent to:
    # mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0)
    # tt_host = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16)
    # tt_dev_sharded = ttnn.distribute_tensor(tt_host, mapper, mesh_device)
    # }[INFO]

    # Validate that auto-composition returns original torch input
    torch_auto = to_torch_auto_compose(tt_dev_sharded, device=mesh_device)

    # Reference using explicit composer through high-level API
    torch_ref = ttnn.to_torch(tt_dev_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert torch.equal(torch_ref, torch_in), "Explicit composer mismatch (device-sharded)"
    assert torch.equal(torch_auto, torch_in), "Auto-composer mismatch on device-sharded tensor"
