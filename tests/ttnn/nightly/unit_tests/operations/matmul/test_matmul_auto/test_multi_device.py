# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-device matmul_auto tests (bounty req #1: multi-device support)."""

import os

import pytest
import torch

import ttnn

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")


def _get_mesh_device():
    """Try to open a mesh device. Returns None if unavailable."""
    try:
        num_devices = ttnn.GetNumAvailableDevices()
        if num_devices < 2:
            return None
        device_ids = list(range(num_devices))
        return ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices), device_ids=device_ids)
    except Exception:
        return None


@pytest.fixture(scope="module")
def mesh_device():
    dev = _get_mesh_device()
    if dev is None:
        pytest.skip("Multi-device not available (need >= 2 devices)")
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def single_device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_matmul_auto_detects_mesh(mesh_device):
    """matmul_auto should work on mesh tensors without crashing."""
    from ttnn._experimental.auto_config import matmul_auto

    a = torch.randn(1, 1, 512, 512)
    b = torch.randn(1, 1, 512, 512)
    ta = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tb = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    ta = ttnn.to_device(ta, mesh_device)
    tb = ttnn.to_device(tb, mesh_device)

    out = matmul_auto(ta, tb)
    assert out is not None


def test_matmul_auto_mesh_correctness(mesh_device):
    """Multi-device matmul_auto output should match torch reference."""
    from ttnn._experimental.auto_config import matmul_auto

    a = torch.randn(1, 1, 256, 256)
    b = torch.randn(1, 1, 256, 256)
    torch_out = torch.matmul(a, b)

    ta = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tb = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    ta = ttnn.to_device(ta, mesh_device)
    tb = ttnn.to_device(tb, mesh_device)

    tt_out = matmul_auto(ta, tb)
    output = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Check first device's output against reference
    single_out = output[0:1]
    from tests.ttnn.utils_for_testing import check_with_pcc

    passed, msg = check_with_pcc(torch_out, single_out, pcc=0.99)
    assert passed, f"Multi-device PCC check failed: {msg}"


def test_single_device_still_works(single_device):
    """Ensure single-device path is not broken by multi-device additions."""
    from ttnn._experimental.auto_config import matmul_auto

    a = torch.randn(1, 1, 128, 128)
    b = torch.randn(1, 1, 128, 128)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=single_device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=single_device)

    out = matmul_auto(ta, tb)
    assert out is not None
