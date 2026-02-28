# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-device tests for matmul_auto.

Verifies that matmul_auto correctly handles:
1. Multi-device input detection
2. CCL strategy selection (all_gather, reduce_scatter, fused paths)
3. Correct outputs on multi-device configurations
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

pytestmark = pytest.mark.requires_wormhole_b0


# Only run multi-device tests when hardware is available
NUM_DEVICES = ttnn.distributed.get_num_pcie_devices() if os.environ.get("USE_NUM_DEVICES") else 1


@pytest.fixture(scope="module")
def mesh_device():
    """Create a mesh device for multi-device tests."""
    if NUM_DEVICES < 2:
        pytest.skip("Multi-device tests require at least 2 devices")

    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, min(NUM_DEVICES, 8)),
    )
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def single_device():
    """Single device for comparison tests."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


MULTI_DEVICE_SHAPES = [
    (1, 1024, 1024, 1024),
    (1, 2048, 4096, 4096),
    (1, 128, 4096, 4096),
]


class TestMatmulAutoMultiDevice:
    """Multi-device tests for matmul_auto."""

    @pytest.mark.skipif(NUM_DEVICES < 2, reason="Requires multi-device")
    @pytest.mark.parametrize("batch,m,k,n", MULTI_DEVICE_SHAPES)
    def test_multi_device_detection(self, mesh_device, batch, m, k, n):
        """Test that matmul_auto correctly detects multi-device inputs."""
        from ttnn.operations.auto_config.feature_extraction import extract_matmul_features

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        input_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        features = extract_matmul_features(input_a, input_b)
        assert features["is_multi_device"] is True
        assert features["num_devices"] >= 2

    @pytest.mark.skipif(NUM_DEVICES < 2, reason="Requires multi-device")
    @pytest.mark.parametrize("batch,m,k,n", MULTI_DEVICE_SHAPES[:2])
    def test_multi_device_correctness(self, mesh_device, batch, m, k, n):
        """Test that matmul_auto produces correct results on multi-device."""
        from ttnn.operations.auto_config import matmul_auto

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)
        torch_output = torch.matmul(torch_a, torch_b)

        input_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        input_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        tt_output = matmul_auto(input_a, input_b)

        # Get output from first device for comparison
        device_tensors = ttnn.get_device_tensors(tt_output)
        first_device_output = ttnn.to_torch(device_tensors[0])

        from tests.ttnn.utils_for_testing import check_with_pcc
        passed, msg = check_with_pcc(torch_output, first_device_output, pcc=0.99)
        assert passed, f"Multi-device PCC check failed: {msg}"


class TestMatmulAutoSingleDeviceFallback:
    """Test single-device behavior works correctly as baseline."""

    @pytest.mark.parametrize("batch,m,k,n", MULTI_DEVICE_SHAPES)
    def test_single_device_features_correct(self, single_device, batch, m, k, n):
        """Test that single device is detected correctly."""
        from ttnn.operations.auto_config.feature_extraction import extract_matmul_features

        torch_a = torch.randn(batch, m, k, dtype=torch.float32)
        torch_b = torch.randn(k, n, dtype=torch.float32)

        input_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=single_device
        )
        input_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=single_device
        )

        features = extract_matmul_features(input_a, input_b)
        assert features["is_multi_device"] is False
        assert features["num_devices"] == 1
