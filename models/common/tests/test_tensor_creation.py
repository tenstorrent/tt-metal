# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.tensor_creation import allocate_replicated_zeros


@pytest.mark.parametrize("mesh_device", [(1, 1), (1, 2), (2, 2)], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, None])
@pytest.mark.parametrize("shape", [(32, 1, 64, 128), (32, 1, 65, 96)])
def test_replicated_zeros_match_host_cache(mesh_device, dtype, shape):
    """Check cache values, padded storage, and the legacy mesh topology."""
    expected = ttnn.from_torch(
        torch.zeros(shape),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    actual = allocate_replicated_zeros(shape, device=mesh_device, dtype=dtype)
    try:
        assert actual.shape == expected.shape
        assert actual.padded_shape == expected.padded_shape
        assert actual.dtype == expected.dtype
        assert actual.memory_config() == expected.memory_config()
        assert actual.tensor_topology() == expected.tensor_topology()
        # View the physical tile-padded shape so the check includes padding.
        padded = ttnn.reshape(actual, actual.padded_shape)
        host = ttnn.to_torch(padded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        assert torch.count_nonzero(host).item() == 0
    finally:
        ttnn.deallocate(actual)
        ttnn.deallocate(expected)
