# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproduction test for ttnn.std() with multi-dimensional dim parameter.

This test reproduces the TypeError encountered in Stable Diffusion XL Base:
    TypeError: ttnn.std(): incompatible function arguments.
    Called with: std(ttnn.Tensor, dim=[1, 2, 3], keepdim=True)

The issue: ttnn.std() fails to accept multi-dimensional dim as a list [1, 2, 3],
even though other reduction ops like ttnn.sum() accept this format successfully.
"""

import pytest
import torch
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 8, 8),  # Simple 4D case
        (1, 4, 16, 16),  # Closer to SDXL noise_pred dimensions
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        [1, 2, 3],  # The failing case from SDXL
        [0, 1, 2],  # Alternative multi-dim
        [-3, -2, -1],  # Negative indices
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_std_multidim_list(device, shape, dim, keepdim):
    """Minimal repro for ttnn.std() with multi-dimensional dim as list."""
    torch.manual_seed(0)

    # Create torch reference
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch.std(torch_input, dim=dim, keepdim=keepdim)

    # Create ttnn tensor
    ttnn_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

    # This should work but currently fails with TypeError
    logger.info(f"Calling ttnn.std with shape={shape}, dim={dim}, keepdim={keepdim}")
    ttnn_output = ttnn.std(ttnn_input, dim=dim, keepdim=keepdim)

    # Convert back and validate
    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.TILE_LAYOUT)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    logger.info(f"torch output shape: {torch_output.shape}, ttnn output shape: {ttnn_output.shape}")
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize("shape", [(2, 4, 8, 8)])
@pytest.mark.parametrize(
    "dim",
    [
        (1, 2, 3),  # Tuple instead of list
    ],
)
@pytest.mark.parametrize("keepdim", [True])
def test_std_multidim_tuple(device, shape, dim, keepdim):
    """Test if tuple works where list fails."""
    torch.manual_seed(0)

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch.std(torch_input, dim=dim, keepdim=keepdim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

    logger.info(f"Calling ttnn.std with tuple dim={dim}")
    ttnn_output = ttnn.std(ttnn_input, dim=dim, keepdim=keepdim)

    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.TILE_LAYOUT)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize("shape", [(2, 4, 8, 8)])
def test_compare_sum_vs_std_multidim(device, shape):
    """Compare behavior: ttnn.sum works with lists, does ttnn.std?"""
    torch.manual_seed(0)

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

    dim = [1, 2, 3]
    keepdim = True

    # This should work (verified in existing tests)
    logger.info("Testing ttnn.sum with list dim=[1, 2, 3]")
    sum_output = ttnn.sum(ttnn_input, dim=dim, keepdim=keepdim)
    logger.info(f"ttnn.sum succeeded with list dim")

    # This fails in SDXL
    logger.info("Testing ttnn.std with list dim=[1, 2, 3]")
    std_output = ttnn.std(ttnn_input, dim=dim, keepdim=keepdim)
    logger.info(f"ttnn.std succeeded with list dim")

    # If we get here, both work
    assert sum_output is not None
    assert std_output is not None


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("shape", [(2, 4, 8, 8)])
@pytest.mark.parametrize("dim", [[1, 2, 3]])
@pytest.mark.parametrize("keepdim", [True])
def test_std_multidim_mesh_tensor(mesh_device, shape, dim, keepdim):
    """Test ttnn.std() with multi-dimensional dim on MESH tensors (the SDXL case)."""
    torch.manual_seed(0)

    # Create torch reference
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch.std(torch_input, dim=dim, keepdim=keepdim)

    # Create mesh tensor (replicated across devices)
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    ttnn_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    # This is the SDXL failing case: mesh tensor + multi-dim list
    logger.info(f"Calling ttnn.std on MESH tensor with shape={shape}, dim={dim}, keepdim={keepdim}")
    logger.info(f"Mesh device shape: {mesh_device.shape}, num_devices: {mesh_device.get_num_devices()}")

    ttnn_output = ttnn.std(ttnn_input, dim=dim, keepdim=keepdim)

    # Convert back and validate
    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.TILE_LAYOUT)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    logger.info(f"torch output shape: {torch_output.shape}, ttnn output shape: {ttnn_output_torch.shape}")
    assert_with_pcc(torch_output, ttnn_output_torch[0], pcc=0.99)
