# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose


def test_manual_seed(device):
    """Test manual_seed with explicit keyword arguments for device and seeds (integer scalar)."""
    ttnn.manual_seed(seeds=42, device=device)


def test_manual_seed_with_user_id(device):
    """Test manual_seed with both seed and user_id as integer scalars using keyword arguments."""
    ttnn.manual_seed(seeds=42, device=device, user_ids=7)


def test_manual_short(device):
    """Test manual_seed using positional arguments with only seed (shorthand syntax)."""
    ttnn.manual_seed(42, device=device)


def test_manual_seed_with_tensor_user_ids(device):
    """Test manual_seed with tensor inputs for both seeds and user_ids, verifying tensor-based API."""
    user_id_tensor = ttnn.from_torch(
        torch.Tensor([0, 1, 2]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )
    ttnn.manual_seed(seeds=7, device=device, user_ids=user_id_tensor)


def test_manual_tensors(device):
    """Test manual_seed with tensor inputs for both seeds and user_ids, verifying tensor-based API."""
    seed_tensor = ttnn.from_torch(
        torch.Tensor([42, 1, 4]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )
    user_id_tensor = ttnn.from_torch(
        torch.Tensor([0, 1, 2]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )
    ttnn.manual_seed(seeds=seed_tensor, device=device, user_ids=user_id_tensor)


def test_manual_tensors_wrong_config(device):
    """Test that manual_seed raises ValueError when mixing tensor seeds with scalar user_ids (invalid configuration)."""
    seed_tensor = ttnn.from_torch(torch.Tensor([42]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)
    with pytest.raises(
        Exception, match="Seeds were provided as a tensor, so user_ids must not be provided as a scalar."
    ):
        ttnn.manual_seed(seeds=seed_tensor, device=device, user_ids=7)


def test_manual_seed_functionality(device):
    """Test that manual_seed produces consistent results in a sampling operation."""
    shape = (1, 1, 32, 64)
    input_values = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_indices = ttnn.from_torch(
        torch.arange(0, shape[-1], dtype=torch.int32).expand(shape),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    k_tensor = ttnn.from_torch(
        torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    p_tensor = ttnn.from_torch(
        torch.tensor([1.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    temp_tensor = ttnn.ones([32], layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Get first sampling result with seed 42
    ttnn.manual_seed(seeds=42, device=device)
    tensor_1 = ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor)

    # Run sampling multiple times with different seeds to change internal state
    ttnn.manual_seed(seeds=1, device=device)
    for i in range(10):
        ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor, seed=i + 1)

    # Get first sampling result with seed 42
    ttnn.manual_seed(seeds=42, device=device)
    tensor_2 = ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor)

    assert_allclose(tensor_1, tensor_2)
