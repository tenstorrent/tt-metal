# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose


def test_manual_seed_different_argument_calls(device):
    """
    Test that manual_seed accepts various valid argument configurations.
    """
    # Keyword minimum arguments
    ttnn.manual_seed(seeds=42, device=device)

    # user_ids uint32_t
    ttnn.manual_seed(seeds=42, device=device, user_ids=7)

    # No keyword seed arguments
    ttnn.manual_seed(42, device=device)

    # user_ids as tensor
    user_id_tensor = ttnn.from_torch(
        torch.Tensor([0, 1, 2]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )
    ttnn.manual_seed(seeds=7, device=device, user_ids=user_id_tensor)

    # Both seeds and user_ids as tensors
    seed_tensor = ttnn.from_torch(
        torch.Tensor([42, 1, 4]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )
    ttnn.manual_seed(seeds=seed_tensor, device=device, user_ids=user_id_tensor)


def test_manual_tensors_wrong_config(device):
    """
    Test that manual_seed correctly rejects invalid argument combinations.
    """
    seed_tensor = ttnn.from_torch(torch.Tensor([42]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)
    with pytest.raises(
        Exception, match="Seeds were provided as a tensor, so user_ids must not be provided as a scalar."
    ):
        ttnn.manual_seed(seeds=seed_tensor, device=device, user_ids=7)


def test_manual_seed_base_functionality(device):
    """
    Test that manual_seed produces deterministic and reproducible results.
    """
    # Prepare test data
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
    for i in range(5):
        ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor, seed=i + 1)

    # Get second sampling result with seed 42
    ttnn.manual_seed(seeds=42, device=device)
    tensor_2 = ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor)

    assert_allclose(tensor_1, tensor_2)


def test_manual_seed_mapping_functionality(device):
    """
    Test that manual_seed correctly handles per-core seed mapping.
    """
    # Prepare test data
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
    # Set all cores PRNG
    ttnn.manual_seed(seeds=42, device=device)

    # Prepare seed and user_id tensors for mapping
    user_id_tensor = ttnn.arange(0, 32, dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)
    seed_tensor = ttnn.rand([32], dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)

    # Get first sampling result with mapped seeds
    ttnn.manual_seed(seeds=seed_tensor, user_ids=user_id_tensor)
    tensor_1 = ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor)

    # Run sampling multiple times with different seeds to change internal state
    for i in range(5):
        ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor, seed=i + 1)

    # Get second sampling result with mapped seeds
    ttnn.manual_seed(seeds=seed_tensor, user_ids=user_id_tensor)
    tensor_2 = ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor)

    assert_allclose(tensor_1, tensor_2)


def test_manual_seed_mapping_functionality_sub_core_grids(device):
    """
    Test that manual_seed correctly handles per-core seed mapping.
    """
    # Prepare test data
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
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 3)),
        ]
    )
    # Set all cores PRNG
    ttnn.manual_seed(seeds=42, device=device, sub_core_grids=sub_core_grids)

    # Prepare seed and user_id tensors for mapping
    user_id_tensor = ttnn.arange(0, 32, dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)
    seed_tensor = ttnn.rand([32], dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)

    # Get first sampling result with mapped seeds
    ttnn.manual_seed(seeds=seed_tensor, user_ids=user_id_tensor, sub_core_grids=sub_core_grids)
    tensor_1 = ttnn.sampling(
        input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor, sub_core_grids=sub_core_grids
    )

    # Run sampling multiple times with different seeds to change internal state
    for i in range(5):
        ttnn.sampling(
            input_values,
            input_indices,
            k=k_tensor,
            p=p_tensor,
            temp=temp_tensor,
            seed=i + 1,
            sub_core_grids=sub_core_grids,
        )

    # Get second sampling result with mapped seeds
    ttnn.manual_seed(seeds=seed_tensor, user_ids=user_id_tensor, sub_core_grids=sub_core_grids)
    tensor_2 = ttnn.sampling(
        input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor, sub_core_grids=sub_core_grids
    )
    assert_allclose(tensor_1, tensor_2)
