# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import pytest
import ttnn
import numpy as np
from loguru import logger
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)
from models.utility_functions import skip_for_blackhole


def check_determinism(input_values_tensor, input_indices_tensor, k, p, seed, device, sub_core_grids):
    """
    Check that the sampling operation is deterministic for the same seed.
    """
    # Run the operation twice with the same seed
    output_tensor_1 = ttnn.sampling(
        input_values_tensor, input_indices_tensor, k=k, p=p, seed=seed, sub_core_grids=sub_core_grids
    )
    output_1 = ttnn.to_torch(output_tensor_1)

    output_tensor_2 = ttnn.sampling(
        input_values_tensor, input_indices_tensor, k=k, p=p, seed=seed, sub_core_grids=sub_core_grids
    )
    output_2 = ttnn.to_torch(output_tensor_2)

    # Ensure the outputs match for all users
    assert torch.allclose(output_1, output_2), "Output is not deterministic for the same seed"


def validate_statistics(input_values, output, k, p):
    """
    Validate the statistical properties of the sampling operation.
    """
    num_users = input_values.shape[-2]
    for user_idx in range(num_users):
        user_input_values = input_values[:, :, user_idx, :]
        user_output = output[:, :, :, user_idx]
        user_k = k[user_idx]
        user_p = p[user_idx]

        # For top-k sampling, ensure the output contains the top-k values
        if user_p == 0.0:
            top_k_values, top_k_ind = torch.topk(user_input_values, k=user_k, dim=-1)
            assert torch.all(
                torch.isin(user_output.int(), top_k_ind.int())
            ), f"Output values for user {user_idx} are not within the top-{user_k} values"

        # For top-p sampling, ensure the cumulative probability of the selected values is >= p
        if user_p != 0.0:
            probs = F.softmax(user_input_values, dim=-1)
            sorted_probs, _ = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = torch.searchsorted(cumulative_probs.squeeze(0).squeeze(0), user_p) + 1
            top_k_values, top_k_ind = torch.topk(user_input_values, k=cutoff_index, dim=-1)
            assert torch.all(
                torch.isin(user_output.int(), top_k_ind.int())
            ), f"Output values for user {user_idx} are not within the top-{cutoff_index} values"


def run_edge_cases(input_values, input_values_tensor, input_indices_tensor, k, p, seed, device, sub_core_grids):
    """
    Test edge cases for the sampling operation.
    """
    num_users = len(k)
    output_tensor_k1 = ttnn.sampling(
        input_values_tensor, input_indices_tensor, k=[1] * num_users, p=p, seed=seed, sub_core_grids=sub_core_grids
    )
    output_k1 = ttnn.to_torch(output_tensor_k1)
    top_1_value, top_1_ind = torch.topk(input_values, k=1, dim=-1)
    assert torch.allclose(
        output_k1.int().view(num_users), top_1_ind.int().view(num_users)
    ), f"Output for users does not match top-1 value"


def validate_sampling(input_values, input_indices, k, p, seed, device, sub_core_grids=None):
    # Convert input tensors to ttnn tensors
    input_values_tensor = ttnn.from_torch(input_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_indices_tensor = ttnn.from_torch(input_indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Call the sampling operation
    output_tensor = ttnn.sampling(
        input_values_tensor, input_indices_tensor, k=k, p=p, seed=seed, sub_core_grids=sub_core_grids
    )

    # Convert the output tensor back to torch
    output = ttnn.to_torch(output_tensor)

    # Perform determinism check
    check_determinism(input_values_tensor, input_indices_tensor, k, p, seed, device, sub_core_grids)

    # Perform statistical validation
    validate_statistics(input_values, output, k, p)

    # Perform edge case testing
    run_edge_cases(input_values, input_values_tensor, input_indices_tensor, k, p, seed, device, sub_core_grids)


def run_sampling(shape, k, p, seed, device, sub_core_grids=None):
    # Generate random input values and indices
    input_values = torch.randn(shape)
    input_indices = torch.arange(0, shape[-1], dtype=torch.int32).expand(shape)

    # Validate the sampling operation
    validate_sampling(
        input_values=input_values,
        input_indices=input_indices,
        k=k,
        p=p,
        seed=seed,
        device=device,
        sub_core_grids=sub_core_grids,
    )


@skip_for_blackhole("Requires wormhole_b0 to run. Issue #19640")
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32 * 8],  # llama on TG and T3K
        [1, 1, 32, 32 * 2],  # llama on N300
    ],
)
@pytest.mark.parametrize("k", [[10, 15, 20, 25, 30] * 6 + [10, 20]])  # Example of per-user k
@pytest.mark.parametrize("p", [[0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]])  # Example of per-user p
@pytest.mark.parametrize("seed", [2024, 11, 0])
def test_sampling_callback(shape, k, p, seed, device, use_program_cache):
    torch.manual_seed(seed)
    num_program_cache_entries_list = []
    for _ in range(2):
        run_sampling(shape, k, p, seed, device)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@skip_for_blackhole("Requires wormhole_b0 to run. Issue #19640")
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32 * 2],  # llama on N300
    ],
)
@pytest.mark.parametrize("k", [[10, 15, 20, 25, 30] * 6 + [10, 20]])  # Example of per-user k
@pytest.mark.parametrize("p", [[0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]])  # Example of per-user p
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize(
    "sub_core_grids", [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(8 - 1, 4 - 1))})]
)
def test_sampling_subcores_callback(shape, k, p, seed, device, sub_core_grids, use_program_cache):
    torch.manual_seed(seed)
    num_program_cache_entries_list = []
    for _ in range(2):
        run_sampling(shape, k, p, seed, device, sub_core_grids)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
