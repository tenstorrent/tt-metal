# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import pytest
import ttnn
from loguru import logger
from models.common.utility_functions import is_wormhole_b0, is_blackhole


def check_determinism(input_values_tensor, input_indices_tensor, k, p, seed, sub_core_grids, device, k_dtype):
    """
    Check that the sampling operation is deterministic for the same seed.
    """
    # Run the operation twice with the same seed
    k_tensor = ttnn.from_torch(torch.tensor(k), device=device, dtype=k_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(torch.tensor(p), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp = ttnn.from_torch(torch.ones(32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor_1 = ttnn.sampling(
        input_values_tensor,
        input_indices_tensor,
        k=k_tensor,
        p=p_tensor,
        temp=temp,
        seed=seed,
        sub_core_grids=sub_core_grids,
    )
    output_1 = ttnn.to_torch(output_tensor_1)

    output_tensor_2 = ttnn.sampling(
        input_values_tensor,
        input_indices_tensor,
        k=k_tensor,
        p=p_tensor,
        temp=temp,
        seed=seed,
        sub_core_grids=sub_core_grids,
    )
    output_2 = ttnn.to_torch(output_tensor_2)

    # Ensure the outputs match for all users
    assert torch.allclose(output_1, output_2), "Output is not deterministic for the same seed"


def check_randomness(input_values_tensor, input_indices_tensor, k, p, sub_core_grids, device, k_dtype):
    """
    Check that the sampling operation is random without setting the seed.
    """
    # Run the operation twice with the same seed
    k_tensor = ttnn.from_torch(torch.tensor(k), device=device, dtype=k_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(torch.tensor(p), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp = ttnn.from_torch(torch.ones(32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor_1 = ttnn.sampling(
        input_values_tensor, input_indices_tensor, k=k_tensor, p=p_tensor, temp=temp, sub_core_grids=sub_core_grids
    )
    output_1 = ttnn.to_torch(output_tensor_1)

    output_tensor_2 = ttnn.sampling(
        input_values_tensor, input_indices_tensor, k=k_tensor, p=p_tensor, temp=temp, sub_core_grids=sub_core_grids
    )
    output_2 = ttnn.to_torch(output_tensor_2)

    # Ensure different outputs
    assert not torch.allclose(output_1, output_2), "Output is deterministic without setting the seed"


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
            top_k_values, top_k_ind = torch.topk(user_input_values, k=cutoff_index + 1, dim=-1)
            assert torch.all(
                torch.isin(user_output.int(), top_k_ind.int())
            ), f"Output values for user {user_idx} are not within the top-{cutoff_index} values"


def run_edge_cases(
    input_values, input_values_tensor, input_indices_tensor, k, p, seed, device, sub_core_grids, k_dtype
):
    """
    Test edge cases for the sampling operation.
    """
    num_users = len(k)
    k_tensor = ttnn.from_torch(
        torch.tensor([32] * num_users), device=device, dtype=k_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    p_tensor = ttnn.from_torch(torch.tensor(p) * 0.0, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp = ttnn.from_torch(torch.ones(32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor_k1 = ttnn.sampling(
        input_values_tensor,
        input_indices_tensor,
        k=k_tensor,
        p=p_tensor,
        temp=temp,
        seed=seed,
        sub_core_grids=sub_core_grids,
    )
    output_k1 = ttnn.to_torch(output_tensor_k1)
    top_1_value, top_1_ind = torch.topk(input_values, k=1, dim=-1)
    assert torch.allclose(
        output_k1.int().view(num_users), top_1_ind.int().view(num_users)
    ), f"Output for users does not match top-1 value"


def validate_sampling(input_values, input_indices, k, p, seed, device, sub_core_grids=None, *, k_dtype):
    # Convert input tensors to ttnn tensors
    input_values_tensor = ttnn.from_torch(input_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_indices_tensor = ttnn.from_torch(input_indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    temp = ttnn.from_torch(torch.ones(32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_tensor = ttnn.from_torch(torch.tensor(k), device=device, dtype=k_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(torch.tensor(p), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # Call the sampling operation
    with device.cache_entries_counter.measure():
        output_tensor = ttnn.sampling(
            input_values_tensor,
            input_indices_tensor,
            k=k_tensor,
            p=p_tensor,
            temp=temp,
            seed=seed,
            sub_core_grids=sub_core_grids,
        )

    # Convert the output tensor back to torch
    output = ttnn.to_torch(output_tensor)

    # Perform determinism check
    check_determinism(input_values_tensor, input_indices_tensor, k, p, seed, sub_core_grids, device, k_dtype=k_dtype)

    # Perform randomness check
    check_randomness(input_values_tensor, input_indices_tensor, k, p, sub_core_grids, device, k_dtype=k_dtype)

    # Perform statistical validation
    validate_statistics(input_values, output, k, p)

    # Perform edge case testing
    run_edge_cases(
        input_values, input_values_tensor, input_indices_tensor, k, p, seed, device, sub_core_grids, k_dtype=k_dtype
    )


def run_sampling(shape, k, p, seed, device, sub_core_grids=None, *, k_dtype):
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
        k_dtype=k_dtype,
    )


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32 * 8],  # llama on TG and T3K
        [1, 1, 32, 32 * 2],  # llama on N300
    ],
)
@pytest.mark.parametrize("k", [[10, 15, 20, 25, 30] * 6 + [10, 20]])  # Example of per-user k
@pytest.mark.parametrize("p", [[0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]])  # Example of per-user p
@pytest.mark.parametrize("seed", [2024, 11, 123])
@pytest.mark.parametrize("k_dtype", [ttnn.uint32, ttnn.int32])
def test_sampling_callback(shape, k, p, seed, k_dtype, device):
    # UINT32 k is only supported on Wormhole/Blackhole; other archs (e.g. Quasar) run the
    # INT32-only path, so skip the UINT32 cases there.
    if k_dtype == ttnn.uint32 and not (is_wormhole_b0() or is_blackhole()):
        pytest.skip("UINT32 dtype is only supported on Wormhole/Blackhole")

    torch.manual_seed(seed)
    for i in range(2):
        run_sampling(shape, k, p, seed, device, k_dtype=k_dtype)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        if i == 0:
            first_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == first_count

    logger.info(f"cache_entries_counter.total={device.cache_entries_counter.total}")
    assert device.cache_entries_counter.total > 0


# Test to run with fewer than 32 users while still mapping one core per user. The kernels
# still process a single padded 32-row tile, but only `num_users` core instances run, so the output
# last dim is `num_users`. Each (num_users, grid) config is run with two sub_core_grids modes:
#   - "explicit_grid": an explicit `grid_rows x grid_cols` CoreRangeSet with at least `num_users`
#     cores. The grid may be over-provisioned (e.g. 13 users on a 3x5=15 core grid); only the first
#     `num_users` cores are used and any extras are ignored.
#   - "full_grid": sub_core_grids=None, so the op auto-selects `num_users` cores from the device grid.
@pytest.mark.parametrize(
    "num_users, grid_rows, grid_cols",
    [
        (2, 1, 2),  # 1x2 grid, exactly sized
        (7, 1, 7),  # 1x7 grid, exactly sized
        (13, 3, 5),  # 3x5=15 core grid, over-provisioned for 13 users
    ],
    ids=["2_users_1x2", "7_users_1x7", "13_users_3x5"],
)
@pytest.mark.parametrize("Wt", [2])  # last dim = 32 * Wt; Wt must be a power of 2
@pytest.mark.parametrize("seed", [2024])
# input_indices and k are the two int-typed inputs (both accept UINT32, as well as INT32 on WH/BH)
@pytest.mark.parametrize("index_dtype", [ttnn.uint32, ttnn.int32])
@pytest.mark.parametrize("grid_mode", ["explicit", "full_grid"], ids=["explicit_grid", "full_grid"])
def test_sampling_sub_32_users(num_users, grid_rows, grid_cols, Wt, seed, index_dtype, grid_mode, device):
    # UINT32 indices/k are only supported on Wormhole/Blackhole; other archs (e.g. Quasar) only support INT32 path
    if index_dtype == ttnn.uint32 and not (is_wormhole_b0() or is_blackhole()):
        pytest.skip("UINT32 index dtype is only supported on Wormhole/Blackhole")

    torch.manual_seed(seed)
    shape = [1, 1, num_users, 32 * Wt]

    # per-user k; p == 0 -> pure top-k, which lets us validate top-k membership of each pick.
    k = [10 + 5 * (i % 5) for i in range(num_users)]
    p = [0.0] * num_users

    if grid_mode == "explicit":
        # A grid_rows x grid_cols grid with at least `num_users` cores (may be over-provisioned).
        sub_core_grids = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_cols - 1, grid_rows - 1))}
        )
        assert sub_core_grids.num_cores() >= num_users
    else:
        # Let the op auto-select `num_users` cores from the full device grid.
        sub_core_grids = None

    input_values = torch.randn(shape)
    input_indices = torch.arange(0, shape[-1], dtype=torch.int32).expand(shape)
    input_values_tensor = ttnn.from_torch(input_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_indices_tensor = ttnn.from_torch(
        input_indices, device=device, dtype=index_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    # k/p/temp carry one entry per user, so they are length `num_users`
    k_tensor = ttnn.from_torch(torch.tensor(k), device=device, dtype=index_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(torch.tensor(p), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp = ttnn.from_torch(torch.ones(num_users), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.to_torch(
        ttnn.sampling(
            input_values_tensor,
            input_indices_tensor,
            k=k_tensor,
            p=p_tensor,
            temp=temp,
            seed=seed,
            sub_core_grids=sub_core_grids,
        )
    )
    assert output.shape[-1] == num_users, f"Expected output last dim = {num_users}, got {output.shape}"

    # Same seed -> identical output.
    output_rerun = ttnn.to_torch(
        ttnn.sampling(
            input_values_tensor,
            input_indices_tensor,
            k=k_tensor,
            p=p_tensor,
            temp=temp,
            seed=seed,
            sub_core_grids=sub_core_grids,
        )
    )
    assert torch.allclose(output, output_rerun), "Output is not deterministic for the same seed"

    # With p == 0 this is pure top-k, so each user's pick must lie within its top-k set.
    validate_statistics(input_values, output, k, p)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32 * 2],  # llama on N300
    ],
)
@pytest.mark.parametrize("k", [[10, 15, 20, 25, 30] * 6 + [10, 20]])  # Example of per-user k
@pytest.mark.parametrize("p", [[0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]])  # Example of per-user p
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("k_dtype", [ttnn.uint32, ttnn.int32])
@pytest.mark.parametrize(
    "sub_core_grids", [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(8 - 1, 4 - 1))})]
)
def test_sampling_subcores_callback(shape, k, p, seed, k_dtype, device, sub_core_grids):
    # UINT32 k is only supported on Wormhole/Blackhole; other archs (e.g. Quasar) run the
    # INT32-only path, so skip the UINT32 cases there.
    if k_dtype == ttnn.uint32 and not (is_wormhole_b0() or is_blackhole()):
        pytest.skip("UINT32 dtype is only supported on Wormhole/Blackhole")

    torch.manual_seed(seed)
    for i in range(2):
        run_sampling(shape, k, p, seed, device, sub_core_grids, k_dtype=k_dtype)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        if i == 0:
            first_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == first_count

    logger.info(f"cache_entries_counter.total={device.cache_entries_counter.total}")
    assert device.cache_entries_counter.total > 0
