# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import pytest
import ttnn
import numpy as np
from loguru import logger
from models.common.utils import LogProbsCalculator
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)
from models.common.utility_functions import comp_pcc


def check_determinism(input_values_tensor, input_indices_tensor, k, p, seed, sub_core_grids, device):
    """
    Check that the sampling operation is deterministic for the same seed.
    """
    # Run the operation twice with the same seed
    k_tensor = ttnn.from_torch(torch.tensor(k), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
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


def check_randomness(input_values_tensor, input_indices_tensor, k, p, sub_core_grids, device):
    """
    Check that the sampling operation is random without setting the seed.
    """
    # Run the operation twice with the same seed
    k_tensor = ttnn.from_torch(torch.tensor(k), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
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


def run_edge_cases(input_values, input_values_tensor, input_indices_tensor, k, p, seed, device, sub_core_grids):
    """
    Test edge cases for the sampling operation.
    """
    num_users = len(k)
    k_tensor = ttnn.from_torch(
        torch.tensor([32] * num_users), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
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


def validate_sampling(input_values, input_indices, k, p, seed, device, sub_core_grids=None):
    # Convert input tensors to ttnn tensors
    input_values_tensor = ttnn.from_torch(input_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_indices_tensor = ttnn.from_torch(input_indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    temp = ttnn.from_torch(torch.ones(32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_tensor = ttnn.from_torch(torch.tensor(k), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(torch.tensor(p), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # Call the sampling operation
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
    check_determinism(input_values_tensor, input_indices_tensor, k, p, seed, sub_core_grids, device)

    # Perform randomness check
    check_randomness(input_values_tensor, input_indices_tensor, k, p, sub_core_grids, device)

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
def test_sampling_callback(shape, k, p, seed, device):
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
def test_sampling_subcores_callback(shape, k, p, seed, device, sub_core_grids):
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


@pytest.mark.parametrize(
    "shape",
    [
        # [1, 1, 256, 256],  # llama on TG and T3K
        [1, 1, 32, 128 * 1024],  # llama on N300
        # [1, 1, 32, 128256],  # llama on T3K with 8 chips
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_calculation(shape, mesh_device):
    seed = 1234
    torch.manual_seed(seed)

    input_values = torch.randn(shape, dtype=torch.bfloat16)
    tt_input_values_tensor = ttnn.from_torch(
        input_values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator = LogProbsCalculator(input_values.shape[-1], mesh_device)
    # calculate global stats
    log_probs_calculator.compute_global_stats(tt_input_values_tensor)

    # reference implementation of log-probability computation in pytorch
    ref_log_softmax = F.log_softmax(input_values, dim=-1)

    # calculate on device log-probability computation
    log_probs = log_probs_calculator.calculate_log_probs(tt_input_values_tensor)
    # Return the log-probability tensor to host
    log_probs_host = ttnn.to_torch(log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    # Compare the log-probability computation with the reference implementation
    pcc_pass = comp_pcc(ref_log_softmax, log_probs_host, pcc=0.99)
    assert pcc_pass, "Log-probability computation is not correct, pcc={pcc_pass}"


@pytest.mark.parametrize(
    "shape",
    [
        # [1, 1, 256, 256],  # llama on TG and T3K
        [1, 1, 32, 8 * 18992],  # llama on N300
        # [1, 1, 32, 128256],  # llama on T3K with 8 chips
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_calculation2(shape, mesh_device):
    seed = 1234
    torch.manual_seed(seed)

    log_probs_calculator = LogProbsCalculator(shape[-1], mesh_device)

    torch_tensor = torch.arange(0, shape[-1]).repeat(1, 1, 32, 1)
    # shuffle the tensor in last 2 dimensions
    for i in range(shape[-2]):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    # torch_tensor = torch_tensor[:, :, :, torch.randperm(shape[-1])]

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.compute_global_stats(logits_tensor)

    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    argmax_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )

    ttnn_indices_tensor = ttnn.from_torch(
        argmax_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # prepare correct logits
    correct_logits_tensor = log_probs_calculator.prepare_correct_logits(logits_tensor, ttnn_indices_tensor)

    tt_log_probs = log_probs_calculator.calculate_log_probs(correct_logits_tensor)

    log_probs_tt_host = ttnn.to_torch(tt_log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    # slice from (1,1,32,256) -> (1,1,1,32)
    log_probs_tt_host = log_probs_tt_host[:, :, :1, :32]
    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)

    log_probs_torch_argmax = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)

    passing, pcc = comp_pcc(log_probs_torch_argmax, log_probs_tt_host, pcc=0.99)
    print(f"pcc={pcc}")

    assert passing, f"Assertion failed, PCC={pcc}"


@pytest.mark.parametrize(
    "shape",
    [
        # [1, 1, 256, 256],  # llama on TG and T3K
        [1, 1, 32, 256],  # llama on N300
    ],
)
@pytest.mark.parametrize("set_seed", [True, False])
def test_produce_sampling_output(shape, device, set_seed):
    seed = 1234
    torch.manual_seed(seed)

    input_values = torch.randn(shape)
    input_indices = torch.arange(0, shape[-1], dtype=torch.int32).expand(shape)

    input_values_tensor = ttnn.from_torch(input_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_indices_tensor = ttnn.from_torch(input_indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    temp = ttnn.from_torch(torch.ones(32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_tensor = ttnn.from_torch(torch.tensor([32] * 32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(
        torch.tensor([0.9] * 32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    if set_seed:
        output_tensor = ttnn.sampling(
            input_values_tensor,
            input_indices_tensor,
            k=k_tensor,
            p=p_tensor,
            temp=temp,
            seed=seed,
        )

    output_tensor = ttnn.sampling(
        input_values_tensor,
        input_indices_tensor,
        k=k_tensor,
        p=p_tensor,
        temp=temp,
    )

    output = ttnn.to_torch(output_tensor)
    print(f"output={output}")

    # ------------------------------------------------------------------
    # Reference log-probability computation for the sampled tokens.
    #
    # Part 1: single-device reference where the full vocab fits on one
    # device. This uses TTNN reductions over the full vocab and checks
    # that the probabilities implied by sampling match a true softmax.
    #
    # Part 2: multi-"chip" style computation that mimics a vocab sharded
    # across 8 chips. Each chip computes local max and sum(exp(.)) for a
    # single user, and we reconstruct the global softmax log-prob using
    # only those local statistics, entirely with TTNN ops on device.
    # ------------------------------------------------------------------

    num_users = shape[-2]  # H dimension
    vocab_size = shape[-1]
    logits_host = input_values[0, 0]  # [H, W]

    # === Part 1: full-vocab per-user reference on a single device ===
    max_tensor = ttnn.max(input_values_tensor, dim=3, keepdim=True)
    centered_logits = ttnn.subtract(input_values_tensor, max_tensor)
    exp_centered = ttnn.exp(centered_logits)
    sum_exp_tensor = ttnn.sum(exp_centered, dim=3, keepdim=True)

    max_host = ttnn.to_torch(max_tensor)  # [1, 1, H, 1]
    sum_exp_host = ttnn.to_torch(sum_exp_tensor)  # [1, 1, H, 1]

    max_per_user = max_host[0, 0, :, 0]  # [H]
    sum_exp_per_user = sum_exp_host[0, 0, :, 0]  # [H]

    # Output of sampling has shape [1, 1, 1, H], where the last dim is user.
    sampled_indices = output[0, 0, 0, :].to(torch.long)  # [H]
    user_ids = torch.arange(num_users, dtype=torch.long)
    sampled_logits = logits_host[user_ids, sampled_indices]  # [H]

    # log p(token) = logit - max - log(sum_exp)
    log_probs_full = sampled_logits - max_per_user - torch.log(sum_exp_per_user)

    # Compare against PyTorch log_softmax over the full vocab.
    ref_log_softmax = F.log_softmax(logits_host, dim=-1)  # [H, W]
    ref_log_probs_full = ref_log_softmax[user_ids, sampled_indices]
    assert torch.allclose(log_probs_full, ref_log_probs_full, atol=1e-2, rtol=1e-2)

    # === Part 2: multi-chip style reference for a single user ========
    #
    # We now mimic an 8-chip sharded vocab:
    #   - vocab is split evenly into num_shards slices
    #   - each "chip" holds local_vocab logits for the same user
    #   - each chip computes:
    #         m_i = max(local_logits)
    #         s_i = sum(exp(local_logits - m_i))
    #   - we reconstruct:
    #         m = max_i m_i
    #         S = sum_i exp(m_i - m) * s_i
    #     and then use:
    #         log p(token) = logit_token - m - log(S)
    #
    # All max/sum/exp/log/multiply ops are done with TTNN on device;
    # only the final scalar log-prob is moved back for validation.
    # ------------------------------------------------------------------

    num_shards = 8
    assert vocab_size % num_shards == 0
    local_vocab = vocab_size // num_shards

    # Choose a specific user to compute the log-prob for.
    target_user = 0
    logits_user = logits_host[target_user]  # [W]

    local_max_tensors = []
    local_sum_tensors = []

    for shard_idx in range(num_shards):
        start = shard_idx * local_vocab
        end = (shard_idx + 1) * local_vocab
        shard_logits = logits_user[start:end].view(1, 1, 1, -1)  # [1,1,1,local_vocab]

        shard_tensor = ttnn.from_torch(
            shard_logits,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        local_max = ttnn.max(shard_tensor, dim=3, keepdim=True)  # [1,1,1,1]
        centered_local = ttnn.subtract(shard_tensor, local_max)
        exp_centered_local = ttnn.exp(centered_local)
        local_sum = ttnn.sum(exp_centered_local, dim=3, keepdim=True)  # [1,1,1,1]

        local_max_tensors.append(local_max)
        local_sum_tensors.append(local_sum)

    # Stack local stats to get vectors of size num_shards on device.
    local_max_vec = ttnn.concat(local_max_tensors, dim=3)  # [1,1,1,num_shards]
    local_sum_vec = ttnn.concat(local_sum_tensors, dim=3)  # [1,1,1,num_shards]

    # Global max over chips: m = max_i m_i
    global_max = ttnn.max(local_max_vec, dim=3, keepdim=True)  # [1,1,1,1]

    # Global sum using stable log-sum-exp over chip-local stats:
    #   S = sum_i exp(m_i - m) * s_i
    delta = ttnn.subtract(local_max_vec, global_max)
    exp_delta = ttnn.exp(delta)
    weighted_local_sums = ttnn.multiply(exp_delta, local_sum_vec)
    global_sum = ttnn.sum(weighted_local_sums, dim=3, keepdim=True)  # [1,1,1,1]

    # Identify the sampled token for this user and its original logit.
    sampled_index_user = sampled_indices[target_user].item()
    logit_token = logits_user[sampled_index_user].view(1, 1, 1, 1)
    logit_token_tt = ttnn.from_torch(
        logit_token,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )  # [1,1,1,1]

    # Compute log p(token) on device:
    #   log p = logit - m - log(S)
    logp_centered = ttnn.subtract(logit_token_tt, global_max)
    global_log_sum = ttnn.log(global_sum, fast_and_approximate_mode=True)
    logp_tt = ttnn.subtract(logp_centered, global_log_sum)

    # Move the scalar back and compare with the full-vocab reference.
    logp = ttnn.to_torch(logp_tt)[0, 0, 0, 0]
    ref_logp = ref_log_softmax[target_user, sampled_index_user]
    assert torch.allclose(logp, ref_logp, atol=1e-2, rtol=1e-2)

    return output
