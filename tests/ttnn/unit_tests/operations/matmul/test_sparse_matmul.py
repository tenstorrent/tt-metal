# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools

from loguru import logger
import pytest
import random
import torch
import ttnn

from models.utility_functions import skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mkn", [(16, 128, 512)])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_tokens", [(1, 4)])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
def test_sparse_matmul_with_nnz(device, mkn, num_experts, num_tokens, tile_h, tile_w, in1_dtype):
    torch.manual_seed(0)
    m, k, n = mkn
    b, s = num_tokens
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_shape = (1, b, s, num_experts)
    sparsity = torch.rand(sparsity_shape)

    # Mark some as 0 to test the sparsity
    sparsity[(sparsity == 0)] = 0.1  # First make sure there are no zeros
    number_of_zeros = random.randint(0, sparsity.numel() - 1)
    zero_indices = torch.randperm(sparsity.numel())[:number_of_zeros]
    sparsity.view(-1)[zero_indices] = 0.0

    sparsity = sparsity.to(dtype=torch.bfloat16)

    nnz = int((sparsity != 0).sum().item())
    logger.info(f"nnz: {nnz}")

    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sparsity_t = ttnn.from_torch(
        sparsity,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
    )

    output_tensor = ttnn.to_torch(output_t)

    # Compute matmul using torch for each batch and concatenate the results
    for b, s, e in itertools.product(range(b), range(s), range(num_experts)):
        if sparsity[0, b, s, e] == 0.0:
            continue
        in0_batch = in0[b, s, :, :]
        in1_batch = in1[0, e, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        expected_pcc = 0.999
        assert_with_pcc(pt_out, output_tensor[b, s, 0, e, :, :], expected_pcc)


@pytest.mark.parametrize("mkn", [(32, 128, 512)])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_tokens", [(1, 4)])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
@skip_for_blackhole("Semaphores used to broadcast sparsity is not propagating on BH, Issue #27979")
def test_sparse_matmul_without_nnz(device, mkn, num_experts, num_tokens, tile_h, tile_w, in1_dtype):
    # torch.manual_seed(0)
    m, k, n = mkn
    b, s = num_tokens
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_shape = (1, b, s, num_experts)
    sparsity = torch.rand(sparsity_shape)

    # Mark some as 0 to test the sparsity
    sparsity[(sparsity == 0)] = 0.1  # First make sure there are no zeros
    number_of_zeros = random.randint(0, sparsity.numel() - 1)
    zero_indices = torch.randperm(sparsity.numel())[:number_of_zeros]
    sparsity.view(-1)[zero_indices] = 0.0

    sparsity = sparsity.to(dtype=torch.bfloat16)

    nnz = int((sparsity != 0).sum().item())
    logger.info(f"nnz: {nnz}")

    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sparsity_t = ttnn.from_torch(
        sparsity,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
    )

    output_tensor = ttnn.to_torch(output_t)
    logger.info(output_tensor.shape)

    # Compute matmul using torch for each batch and concatenate the results
    for b, s, e in itertools.product(range(b), range(s), range(num_experts)):
        if sparsity[0, b, s, e] == 0.0:
            continue
        in0_batch = in0[b, s, :, :]
        in1_batch = in1[0, e, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        expected_pcc = 0.999
        assert_with_pcc(pt_out, output_tensor[b, s, 0, e, :, :], expected_pcc)
