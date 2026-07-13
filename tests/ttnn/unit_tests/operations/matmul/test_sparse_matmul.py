# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import os

from loguru import logger
import pytest
import random
import torch
import ttnn

from models.common.utility_functions import skip_for_slow_dispatch
from tests.ttnn.utils_for_testing import assert_numeric_metrics


@pytest.mark.parametrize("mkn", [(16, 128, 512)])
@pytest.mark.parametrize("num_experts", [8, 32])
@pytest.mark.parametrize("num_batches", [(1, 4)])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("core_grid", [(4, 4)])
def test_sparse_matmul_with_nnz(device, mkn, num_experts, num_batches, tile_h, tile_w, in1_dtype, core_grid):
    torch.manual_seed(0)
    m, k, n = mkn
    b, s = num_batches
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_shape = (b, s, 1, num_experts)
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

    core_x, core_y = core_grid
    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=nnz,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // tile_h,
            per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )

    output_tensor = ttnn.to_torch(output_t)

    # Compute matmul using torch for each batch and check the results
    for b_i, s_i, e_i in itertools.product(range(b), range(s), range(num_experts)):
        if sparsity[b_i, s_i, 0, e_i] == 0.0:
            continue
        in0_batch = in0[b_i, s_i, :, :]
        in1_batch = in1[0, e_i, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        assert_numeric_metrics(
            pt_out,
            output_tensor[b_i, s_i, 0, e_i, :, :],
            atol=0.01 * k,
            rtol=10.188 * k,
            frobenius_threshold=0.001 * k,
            pcc_threshold=0.999,
        )


@pytest.mark.parametrize("mkn", [(16, 128, 512)])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_batches", [(1, 4)])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("core_grid", [(4, 4)])
def test_sparse_matmul_without_nnz(device, mkn, num_experts, num_batches, tile_h, tile_w, in1_dtype, core_grid):
    torch.manual_seed(0)
    m, k, n = mkn
    b, s = num_batches
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_shape = (b, s, 1, num_experts)
    sparsity = torch.rand(sparsity_shape)

    # Mark some as 0 to test the sparsity
    sparsity[(sparsity == 0)] = 0.1  # First make sure there are no zeros
    number_of_zeros = torch.randint(0, sparsity.numel(), ()).item()
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

    core_x, core_y = core_grid
    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // tile_h,
            per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )

    output_tensor = ttnn.to_torch(output_t)

    # Compute matmul using torch for each batch and check the results
    for b_i, s_i, e_i in itertools.product(range(b), range(s), range(num_experts)):
        if sparsity[b_i, s_i, 0, e_i] == 0.0:
            continue
        in0_batch = in0[b_i, s_i, :, :]
        in1_batch = in1[0, e_i, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        if in1_dtype == ttnn.bfloat8_b:
            assert_numeric_metrics(
                pt_out,
                output_tensor[b_i, s_i, 0, e_i, :, :],
                atol=0.008 * k,
                rtol=6.313 * k,
                frobenius_threshold=0.001 * k,
                pcc_threshold=0.999,
                check_ulp=False,
            )
        else:
            assert_numeric_metrics(
                pt_out,
                output_tensor[b_i, s_i, 0, e_i, :, :],
                atol=0.01 * k,
                rtol=10.188 * k,
                frobenius_threshold=0.001 * k,
                pcc_threshold=0.999,
                check_ulp=False,
            )


@pytest.mark.parametrize("mkn", [(16, 128, 512)])
@pytest.mark.parametrize("num_experts", [(1, 32), (1, 128)])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("core_grid", [(4, 4)])
def test_batched_sparse_matmul_with_nnz(device, mkn, num_experts, tile_h, tile_w, in1_dtype, core_grid):
    torch.manual_seed(0)
    m, k, n = mkn
    b, s = num_experts
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((b, s, k, n), dtype=torch.bfloat16)

    sparsity_shape = (1, 1, b, s)
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

    core_x, core_y = core_grid
    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // tile_h,
            per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )

    output_tensor = ttnn.to_torch(output_t)

    # Compute matmul using torch for each batch and check the results
    for b_i, s_i in itertools.product(range(b), range(s)):
        if sparsity[0, 0, b_i, s_i] == 0.0:
            continue
        in0_batch = in0[b_i, s_i, :, :]
        in1_batch = in1[b_i, s_i, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        assert_numeric_metrics(
            pt_out,
            output_tensor[b_i, s_i, :, :],
            atol=0.01 * k,
            rtol=21.25 * k,
            frobenius_threshold=0.001 * k,
            pcc_threshold=0.999,
            check_ulp=False,
        )


@pytest.mark.parametrize("mkn", [(16, 128, 512)])
@pytest.mark.parametrize("num_experts", [(1, 32), (1, 128)])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("core_grid", [(4, 4)])
def test_batched_sparse_matmul_without_nnz(device, mkn, num_experts, tile_h, tile_w, in1_dtype, core_grid):
    torch.manual_seed(0)
    m, k, n = mkn
    b, s = num_experts
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((b, s, k, n), dtype=torch.bfloat16)

    sparsity_shape = (1, 1, b, s)
    sparsity = torch.rand(sparsity_shape)

    # Mark some as 0 to test the sparsity
    sparsity[(sparsity == 0)] = 0.1  # First make sure there are no zeros
    number_of_zeros = random.randint(0, sparsity.numel() - 1)
    zero_indices = torch.randperm(sparsity.numel())[:number_of_zeros]
    sparsity.view(-1)[zero_indices] = 0.0

    sparsity = sparsity.to(dtype=torch.bfloat16)

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

    core_x, core_y = core_grid
    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // tile_h,
            per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )

    output_tensor = ttnn.to_torch(output_t)

    # Compute matmul using torch for each batch and check the results
    for b_i, s_i in itertools.product(range(b), range(s)):
        if sparsity[0, 0, b_i, s_i] == 0.0:
            continue
        in0_batch = in0[b_i, s_i, :, :]
        in1_batch = in1[b_i, s_i, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        assert_numeric_metrics(
            pt_out,
            output_tensor[b_i, s_i, :, :],
            atol=0.01 * k,
            rtol=25.875 * k,
            frobenius_threshold=0.001 * k,
            pcc_threshold=0.999,
            check_ulp=False,
        )


@pytest.mark.parametrize("mkn", [(16, 128, 512)])
@pytest.mark.parametrize("num_experts", [8, 32])
@pytest.mark.parametrize("num_batches", [4])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("core_grid", [(4, 4)])
def test_sparse_matmul_inputA_with_nnz(device, mkn, num_experts, num_batches, tile_h, tile_w, in1_dtype, core_grid):
    torch.manual_seed(0)
    m, k, n = mkn
    b = num_batches
    in0 = torch.randn((b, num_experts, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_shape = (1, 1, b, num_experts)
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

    core_x, core_y = core_grid
    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=nnz,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // tile_h,
            per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )

    output_tensor = ttnn.to_torch(output_t)
    logger.info(f"output_tensor.shape: {output_tensor.shape}")

    # Compute matmul using torch for each batch and check the results
    for b_i, e_i in itertools.product(range(b), range(num_experts)):
        if sparsity[0, 0, b_i, e_i] == 0.0:
            continue
        in0_batch = in0[b_i, e_i, :, :]
        in1_batch = in1[0, e_i, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        assert_numeric_metrics(
            pt_out,
            output_tensor[b_i, e_i, :, :],
            atol=0.012 * k,
            rtol=22.25 * k,
            frobenius_threshold=0.001 * k,
            pcc_threshold=0.999,
            check_ulp=False,
        )


@pytest.mark.parametrize("mkn", [(16, 128, 512)])
@pytest.mark.parametrize("num_experts", [8, 32])
@pytest.mark.parametrize("num_batches", [4])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("core_grid", [(4, 4)])
def test_sparse_matmul_inputA_without_nnz(device, mkn, num_experts, num_batches, tile_h, tile_w, in1_dtype, core_grid):
    torch.manual_seed(0)
    m, k, n = mkn
    b = num_batches
    in0 = torch.randn((b, num_experts, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_shape = (1, 1, b, num_experts)
    sparsity = torch.rand(sparsity_shape)

    # Mark some as 0 to test the sparsity
    sparsity[(sparsity == 0)] = 0.1  # First make sure there are no zeros
    number_of_zeros = random.randint(0, sparsity.numel() - 1)
    zero_indices = torch.randperm(sparsity.numel())[:number_of_zeros]
    sparsity.view(-1)[zero_indices] = 0.0

    sparsity = sparsity.to(dtype=torch.bfloat16)

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

    core_x, core_y = core_grid
    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // tile_h,
            per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )

    output_tensor = ttnn.to_torch(output_t)
    logger.info(f"output_tensor.shape: {output_tensor.shape}")

    # Compute matmul using torch for each batch and check the results
    for b_i, e_i in itertools.product(range(b), range(num_experts)):
        if sparsity[0, 0, b_i, e_i] == 0.0:
            continue
        in0_batch = in0[b_i, e_i, :, :]
        in1_batch = in1[0, e_i, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)

        # Compare with output tensor
        assert_numeric_metrics(
            pt_out,
            output_tensor[b_i, e_i, :, :],
            atol=0.01 * k,
            rtol=22.25 * k,
            frobenius_threshold=0.001 * k,
            pcc_threshold=0.999,
            check_ulp=False,
        )


def _make_sparse_inputs(device, b=1, s=4, m=32, k=128, n=512, num_experts=8, tile_h=32, tile_w=32):
    torch.manual_seed(0)
    in0 = ttnn.from_torch(
        torch.randn((b, s, m, k), dtype=torch.bfloat16),
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1 = ttnn.from_torch(
        torch.randn((1, num_experts, k, n), dtype=torch.bfloat16),
        tile=ttnn.Tile((32, tile_w)),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sparsity = ttnn.from_torch(
        torch.ones((b, s, 1, num_experts), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    nnz = b * s * num_experts
    core_x, core_y = 4, 4
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=m // tile_h,
        per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    return in0, in1, sparsity, nnz, program_config, (m, k, n, num_experts, tile_h, tile_w)


def test_sparse_matmul_requires_at_least_one_sparse_flag(device, expect_error):
    """At least one of is_input_a_sparse / is_input_b_sparse must be true."""
    in0, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    _, _, _, _, tile_h, tile_w = dims

    with expect_error(
        RuntimeError,
        "sparse_matmul requires at least one of is_input_a_sparse or is_input_b_sparse to be true",
    ):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=pc,
        )


def test_sparse_matmul_volume_must_match_batch_length(device, expect_error):
    """sparsity logical_volume must equal product of all batch dimensions."""
    in0, in1, _, nnz, pc, dims = _make_sparse_inputs(device)
    _, _, _, num_experts, tile_h, tile_w = dims

    wrong_sparsity = ttnn.from_torch(
        torch.ones((1, 2, 1, num_experts), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with expect_error(RuntimeError, "sparsity logical_volume"):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=wrong_sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=pc,
        )


def test_sparse_matmul_inputA_wrong_layout(device, expect_error):
    """Input tensor A must be TILE layout, ROW_MAJOR must be rejected."""
    _, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    m, k, _, _, tile_h, tile_w = dims

    in0_row_major = ttnn.from_torch(
        torch.randn((1, 4, m, k), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "Input tensor A must be TILE layout"):
        ttnn.sparse_matmul(
            in0_row_major,
            in1,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=pc,
        )


def test_sparse_matmul_inputB_wrong_layout(device, expect_error):
    """Input tensor B must be TILE layout, ROW_MAJOR must be rejected."""
    in0, _, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    _, k, n, num_experts, tile_h, tile_w = dims

    in1_row_major = ttnn.from_torch(
        torch.randn((1, num_experts, k, n), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "Input tensor B must be TILE layout"):
        ttnn.sparse_matmul(
            in0,
            in1_row_major,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=pc,
        )


def test_sparse_matmul_inputA_wrong_dtype(device, expect_error):
    """Input tensor A must be floating point, integer types must be rejected."""
    _, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    m, k, _, _, tile_h, tile_w = dims

    in0_int = ttnn.from_torch(
        torch.ones((1, 4, m, k), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "Input tensor A must be a floating point type"):
        ttnn.sparse_matmul(
            in0_int,
            in1,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=pc,
        )


def test_sparse_matmul_inputB_wrong_dtype(device, expect_error):
    """Input tensor B must be floating point, integer types must be rejected."""
    in0, _, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    _, k, n, num_experts, tile_h, tile_w = dims

    in1_int = ttnn.from_torch(
        torch.ones((1, num_experts, k, n), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "Input tensor B must be a floating point type"):
        ttnn.sparse_matmul(
            in0,
            in1_int,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=pc,
        )


def test_sparse_matmul_sparsity_wrong_layout(device, expect_error):
    """Sparsity tensor must be ROW_MAJOR, TILE layout must be rejected."""
    in0, in1, _, nnz, pc, dims = _make_sparse_inputs(device)
    _, _, _, _, tile_h, tile_w = dims

    sparsity_tile = ttnn.from_torch(
        torch.ones((1, 4, 32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "Sparsity tensor must be ROW_MAJOR layout"):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity_tile,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=pc,
        )


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_dense_mask_sparse_matmul_on_independent_subdevices(device):
    """Dense sparse masks can run concurrently on offset sub-devices.

    When nnz equals the sparsity volume, sparse_matmul fully overwrites its
    output. This must not enqueue a redundant zeros_like on the default
    sub-device, and the sparse 1D factory must anchor its program at the
    selected sub-device's worker-grid origin.
    """
    grid = device.compute_with_storage_grid_size()
    if grid.y < 2:
        pytest.skip("Test requires at least two worker rows")

    torch.manual_seed(0)
    in0_torch = torch.randn((1, 1, 32, 128), dtype=torch.bfloat16)
    in1_torch = torch.randn((1, 128, 128, 192), dtype=torch.bfloat16)
    expected = in0_torch[0, 0] @ in1_torch[0]
    in0 = ttnn.from_torch(in0_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1 = ttnn.from_torch(in1_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity = ttnn.from_torch(
        torch.ones((1, 1, 1, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    outputs = [
        ttnn.from_torch(
            torch.full((1, 1, 1, 128, 32, 192), 99.0, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for _ in range(2)
    ]

    row0 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 0))})
    remaining_rows = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = device.create_sub_device_manager([ttnn.SubDevice([row0]), ttnn.SubDevice([remaining_rows])], 0)
    sub_device_ids = [ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)]

    def program_config(start_y):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(6, 1),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            allowed_worker_cores=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, start_y), ttnn.CoreCoord(5, start_y))}
            ),
        )

    try:
        device.load_sub_device_manager(manager)
        device.set_sub_device_stall_group(sub_device_ids)
        for cq_id, sub_device_id in enumerate(sub_device_ids):
            with ttnn.command_queue(cq_id):
                ttnn.sparse_matmul(
                    in0,
                    in1,
                    sparsity=sparsity,
                    nnz=128,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=program_config(cq_id),
                    optional_output_tensor=outputs[cq_id],
                    sub_device_id=sub_device_id,
                )
        ttnn.synchronize_device(device)
    finally:
        device.reset_sub_device_stall_group()
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)

    for output in outputs:
        torch.testing.assert_close(ttnn.to_torch(output)[0, 0, 0], expected, rtol=0.1, atol=1.5)


def test_sparse_matmul_compact_optional_output(device):
    """A compact optional output stores only nonzero batch pairs in scan order."""
    torch.manual_seed(0)
    num_blocks, num_experts = 4, 8
    m, k, n = 32, 128, 192
    expert_for_block = [3, 1, 7, 2]
    in0_torch = torch.randn((1, num_blocks, m, k), dtype=torch.bfloat16)
    in1_torch = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    sparsity_torch = torch.zeros((1, 1, num_blocks, num_experts), dtype=torch.bfloat16)
    for block, expert in enumerate(expert_for_block):
        sparsity_torch[0, 0, block, expert] = 1

    in0 = ttnn.from_torch(in0_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1 = ttnn.from_torch(in1_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity = ttnn.from_torch(
        sparsity_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    compact_output = ttnn.from_torch(
        torch.full((1, num_blocks, m, n), 99.0, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(6, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )

    output = ttnn.sparse_matmul(
        in0,
        in1,
        sparsity=sparsity,
        nnz=num_blocks,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        dtype=ttnn.bfloat16,
        optional_output_tensor=compact_output,
    )
    output_torch = ttnn.to_torch(output).float()
    reference = torch.stack(
        [in0_torch[0, block].float() @ in1_torch[0, expert].float() for block, expert in enumerate(expert_for_block)]
    ).unsqueeze(0)

    assert tuple(output.shape) == (1, num_blocks, m, n)
    assert not torch.any(output_torch == 99)
    torch.testing.assert_close(output_torch, reference, rtol=0.1, atol=1.5)


# DiffusionGemma fused-MoE increment 1: WRITER_SCALE.
#
# The sparse writer kernel, when TTNN_SPARSE_MATMUL_WRITER_SCALE is set, folds the per-batch value
# carried in the cb_sparsity page into the output write (a real step toward fusing the MoE
# combine's route-weight multiply -- see
# models/experimental/diffusion_gemma/doc/optimize_perf/fused_moe_kernel.md). Today the op uses only
# the sparsity `== 0` gate; the nonzero magnitude is ignored. This test asserts the OPPOSITE
# behavior under the flag: each active output batch is scaled by its (nonzero) sparsity value.
#
# Requires a Tenstorrent device (the writer kernel is JIT-compiled on device). The env var is read
# inside the program factory and is NOT part of the program hash, so this test uses a distinct
# num_experts and sets the flag via monkeypatch before the first op -- do not run it in the same
# process after a same-shape sparse_matmul that built the program with the flag off.
@pytest.mark.parametrize("mkn", [(16, 128, 512)])
@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("num_batches", [(1, 4)])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("core_grid", [(4, 4)])
def test_sparse_matmul_writer_scale(monkeypatch, device, mkn, num_experts, num_batches, tile_h, tile_w, core_grid):
    monkeypatch.setenv("TTNN_SPARSE_MATMUL_WRITER_SCALE", "1")
    assert os.environ.get("TTNN_SPARSE_MATMUL_WRITER_SCALE") == "1"

    torch.manual_seed(0)
    m, k, n = mkn
    b, s = num_batches
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    # Per-batch route-weights carried in the sparsity tensor: nonzero values are the scale, some
    # entries are zeroed (dropped/inactive) exactly as in the other sparse_matmul tests.
    sparsity_shape = (b, s, 1, num_experts)
    sparsity = torch.rand(sparsity_shape) * 0.9 + 0.1  # nonzero scales in [0.1, 1.0)
    number_of_zeros = random.randint(0, sparsity.numel() - 1)
    zero_indices = torch.randperm(sparsity.numel())[:number_of_zeros]
    sparsity.view(-1)[zero_indices] = 0.0
    sparsity = sparsity.to(dtype=torch.bfloat16)

    nnz = int((sparsity != 0).sum().item())
    logger.info(f"writer_scale nnz: {nnz}")

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
        dtype=ttnn.bfloat16,
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

    core_x, core_y = core_grid
    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=nnz,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        dtype=ttnn.bfloat16,  # bf16 output -> exercises the bf16 in-place scale path
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // tile_h,
            per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )

    output_tensor = ttnn.to_torch(output_t)

    # Reference: matmul scaled by the per-batch route-weight (the WRITER_SCALE contract).
    for b_i, s_i, e_i in itertools.product(range(b), range(s), range(num_experts)):
        scale = float(sparsity[b_i, s_i, 0, e_i])
        if scale == 0.0:
            continue
        pt_out = torch.matmul(in0[b_i, s_i, :, :].float(), in1[0, e_i, :, :].float()) * scale
        assert_numeric_metrics(
            pt_out,
            output_tensor[b_i, s_i, 0, e_i, :, :],
            atol=0.02 * k,
            rtol=10.188 * k,
            frobenius_threshold=0.002 * k,
            pcc_threshold=0.999,
        )
