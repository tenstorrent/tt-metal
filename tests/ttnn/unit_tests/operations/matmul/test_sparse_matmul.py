# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import math

from loguru import logger
import pytest
import random
import torch
import ttnn

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


def test_sparse_matmul_rejects_indivisible_subblock(device, expect_error):
    """out_subblock_w must divide out_block_w, otherwise in1_num_subblocks is 0 and mcast_in0 deadlocks."""
    in0, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    _, _, _, _, tile_h, tile_w = dims

    bad_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
        in0_block_w=pc.in0_block_w,
        out_subblock_h=pc.out_subblock_h,
        out_subblock_w=4,
        out_block_h=pc.out_block_h,
        out_block_w=1,
        per_core_M=pc.per_core_M,
        per_core_N=pc.per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    with expect_error(RuntimeError, "must be divisible by out_subblock_w"):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=bad_pc,
        )


def test_sparse_matmul_wide_subblock(device):
    """Positive counterpart to test_sparse_matmul_rejects_indivisible_subblock.

    A config with out_subblock_w = out_block_w > 1 is newly legal (Part B) and must
    actually run and be numerically correct. Choose shapes so per_core_N is a multiple
    of out_subblock_w:
        n = 1024, tile_w = 32  -> Nt = ceil(1024/32) = 32
        core grid 4x4 = 16 cores -> per_core_N = 32 // 16 = 2
        out_subblock_w = out_block_w = 2  ->  per_core_N (2) % out_block_w (2) == 0
                                              out_block_w (2) % out_subblock_w (2) == 0
    so in1_num_subblocks = out_block_w / out_subblock_w = 1 (non-zero, no deadlock).
    """
    in0, in1, sparsity, nnz, _pc, dims = _make_sparse_inputs(device, n=1024)
    m, k, n, num_experts, tile_h, tile_w = dims

    core_x, core_y = 4, 4
    per_core_N = int(math.ceil(n / tile_w)) // (core_x * core_y)
    assert per_core_N == 2, f"expected per_core_N=2, got {per_core_N}"
    out_subblock_w = 2
    assert per_core_N % out_subblock_w == 0

    wide_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_subblock_w,
        per_core_M=m // tile_h,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )

    output_t = ttnn.sparse_matmul(
        in0,
        in1,
        sparsity=sparsity,
        nnz=nnz,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=ttnn.Tile([tile_h, tile_w]),
        program_config=wide_pc,
    )
    output_tensor = ttnn.to_torch(output_t)

    # Reference from the actual (dequantized) device inputs; sparsity is all-ones.
    in0_ref = ttnn.to_torch(in0).float()
    in1_ref = ttnn.to_torch(in1).float()
    b, s = in0_ref.shape[0], in0_ref.shape[1]
    for b_i, s_i, e_i in itertools.product(range(b), range(s), range(num_experts)):
        pt_out = torch.matmul(in0_ref[b_i, s_i, :, :], in1_ref[0, e_i, :, :])
        assert_numeric_metrics(
            pt_out,
            output_tensor[b_i, s_i, 0, e_i, :, :],
            atol=0.008 * k,
            rtol=6.313 * k,
            frobenius_threshold=0.001 * k,
            pcc_threshold=0.999,
            check_ulp=False,
        )


def test_sparse_matmul_rejects_zero_out_block_w(device, expect_error):
    """out_block_w must be non-zero.

    A zero width satisfies out_block_w % out_subblock_w == 0, so without an explicit
    non-zero check it reaches per_core_N % out_block_w and faults the host with a
    divide-by-zero instead of raising the guard's TT_FATAL.
    """
    in0, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    _, _, _, _, tile_h, tile_w = dims

    bad_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
        in0_block_w=pc.in0_block_w,
        out_subblock_h=pc.out_subblock_h,
        out_subblock_w=pc.out_subblock_w,
        out_block_h=pc.out_block_h,
        out_block_w=0,
        per_core_M=pc.per_core_M,
        per_core_N=pc.per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    with expect_error(RuntimeError, "out_block_w and out_block_h must be non-zero"):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=bad_pc,
        )


def test_sparse_matmul_rejects_indivisible_per_core_M(device, expect_error):
    """per_core_M must be divisible by out_block_h.

    The program factory computes in0_num_blocks_y = per_core_M / out_block_h, so a
    non-divisible height is silently truncated and leaves output rows uncomputed.
    """
    in0, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device, m=64)
    _, _, _, _, tile_h, tile_w = dims

    # per_core_M = 64 // 32 = 2, which is not divisible by out_block_h = 3.
    bad_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
        in0_block_w=pc.in0_block_w,
        out_subblock_h=1,
        out_subblock_w=pc.out_subblock_w,
        out_block_h=3,
        out_block_w=pc.out_block_w,
        per_core_M=pc.per_core_M,
        per_core_N=pc.per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    with expect_error(RuntimeError, "must be divisible by out_block_h"):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, tile_w]),
            program_config=bad_pc,
        )


def test_sparse_matmul_golden_contract_host_only():
    golden = ttnn.get_golden_function(ttnn.sparse_matmul)

    dense = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    experts = torch.arange(1 * 5 * 4 * 2, dtype=torch.float32).reshape(1, 5, 4, 2)
    mask = torch.tensor(
        [
            [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 1], [1, 0, 0, 1, 0], [0, 1, 1, 0, 1]],
        ],
        dtype=torch.bfloat16,
    ).unsqueeze(-2)

    actual = golden(dense, experts, sparsity=mask, is_input_a_sparse=False, is_input_b_sparse=True)
    expected = torch.einsum("abmk,ekn->abemn", dense, experts[0])
    expected = expected.unsqueeze(2) * mask[..., None, None]
    assert torch.equal(actual, expected)

    output = torch.empty_like(expected)
    result = golden(
        dense,
        experts,
        sparsity=mask,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        optional_output_tensor=output,
    )
    assert result is output
    assert torch.equal(result, expected)

    paired_a = dense[:, :1].expand(2, 5, 2, 4)
    paired_b = experts.expand(2, 5, 4, 2)
    paired_mask = torch.tensor([[[[1, 0, 1, 0, 1], [0, 1, 1, 0, 1]]]], dtype=torch.bfloat16)
    actual = golden(
        paired_a,
        paired_b,
        sparsity=paired_mask,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
    )
    expected = torch.matmul(paired_a, paired_b) * paired_mask.reshape(2, 5)[..., None, None]
    assert torch.equal(actual, expected)


def test_matmul_batched_weights_golden_contract_host_only():
    golden = ttnn.get_golden_function(ttnn.matmul_batched_weights)
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    weights = [
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
    ]

    outputs = golden(input_tensor, weights)
    assert isinstance(outputs, list)
    assert len(outputs) == len(weights)
    for output, weight in zip(outputs, weights):
        assert torch.equal(output, input_tensor @ weight)

    outputs_bfloat16 = golden(input_tensor, weights, dtype=ttnn.bfloat16)
    assert all(output.dtype == torch.bfloat16 for output in outputs_bfloat16)
