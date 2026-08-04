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
