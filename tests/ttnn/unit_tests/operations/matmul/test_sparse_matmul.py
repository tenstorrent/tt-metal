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


def test_sparse_matmul_rejects_same_volume_wrong_compact_shape(device, expect_error):
    """Compact detection must validate geometry, not only logical volume."""
    in0, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    m, _, n, _, tile_h, tile_w = dims
    wrong_geometry = ttnn.from_torch(
        torch.zeros((1, nnz // 2, m * 2, n), dtype=torch.bfloat16),
        tile=ttnn.Tile([tile_h, tile_w]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    with expect_error(RuntimeError, "Optional output tensor shape"):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            dtype=ttnn.bfloat16,
            optional_output_tensor=wrong_geometry,
        )


def test_sparse_matmul_rejects_compact_optional_output_without_nnz(device, expect_error):
    """Without nnz, an optional output must use the expanded sparse layout."""
    in0, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    m, _, n, _, tile_h, tile_w = dims
    compact_output = ttnn.from_torch(
        torch.zeros((1, nnz, m, n), dtype=torch.bfloat16),
        tile=ttnn.Tile([tile_h, tile_w]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    with expect_error(RuntimeError, "when nnz is not provided"):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            dtype=ttnn.bfloat16,
            optional_output_tensor=compact_output,
        )


def test_sparse_matmul_compact_optional_output(device):
    """Compact output packs active pairs; expanded output preserves sparse positions."""
    torch.manual_seed(0)
    num_blocks, num_experts = 4, 8
    m, k, n = 32, 128, 192
    compact_sentinel = 99.0
    cached_sentinel = 98.0
    expanded_sentinel = 97.0
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
        torch.full((1, num_blocks, m, n), compact_sentinel, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    compact_output_cached = ttnn.from_torch(
        torch.full((1, num_blocks, m, n), cached_sentinel, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    expanded_output = ttnn.from_torch(
        torch.full((1, num_blocks, 1, num_experts, m, n), expanded_sentinel, dtype=torch.bfloat16),
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
    cache_entries_after_first = device.num_program_cache_entries()
    cached_output = ttnn.sparse_matmul(
        in0,
        in1,
        sparsity=sparsity,
        nnz=num_blocks,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        dtype=ttnn.bfloat16,
        optional_output_tensor=compact_output_cached,
    )
    cache_entries_after_second = device.num_program_cache_entries()
    assert cache_entries_after_second == cache_entries_after_first, "compact output should reuse the cached program"

    output_torch = ttnn.to_torch(output).float()
    cached_output_torch = ttnn.to_torch(cached_output).float()
    reference = torch.stack(
        [in0_torch[0, block].float() @ in1_torch[0, expert].float() for block, expert in enumerate(expert_for_block)]
    ).unsqueeze(0)

    expanded = ttnn.sparse_matmul(
        in0,
        in1,
        sparsity=sparsity,
        nnz=num_blocks,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        dtype=ttnn.bfloat16,
        optional_output_tensor=expanded_output,
    )
    expanded_torch = ttnn.to_torch(expanded).float()
    expanded_reference = torch.zeros((1, num_blocks, 1, num_experts, m, n))
    for block, expert in enumerate(expert_for_block):
        expanded_reference[0, block, 0, expert] = reference[0, block]

    # Compact output skips the pre-zero fill, so the writer must have covered every element;
    # the expanded output's sentinel must instead be cleared by the zero-fill.
    assert not (output_torch == compact_sentinel).any()
    assert not (cached_output_torch == cached_sentinel).any()
    assert not (expanded_torch == expanded_sentinel).any()
    torch.testing.assert_close(output_torch, reference, rtol=0.1, atol=1.5)
    torch.testing.assert_close(cached_output_torch, reference, rtol=0.1, atol=1.5)
    torch.testing.assert_close(expanded_torch, expanded_reference, rtol=0.1, atol=1.5)


def test_sparse_matmul_rejects_optional_output_tile_mismatch(device, expect_error):
    """The writer pages the output with the input-derived tile, so any other tile is rejected."""
    in0, in1, sparsity, nnz, pc, dims = _make_sparse_inputs(device)
    m, _, n, _, _, _ = dims
    mismatched_tile_output = ttnn.from_torch(
        torch.zeros((1, nnz, m, n), dtype=torch.bfloat16),
        tile=ttnn.Tile([16, 32]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    with expect_error(RuntimeError, "must equal to the in0 tile height"):
        ttnn.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            dtype=ttnn.bfloat16,
            optional_output_tensor=mismatched_tile_output,
        )


def test_sparse_matmul_compact_shape_coincides_with_expanded(device):
    """Both-inputs-sparse with every entry active: compact [1, E, M, N] equals the expanded
    shape, so the output is classified compact (zero-fill skipped) — benign under the
    exact-nnz contract because no batch is skipped and the writer covers every element."""
    torch.manual_seed(0)
    num_experts = 8
    m, k, n = 32, 128, 192
    sentinel = 96.0
    in0_torch = torch.randn((1, num_experts, m, k), dtype=torch.bfloat16)
    in1_torch = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    sparsity_torch = torch.ones((1, 1, 1, num_experts), dtype=torch.bfloat16)

    in0 = ttnn.from_torch(in0_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1 = ttnn.from_torch(in1_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity = ttnn.from_torch(
        sparsity_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    output_tensor = ttnn.from_torch(
        torch.full((1, num_experts, m, n), sentinel, dtype=torch.bfloat16),
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
        nnz=num_experts,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        dtype=ttnn.bfloat16,
        optional_output_tensor=output_tensor,
    )

    output_torch = ttnn.to_torch(output).float()
    reference = torch.stack([in0_torch[0, e].float() @ in1_torch[0, e].float() for e in range(num_experts)]).unsqueeze(
        0
    )
    assert not (output_torch == sentinel).any()
    torch.testing.assert_close(output_torch, reference, rtol=0.1, atol=1.5)


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
