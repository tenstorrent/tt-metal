# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


def test_matmul(device):
    # matrix x matrix - no batch dimensions
    tensor1 = ttnn.rand((64, 32), dtype=ttnn.bfloat16, device=device)
    tensor2 = ttnn.rand((32, 64), dtype=ttnn.bfloat16, device=device)
    output = ttnn.matmul(tensor1, tensor2)
    logger.info(f"Output matrix shape: {output.shape}")  # Output matrix shape: Shape([64, 64])

    # extended matrix x extended matrix - all batch dimensions of size 1
    tensor1 = ttnn.rand((1, 1, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(tensor1, tensor2)
    logger.info(f"Output matrix shape: {output.shape}")  # Output matrix shape: Shape([1, 1, 64, 64])

    # extended matrix x extended matrix - all batch dimensions of size 1
    tensor1 = ttnn.rand((1, 1, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.rand((1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(tensor1, tensor2)
    logger.info(f"Output matrix shape: {output.shape}")  # Output matrix shape: Shape([1, 1, 64, 64])

    # batched matrix x broadcasted matrix - first input has batch dimensions not of size 1
    tensor1 = ttnn.rand((10, 64, 32), dtype=ttnn.bfloat16, device=device)
    tensor2 = ttnn.rand((32, 64), dtype=ttnn.bfloat16, device=device)
    output = ttnn.matmul(tensor1, tensor2)
    logger.info(f"Output matrix shape: {output.shape}")  # Output matrix shape: Shape([10, 64, 64])

    # batched matrix x batched matrix - both inputs have batch dimensions
    tensor1 = ttnn.rand((10, 64, 32), dtype=ttnn.bfloat16, device=device)
    tensor2 = ttnn.rand((10, 32, 128), dtype=ttnn.bfloat16, device=device)
    output = tensor1 @ tensor2  # alternative to ttnn.matmul(tensor1, tensor2)
    logger.info(f"Output matrix shape: {output.shape}")  # Output matrix shape: Shape([10, 64, 128])

    # batched matrix x broadcasted extended matrix - first input has batch dimensions not of size 1
    tensor1 = ttnn.rand((10, 64, 32), dtype=ttnn.bfloat16, device=device)
    tensor2 = ttnn.rand((1, 1, 32, 128), dtype=ttnn.bfloat16, device=device)
    output = tensor1 @ tensor2
    logger.info(f"Output matrix shape: {output.shape}")  # Output matrix shape: Shape([1, 10, 64, 128])


def test_linear(device):
    # Define input tensors
    activations = ttnn.rand((10, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.rand((32, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.rand((128,), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Perform linear transformation

    output = ttnn.linear(activations, weight, bias=bias)
    logger.info(f"Output tensor shape: {output.shape}")  # Output tensor shape: Shape([10, 64, 128])


def test_addmm(device):
    # Define input tensors
    input_tensor = ttnn.rand((32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor1 = ttnn.rand((32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.rand((32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Perform addmm operation
    output = ttnn.addmm(input_tensor, tensor1, tensor2, beta=1.0, alpha=1.0)
    logger.info(f"Output tensor shape: {output.shape}")  # Output tensor shape: Shape([10, 64, 128])


def test_sparse_matmul(device):
    # Define program configuration
    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 2),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=2,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    nnz = 4

    #
    # Case 1: When `is_input_a_sparse` is True and `is_input_b_sparse` is True
    #
    tensor1 = ttnn.rand((1, 8, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.rand((1, 8, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Create a sparsity tensor
    sparsity_bitmask = torch.zeros((1, 1, 1, 8), dtype=torch.bfloat16)
    sparsity_bitmask.view(-1)[torch.randperm(sparsity_bitmask.numel())[:nnz]] = 1.0
    sparsity_bitmask = ttnn.to_device(ttnn.from_torch(sparsity_bitmask), device)
    output = ttnn.sparse_matmul(
        tensor1,
        tensor2,
        sparsity=sparsity_bitmask,
        nnz=nnz,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        program_config=config,
    )
    logger.info(f"Output shape: {output.shape}")  # Output shape: Shape([1, 8, 64, 64])

    # When nnz is not provided, it will be inferred from the sparsity tensor at runtime
    output = ttnn.sparse_matmul(
        tensor1,
        tensor2,
        sparsity=sparsity_bitmask,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        program_config=config,
    )
    logger.info(f"Output shape: {output.shape}")  # Output shape: Shape([1, 8, 64, 64])

    #
    # Case 2: When `is_input_a_sparse` is False and `is_input_b_sparse` is True
    #
    tensor1 = ttnn.rand((2, 16, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.rand((1, 8, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Create a sparsity tensor
    sparsity_bitmask = torch.zeros((2, 16, 1, 8), dtype=torch.bfloat16)
    sparsity_bitmask.view(-1)[torch.randperm(sparsity_bitmask.numel())[:nnz]] = 1.0
    sparsity_bitmask = ttnn.to_device(ttnn.from_torch(sparsity_bitmask), device)
    output = ttnn.sparse_matmul(
        tensor1,
        tensor2,
        sparsity=sparsity_bitmask,
        nnz=nnz,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        program_config=config,
    )
    logger.info(f"Output shape: {output.shape}")  # Output shape: Shape([2, 16, 1, 8, 64, 64])

    # When nnz is not provided, it will be inferred from the sparsity tensor at runtime
    output = ttnn.sparse_matmul(
        tensor1,
        tensor2,
        sparsity=sparsity_bitmask,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        program_config=config,
    )
    logger.info(f"Output shape: {output.shape}")  # Output shape: Shape([2, 16, 1, 8, 64, 64])

    #
    # Case 3: When `is_input_a_sparse` is True and `is_input_b_sparse` is False
    #
    tensor1 = ttnn.rand((4, 8, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor2 = ttnn.rand((1, 8, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Create a sparsity tensor
    sparsity_bitmask = torch.zeros((1, 1, 4, 8), dtype=torch.bfloat16)
    sparsity_bitmask.view(-1)[torch.randperm(sparsity_bitmask.numel())[:nnz]] = 1.0
    sparsity_bitmask = ttnn.to_device(ttnn.from_torch(sparsity_bitmask), device)
    output = ttnn.sparse_matmul(
        tensor1,
        tensor2,
        sparsity=sparsity_bitmask,
        nnz=nnz,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        program_config=config,
    )
    logger.info(f"Output shape: {output.shape}")  # Output shape: Shape([4, 8, 64, 64])
    # When nnz is not provided, it will be inferred from the sparsity tensor at runtime
    output = ttnn.sparse_matmul(
        tensor1,
        tensor2,
        sparsity=sparsity_bitmask,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        program_config=config,
    )
    logger.info(f"Output shape: {output.shape}")  # Output shape: Shape([4, 8, 64, 64])

    #
    # Case 4: When `is_input_a_sparse` is False and `is_input_b_sparse` is False
    #
    # This is invalid
