# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


def test_cumprod(device):
    # Create tensor
    tensor_input = ttnn.rand((2, 3, 4), device=device)

    # Apply ttnn.cumprod() on dim=0
    tensor_output = ttnn.cumprod(tensor_input, dim=0)
    logger.info(f"Cumprod result: {tensor_output}")

    # With preallocated output and dtype
    preallocated_output = ttnn.rand([2, 3, 4], dtype=ttnn.bfloat16, device=device)

    # Apply ttnn.cumprod() with out and dtype
    tensor_output = ttnn.cumprod(tensor_input, dim=0, dtype=ttnn.bfloat16, out=preallocated_output)
    logger.info(f"Cumprod with preallocated output result: {tensor_output}")


def test_max(device):
    # Create tensor
    tensor_input = ttnn.rand((2, 3, 4), device=device)

    # Apply ttnn.max() on dim=1
    tensor_output = ttnn.max(tensor_input, dim=1)
    logger.info(f"Max result: {tensor_output}")


def test_mean(device):
    # Create tensor
    tensor_input = ttnn.rand((2, 3, 4), device=device)

    # Apply ttnn.mean() on dim=2
    tensor_output = ttnn.mean(tensor_input, dim=2)
    logger.info(f"Mean result: {tensor_output}")


def test_min(device):
    # Create tensor
    tensor_input = ttnn.rand((2, 3, 4), device=device)

    # Apply ttnn.min() on dim=0
    tensor_output = ttnn.min(tensor_input, dim=0)
    logger.info(f"Min result: {tensor_output}")


def test_std(device):
    # Create tensor
    tensor_input = ttnn.rand((2, 3, 4), device=device)

    # Apply ttnn.std() on dim=1
    tensor_output = ttnn.std(tensor_input, dim=1)
    logger.info(f"Std result: {tensor_output}")


def test_sum(device):
    # Create tensor
    tensor_input = ttnn.rand((2, 3, 4), device=device)

    # Apply ttnn.sum() on dim=2
    tensor_output = ttnn.sum(tensor_input, dim=2)
    logger.info(f"Sum result: {tensor_output}")


def test_var(device):
    # Create tensor
    tensor_input = ttnn.rand((2, 3, 4), device=device)

    # Apply ttnn.var() on dim=0
    tensor_output = ttnn.var(tensor_input, dim=0)
    logger.info(f"Var result: {tensor_output}")


def test_argmax(device):
    # Create tensor
    tensor_input = ttnn.rand([1, 1, 32, 64], device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Last dim reduction yields shape of [1, 1, 32, 1]
    output_onedim = ttnn.argmax(tensor_input, dim=-1, keepdim=True)
    logger.info(f"Argmax onedim result: {output_onedim}")

    # All dim reduction yields shape of []
    output_alldim = ttnn.argmax(tensor_input)
    logger.info(f"Argmax alldim result: {output_alldim}")


def test_prod(device):
    # 1D Product
    tensor = ttnn.rand((1, 2), device=device)

    # Apply ttnn.prod() on dim=0
    output = ttnn.prod(tensor, dim=0)
    logger.info(f"Prod result: {output}")
    # All Dims Product
    output_all_dims = ttnn.prod(tensor)
    logger.info(f"Prod all dims result: {output_all_dims}")

    # NC Product
    # Define reduction dims, input and output shapes
    dims = [0, 1]
    input_shape = [2, 3, 4, 5]
    output_shape = [1, 1, 4, 5]  # shape on any dimension being reduced will be 1

    # Create input and output tensors
    input_tensor = ttnn.rand(input_shape, device=device)
    output_tensor = ttnn.rand(output_shape, device=device)

    # Apply ttnn.prod() on specified dims
    output = ttnn.prod(input_tensor=input_tensor, output_tensor=output_tensor, dims=dims)
    logger.info(f"Prod result: {output}")


def test_sampling(device):
    # Input values tensor for N*C*H = 32 users
    input_tensor = ttnn.rand([1, 1, 32, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Input indices tensor: this example uses sequential indices [0, 1, 2, ..., W-1] for each of the 32 users
    # Resulting in a final shape of [1, 1, 32, 64]
    indices_1d = ttnn.arange(0, 64, dtype=ttnn.int32, device=device)
    indices_reshaped = ttnn.reshape(indices_1d, [1, 1, 1, 64])
    input_indices_tensor = ttnn.repeat(indices_reshaped, (1, 1, 32, 1))

    # k tensor: 32 values in range (0, 32] for top-k sampling
    k_tensor = ttnn.full([32], fill_value=10, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # p tensor: 32 values in range [0.0, 1.0] for top-p sampling
    p_tensor = ttnn.full([32], fill_value=0.9, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # temp tensor: 32 temperature values in range [0.0, 1.0]
    temp_tensor = ttnn.ones([32], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output = ttnn.sampling(input_tensor, input_indices_tensor, k=k_tensor, p=p_tensor, temp=temp_tensor)
    logger.info(f"Sampling result: {output}")


def test_topk(device):
    # Create tensor
    tensor_input = ttnn.rand([1, 1, 32, 64], device=device)

    # Apply ttnn.topk() to get top 3 values along dim=1
    values, indices = ttnn.topk(tensor_input, k=32, dim=-1, largest=True, sorted=True)
    logger.info(f"Topk values: {values}")
    logger.info(f"Topk indices: {indices}")


def test_cumsum(device):
    # Create tensor
    tensor_input = ttnn.rand((2, 3, 4), device=device)

    # Apply ttnn.cumsum() on dim=0
    tensor_output = ttnn.cumsum(tensor_input, dim=0)
    logger.info(f"Cumsum result: {tensor_output}")

    # With preallocated output and dtype
    preallocated_output = ttnn.rand([2, 3, 4], dtype=ttnn.bfloat16, device=device)

    tensor_output = ttnn.cumsum(tensor_input, dim=0, dtype=ttnn.bfloat16, out=preallocated_output)
    logger.info(f"Cumsum result: {tensor_output}")


def test_ema(device):
    # Create tensor
    tensor_input = ttnn.rand((1, 2, 64, 128), device=device, layout=ttnn.TILE_LAYOUT)

    # Apply ttnn.ema() with alpha=0.99
    tensor_output = ttnn.ema(tensor_input, 0.99)
    logger.info(f"EMA result: {tensor_output}")

    # With preallocated output
    preallocated_output = ttnn.rand([1, 2, 64, 128], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tensor_output = ttnn.ema(tensor_input, 0.99, out=preallocated_output)
    logger.info(f"EMA with preallocated output result: {tensor_output}")


def test_moe(device):
    N, C, H, W = 1, 1, 32, 64
    k = 32

    input_tensor = ttnn.rand([N, C, H, W], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    expert_mask = ttnn.zeros([N, C, 1, W], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    topE_mask = ttnn.zeros([N, C, 1, k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tensor_output = ttnn.moe(input_tensor, expert_mask, topE_mask, k)
    logger.info(f"MOE result: {tensor_output}")


def test_manual_seed(device):
    # Set manual seed with scalar seed value for all cores
    ttnn.manual_seed(seeds=42, device=device)

    # Set manual seed for specific core
    ttnn.manual_seed(seeds=42, device=device, user_ids=7)

    # Set manual seed with tensor of seeds and tensor of user IDs
    # Maps user_id to seed value e.g., user_id 0 -> seed 42, user_id 1 -> seed 1, user_id 2 -> seed 4
    seed_tensor = ttnn.from_torch(
        torch.Tensor([42, 1, 4]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )
    user_id_tensor = ttnn.from_torch(
        torch.Tensor([0, 1, 2]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )
    ttnn.manual_seed(seeds=seed_tensor, user_ids=user_id_tensor)
