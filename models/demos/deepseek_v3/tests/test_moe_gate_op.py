# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.deepseek_moe_gate.op import DeepseekMoeGateOp


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("enable_sigmoid", [True])
def test_deepseek_moe_gate_op(device, batch_size, enable_sigmoid):
    """Test TTNN Deepseek Moe Gate operation on a 32x32 tile"""

    # Tensor dimensions - full 32x32 tile
    input_shape = (batch_size, 8, 32)
    reshaped_input_shape = (batch_size, 16, 16)
    input_shard_shape = (32, 32)
    input_tile = ttnn.Tile(input_shard_shape)
    output_shape = (batch_size, 1, 16)
    output_shard_shape = (32, 32)
    output_tile = ttnn.Tile(output_shard_shape)

    logger.info(f"Testing Deepseek Moe Gate with input shape {input_shape}")
    logger.info(f"Input tile size: {input_tile.tile_shape}")
    logger.info(f"Output tile size: {output_tile.tile_shape}")

    # Create input PyTorch tensor with random values
    # this is where tensors are loaded
    torch_input_loaded = torch.load("/work/logits.pt")
    torch_input = torch.reshape(torch_input_loaded, input_shape)
    torch_bias_loaded = torch.load("/work/scores_correction_bias.pt")
    torch_bias = torch.reshape(torch_bias_loaded, input_shape)
    eps = 1e-20
    scaling_factor = 2.5

    # Compute reference output using PyTorch
    top8_scores, top8_indices = DeepseekMoeGateOp.golden(torch_input, torch_bias, eps, scaling_factor, enable_sigmoid)

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(
        batch_size,
        ttnn.CoreCoord(grid.x, grid.y),
        row_wise=True,
    )
    # Shard spec: single core
    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create input values tensor sharded on single core
    reshaped_input = torch.reshape(torch_input, reshaped_input_shape)
    ttnn_input = ttnn.from_torch(
        reshaped_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    reshaped_bias = torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1)
    ttnn_bias = ttnn.from_torch(
        reshaped_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    torch_input_indices = torch.arange(reshaped_input_shape[1] * reshaped_input_shape[2], dtype=torch.int32)
    torch_input_indices = torch_input_indices.unsqueeze(0).expand(reshaped_input_shape[0], -1)
    torch_input_indices = torch_input_indices.reshape(reshaped_input_shape)
    torch_input_indices = torch.transpose(torch_input_indices, -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        torch_input_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Create output tensor sharded on same core
    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    torch_output_indices = torch.zeros(output_shape, dtype=torch.uint16)
    ttnn_output_indices = ttnn.from_torch(
        torch_output_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    logger.info(
        f"Created tensors sharded on single core with input shard shape {input_shard_shape} and output shard shape {output_shard_shape}"
    )

    # Run Deepseek Moe Gate operation
    logger.info("Running Deepseek Moe Gate operation...")

    ttnn_result, ttnn_result_indices = DeepseekMoeGateOp.op(
        ttnn_input,
        ttnn_bias,
        ttnn_output,
        ttnn_input_indices,
        ttnn_output_indices,
        eps,
        scaling_factor,
        enable_sigmoid,
    )

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)
    output_indices_torch = ttnn.to_torch(ttnn_result_indices)

    output_torch = output_torch[:, 0, :8]
    output_indices_torch = output_indices_torch[:, 0, :8]

    sorted_output_indices_torch, i = torch.sort(output_indices_torch, dim=-1)
    sorted_output_indices_torch = sorted_output_indices_torch.squeeze(1)
    sorted_output_torch = torch.gather(output_torch, dim=-1, index=i)

    top8_indices, i = torch.sort(top8_indices, dim=-1)
    top8_scores = torch.gather(top8_scores, dim=-1, index=i)
    breakpoint()

    assert torch.equal(sorted_output_indices_torch.to(top8_indices.dtype), top8_indices), "Output indices do not match"
    assert torch.allclose(sorted_output_torch, top8_scores, atol=1e-2, rtol=1e-4), "Output scores do not match"


"""
Original op
(Pdb) sorted_output_indices_torch
tensor([[  9,  84,  88,  92, 192, 196, 197, 210]], dtype=torch.uint16)
(Pdb) top8_indices
tensor([[  9,  84,  88,  92, 196, 210, 230, 237]])
(Pdb) sorted_output_torch
tensor([[0.0000e+00, 1.8000e-38, 9.0206e-16, 2.7551e-40, 9.0206e-16, 9.9182e-39,
         0.0000e+00, 7.7142e-39]], dtype=torch.bfloat16)
(Pdb) top8_scores
tensor([[0.3526, 0.2854, 0.3079, 0.3112, 0.2906, 0.3816, 0.2687, 0.3019]])
(Pdb) q

This is from the updated op
(Pdb) sorted_output_indices_torch
tensor([[  9,  84,  88,  92, 192, 196, 197, 210]], dtype=torch.uint16)
(Pdb) sorted_output_torch
tensor([[0.3574, 0.2891, 0.3105, 0.3145, 0.2891, 0.2949, 0.2500, 0.3867]],
       dtype=torch.bfloat16)
(Pdb) top8_indices
tensor([[  9,  84,  88,  92, 196, 210, 230, 237]])
(Pdb) top8_scores
tensor([[0.3526, 0.2854, 0.3079, 0.3112, 0.2906, 0.3816, 0.2687, 0.3019]])

Output from test_moe_gate.py
(Pdb) ref_sorted_indices[0]
tensor([210,   9,  92,  88, 237, 196,  84, 230], dtype=torch.int32)
(Pdb) ref_sorted_weights[0]
tensor([0.3809, 0.3535, 0.3105, 0.3086, 0.3027, 0.2910, 0.2852, 0.2695],
       dtype=torch.bfloat16)
(Pdb) tt_sorted_indices[0]
tensor([210,   9,  92,  88, 196,  84, 192, 197], dtype=torch.int32)
(Pdb) tt_sorted_weights[0]
tensor([0.3867, 0.3574, 0.3145, 0.3105, 0.2949, 0.2891, 0.2891, 0.2500],
       dtype=torch.bfloat16)

First, we can make sure that the torch version outputs from op and test_moe_gate are close.
Op:
tensor([[  9,  84,  88,  92, 196, 210, 230, 237]])
tensor([[0.3526, 0.2854, 0.3079, 0.3112, 0.2906, 0.3816, 0.2687, 0.3019]])
tensor([210,   9,  92,  88, 237, 196,  84, 230])
tensor([0.3816, 0.3526, 0.3112, 0.3079, 0.3019, 0.2906, 0.2854, 0.2687])
Test_moe_gate:
tensor([210,   9,  92,  88, 237, 196,  84, 230])
tensor([0.3809, 0.3535, 0.3105, 0.3086, 0.3027, 0.2910, 0.2852, 0.2695])

Second, the outputs from test_moe_gate and op are the same.
Op:
tensor([[  9,  84,  88,  92, 192, 196, 197, 210]])
tensor([[0.3574, 0.2891, 0.3105, 0.3145, 0.2891, 0.2949, 0.2500, 0.3867]])
Test_moe_gate:
tensor([210,   9,  92,  88, 196,  84, 192, 197])
tensor([0.3867, 0.3574, 0.3145, 0.3105, 0.2949, 0.2891, 0.2891, 0.2500])
tensor([[  9,  84,  88,  92, 192, 196, 197, 210]])
tensor([[0.3574, 0.2891, 0.3105, 0.3145, 0.2891, 0.2949, 0.2500, 0.3867]])

Third, the problem happens on one group:
In ref:
tensor([[  9,  84,  88,  92, 196, 210, 230, 237]])
tensor([[0.3526, 0.2854, 0.3079, 0.3112, 0.2906, 0.3816, 0.2687, 0.3019]])
In op:
tensor([[  9,  84,  88,  92, 192, 196, 197, 210]])
tensor([[0.3574, 0.2891, 0.3105, 0.3145, 0.2891, 0.2949, 0.2500, 0.3867]])
"""
