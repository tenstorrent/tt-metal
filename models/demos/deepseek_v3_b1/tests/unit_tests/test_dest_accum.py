# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests element-wise addition of N tiles from single CB: output = in[0] + in[1] + ... + in[n-1]

All input tiles come from the same circular buffer. We use the dest accum add operation to perform the reduction.

This is used in MOE-shared expert / Dense MLP when we shard matmul on inner-dim across multiple cores. We'll
use this operation on the gather/mcast core to accumulate the results.

"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.dest_accum.op import DestAccumOp


@pytest.mark.parametrize(
    "tile_h, tile_w, num_tiles",
    [
        (32, 32, 2),  # Standard tiles, 2 inputs
        (32, 32, 4),  # Standard tiles, 4 inputs
        (32, 32, 8),  # Standard tiles, 8 inputs
        (16, 16, 4),  # Small tiles, 4 inputs
        (1, 32, 6),  # Tiny tiles, 6 inputs
    ],
)
def test_dest_accum_add(device, tile_h, tile_w, num_tiles):
    tile = ttnn.Tile([tile_h, tile_w])

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    logger.info(f"Testing dest accum add: tile=[{tile_h}, {tile_w}], num_tiles={num_tiles}")

    # Create N input tensors
    torch.manual_seed(42)
    torch_inputs = [torch.randn((tile_h, tile_w), dtype=torch.bfloat16) for _ in range(num_tiles)]

    # Golden reference: sum of all inputs
    torch_expected = DestAccumOp.golden(*[t.float() for t in torch_inputs]).bfloat16()

    # Stack inputs into single tensor: [N*tile_h, tile_w] (N tiles)
    torch_input_stacked = torch.cat(torch_inputs, dim=0)

    logger.info(f"Input (stacked) shape: {torch_input_stacked.shape}")
    logger.info(f"Output shape: {torch_expected.shape}")

    # Create sharded memory config for input (N tiles)
    input_shard_shape = (num_tiles * tile_h, tile_w)
    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input_stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )

    output_shard_shape = (tile_h, tile_w)
    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    ttnn_output = ttnn.from_torch(
        torch.zeros((tile_h, tile_w), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    logger.info("Running dest accum add operation...")
    ttnn_result = DestAccumOp.op(ttnn_input, ttnn_output, num_tiles)

    # Convert back to torch and verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (tile_h, tile_w), f"Expected shape ({tile_h}, {tile_w}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.999)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info(f"Dest accum add test passed! (tile=[{tile_h}, {tile_w}], num_tiles={num_tiles})")


@pytest.mark.parametrize("num_faces", [2, 4, 6, 8])
def test_dest_accum_face_view(device, num_faces):
    """
    Test automatic face-view optimization at the op/kernel level.

    Input: N*8 tiles of [1,32] = N*256 elements = N faces
    The op detects this can be viewed as N [16,16] faces and:
    1. Configures CB with [16,16] tile descriptor
    2. Passes num_tiles=N to kernel
    3. N/2 add_tiles calls instead of N*4

    Memory layout: Elements are grouped into faces of 256 elements each.
    Output: sum of all faces element-wise.
    """
    # Original view: N*8 tiles of [1,32] (8 tiles per face)
    orig_tile_h, orig_tile_w = 1, 32
    tiles_per_face = 256 // (orig_tile_h * orig_tile_w)  # 8 tiles per face
    orig_num_tiles = num_faces * tiles_per_face
    orig_tile = ttnn.Tile([orig_tile_h, orig_tile_w])

    # Output will be a [16,16] face
    output_tile_h, output_tile_w = 16, 16
    output_tile = ttnn.Tile([output_tile_h, output_tile_w])

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    logger.info(f"Testing face-view: {orig_num_tiles}x[{orig_tile_h},{orig_tile_w}] -> {num_faces}x[16,16] faces")

    # Create input data
    torch.manual_seed(42)
    torch_inputs = [torch.randn((orig_tile_h, orig_tile_w), dtype=torch.bfloat16) for _ in range(orig_num_tiles)]

    # Stack inputs
    torch_input_stacked = torch.cat(torch_inputs, dim=0)
    total_elements = orig_num_tiles * orig_tile_h * orig_tile_w
    assert torch_input_stacked.numel() == total_elements

    # Golden: element-wise sum of all faces
    # Each face is 256 elements
    input_flat = torch_input_stacked.flatten().float()
    faces = [input_flat[i * 256 : (i + 1) * 256] for i in range(num_faces)]
    torch_expected = sum(faces).view(output_tile_h, output_tile_w).bfloat16()

    logger.info(f"Input: {orig_num_tiles} tiles, {total_elements} elements, {num_faces} faces")
    logger.info(f"Output shape: {torch_expected.shape}")

    # Create sharded memory config
    input_shard_shape = (orig_num_tiles * orig_tile_h, orig_tile_w)
    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input_stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=orig_tile,
    )

    # Create output tensor - [16,16] face
    output_shard_shape = (output_tile_h, output_tile_w)
    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    ttnn_output = ttnn.from_torch(
        torch.zeros((output_tile_h, output_tile_w), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    # Verify face-view optimization will be applied
    assert DestAccumOp.can_use_face_view(
        orig_tile_h, orig_tile_w, orig_num_tiles
    ), f"Face-view optimization should be applicable for {orig_num_tiles} [1,32] tiles"

    # Get optimization info and verify expected add_tiles reduction
    opt_info = DestAccumOp.get_optimization_info(orig_tile_h, orig_tile_w, orig_num_tiles, use_face_view=True)
    assert opt_info["use_face_view"] is True
    assert (
        opt_info["kernel_num_tiles"] == num_faces
    ), f"Expected kernel to see {num_faces} tiles, got {opt_info['kernel_num_tiles']}"
    assert (
        opt_info["num_add_calls"] == num_faces // 2
    ), f"Expected {num_faces // 2} add_tiles calls, got {opt_info['num_add_calls']}"

    logger.info(
        f"Optimization: {opt_info['original_add_calls']} add_tiles -> {opt_info['num_add_calls']} add_tiles "
        f"({opt_info['original_add_calls'] - opt_info['num_add_calls']} fewer calls)"
    )

    # Run operation with face-view optimization
    logger.info(f"Running dest accum with {num_faces} faces...")
    ttnn_result = DestAccumOp.op(ttnn_input, ttnn_output, orig_num_tiles, use_face_view=True)

    # Convert back to torch and verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (
        output_tile_h,
        output_tile_w,
    ), f"Expected shape ({output_tile_h}, {output_tile_w}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.999)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info(f"Face-view test passed! ({num_faces} faces)")
