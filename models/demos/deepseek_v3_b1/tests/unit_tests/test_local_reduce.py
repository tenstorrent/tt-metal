# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests element-wise reduction of N tiles with optional SiLU activation.

output = SiLU(in[0] + in[1] + ... + in[n-1])  if apply_silu=True
output = in[0] + in[1] + ... + in[n-1]        if apply_silu=False

All input tiles come from the same circular buffer. We use the local reduce operation
to perform the reduction.

This is used in MOE-shared expert / Dense MLP when we shard matmul on inner-dim
across multiple cores. We accumulate results and optionally apply activation.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.local_reduce.op import LocalReduceOp


@pytest.mark.parametrize(
    "tile_h, tile_w, num_tiles, apply_silu",
    [
        # Without SiLU
        (32, 32, 2, False),  # Standard tiles, 2 inputs
        (32, 32, 4, False),  # Standard tiles, 4 inputs
        (32, 32, 8, False),  # Standard tiles, 8 inputs
        (16, 16, 4, False),  # Small tiles, 4 inputs
        (1, 32, 6, False),  # Tiny tiles, 6 inputs
        # With SiLU
        (32, 32, 2, True),  # Standard tiles, with SiLU
        (32, 32, 4, True),  # Standard tiles, 4 inputs, with SiLU
        (16, 16, 4, True),  # Small tiles, with SiLU
        (1, 32, 6, True),  # Tiny tiles, with SiLU
    ],
)
def test_local_reduce(device, tile_h, tile_w, num_tiles, apply_silu):
    tile = ttnn.Tile([tile_h, tile_w])

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    silu_str = "with SiLU" if apply_silu else "no SiLU"
    logger.info(f"Testing local reduce: tile=[{tile_h}, {tile_w}], num_tiles={num_tiles}, {silu_str}")

    # Create N input tensors
    torch.manual_seed(42)
    torch_inputs = [torch.randn((tile_h, tile_w), dtype=torch.bfloat16) for _ in range(num_tiles)]

    # Golden reference: sum of all inputs with optional SiLU
    torch_expected = LocalReduceOp.golden(*[t.float() for t in torch_inputs], apply_silu=apply_silu).bfloat16()

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

    logger.info(f"Running local reduce {'+ SiLU ' if apply_silu else ''}operation...")
    ttnn_result = LocalReduceOp.op(ttnn_input, ttnn_output, num_tiles, apply_silu=apply_silu)

    # Convert back to torch and verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (tile_h, tile_w), f"Expected shape ({tile_h}, {tile_w}), got {output_torch.shape}"

    # SiLU uses approximation, so we use a slightly lower PCC threshold
    pcc_threshold = 0.998 if apply_silu else 0.999
    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info(f"Test passed! (tile=[{tile_h}, {tile_w}], num_tiles={num_tiles}, {silu_str})")


@pytest.mark.parametrize("num_faces", [2, 4, 6, 8])
@pytest.mark.parametrize("apply_silu", [False, True])
def test_local_reduce_face_view(device, num_faces, apply_silu):
    """
    Test automatic face-view optimization at the op/kernel level.

    Input: N*8 tiles of [1,32] = N*256 elements = N faces
    The op detects this can be viewed as N [16,16] faces and:
    1. Configures CB with [16,16] tile descriptor
    2. Passes num_tiles=N to kernel
    3. N/2 add_tiles calls instead of N*4

    Memory layout: Elements are grouped into faces of 256 elements each.
    Output: sum of all faces element-wise, with optional SiLU activation.
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

    silu_str = "with SiLU" if apply_silu else "no SiLU"
    logger.info(
        f"Testing face-view: {orig_num_tiles}x[{orig_tile_h},{orig_tile_w}] -> {num_faces}x[16,16] faces, {silu_str}"
    )

    # Create input data
    torch.manual_seed(42)
    torch_inputs = [torch.randn((orig_tile_h, orig_tile_w), dtype=torch.bfloat16) for _ in range(orig_num_tiles)]

    # Stack inputs
    torch_input_stacked = torch.cat(torch_inputs, dim=0)
    total_elements = orig_num_tiles * orig_tile_h * orig_tile_w
    assert torch_input_stacked.numel() == total_elements

    # Golden: element-wise sum of all faces, then optional SiLU
    input_flat = torch_input_stacked.flatten().float()
    faces = [input_flat[i * 256 : (i + 1) * 256] for i in range(num_faces)]
    sum_result = sum(faces)
    if apply_silu:
        torch_expected = torch.nn.functional.silu(sum_result).view(output_tile_h, output_tile_w).bfloat16()
    else:
        torch_expected = sum_result.view(output_tile_h, output_tile_w).bfloat16()

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
    assert LocalReduceOp.can_use_face_view(
        orig_tile_h, orig_tile_w, orig_num_tiles
    ), f"Face-view optimization should be applicable for {orig_num_tiles} [1,32] tiles"

    # Log the optimization benefit
    original_add_calls = orig_num_tiles // 2
    optimized_add_calls = num_faces // 2
    logger.info(
        f"Optimization: {original_add_calls} add_tiles -> {optimized_add_calls} add_tiles "
        f"({original_add_calls - optimized_add_calls} fewer calls)"
    )

    # Run operation with face-view optimization
    logger.info(f"Running local reduce {'+ SiLU ' if apply_silu else ''}with {num_faces} faces...")
    ttnn_result = LocalReduceOp.op(ttnn_input, ttnn_output, orig_num_tiles, apply_silu=apply_silu, use_face_view=True)

    # Convert back to torch and verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (
        output_tile_h,
        output_tile_w,
    ), f"Expected shape ({output_tile_h}, {output_tile_w}), got {output_torch.shape}"

    # SiLU uses approximation, so we use a slightly lower PCC threshold
    pcc_threshold = 0.998 if apply_silu else 0.999
    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info(f"Face-view test passed! ({num_faces} faces, {silu_str})")
