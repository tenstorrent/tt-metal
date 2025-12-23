# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import math
from loguru import logger


def test_group_norm(device):
    #
    # Sharded Input Tensor Example
    #
    N, C, H, W = 1, 64, 32, 1
    num_groups = 2

    # Prepare random inputs
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    # Generate random inputs and prepare reference output
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )

    # Permute the torch output to match the TTNN format, so they can be compared
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Prepare TTNN input
    # Determine how to shard the input tensor
    sharded_mem_config, grid_size = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
        device=device,
        num_channels=C,
        num_groups=num_groups,
        input_nhw=N * H * W,
        is_height_sharded=True,
        is_row_major=True,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    # Input mask - this is needed for each group to be able to select the correct elements of the input tensor
    # In general, it will have dimensions of [1, num_groups, 32, 32*block_wt]

    # In this example, C=64 and num_groups=2, so each group is 32 channels (i.e. one tile) wide
    # As a result, the input_mask_tensor is a [1, 2, 32, 32] tensor where every value is 1

    # If instead num_groups was 4, each group would be 16 channels (i.e. half a tile) wide
    # As a result, the input_mask_tensor would be a [1, 4, 32, 32] tensor that selects either the first or second half of the tile
    # e.g. The mask at [0][0][:][:] would be a 32x32 tensor where the left half is 1 and the right half is 0
    # While [0][1][:][:] would be a 32x32 tensor where the left half is 0 and the right half is 1
    input_mask_tensor = ttnn.create_group_norm_input_mask(
        num_channel=C,
        num_groups=num_groups,
        num_cores_across_channel=1,  # As explained in the Limitations, supply 1 for height sharded input tensors
        data_type=ttnn.bfloat8_b,
    )
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # Prepare gamma and beta for TTNN. Currently these are just 1D tensors of size [C], which isn't compatible with tile based processing
    # First they will zero padded if needed (does not apply to this example)
    # Then reshaped to be [1, 1, tiles_per_core_total, 32], which in this case will be [1, 1, 2, 32]

    # As with the input mask, we supply a core count of 1 for height sharded input tensors
    gamma = ttnn.create_group_norm_weight_bias_rm(input_tensor=torch_weight, num_channels=C, num_cores_x=1)
    beta = ttnn.create_group_norm_weight_bias_rm(input_tensor=torch_bias, num_channels=C, num_cores_x=1)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Compute the TTNN output and compare with the reference output
    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    logger.info(f"Group Norm result: {output_tensor}")

    #
    # Base example with tilized input
    #
    tile_size = 32
    N, C, H, W = 1, 480, 1, 64
    grid_size = ttnn.CoreGrid(y=1, x=1)
    num_out_blocks = 1

    num_groups = 8  # This must be a multiple of grid_size.y (1 in this example)

    input_tensor_row_major = ttnn.rand(
        [N, 1, H * W, C], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    # input mask
    width_per_group = C // num_groups  # C must be a multiple of num_groups
    max_tiles_group_can_span = 1 + math.ceil((width_per_group - 1) / tile_size)
    input_mask_tensor = ttnn.zeros(
        [1, num_groups, tile_size, max_tiles_group_can_span * tile_size],
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # gamma/beta
    values_per_chunk = (
        C // grid_size.y
    )  # 480 / 1 = 480. Note that 480 is a multiple of 32, so no padding up to the next tile is needed.
    values_per_chunk_per_tile = values_per_chunk // tile_size  # 480 / 32 = 15

    gamma_beta = ttnn.rand(
        [1, 1, values_per_chunk_per_tile, 32], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor_tilized,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_beta,
        bias=gamma_beta,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=num_out_blocks,
    )
    logger.info(f"Group Norm result: {output_tensor}")


def test_layer_norm(device):
    # Create input tensor
    input_tensor = ttnn.rand([32, 64], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply layer normalization
    output_tensor = ttnn.layer_norm(input_tensor)
    logger.info(f"Layer Norm result: {output_tensor}")


def test_layernorm_distributed(device):
    # Create input tensor
    input_tensor = ttnn.rand([1, 1, 32, 32], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply pre-all-gather layer normalization
    stats = ttnn.layer_norm_pre_all_gather(input_tensor)
    logger.info(f"Layer Norm Pre All Gather result: {stats}")

    # On a distributed setup, all gather would go here to collect the stats from all devices
    # See documentation for ttnn.all_gather for example usage of all_gather

    # Now apply the post-all-gather layer normalization
    output = ttnn.layer_norm_post_all_gather(input_tensor, stats)
    logger.info(f"Layer Norm Post All Gather result: {output}")

    # For reference, this two-step process is equivalent to the following
    # output = ttnn.layer_norm(input_tensor)


def test_rms_norm(device):
    # Setup input tensor and weight
    h, w = 32, 64
    batch_size = 1
    input_tensor = ttnn.rand([batch_size, h, w], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.rand([w], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply RMS normalization
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight)
    logger.info(f"RMS Norm result: {output_tensor}")


def test_rms_norm_distributed(device):
    # Create input tensor
    input_tensor = ttnn.rand([1, 1, 32, 32], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.rand([32], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Apply pre-all-gather RMS normalization
    stats = ttnn.rms_norm_pre_all_gather(input_tensor)
    logger.info(f"RMS Norm Pre All Gather result: {stats}")

    # On a distributed setup, an all gather would go here to collect the stats from all the devices
    # See documentation for ttnn.all_gather for example usage of all_gather

    # Now apply the post-all-gather RMS normalization
    output = ttnn.rms_norm_post_all_gather(input_tensor, stats, weight=weight)
    logger.info(f"RMS Norm Post All Gather result: {output}")

    # For reference, this two-step process is equivalent to the following
    # output = ttnn.rms_norm(input_tensor, weight=weight)


def test_batch_norm(device):
    # Setup input tensor and parameters
    N, C, H, W = 2, 3, 4, 5

    input_tensor = ttnn.rand([N, C, H, W], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    running_mean = ttnn.rand([1, C, 1, 1], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    running_var = ttnn.rand([1, C, 1, 1], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.rand([1, C, 1, 1], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch.rand([1, C, 1, 1], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    # Apply batch normalization
    output = ttnn.batch_norm(
        input_tensor,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        eps=1e-05,
        momentum=0.1,
        training=True,
    )
    logger.info(f"Batch Norm result: {output}")


def test_softmax(device):
    # Create input tensor
    tensor = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply softmax on dim=-1
    output_tensor = ttnn.softmax(tensor, dim=-1)
    logger.info(f"Softmax result: {output_tensor}")


def test_softmax_default_program_config(device):
    # Create input tensor
    tensor = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Explicitly specify a default config
    ttnn.softmax_in_place(tensor, dim=-1, program_config=ttnn.SoftmaxDefaultProgramConfig())


def test_scale_mask_softmax(device):
    # Setup input tensor and mask
    compute_grid_size = device.compute_with_storage_grid_size()
    fuse_head = 2
    batch = compute_grid_size.x
    num_cores_r = compute_grid_size.y

    input_shape = (batch, num_cores_r, fuse_head * 384, 768)
    attention_mask_t = ttnn.rand((batch, 1, 1, 768), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply scale mask softmax
    tt_output = ttnn.scale_mask_softmax(
        input_tensor=input_tensor,
        scale=1.0,
        mask=attention_mask_t,
    )
    logger.info(f"Scale Mask Softmax result: {tt_output}")


def test_softmax_in_place(device):
    # Create input tensor
    shape = [1, 1, 32, 32]
    input_tensor = ttnn.rand(shape, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply in-place softmax
    logger.info(f"Input tensor before softmax in place: {input_tensor}")
    ttnn.softmax_in_place(input_tensor)
    logger.info(f"Input tensor after softmax in place: {input_tensor}")


def test_scale_mask_softmax_in_place(device):
    # Setup input tensor and mask
    input_shape = (1, 1, 32, 32)

    attention_mask_t = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply in-place scale mask softmax
    tt_output = ttnn.scale_mask_softmax_in_place(
        input_tensor=input_tensor,
        scale=1.0,
        mask=attention_mask_t,
    )
    logger.info(f"Scale Mask Softmax In Place result: {tt_output}")

    compute_grid_size = device.compute_with_storage_grid_size()
    fuse_head = 2
    batch = compute_grid_size.x
    num_cores_r = compute_grid_size.y

    input_shape = (batch, num_cores_r, fuse_head * 384, 768)

    attention_mask_t = ttnn.rand((batch, 1, 384, 768), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Shard the input tensor
    grid_coord = ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = [fuse_head * 384, 768]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    input_sharded = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    # Create sharded program config
    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=compute_grid_size,
        subblock_w=8,
        block_h=12 * fuse_head,
        block_w=24,
    )

    tt_output = ttnn.scale_mask_softmax_in_place(
        input_tensor=input_sharded,
        scale=1.0,
        mask=attention_mask_t,
        program_config=program_config,
    )
    logger.info(f"Scale Mask Softmax In Place result: {tt_output}")


def test_scale_causal_mask_hw_dims_softmax_in_place(device):
    # Setup input tensor and mask
    compute_grid_size = device.compute_with_storage_grid_size()
    batch = compute_grid_size.x
    num_cores_r = compute_grid_size.y

    input_shape = (batch, num_cores_r, 384, 768)
    attention_mask_t = ttnn.rand((1, 1, 384, 768), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    input_tiled = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # We must shard the input tensor in ROW_MAJOR orientation
    grid_coord = ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = [384, 768]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    input_sharded = ttnn.to_memory_config(input_tiled, sharded_mem_config)

    # We must also use the sharded softmax program config
    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=compute_grid_size,
        subblock_w=8,
        block_h=12,
        block_w=24,
    )

    tt_output_sharded = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
        input_tensor=input_sharded,
        scale=1.0,
        mask=attention_mask_t,
        program_config=program_config,
    )
    logger.info(f"Scale Causal Mask HW Dims Softmax In Place result: {tt_output_sharded}")
