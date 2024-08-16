# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tt_lib.fallback_ops import fallback_ops
import math

import torch
from typing import Optional, Dict

conv_cache = {}


def round_up_to_tile_dim(n):
    return ((n + 31) // 32) * 32


def is_tile_dim_alligned(dim):
    return dim % 32 == 0


def pre_process_input(device, tensor):
    batch_size = tensor.shape[0]
    input_channels = tensor.shape[1]
    input_height = tensor.shape[2]
    input_width = tensor.shape[3]
    tensor = fallback_ops.permute(tensor, (0, 2, 3, 1), output_layout=ttnn.ROW_MAJOR_LAYOUT, output_on_device=False)
    tensor = ttnn.to_device(tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    return tensor
    import math

    assert input_channels == tensor.get_legacy_shape()[3]
    padded_input_channels = math.ceil(input_channels / 32) * 32
    if padded_input_channels != input_channels:
        tensor = fallback_ops.pad(
            tensor,
            (0, padded_input_channels - input_channels, 0, 0, 0, 0),
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            output_on_device=False,
        )
    # Reshape 4d to 2d
    tensor = fallback_ops.reshape(
        tensor,
        1,
        1,
        batch_size * input_height * input_width,
        padded_input_channels,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_on_device=False,
    )
    tensor = ttnn.Tensor(tensor)
    tensor = ttnn.to_device(tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    return tensor


def pad_encoder_hidden_states(device, tensor, required_sequence_length):
    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    assert tensor.shape[0] == 1
    batch_size = tensor.shape[1]
    sequence_length = tensor.shape[2]
    hidden_dim = tensor.shape[3]
    if sequence_length < required_sequence_length:
        assert (required_sequence_length % batch_size) == 0
        sequence_length = required_sequence_length
        tensor = ttnn.Tensor(
            fallback_ops.pad(
                tensor,
                (0, 0, 0, sequence_length - tensor.shape[2]),
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                output_on_device=False,
            )
        )
        # TODO: change above code to below
        # tensor = ttnn.pad(tensor, (0, 0, 0, sequence_length - tensor.shape[2]), 0)
    # tensor = ttnn.Tensor(
    #     fallback_ops.reshape(
    #         tensor.value,
    #         1,
    #         1,
    #         batch_size * sequence_length,
    #         hidden_dim,
    #         output_layout=ttnn.ROW_MAJOR_LAYOUT,
    #         output_on_device=False,
    #     )
    # )
    # breakpoint()
    tensor = ttnn.to_device(tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return tensor


def post_process_output(device, tensor, batch_size, output_height, output_width, output_channels):
    tensor = ttnn.to_layout(
        tensor,
        ttnn.ROW_MAJOR_LAYOUT,  # use_multicore=ttnn.get_memory_config(tensor).shard_spec is not None
    )
    tensor = ttnn.from_device(tensor)
    assert output_channels == tensor.shape[3]
    tensor = fallback_ops.reshape(
        tensor,
        batch_size,
        output_height,
        output_width,
        output_channels,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_on_device=False,
    )
    tensor = fallback_ops.permute(tensor, (0, 3, 1, 2), output_layout=ttnn.ROW_MAJOR_LAYOUT, output_on_device=False)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    tensor = ttnn.to_device(tensor, device)
    return tensor


def run_ttnn_conv_with_pre_and_post_tensor_formatting(
    device, ttnn_conv_op, tensor: ttnn.Tensor, batch_size, output_height, output_width, output_channels
) -> ttnn.Tensor:
    tensor = pre_process_input(device, tensor)
    # print("Running conv op")
    tensor = ttnn_conv_op(tensor)
    tensor = post_process_output(device, tensor, batch_size, output_height, output_width, output_channels)
    return tensor


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def weight_to_bfp8(weight):
    device = weight.device()
    memory_config = ttnn.get_memory_config(weight)
    weight = ttnn_to_torch(weight)
    weight = ttnn.from_torch(weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.to_device(weight, device, memory_config=memory_config)
    return weight


def pad_group_norm_weight(weight, groups, channels):
    device = weight.device()
    memory_config = ttnn.get_memory_config(weight)
    weight = ttnn_to_torch(weight)
    elems_per_group = channels // groups
    padding_needed = round_up_to_tile_dim(elems_per_group) - elems_per_group
    weight = weight.view(-1, elems_per_group)
    weight = torch.nn.functional.pad(weight, (0, padding_needed))
    weight = weight.flatten()
    weight = weight[: channels + padding_needed * (channels // elems_per_group)]
    weight = weight.reshape(1, 1, -1, 32)
    weight = ttnn.from_torch(weight, ttnn.bfloat16)
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_device(weight, device, memory_config=memory_config)
    return weight


def permute_conv_parameters(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


def dealloc_input(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    for a in args:
        if type(a) is list:
            for e in a:
                if type(a) is ttnn.Tensor:
                    if a.is_allocated():
                        ttnn.deallocate(a)
        if type(a) is ttnn.Tensor:
            if a.is_allocated():
                ttnn.deallocate(a)
    return out


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:  # h is a divisor of out_block_h
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:  # w is a divisor and product condition met
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


def determine_largest_subblock_size(block_height, block_width, fp32_accum=False):
    subblocks = [
        (2, 4),
        (4, 2),
        (1, 8),
        (8, 1),
        (1, 7),
        (7, 1),
        (2, 3),
        (3, 2),
        (1, 6),
        (6, 1),
        (1, 5),
        (5, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 3),
        (3, 1),
        (1, 2),
        (2, 1),
        (1, 1),
    ]
    for subblock_height, subblock_width in subblocks:
        if fp32_accum and subblock_height * subblock_width > 4:
            continue
        if block_height % subblock_height == 0 and block_width % subblock_width == 0:
            if subblock_width != block_width and subblock_height != 1:
                continue
            break
    return subblock_height, subblock_width


def determine_blocking(M, K, N, grid_size, transpose_mcast=False):
    logical_grid_size = grid_size if transpose_mcast == False else (grid_size[1], grid_size[0])

    in0_block_h = M // logical_grid_size[1] // 32
    in0_block_w = K // logical_grid_size[0] // 32
    out_block_h = math.ceil(M / logical_grid_size[1] / 32)
    out_block_w = math.ceil(N / logical_grid_size[0] / 32)
    out_subblock_h, out_subblock_w = determine_largest_subblock_size(out_block_h, out_block_w)
    # TODO: https://github.com/tenstorrent/tt-metal/issues/7560
    # There's a bug that causes an ND hang, until it's solved reduce subblock sizes to 1, if we're not
    import os

    if os.environ.get("SLOW_MATMULS", "0") == "1":
        out_subblock_h = 1
        out_subblock_w = 1
    return in0_block_h, in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w


def reshard_to(tensor, grid_size, layout, col_major=False, shape=None):
    if shape is not None:
        shape = list(shape)
        volume = math.prod(shape)
    else:
        shape = tensor.shape
        volume = tensor.volume()
    if layout == ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED:
        logical_grid_size = list(grid_size)
        if col_major:
            logical_grid_size[0], logical_grid_size[1] = grid_size[1], grid_size[0]
        shard_spec = [
            volume // shape[-1] // logical_grid_size[1],
            shape[-1] // logical_grid_size[0],
        ]
    elif layout == ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores = grid_size[0] * grid_size[1]
        shard_spec = [volume // shape[-1] // num_cores, shape[-1]]
    output_shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {
            ttnn.experimental.tensor.CoreRange(
                ttnn.experimental.tensor.CoreCoord(0, 0),
                ttnn.experimental.tensor.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
            )
        }
    )
    output_shard_spec = ttnn.experimental.tensor.ShardSpec(
        output_shard_grid,
        shard_spec,
        (
            ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
            if col_major
            else ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR
        ),
        False,
    )
    output_mem_config = ttnn.experimental.tensor.MemoryConfig(
        layout,
        ttnn.experimental.tensor.BufferType.L1,
        output_shard_spec,
    )
    if tensor.is_sharded():
        tensor = ttnn.reshard(
            tensor,
            output_mem_config,
        )
    else:
        tensor = ttnn.interleaved_to_sharded(
            tensor,
            grid_size,
            shard_spec,
            layout,
            ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
        )
    return tensor
