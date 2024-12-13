# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
import math
from tt_lib.utils import (
    _nearest_y,
)
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


def ResnetBlock2D(
    conifg,
    input_tensor=None,
    temb=None,
    in_channels=None,
    input_height=None,
    input_width=None,
    parameters=None,
    device=None,
    eps=1e-5,
    groups=32,
    time_embedding_norm="default",
    non_linearity="silu",
    conv_shortcut=False,
    output_scale_factor=1.0,
    use_torch_conv=False,
):
    hidden_states = input_tensor
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    N = hidden_states.shape[0]
    batch_size = N
    C = in_channels
    H = input_height
    W = input_width
    grid_size = ttnn.CoreGrid(y=4, x=8)

    use_torch_silu = False
    use_torch_gn = False
    if (C == 960 and H == 128) or (C == 640 and H == 128) or (C == 1920 and H == 64):
        use_torch_silu = True
    if H >= 128 or (C == 1920 and H == 64) or (C == 1280 and H == 64) or (C == 960 and H == 64):
        use_torch_gn = True
    if C == 960 and H == 128:
        use_torch_conv = True

    if use_torch_gn:
        hidden_states = ttnn.to_torch(hidden_states)
        torch_weight = ttnn.to_torch(parameters.norm1.weight)
        torch_bias = ttnn.to_torch(parameters.norm1.bias)
        hidden_states = (
            torch.nn.functional.group_norm(hidden_states, groups, weight=torch_weight, bias=torch_bias)
            .permute(0, 2, 3, 1)
            .view(N, 1, W * H, C)
        )
        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
        hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, input_width * input_height, in_channels))
        input_mask_tensor = get_mask_tensor(C, groups, grid_size.y, device)
        gamma_t, beta_t = get_weights(parameters.norm1.weight, parameters.norm1.bias, C, grid_size.y, device)

        # shard config
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = N * H * W // grid_size.x, C // grid_size.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            epsilon=eps,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
        )

    if non_linearity == "silu":
        if use_torch_silu:
            torch_silu = torch.nn.SiLU()
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = torch_silu(hidden_states)
            hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            hidden_states = get_inputs(device, hidden_states, grid_size)
            hidden_states = ttnn.silu(hidden_states)

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, C))
    batch_size = hidden_states.shape[0]

    if use_torch_conv:
        weight = ttnn.to_torch(parameters.conv1.weight).to(torch.float)
        bias = ttnn.to_torch(parameters.conv1.bias).to(torch.float)
        conv = nn.Conv2d(
            in_channels=C, out_channels=parameters.conv1.bias.shape[-1], kernel_size=3, stride=1, padding=1
        )
        conv.weight = nn.Parameter(weight)
        conv.bias = nn.Parameter(bias)
        hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
        hidden_states = conv(hidden_states)
        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        if parameters.conv1.use_split_conv:
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            hidden_states = run_conv_with_split(
                device,
                hidden_states,
                hidden_states.shape[0],
                parameters,
                kernel_size=3,
                stride=1,
                pad=1,
                split_factor=parameters.conv1.split_factor,
                ttnn_weight=parameters.conv1.weight,
                ttnn_bias=parameters.conv1.bias,
            )
        else:
            hidden_states = run_conv(
                device,
                output_channels=parameters.conv1.bias.shape[-1],
                input_channels=C,
                input_height=H,
                input_width=W,
                filter_height=3,
                stride_h=1,
                pad_h=1,
                tt_input_tensor=hidden_states,
                tt_weight_tensor=parameters.conv1.weight,
                tt_bias_tensor=parameters.conv1.bias,
            )

    if temb is not None:
        temb = ttnn.silu(temb)
        temb = ttnn.linear(
            temb,
            parameters.time_emb_proj.weight,
            bias=parameters.time_emb_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
        )

    if temb is not None and time_embedding_norm == "default":
        temb = ttnn.reshape(temb, (temb.shape[0], temb.shape[1], 1, 1))
        hidden_states = ttnn.add(hidden_states, temb)

    N = hidden_states.shape[0]
    C = hidden_states.shape[1]
    H = hidden_states.shape[2]
    W = hidden_states.shape[3]

    if use_torch_gn:
        hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
        torch_weight = ttnn.to_torch(parameters.norm2.weight).to(torch.float)
        torch_bias = ttnn.to_torch(parameters.norm2.bias).to(torch.float)
        hidden_states = (
            torch.nn.functional.group_norm(hidden_states, 32, weight=torch_weight, bias=torch_bias)
            .permute(0, 2, 3, 1)
            .view(N, 1, W * H, C)
        )

        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        input_mask_tensor = get_mask_tensor(C, groups, grid_size.y, device)

        input_mask_tensor = ttnn.create_group_norm_input_mask(C, groups, grid_size.y)
        input_mask_tensor = ttnn.from_torch(
            input_mask_tensor,
            dtype=ttnn.DataType.BFLOAT8_B,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gamma_t, beta_t = get_weights(parameters.norm2.weight, parameters.norm2.bias, C, grid_size.y, device)

        hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (N, 1, W * H, C))

        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = N * H * W // grid_size.x, C // grid_size.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )

        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            epsilon=eps,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
        )

    if non_linearity == "silu":
        if use_torch_silu:
            torch_silu = torch.nn.SiLU()
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = torch_silu(hidden_states)
            hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            hidden_states = get_inputs(device, hidden_states, grid_size)
            hidden_states = ttnn.silu(hidden_states)

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, C))
    batch_size = hidden_states.shape[0]

    if use_torch_conv:
        weight = ttnn.to_torch(parameters.conv2.weight).to(torch.float)
        bias = ttnn.to_torch(parameters.conv2.bias).to(torch.float)
        conv = nn.Conv2d(
            in_channels=C, out_channels=parameters.conv2.bias.shape[-1], kernel_size=3, stride=1, padding=1
        )
        conv.weight = nn.Parameter(weight)
        conv.bias = nn.Parameter(bias)
        hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
        hidden_states = conv(hidden_states)
        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    else:
        if parameters.conv2.use_split_conv:
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            hidden_states = run_conv_with_split(
                device,
                hidden_states,
                hidden_states.shape[0],
                parameters,
                kernel_size=3,
                stride=1,
                pad=1,
                split_factor=parameters.conv2.split_factor,
                ttnn_weight=parameters.conv2.weight,
                ttnn_bias=parameters.conv2.bias,
            )
        else:
            hidden_states = run_conv(
                device,
                output_channels=parameters.conv2.bias.shape[-1],
                input_channels=C,
                input_height=H,
                input_width=W,
                filter_height=3,
                stride_h=1,
                pad_h=1,
                tt_input_tensor=hidden_states,
                tt_weight_tensor=parameters.conv2.weight,
                tt_bias_tensor=parameters.conv2.bias,
            )

    if "conv_shortcut" in parameters:
        if use_torch_conv:
            input_tensor = ttnn.to_torch(input_tensor).to(torch.float)
            weight = ttnn.to_torch(parameters.conv_shortcut.weight).to(torch.float)
            bias = ttnn.to_torch(parameters.conv_shortcut.bias).to(torch.float)
            conv = nn.Conv2d(
                in_channels=C, out_channels=parameters.conv_shortcut.bias.shape[-1], kernel_size=1, stride=1
            )
            conv.weight = nn.Parameter(weight)
            conv.bias = nn.Parameter(bias)
            input_tensor = conv(input_tensor)
            input_tensor = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        else:
            if parameters.conv_shortcut.use_split_conv:
                input_tensor = run_conv_with_split(
                    device,
                    input_tensor,
                    input_tensor.shape[0],
                    parameters,
                    kernel_size=1,
                    stride=1,
                    pad=0,
                    split_factor=parameters.conv_shortcut.split_factor,
                    ttnn_weight=parameters.conv_shortcut.weight,
                    ttnn_bias=parameters.conv_shortcut.bias,
                )
            else:
                input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
                input_tensor = run_conv(
                    device,
                    output_channels=parameters.conv_shortcut.bias.shape[-1],
                    input_channels=C,
                    input_height=H,
                    input_width=W,
                    filter_height=1,
                    stride_h=1,
                    pad_h=0,
                    tt_input_tensor=input_tensor,
                    tt_weight_tensor=parameters.conv_shortcut.weight,
                    tt_bias_tensor=parameters.conv_shortcut.bias,
                )
    output_tensor = ttnn.add(input_tensor, hidden_states)
    output_tensor = ttnn.mul(output_tensor, (1 / output_scale_factor))
    return output_tensor


def get_mask_tensor(C, groups, grid_size, device):
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, groups, grid_size)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return input_mask_tensor


def get_weights(weight, bias, C, grid_size, device):
    gamma = ttnn.create_group_norm_weight_bias_rm(ttnn.to_torch(weight), C, grid_size)
    beta = ttnn.create_group_norm_weight_bias_rm(ttnn.to_torch(bias), C, grid_size)

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
    return gamma_t, beta_t


def get_inputs(device, input_tensor, grid_size):
    ncores = 32
    interleaved_mem_config = ttnn.L1_MEMORY_CONFIG
    input_tensor = ttnn.to_memory_config(input_tensor, interleaved_mem_config)

    input_2d_height = input_tensor.shape.with_tile_padding()[2]
    input_2d_width = input_tensor.shape.with_tile_padding()[3]
    shard_strategy = ttnn.ShardStrategy.HEIGHT

    ## input shard

    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        input_2d_height_padded = _nearest_y(input_2d_height, grid_size[0] * 32)
        shard_height = math.ceil(input_2d_height_padded / grid_size[0])
        shard_width = math.ceil(input_2d_width / grid_size[1])
        shard_orientation = ttnn.ShardOrientation.COL_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        input_2d_height_padded = _nearest_y(input_2d_height, ncores * 32)
        shard_height = math.ceil(input_2d_height_padded / ncores)
        shard_grid = get_shard_grid_from_num_cores(ncores, device)
        shard_width = input_2d_width
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        shard_height = input_2d_height
        input_2d_width_padded = _nearest_y(input_2d_width, ncores * 32)
        shard_width = math.ceil(input_2d_width_padded / ncores)
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        shard_grid = get_shard_grid_from_num_cores(ncores, device)

    shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.BufferType.L1, shard_spec)

    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)
    return input_tensor


def update_params(parameters):
    print("Parms are updating")
    down_block = {
        (0, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": False,
        },
        (0, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": False,
        },
        (1, 0): {
            "split_conv_1": False,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 2,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (1, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 2,
            "split_conv_2": True,
            "conv2_split_factor": 2,
            "conv_shortcut": False,
        },
        (2, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 8,
        },
        (2, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": False,
        },
    }

    mid_block = {
        (0, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": False,
        },
        (1, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": False,
        },
    }

    up_block = {
        (0, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 16,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (0, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 16,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (0, 2): {
            "split_conv_1": True,
            "conv1_split_factor": 16,
            "split_conv_2": True,
            "conv2_split_factor": 16,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 16,
        },
        (1, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 12,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (1, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (1, 2): {
            "split_conv_1": True,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 8,
        },
        (2, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 24,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (2, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 8,
        },
        (2, 2): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 8,
        },
    }

    for k, v in down_block.items():
        index1 = k[0]
        index2 = k[1]
        parameters.down_blocks[index1].resnets[index2].conv1["use_split_conv"] = down_block[k]["split_conv_1"]
        parameters.down_blocks[index1].resnets[index2].conv1["split_factor"] = down_block[k]["conv1_split_factor"]
        parameters.down_blocks[index1].resnets[index2].conv2["use_split_conv"] = down_block[k]["split_conv_2"]
        parameters.down_blocks[index1].resnets[index2].conv2["split_factor"] = down_block[k]["conv2_split_factor"]
        if down_block[k]["conv_shortcut"]:
            parameters.down_blocks[index1].resnets[index2].conv_shortcut["use_split_conv"] = down_block[k][
                "split_conv_3"
            ]
            parameters.down_blocks[index1].resnets[index2].conv_shortcut["split_factor"] = down_block[k][
                "conv3_split_factor"
            ]

    for k, v in mid_block.items():
        index1 = k[0]
        index2 = k[1]
        parameters.mid_block.resnets[index1].conv1["use_split_conv"] = mid_block[k]["split_conv_1"]
        parameters.mid_block.resnets[index1].conv1["split_factor"] = mid_block[k]["conv1_split_factor"]
        parameters.mid_block.resnets[index1].conv2["use_split_conv"] = mid_block[k]["split_conv_2"]
        parameters.mid_block.resnets[index1].conv2["split_factor"] = mid_block[k]["conv2_split_factor"]
        if mid_block[k]["conv_shortcut"]:
            parameters.mid_block.resnets[index1].conv_shortcut["use_split_conv"] = mid_block[k]["split_conv_3"]
            parameters.mid_block.resnets[index1].conv_shortcut["split_factor"] = mid_block[k]["conv3_split_factor"]

    for k, v in up_block.items():
        index1 = k[0]
        index2 = k[1]
        parameters.up_blocks[index1].resnets[index2].conv1["use_split_conv"] = up_block[k]["split_conv_1"]
        parameters.up_blocks[index1].resnets[index2].conv1["split_factor"] = up_block[k]["conv1_split_factor"]
        parameters.up_blocks[index1].resnets[index2].conv2["use_split_conv"] = up_block[k]["split_conv_2"]
        parameters.up_blocks[index1].resnets[index2].conv2["split_factor"] = up_block[k]["conv2_split_factor"]
        if up_block[k]["conv_shortcut"]:
            parameters.up_blocks[index1].resnets[index2].conv_shortcut["use_split_conv"] = up_block[k]["split_conv_3"]
            parameters.up_blocks[index1].resnets[index2].conv_shortcut["split_factor"] = up_block[k][
                "conv3_split_factor"
            ]

    return parameters


def run_conv_with_split(
    device,
    input_tensor,
    batch_size,
    parameters,
    kernel_size,
    stride,
    pad,
    split_factor,
    weights_dtype=ttnn.bfloat8_b,
    ttnn_weight=None,
    ttnn_bias=None,
):
    input_channels = input_tensor.shape[1]
    assert input_channels % split_factor == 0
    split_input_channels = input_channels // split_factor
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_weight = ttnn.to_torch(ttnn_weight)
    ttnn_weight = ttnn.from_torch(ttnn_weight, dtype=ttnn.float32)
    ttnn_bias = ttnn.to_torch(ttnn_bias)
    ttnn_bias = ttnn.from_torch(ttnn_bias, dtype=ttnn.float32)
    ttnn_bias = ttnn.reshape(ttnn_bias, (1, 1, 1, ttnn_bias.shape[0]))

    split_input_tensors = ttnn.split(input_tensor, split_factor, 1)
    split_weight_tensors = ttnn.split(ttnn_weight, split_factor, 1)
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_weight_tensor = split_weight_tensors
    out_channels = tt_weight_tensor[1].shape[0]
    for i in range(split_factor):
        tt_input_tensor = ttnn.permute(split_input_tensors[i], (0, 2, 3, 1))
        tt_input_tensor = ttnn.from_device(tt_input_tensor)
        [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor[i],
            in_channels=split_input_channels,
            out_channels=tt_weight_tensor[i].shape[0],
            device=device,
            bias_tensor=ttnn_bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(pad, pad),
            batch_size=batch_size,
            input_height=tt_input_tensor.shape[1],
            input_width=tt_input_tensor.shape[2],
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache=reader_patterns_cache,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        conv_output_tensor = ttnn.reshape(tt_output_tensor_on_device, (batch_size, out_height, out_width, out_channels))
        conv_output_tensor = ttnn.sharded_to_interleaved(conv_output_tensor)
        conv_output_tensor = ttnn.permute(conv_output_tensor, (0, 3, 1, 2))
        if i == 0:
            output_tensor = conv_output_tensor
        else:
            output_tensor = ttnn.add(output_tensor, conv_output_tensor)

    return output_tensor


def run_conv(
    device,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    stride_h,
    pad_h,
    tt_input_tensor=None,
    tt_weight_tensor=None,
    tt_bias_tensor=None,
    math_fidelity=ttnn.MathFidelity.LoFi,
    activations_dtype=ttnn.bfloat8_b,
    weights_dtype=ttnn.bfloat8_b,
    use_1d_systolic_array=True,
    config_override=None,
    use_shallow_conv_variant=False,
    dilation=1,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    debug=False,
    groups=1,
    shard_layout=None,
    auto_shard=False,
):
    batch_size = tt_input_tensor.shape[0]
    tt_weight_tensor = ttnn.to_torch(tt_weight_tensor)
    tt_weight_tensor = ttnn.from_torch(tt_weight_tensor, dtype=ttnn.float32)
    tt_bias_tensor = ttnn.to_torch(tt_bias_tensor)
    tt_bias_tensor = ttnn.from_torch(tt_bias_tensor, dtype=ttnn.float32)
    tt_bias_tensor = ttnn.reshape(tt_bias_tensor, (1, 1, 1, tt_bias_tensor.shape[0]))
    reader_patterns_cache = {}

    if shard_layout is None and not auto_shard:
        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        input_channels_alignment=32,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=output_layout,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override and not auto_shard:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override and not auto_shard:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
    tt_output_tensor_on_device, [out_height, out_width] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_height),
        stride=(stride_h, stride_h),
        padding=(pad_h, pad_h),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    tt_output_tensor_on_device = ttnn.reshape(
        tt_output_tensor_on_device,
        (tt_output_tensor_on_device.shape[0], out_height, out_width, tt_output_tensor_on_device.shape[-1]),
    )
    tt_output_tensor_on_device = ttnn.to_torch(tt_output_tensor_on_device)
    tt_output_tensor_on_device = ttnn.from_torch(
        tt_output_tensor_on_device, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    tt_output_tensor_on_device = ttnn.permute(tt_output_tensor_on_device, (0, 3, 1, 2))
    return tt_output_tensor_on_device
