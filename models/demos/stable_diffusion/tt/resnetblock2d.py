# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.demos.stable_diffusion.tt.resnetblock2d_utils import (
    get_inputs,
    get_weights,
    get_mask_tensor,
    run_conv,
    run_conv_with_split,
)


def ResnetBlock2D(
    conifg,
    input_tensor=None,
    temb=None,
    parameters=None,
    device=None,
    eps=1e-5,
    groups=32,
    time_embedding_norm="default",
    non_linearity="silu",
    output_scale_factor=1.0,
    use_torch_conv=False,
):
    hidden_states = input_tensor
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    N = input_tensor.shape[0]
    batch_size = N
    C = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    in_channels = C
    input_height = H
    input_width = W
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

    if parameters.conv2.conv_shortcut:
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
