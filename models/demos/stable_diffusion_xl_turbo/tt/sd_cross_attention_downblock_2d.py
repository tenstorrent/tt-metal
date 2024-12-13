# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn

from tt_lib.utils import (
    _nearest_y,
)
import math
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from models.demos.stable_diffusion_xl_turbo.tt.resnetblock2d_utils import (
    get_inputs,
    get_weights,
    get_mask_tensor,
    run_conv,
    run_conv_with_split_resnet,
)


def run_conv_with_split(
    device,
    input_tensor,
    batch_size,
    parameters,
    conv_params,
    kernel_size,
    weights_dtype=ttnn.bfloat8_b,
    split_factor=2,
):
    input_channels = input_tensor.shape[1]
    assert input_channels % split_factor == 0
    split_input_channels = input_channels // split_factor
    ttnn_weight = parameters.conv.weight
    ttnn_bias = parameters.conv.bias
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    split_input_tensors = ttnn.split(input_tensor, 2, 1)
    split_weight_tensors = ttnn.split(ttnn_weight, 2, 1)
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    torch_output_tensor = None
    tt_weight_tensor = split_weight_tensors
    out_channels = tt_weight_tensor[1].shape[0]
    for i in range(split_factor):
        tt_input_tensor = ttnn.permute(split_input_tensors[i], (0, 2, 3, 1))
        tt_input_tensor = ttnn.from_device(tt_input_tensor)
        tt_weight_tensor[i] = ttnn.from_device(tt_weight_tensor[i])
        ttnn_bias = ttnn.from_device(ttnn_bias)
        ttnn_bias = ttnn.reshape(ttnn_bias, (1, 1, 1, -1))
        tt_weight_tensor[i] = ttnn.to_dtype(tt_weight_tensor[i], ttnn.float32)
        [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor[i],
            in_channels=split_input_channels,
            out_channels=tt_weight_tensor[i].shape[0],
            device=device,
            bias_tensor=ttnn_bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(conv_params[0], conv_params[1]),
            padding=(conv_params[2], conv_params[3]),
            batch_size=batch_size,
            input_height=tt_input_tensor.shape[1],
            input_width=tt_input_tensor.shape[2],
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache=reader_patterns_cache,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        torch_conv_output_tensor = ttnn.reshape(
            tt_output_tensor_on_device, (batch_size, out_height, out_width, out_channels)
        )
        torch_conv_output_tensor = ttnn.sharded_to_interleaved(torch_conv_output_tensor)
        torch_conv_output_tensor = ttnn.permute(torch_conv_output_tensor, (0, 3, 1, 2))
        if i == 0:
            torch_output_tensor = torch_conv_output_tensor
        else:
            torch_output_tensor = ttnn.add(torch_output_tensor, torch_conv_output_tensor)

    return torch_output_tensor


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


def batch_to_head_dim(tensor, heads=8):
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.reshape(tensor, (batch_size // heads, heads, seq_len, dim))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (1, batch_size // heads, seq_len, dim * heads))
    return tensor


def head_to_batch_dim(tensor, heads=8):
    batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.reshape(tensor, (batch_size, seq_len, heads, dim // heads))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (1, batch_size * heads, seq_len, dim // heads))
    return tensor


def get_attention_scores(query, key, attention_mask=None, scale=None, device=None):
    t_key = ttnn.permute(key, (0, 1, 3, 2))
    temp = ttnn.matmul(query, t_key)
    attention_scores = ttnn.mul(temp, scale)
    if attention_mask is not None:
        attention_scores = ttnn.add(attention_scores, attention_mask)
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    return attention_probs


def sd_geglu(
    hidden_states,
    parameters,
    device=None,
):
    x = ttnn.linear(
        hidden_states,
        parameters.proj.weight,
        bias=parameters.proj.bias,
    )
    x = ttnn.unsqueeze(x, 0)
    x = ttnn.geglu(x)
    x = ttnn.squeeze(x, 0)
    return x


def sd_feed_forward(
    hidden_states,
    parameters,
    device,
):
    hidden_states = sd_geglu(hidden_states, parameters.net[0], device)
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.net[2].weight,
        bias=parameters.net[2].bias,
        dtype=ttnn.bfloat16,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return hidden_states


def sd_attention(
    hidden_states,
    encoder_hidden_states,
    query_dim: int = None,
    cross_attention_dim=None,
    heads: int = 8,
    attention_mask=None,
    cross_attention_kwargs={},
    *,
    parameters,
    device,
):
    batch_size, sequence_length, _ = hidden_states.shape

    query = ttnn.linear(
        hidden_states,
        parameters.to_q.weight,
        dtype=ttnn.bfloat16,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    query = head_to_batch_dim(query, heads=heads)

    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

    key = ttnn.linear(
        encoder_hidden_states,
        parameters.to_k.weight,
        dtype=ttnn.bfloat16,
    )

    value = ttnn.linear(
        encoder_hidden_states,
        parameters.to_v.weight,
        dtype=ttnn.bfloat16,
    )

    key = head_to_batch_dim(key, heads=heads)
    value = head_to_batch_dim(value, heads=heads)

    scale = query.shape[-1] ** -0.5

    attention_probs = get_attention_scores(query, key, attention_mask, scale=scale, device=device)

    hidden_states = ttnn.matmul(attention_probs, value)
    hidden_states = batch_to_head_dim(hidden_states, heads=heads)

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.to_out[0].weight,
        bias=parameters.to_out[0].bias,
        dtype=ttnn.bfloat16,
    )
    hidden_states = ttnn.squeeze(hidden_states, 0)

    return hidden_states


def sd_basic_transformer_block(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    class_labels=None,
    config=None,
    num_embeds_ada_norm=False,
    cross_attention_dim: int = None,
    only_cross_attention: bool = False,
    attention_head_dim=None,
    *,
    parameters,
    device,
):
    norm_hidden_states = ttnn.layer_norm(
        hidden_states,
        epsilon=1e-05,
        weight=parameters.norm1.weight,
        bias=parameters.norm1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
    cross_attention_dim = config.cross_attention_dim if cross_attention_dim is None else cross_attention_dim

    attn_output = sd_attention(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if only_cross_attention else None,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        cross_attention_dim=cross_attention_dim,
        heads=attention_head_dim,
        parameters=parameters.attn1,
        device=device,
    )

    hidden_states = ttnn.add(attn_output, hidden_states)

    if cross_attention_dim is not None:
        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=parameters.norm2.weight,
            bias=parameters.norm2.bias,
        )

        attn_output = sd_attention(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            cross_attention_dim=cross_attention_dim,
            heads=attention_head_dim,
            parameters=parameters.attn2,
            device=device,
        )

        hidden_states = ttnn.add(attn_output, hidden_states)

        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=parameters.norm3.weight,
            bias=parameters.norm3.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ff_output = sd_feed_forward(hidden_states=norm_hidden_states, parameters=parameters.ff, device=device)

        hidden_states = ttnn.add(ff_output, hidden_states)

        return hidden_states


def sd_transformer_2d(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    class_labels=None,
    config=None,
    num_embeds_ada_norm=False,
    cross_attention_dim: int = None,
    norm_num_groups=32,
    only_cross_attention: bool = False,
    attention_head_dim=None,
    return_dict=None,
    num_layers=1,
    eps=1e-5,
    transformer_layers_per_block=10,
    *,
    parameters,
    device,
):
    inner_dim = hidden_states.shape[1]

    residual = hidden_states

    N, C, H, W = hidden_states.shape

    grid_size = ttnn.CoreGrid(y=4, x=8)
    input_mask_tensor = get_mask_tensor(C, norm_num_groups, grid_size.y, device)

    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.reshape(hidden_states, (N, 1, W * H, C))

    grid_size = ttnn.CoreGrid(y=4, x=8)
    input_mask_tensor = get_mask_tensor(C, norm_num_groups, grid_size.y, device)

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

    gamma_t, beta_t = get_weights(parameters.norm.weight, parameters.norm.bias, C, grid_size.y, device)

    hidden_states = ttnn.group_norm(
        input_tensor=hidden_states,
        num_groups=norm_num_groups,
        input_mask=input_mask_tensor,
        epsilon=eps,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )
    hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.reshape(hidden_states, (N, H * W, inner_dim))

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.proj_in.weight,
        bias=parameters.proj_in.bias,
    )

    for d in range(transformer_layers_per_block):
        hidden_states = sd_basic_transformer_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
            attention_head_dim=attention_head_dim,
            attention_mask=attention_mask,
            config=config,
            parameters=parameters.transformer_blocks[d],
            device=device,
            cross_attention_dim=cross_attention_dim,
            only_cross_attention=only_cross_attention,
        )

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.proj_out.weight,
        bias=parameters.proj_out.bias,
    )

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, inner_dim))
    hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))
    residual = ttnn.to_layout(residual, ttnn.TILE_LAYOUT)

    output = ttnn.add(
        hidden_states,
        residual,
    )

    return output


def sd_downsample_2(input_tensor, parameters, device):
    tt_output_tensor_on_device = run_conv_with_split(
        device,
        input_tensor,
        batch_size=1,
        parameters=parameters,
        conv_params=[2, 2, 1, 1],
        kernel_size=3,
        weights_dtype=ttnn.bfloat8_b,
        split_factor=2,
    )
    return tt_output_tensor_on_device


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
            hidden_states = run_conv_with_split_resnet(
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
            hidden_states = run_conv_with_split_resnet(
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
                input_tensor = run_conv_with_split_resnet(
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


def downsample_1(input_tensor, parameters, device):
    tt_output_tensor_on_device, [out_height, out_width] = conv(
        device,
        input_tensor,
        batch_size=1,
        parameters=parameters,
        conv_params=[2, 2, 1, 1],
        kernel_size=3,
        weights_dtype=ttnn.bfloat8_b,
    )

    tt_output_tensor_on_device = ttnn.reshape(
        tt_output_tensor_on_device, (1, out_height, out_width, tt_output_tensor_on_device.shape[-1])
    )

    return tt_output_tensor_on_device


def sd_cross_attention_down_blocks2d(
    hidden_states,
    temb=None,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    encoder_attention_mask=None,
    additional_residuals=None,
    config=None,
    conv_shortcut=True,
    use_torch_conv=False,
    class_labels=None,
    add_downsample=False,
    return_dict=None,
    attention_head_dim=None,
    num_layers=None,
    norm_num_groups=32,
    transformer_layers_per_block=10,
    device=None,
    parameters=None,
):
    output_states = ()
    for index, (resnet, attn) in enumerate(zip(parameters.resnets, parameters.attentions)):
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ResnetBlock2D(
            config,
            input_tensor=hidden_states,
            temb=temb,
            parameters=resnet,
            device=device,
            use_torch_conv=use_torch_conv,
        )
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = sd_transformer_2d(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            parameters=attn,
            device=device,
            timestep=timestep,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            norm_num_groups=norm_num_groups,
            attention_mask=attention_mask,
            config=config,
            eps=1e-06,
        )

    if add_downsample:
        hidden_states = sd_downsample_2(hidden_states, parameters.downsamplers[0], device)

    return hidden_states
