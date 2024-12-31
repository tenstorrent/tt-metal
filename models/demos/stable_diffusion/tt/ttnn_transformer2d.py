# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.demos.stable_diffusion.tt.resnetblock2d_utils import get_weights, get_mask_tensor
from models.demos.stable_diffusion.tt.ttnn_basic_transformer_block import sd_basic_transformer_block


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
    print("Shape of hidden states in transformer2d:", hidden_states.shape)
    if (C == 1280 and H == 64) or (C == 640 and H == 64):
        hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
        torch_weight = ttnn.to_torch(parameters.norm.weight).to(torch.float)
        torch_bias = ttnn.to_torch(parameters.norm.bias).to(torch.float)
        hidden_states = (
            torch.nn.functional.group_norm(hidden_states, norm_num_groups, weight=torch_weight, bias=torch_bias)
            .permute(0, 2, 3, 1)
            .view(N, 1, W * H, C)
        )
        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        # input_mask_tensor = get_mask_tensor(C, norm_num_groups, grid_size.y, device)
        hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, (N, 1, W * H, C))
        # grid_size = ttnn.CoreGrid(y=4, x=8)
        # input_mask_tensor = get_mask_tensor(C, norm_num_groups, grid_size.y, device)
        gamma_t, beta_t = parameters.norm.tt_weight, parameters.norm.tt_bias
        input_mask_tensor = parameters.norm.input_mask_tensor
        gamma_t = ttnn.to_device(gamma_t, device)
        beta_t = ttnn.to_device(beta_t, device)
        input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = N * H * W // grid_size.x, C // grid_size.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)
        if hidden_states.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
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
        ttnn.deallocate(input_mask_tensor)
    hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.reshape(hidden_states, (N, H * W, inner_dim))
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.proj_in.weight,
        bias=parameters.proj_in.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    for params in parameters.transformer_blocks:
        hidden_states = sd_basic_transformer_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
            attention_head_dim=attention_head_dim,
            attention_mask=attention_mask,
            config=config,
            parameters=params,
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
    ttnn.deallocate(hidden_states)
    ttnn.deallocate(residual)
    return output
