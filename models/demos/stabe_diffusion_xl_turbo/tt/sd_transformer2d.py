# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


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
    input_mask_tensor=None,
    *,
    parameters,
    device,
):
    inner_dim = hidden_states.shape[1]

    residual = hidden_states

    N, C, H, W = hidden_states.shape

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
    hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.reshape(hidden_states, (N, H * W, inner_dim))

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.proj_in.weight,
        bias=parameters.proj_in.bias,
    )

    for d in range(num_layers):
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

    if not return_dict:
        return (output,)

    return output
