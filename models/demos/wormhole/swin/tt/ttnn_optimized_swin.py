# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from torch import nn
from models.demos.wormhole.swin.tt.swin_utils import window_partition, window_reverse
from ttnn.model_preprocessing import (
    preprocess_linear_bias,
    preprocess_linear_weight,
)


def patch_embeddings(config, pixel_values, parameters, device, mesh_mapper, output_mesh_composer):
    _, num_channels, height, width = pixel_values.shape
    pixel_values = ttnn.to_torch(pixel_values, mesh_composer=output_mesh_composer).to(torch.float)

    weight = ttnn.to_torch(
        parameters.embeddings.patch_embeddings.projection.weight, mesh_composer=output_mesh_composer
    ).to(torch.float)
    bias = ttnn.to_torch(parameters.embeddings.patch_embeddings.projection.bias, mesh_composer=output_mesh_composer).to(
        torch.float
    )
    projection = nn.Conv2d(
        in_channels=3,
        out_channels=96,
        kernel_size=4,
        stride=4,
        padding=0,
    )
    projection.weight = nn.Parameter(weight[:96, :, :, :])
    projection.bias = nn.Parameter(bias[:1, :, :, :].squeeze(0).squeeze(0).squeeze(0))

    embeddings = projection(pixel_values)
    batch, channel, height, width = embeddings.shape
    embeddings = ttnn.from_torch(
        embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
    )
    output_dimensions = (height, width)
    embeddings = ttnn.permute(embeddings, (0, 2, 3, 1))
    embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)
    embeddings = ttnn.reshape(
        embeddings, (embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2], embeddings.shape[3])
    )
    embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
    return embeddings, output_dimensions


def embeddings(
    config,
    pixel_values,
    position_embeddings=None,
    bool_masked_pos=None,
    parameters=None,
    device=None,
    mesh_mapper=None,
    output_mesh_composer=None,
):
    embeddings, output_dimensions = patch_embeddings(
        config, pixel_values, parameters, device, mesh_mapper, output_mesh_composer
    )
    embeddings = ttnn.layer_norm(
        embeddings,
        weight=parameters.embeddings.norm.weight,
        bias=parameters.embeddings.norm.bias,
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if config.use_absolute_embeddings and position_embeddings is not None:
        embeddings = embeddings + position_embeddings

    return embeddings, output_dimensions


def self_attention(
    config,
    dim,
    num_heads,
    window_size,
    hidden_states,
    attention_mask,
    head_mask=None,
    output_attentions=None,
    parameters=None,
    device=None,
    relative_position_bias=None,
    mesh_mapper=None,
    output_mesh_composer=None,
):
    batch_size, c, num_channels = hidden_states.shape
    num_attention_heads = num_heads
    attention_head_size = int(dim / num_heads)
    all_head_size = num_attention_heads * attention_head_size

    query_key_value_output = ttnn.linear(
        hidden_states,
        parameters.query_key_value.weight,
        bias=parameters.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    (
        query_layer,
        key_layer,
        value_layer,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    attention_scores = ttnn.matmul(query_layer, key_layer)
    attention_head_size = int(dim / num_heads)

    attention_scores = ttnn.mul(attention_scores, (1 / (attention_head_size ** (1 / 2))))

    attention_scores = ttnn.add(
        attention_scores,
        relative_position_bias,
    )

    if attention_mask is not None:
        mask_shape = attention_mask.shape[0]
        attention_scores = attention_scores + attention_mask

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask
    context_layer = ttnn.matmul(attention_probs, value_layer)
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    new_context_layer_shape = tuple(context_layer.shape.with_tile_padding())[:-2] + (num_heads * attention_head_size,)
    context_layer = ttnn.to_layout(context_layer, layout=ttnn.ROW_MAJOR_LAYOUT)

    context_layer = ttnn.reshape(
        ttnn.from_device(context_layer),
        (
            new_context_layer_shape[0],
            new_context_layer_shape[1],
            new_context_layer_shape[2],
        ),
    )
    context_layer = ttnn.to_layout(context_layer, layout=ttnn.TILE_LAYOUT)
    context_layer = ttnn.to_device(context_layer, device=device)
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs


def attention(
    config,
    dim,
    num_heads,
    window_size,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    output_attentions=None,
    parameters=None,
    device=None,
    relative_position_bias=None,
    mesh_mapper=None,
    output_mesh_composer=None,
):
    self_output = self_attention(
        config,
        dim,
        num_heads,
        window_size,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions,
        parameters,
        device,
        relative_position_bias=relative_position_bias,
        mesh_mapper=mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    attention_output = ttnn.linear(
        self_output[0],
        parameters.output.weight,
        bias=parameters.output.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )
    outputs = (attention_output,) + self_output[1:]
    return outputs


def maybe_pad(hidden_states, height, width, window_size):
    pad_right = (window_size - width % window_size) % window_size
    pad_bottom = (window_size - height % window_size) % window_size
    pad_values = [(0, 0), (0, pad_right), (0, pad_bottom)]
    hidden_states = ttnn.pad(hidden_states, pad_values, value=0)
    return hidden_states, pad_values


def swin_intermediate(config, dim, hidden_states, parameter, device):
    return ttnn.linear(
        hidden_states,
        parameter.dense.weight,
        bias=parameter.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        activation="gelu",
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )


def swin_layer(
    config,
    dim,
    input_resolution,
    num_heads,
    shift_size,
    hidden_states,
    input_dimensions,
    head_mask=None,
    output_attentions=None,
    parameters=None,
    device=None,
    relative_position_bias=None,
    attn_mask=None,
    mesh_mapper=None,
    output_mesh_composer=None,
):
    height, width = input_dimensions
    window_size = config.window_size
    if min(input_dimensions) < config.window_size:
        shift_size = 0
        window_size = min(input_dimensions)

    batch_size, _, channels = hidden_states.shape
    shortcut = hidden_states

    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, channels))
    hidden_states, pad_values = maybe_pad(hidden_states, height, width, window_size)
    _, height_pad, width_pad, _ = hidden_states.shape

    if shift_size > 0:
        shifted_hidden_states = torch.roll(
            ttnn.to_torch(hidden_states, mesh_composer=output_mesh_composer),
            shifts=(-shift_size, -shift_size),
            dims=(1, 2),
        )
        shifted_hidden_states = ttnn.from_torch(
            shifted_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
        )
    else:
        shifted_hidden_states = hidden_states

    hidden_states_windows = window_partition(
        shifted_hidden_states, window_size, mesh_mapper, device, output_mesh_composer
    )
    hidden_states_windows = ttnn.reshape(hidden_states_windows, (-1, window_size * window_size, channels))
    hidden_states_windows = ttnn.to_layout(hidden_states_windows, layout=ttnn.TILE_LAYOUT)
    hidden_states_windows = ttnn.to_device(hidden_states_windows, device=device)
    attention_outputs = attention(
        config,
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        hidden_states=hidden_states_windows,
        attention_mask=attn_mask,
        head_mask=head_mask,
        output_attentions=output_attentions,
        parameters=parameters.attention,
        device=device,
        relative_position_bias=relative_position_bias,
        mesh_mapper=mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    attention_output = attention_outputs[0]

    attention_output = ttnn.to_layout(attention_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    attention_windows = ttnn.reshape(attention_output, (-1, window_size, window_size, channels))

    shifted_windows = window_reverse(
        attention_windows, window_size, height_pad, width_pad, device, mesh_mapper, output_mesh_composer
    )

    if shift_size > 0:
        attention_windows = torch.roll(
            ttnn.to_torch(shifted_windows, mesh_composer=output_mesh_composer),
            shifts=(shift_size, shift_size),
            dims=(1, 2),
        )
        attention_windows = ttnn.from_torch(
            attention_windows, dtype=ttnn.bfloat16, device=device, mesh_mapper=mesh_mapper
        )
    else:
        attention_windows = shifted_windows

    was_padded = pad_values[1][1] > 0 or pad_values[2][1] > 0
    if was_padded:
        attention_windows = attention_windows[:, :height, :width, :]
    attention_windows = ttnn.reshape(attention_windows, (batch_size, height * width, channels))
    attention_windows = ttnn.to_layout(attention_windows, layout=ttnn.TILE_LAYOUT)
    hidden_states = ttnn.add(shortcut, attention_windows)

    layer_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    layer_output = ttnn.linear(
        layer_output,
        parameters.intermediate.dense.weight,
        bias=parameters.intermediate.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        activation="gelu",
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    layer_output = ttnn.linear(
        layer_output,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )
    layer_output = hidden_states + layer_output
    layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
    return layer_outputs


def patch_merge_pad(input_feature, height, width):
    should_pad = (height % 2 == 1) or (width % 2 == 1)
    if should_pad:
        pad_values = (0, 0, 0, width % 2, 0, height % 2)
        input_feature = ttnn.pad(input_feature, pad_values)

    return input_feature


def patch_merging(config, input_resolution, dim, input_feature, input_dimensions, parameter, device):
    height, width = input_dimensions
    batch_size, dim, num_channels = input_feature.shape
    input_feature = ttnn.to_layout(input_feature, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_feature = ttnn.reshape(input_feature, (batch_size, height, width, num_channels))
    input_feature = patch_merge_pad(input_feature, height, width)

    input_feature_0 = input_feature[:, 0::2, 0::2, :]
    input_feature_1 = input_feature[:, 1::2, 0::2, :]
    input_feature_2 = input_feature[:, 0::2, 1::2, :]
    input_feature_3 = input_feature[:, 1::2, 1::2, :]

    input_feature = ttnn.concat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)

    input_feature = ttnn.to_layout(input_feature, layout=ttnn.ROW_MAJOR_LAYOUT)

    input_feature = ttnn.reshape(input_feature, (batch_size, -1, 4 * num_channels))
    input_feature = ttnn.to_layout(input_feature, layout=ttnn.TILE_LAYOUT)

    input_feature = ttnn.layer_norm(
        input_feature,
        weight=parameter.downsample.norm.weight,
        bias=parameter.downsample.norm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_feature = ttnn.linear(
        input_feature,
        parameter.downsample.reduction.weight,
        bias=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    return input_feature


def swin_stage(
    config,
    dim,
    input_resolution,
    hidden_states,
    input_dimensions,
    depth,
    layer_head_mask=None,
    output_attention=None,
    num_heads=None,
    downsample=None,
    parameter=None,
    device=None,
    relative_position_bias=None,
    attn_mask_list=None,
    mesh_mapper=None,
    output_mesh_composer=None,
):
    height, width = input_dimensions

    for i in range(depth):
        layer_head_mask = None
        layer_outputs = swin_layer(
            config=config,
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            shift_size=0 if i % 2 == 0 else config.window_size // 2,
            hidden_states=hidden_states,
            input_dimensions=input_dimensions,
            head_mask=layer_head_mask,
            output_attentions=output_attention,
            parameters=parameter.blocks[i],
            device=device,
            relative_position_bias=relative_position_bias[i],
            attn_mask=attn_mask_list[i],
            mesh_mapper=mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )
        hidden_states = layer_outputs[0]

    hidden_states_before_downsampling = hidden_states
    if downsample:
        height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
        output_dimensions = (height, width, height_downsampled, width_downsampled)
        hidden_states = patch_merging(
            config, input_resolution, dim, hidden_states_before_downsampling, input_dimensions, parameter, device
        )
    else:
        output_dimensions = (height, width, height, width)

    stage_outputs = (
        hidden_states,
        hidden_states_before_downsampling,
        output_dimensions,
    )

    return stage_outputs


def encoder(
    config,
    hidden_state,
    input_dimension,
    head_mask=None,
    output_attention=None,
    output_hidden_states=None,
    parameters=None,
    device=None,
    bias_table=None,
    attention_mask_list=None,
    mesh_mapper=None,
    output_mesh_composer=None,
):
    if output_hidden_states:
        batch_size, _, hidden_size = hidden_state.shape
    image_size = (config.image_size, config.image_size)
    patch_size = (config.patch_size, config.patch_size)
    num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
    grid_size = (
        image_size[0] // patch_size[0],
        image_size[1] // patch_size[1],
    )
    for i_layer in range(len(config.depths)):
        layer_head_mask = None
        layer_outputs = swin_stage(
            config,
            dim=int(config.embed_dim * 2**i_layer),
            input_resolution=(
                grid_size[0] // (2**i_layer),
                grid_size[1] // (2**i_layer),
            ),
            hidden_states=hidden_state,
            input_dimensions=input_dimension,
            layer_head_mask=layer_head_mask,
            output_attention=output_attention,
            num_heads=config.num_heads[i_layer],
            depth=config.depths[i_layer],
            downsample=True if (i_layer < len(config.depths) - 1) else False,
            parameter=parameters.layers[i_layer],
            device=device,
            relative_position_bias=bias_table[i_layer],
            attn_mask_list=attention_mask_list[i_layer],
            mesh_mapper=mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )
        hidden_state = layer_outputs[0]
        hidden_states_before_downsampling = layer_outputs[1]
        output_dimensions = layer_outputs[2]
        input_dimension = (output_dimensions[-2], output_dimensions[-1])
    return hidden_state


def swin(
    config,
    pixel_values,
    bool_masked_pos=None,
    head_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    parameters=None,
    device=None,
    bias_table=None,
    attention_mask_list=None,
    mesh_mapper=None,
    output_mesh_composer=None,
):
    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
    head_mask = [
        None,
    ] * len(config.depths)

    embedding_output, input_dimensions = embeddings(
        config=config,
        pixel_values=pixel_values,
        bool_masked_pos=bool_masked_pos,
        parameters=parameters,
        device=device,
        mesh_mapper=mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    sequence_output = encoder(
        config,
        embedding_output,
        input_dimensions,
        head_mask=head_mask,
        output_attention=output_attentions,
        output_hidden_states=output_hidden_states,
        parameters=parameters.encoder,
        device=device,
        bias_table=bias_table,
        attention_mask_list=attention_mask_list,
        mesh_mapper=mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    sequence_output = ttnn.to_device(sequence_output, device=device)
    sequence_output = ttnn.layer_norm(
        sequence_output,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=config.layer_norm_eps,
    )

    pooler = nn.AdaptiveAvgPool1d(1)
    sequence_output_1 = ttnn.to_torch(sequence_output, mesh_composer=output_mesh_composer)
    pooled_output = pooler(sequence_output_1.transpose(1, 2))

    pooled_output = torch.reshape(
        pooled_output, (pooled_output.shape[0], pooled_output.shape[1] * pooled_output.shape[2])
    )
    pooled_output = ttnn.from_torch(pooled_output, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper, device=device)
    return sequence_output, pooled_output


def swin_for_image_classification(
    config,
    pixel_values,
    head_mask=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    bias_table=None,
    attention_mask_list=None,
    mesh_mapper=None,
    output_mesh_composer=None,
    *,
    parameters,
    device,
):
    outputs = swin(
        config=config,
        pixel_values=pixel_values,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        parameters=parameters.swin,
        device=device,
        bias_table=bias_table,
        attention_mask_list=attention_mask_list,
        mesh_mapper=mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    pooled_output = outputs[1]

    pooled_output = ttnn.to_layout(pooled_output, layout=ttnn.TILE_LAYOUT)
    pooled_output = ttnn.to_device(pooled_output, device=device)

    logits = ttnn.linear(
        pooled_output,
        parameters.classifier.weight,
        bias=parameters.classifier.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    return logits


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        weight = model.weight
        bias = model.bias
        while weight.dim() < 4:
            weight = weight.unsqueeze(0)
        while bias.dim() < 4:
            bias = bias.unsqueeze(0)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.bfloat16)

    if hasattr(model, "self"):
        qkv_weight = torch.cat(
            [
                model.self.query.weight,
                model.self.key.weight,
                model.self.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [model.self.query.bias, model.self.key.bias, model.self.value.bias],
            dim=0,
        )
        output_weight = model.output.dense.weight
        output_bias = model.output.dense.bias
        parameters = {"query_key_value": {}, "relative_position_bias_table": {}, "output": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16)
        parameters["output"]["weight"] = preprocess_linear_weight(output_weight, dtype=ttnn.bfloat16)
        parameters["output"]["bias"] = preprocess_linear_bias(output_bias, dtype=ttnn.bfloat16)
        parameters["relative_position_bias_table"] = ttnn.from_torch(
            model.self.relative_position_bias_table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
    return parameters
