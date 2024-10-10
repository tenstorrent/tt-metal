# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from torch import nn
import collections.abc
from models.demos.swin.tt.swin_utils import window_partition, window_reverse


def patch_embeddings(config, pixel_values, parameters, device):
    _, num_channels, height, width = pixel_values.shape
    pixel_values = ttnn.to_torch(pixel_values).to(torch.float)
    weight = ttnn.to_torch(parameters.embeddings.patch_embeddings.projection.weight).to(torch.float)
    bias = ttnn.to_torch(parameters.embeddings.patch_embeddings.projection.bias).to(torch.float)
    projection = nn.Conv2d(
        in_channels=3,
        out_channels=96,
        kernel_size=4,
        stride=4,
        padding=0,
    )
    projection.weight = nn.Parameter(weight)
    projection.bias = nn.Parameter(bias.squeeze(0).squeeze(0).squeeze(0))
    embeddings = projection(pixel_values)

    embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16)  # , device = device)

    batch, channel, height, width = embeddings.shape
    output_dimensions = (height, width)
    embeddings = ttnn.reshape(embeddings, (1, batch, channel, height * width))

    embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
    embeddings = ttnn.to_device(embeddings, device=device)
    embeddings = ttnn.permute(embeddings, (0, 1, 3, 2))
    embeddings = ttnn.reshape(embeddings, (embeddings.shape[1], embeddings.shape[2], embeddings.shape[3]))
    return embeddings, output_dimensions


def embeddings(config, pixel_values, position_embeddings=None, bool_masked_pos=None, parameters=None, device=None):
    embeddings, output_dimensions = patch_embeddings(config, pixel_values, parameters, device)
    weight = ttnn.to_device(parameters.embeddings.norm.weight, device=device)
    bias = ttnn.to_device(parameters.embeddings.norm.bias, device=device)
    embeddings = ttnn.layer_norm(
        embeddings,
        weight=weight,
        bias=bias,
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if config.use_absolute_embeddings and position_embeddings is not None:
        embeddings = embeddings + position_embeddings

    return embeddings, output_dimensions


def transpose_for_scores(x, num_attention_heads, attention_head_size, device):
    # x must be 4d originaly
    # 1 is appended to the beggining
    # so create tensor shape by ommiting the first dimension
    new_x_shape = list(x.shape)[:-1] + [
        num_attention_heads,
        attention_head_size,
    ]

    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(ttnn.from_device(x), (new_x_shape))
    x = ttnn.permute(ttnn.to_device(x, device=device), (0, 2, 1, 3))
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    return x


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
    stage_index=None,
    swin_index=None,
    relative_position_bias=None,
):
    batch_size, c, num_channels = hidden_states.shape
    num_attention_heads = num_heads
    attention_head_size = int(dim / num_heads)
    all_head_size = num_attention_heads * attention_head_size

    weight = ttnn.to_device(parameters.query.weight, device=device)
    bias = ttnn.to_device(parameters.query.bias, device=device)

    mixed_query_layer = ttnn.linear(
        hidden_states,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=12),
    )

    weight = ttnn.to_device(parameters.key.weight, device=device)
    bias = ttnn.to_device(parameters.key.bias, device=device)

    key_layer = ttnn.linear(
        hidden_states,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        activation=None,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    weight = ttnn.to_device(parameters.value.weight, device=device)
    bias = ttnn.to_device(parameters.value.bias, device=device)

    weight = ttnn.to_device(parameters.value.weight, device=device)
    bias = ttnn.to_device(parameters.value.bias, device=device)
    value_layer = ttnn.linear(
        hidden_states,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=12),
    )

    key_layer = transpose_for_scores(key_layer, num_attention_heads, attention_head_size, device)
    query_layer = transpose_for_scores(mixed_query_layer, num_attention_heads, attention_head_size, device)
    value_layer = transpose_for_scores(value_layer, num_attention_heads, attention_head_size, device)

    key_layer_transposed = ttnn.permute(key_layer, (0, 1, 3, 2))
    attention_scores = ttnn.matmul(query_layer, key_layer_transposed)
    torch.save(
        ttnn.to_torch(attention_scores), "ttnn_attention_scores_" + str(stage_index) + "_" + str(swin_index) + ".pt"
    )
    attention_head_size = int(dim / num_heads)

    attention_scores = ttnn.mul(attention_scores, (1 / (attention_head_size ** (1 / 2))))

    attention_scores = ttnn.add(
        attention_scores,
        relative_position_bias,
    )

    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in SwinModel forward() function)
        mask_shape = attention_mask.shape[0]

        attention_scores = ttnn.to_layout(ttnn.from_device(attention_scores), layout=ttnn.ROW_MAJOR_LAYOUT)
        attention_scores = ttnn.reshape(
            attention_scores,
            (
                batch_size // mask_shape,
                mask_shape,
                num_heads,
                c,
                c,
            ),
        )
        attention_scores = ttnn.to_layout(attention_scores, layout=ttnn.TILE_LAYOUT)
        attention_scores = ttnn.to_device(attention_scores, device=device)

        attention_mask = ttnn.from_device(attention_mask)
        attention_mask = ttnn.to_layout(attention_mask, layout=ttnn.ROW_MAJOR_LAYOUT)

        attention_mask = ttnn.reshape(
            attention_mask, (1, attention_mask.shape[0], 1, attention_mask.shape[1], attention_mask.shape[2])
        )

        attention_mask = ttnn.to_layout(attention_mask, layout=ttnn.TILE_LAYOUT)
        attention_mask = ttnn.to_device(attention_mask, device=device)

        attention_scores = attention_scores + attention_mask
        attention_scores = ttnn.from_device(attention_scores)
        attention_scores = ttnn.to_layout(attention_scores, layout=ttnn.ROW_MAJOR_LAYOUT)
        attention_scores = ttnn.reshape(attention_scores, (-1, num_heads, c, c))

    attention_scores = ttnn.to_layout(attention_scores, layout=ttnn.TILE_LAYOUT)
    attention_scores = ttnn.to_device(attention_scores, device=device)
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
            1,
            new_context_layer_shape[0],
            new_context_layer_shape[1],
            new_context_layer_shape[2],
        ),
    )
    context_layer = ttnn.to_layout(context_layer, layout=ttnn.TILE_LAYOUT)
    context_layer = ttnn.to_device(context_layer, device=device)
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs


def output(config, hidden_states, parameter, device):
    weight = ttnn.to_device(parameter.dense.weight, device=device)
    bias = ttnn.to_device(parameter.dense.bias, device=device)
    return ttnn.linear(
        hidden_states,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=12),
    )


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
    stage_index=None,
    swin_index=None,
    relative_position_bias=None,
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
        parameters.self,
        device,
        swin_index=swin_index,
        stage_index=stage_index,
        relative_position_bias=relative_position_bias,
    )
    attention_output = output(config, self_output[0], parameters.output, device)
    outputs = (attention_output,) + self_output[1:]
    return outputs


def maybe_pad(hidden_states, height, width, window_size):
    pad_right = (window_size - width % window_size) % window_size
    pad_bottom = (window_size - height % window_size) % window_size
    pad_values = [(0, 0), (0, pad_right), (0, pad_bottom)]
    hidden_states = ttnn.pad(hidden_states, pad_values, value=0)
    return hidden_states, pad_values


def swin_intermediate(config, dim, hidden_states, parameter, device):
    weight = ttnn.to_device(parameter.dense.weight, device=device)
    bias = ttnn.to_device(parameter.dense.bias, device=device)
    return ttnn.linear(
        hidden_states,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        activation="gelu",
        # core_grid=ttnn.CoreGrid(y=batch_size, x=12),
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
    stage_index=None,
    swin_index=None,
    relative_position_bias=None,
    attn_mask=None,
):
    height, width = input_dimensions
    window_size = config.window_size
    if min(input_dimensions) < config.window_size:
        shift_size = 0
        window_size = min(input_dimensions)

    batch_size, _, channels = hidden_states.shape
    shortcut = hidden_states
    weight = ttnn.to_device(parameters.layernorm_before.weight, device=device)
    bias = ttnn.to_device(parameters.layernorm_before.bias, device=device)

    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=weight,
        bias=bias,
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, channels))
    hidden_states, pad_values = maybe_pad(hidden_states, height, width, window_size)
    _, height_pad, width_pad, _ = hidden_states.shape

    if shift_size > 0:
        shifted_hidden_states = torch.roll(ttnn.to_torch(hidden_states), shifts=(-shift_size, -shift_size), dims=(1, 2))
        shifted_hidden_states = ttnn.from_torch(
            shifted_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
    else:
        shifted_hidden_states = hidden_states

    hidden_states_windows = window_partition(shifted_hidden_states, window_size, device)
    hidden_states_windows = ttnn.reshape(hidden_states_windows, (-1, window_size * window_size, channels))
    print("Height and width pad :", stage_index, " ", height_pad, " ", width_pad, " ", shift_size, " ", window_size)
    # attn_mask = get_attn_mask(height_pad, width_pad, shift_size, window_size, dtype=hidden_states.dtype, device=device)
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
        stage_index=stage_index,
        swin_index=swin_index,
        relative_position_bias=relative_position_bias,
    )

    attention_output = attention_outputs[0]
    torch.save(
        ttnn.to_torch(attention_output), "ttnn_attention_output_" + str(stage_index) + "_" + str(swin_index) + ".pt"
    )
    attention_output = ttnn.to_layout(attention_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    attention_windows = ttnn.reshape(attention_output, (-1, window_size, window_size, channels))

    shifted_windows = window_reverse(attention_windows, window_size, height_pad, width_pad, device)

    if shift_size > 0:
        attention_windows = torch.roll(ttnn.to_torch(shifted_windows), shifts=(shift_size, shift_size), dims=(1, 2))
        attention_windows = ttnn.from_torch(attention_windows, dtype=ttnn.bfloat16, device=device)
    else:
        attention_windows = shifted_windows

    was_padded = pad_values[1][1] > 0 or pad_values[2][1] > 0
    if was_padded:
        attention_windows = attention_windows[:, :height, :width, :]
    attention_windows = ttnn.reshape(attention_windows, (batch_size, height * width, channels))
    attention_windows = ttnn.to_layout(attention_windows, layout=ttnn.TILE_LAYOUT)
    hidden_states = ttnn.add(shortcut, attention_windows)

    weight = ttnn.to_device(parameters.layernorm_after.weight, device=device)
    bias = ttnn.to_device(parameters.layernorm_after.bias, device=device)
    layer_output = ttnn.layer_norm(
        hidden_states,
        weight=weight,
        bias=bias,
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    layer_output = swin_intermediate(config, dim, layer_output, parameters.intermediate, device)
    layer_output = output(config, layer_output, parameters.output, device)
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

    weight = ttnn.to_device(parameter.downsample.norm.weight, device=device)
    bias = ttnn.to_device(parameter.downsample.norm.bias, device=device)
    input_feature = ttnn.layer_norm(
        input_feature,
        weight=weight,
        bias=bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    weight = ttnn.to_device(parameter.downsample.reduction.weight, device=device)
    input_feature = ttnn.linear(
        input_feature,
        weight,
        bias=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=12),
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
    stage_index=None,
    relative_position_bias=None,
    attn_mask_list=None,
):
    height, width = input_dimensions

    # for block in parameter.blocks:
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
            stage_index=stage_index,
            swin_index=i,
            relative_position_bias=relative_position_bias,
            attn_mask=attn_mask_list[i],
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
            stage_index=i_layer,
            relative_position_bias=bias_table[i_layer],
            attn_mask_list=attention_mask_list[i_layer],
        )
        hidden_state = layer_outputs[0]
        torch.save(ttnn.to_torch(hidden_state), "ttnn_hidden_state_" + str(i_layer) + ".pt")
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
):
    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
    head_mask = [
        None,
    ] * len(config.depths)

    embedding_output, input_dimensions = embeddings(
        config=config, pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, parameters=parameters, device=device
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
    )
    sequence_output = ttnn.to_device(sequence_output, device=device)
    weight = ttnn.to_device(parameters.layernorm.weight, device=device)
    bias = ttnn.to_device(parameters.layernorm.bias, device=device)

    sequence_output = ttnn.layer_norm(
        sequence_output,
        weight=weight,
        bias=bias,
        epsilon=config.layer_norm_eps,
        # memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    pooler = nn.AdaptiveAvgPool1d(1)
    sequence_output_1 = ttnn.to_torch(sequence_output)
    pooled_output = pooler(sequence_output_1.transpose(1, 2))
    pooled_output = ttnn.from_torch(pooled_output, dtype=ttnn.bfloat16)
    pooled_output = ttnn.reshape(
        pooled_output, (pooled_output.shape[0], pooled_output.shape[1] * pooled_output.shape[2])
    )

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
    )
    pooled_output = outputs[1]

    pooled_output = ttnn.to_layout(pooled_output, layout=ttnn.TILE_LAYOUT)
    pooled_output = ttnn.to_device(pooled_output, device=device)
    weight = ttnn.to_device(parameters.classifier.weight, device=device)
    bias = ttnn.to_device(parameters.classifier.bias, device=device)
    logits = ttnn.linear(
        pooled_output,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=12),
    )

    return logits


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)  # ,layout = ttnn.TILE_LAYOUT)
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
    return parameters
