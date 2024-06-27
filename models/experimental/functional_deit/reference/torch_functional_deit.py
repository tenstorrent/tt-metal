# # SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# # SPDX-License-Identifier: Apache-2.0

import torch
import transformers
from torch import nn


def transpose_for_scores(x, num_attention_heads, attention_head_size):
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)


def deit_patch_embeddings(
    config,
    *,
    parameters,
    pixel_values,
):
    print(f"parameters : {parameters}")
    hidden_size = config.hidden_size
    batch_size, num_channels, height, width = pixel_values.shape
    image_size, patch_size = config.image_size, config.patch_size

    if num_channels != config.num_channels:
        raise ValueError(
            "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
        )
    projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    projection.weight = parameters.projection.weight
    projection.bias = parameters.projection.bias

    x = projection(pixel_values).flatten(2).transpose(1, 2)

    return x


def deit_embeddings(
    config,
    *,
    parameters,
    pixel_values,
    bool_masked_pos=None,
    use_mask_token=False,
    interpolate_pos_encoding=False,
):
    _, _, height, width = pixel_values.shape
    embeddings = deit_patch_embeddings(config=config, parameters=parameters.patch_embeddings, pixel_values=pixel_values)
    batch_size, seq_length, _ = embeddings.size()

    image_size = (config.image_size, config.image_size)
    patch_size = (config.patch_size, config.patch_size)
    num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

    mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
    cls_token = parameters.cls_token
    distillation_token = parameters.distillation_token
    position_embeddings = parameters.position_embeddings

    if bool_masked_pos is not None:
        mask_tokens = mask_token.expand(batch_size, seq_length, -1)
        # replace the masked visual tokens by mask_tokens
        mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
        embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

    cls_tokens = cls_token.expand(batch_size, -1, -1)

    distillation_tokens = distillation_token.expand(batch_size, -1, -1)

    embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
    position_embedding = position_embeddings

    if interpolate_pos_encoding:
        position_embedding = interpolate_pos_encoding(embeddings, height, width)

    embeddings = embeddings + position_embedding

    return embeddings


def fold_torch(input_tensor, stride_h, stride_w):
    N, H, W, C = input_tensor.shape
    reshaped = input_tensor.reshape(N, H // stride_h, stride_h, W // stride_w, stride_w, C)
    transposed = reshaped.permute(0, 1, 3, 2, 4, 5)
    return transposed.reshape(N, H // stride_h, W // stride_w, C * stride_h * stride_w)


def deit_layernorm_before(
    config,
    hidden_states,
    *,
    parameters,
):
    batch_size, sequence_size, hidden_size = hidden_states.shape
    attention_output = torch.nn.functional.layer_norm(
        hidden_states,
        (hidden_size,),
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        eps=config.layer_norm_eps,
    )

    return attention_output


def deit_layernorm_after(
    config,
    hidden_states,
    *,
    parameters,
):
    batch_size, sequence_size, hidden_size = hidden_states.shape
    layernorm_output = torch.nn.functional.layer_norm(
        hidden_states,
        (hidden_size,),
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        eps=config.layer_norm_eps,
    )

    return layernorm_output


def deit_attention(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = hidden_states @ parameters.attention.query.weight
    query = query + parameters.attention.query.bias
    query = torch.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = torch.permute(query, (0, 2, 1, 3))

    key = hidden_states @ parameters.attention.key.weight
    key = key + parameters.attention.key.bias
    key = torch.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = torch.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters.attention.value.weight
    value = value + parameters.attention.value.bias
    value = torch.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = torch.permute(value, (0, 2, 1, 3))

    attention_scores = query @ key
    attention_scores = attention_scores / (head_size**0.5)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value
    context_layer = torch.permute(context_layer, (0, 2, 1, 3))
    context_layer = torch.reshape(context_layer, (batch_size, sequence_size, hidden_size))

    self_output = context_layer
    self_output = self_output @ parameters.output.dense.weight
    self_output = self_output + parameters.output.dense.bias

    return self_output


def deit_intermediate(hidden_states, *, parameters):
    hidden_states = hidden_states @ parameters.dense.weight
    hidden_states = hidden_states + parameters.dense.bias
    hidden_states = torch.nn.functional.gelu(hidden_states)
    return hidden_states


def deit_output(config, hidden_states, residual, *, parameters):
    output = hidden_states @ parameters.dense.weight
    output = output + parameters.dense.bias
    output = output + residual

    return output


def deit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = deit_intermediate(hidden_states, parameters=parameters.intermediate)
    hidden_states = deit_output(config, intermediate, attention_output, parameters=parameters.output)

    return hidden_states


def deit_layer(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    layernorm_before_output = deit_layernorm_before(
        config,
        hidden_states,
        parameters=parameters,
    )
    attention_output = deit_attention(
        config,
        layernorm_before_output,
        attention_mask,
        parameters=parameters.attention,
    )
    attention_output = attention_output + hidden_states

    layernorm_after_output = deit_layernorm_after(
        config,
        attention_output,
        parameters=parameters,
    )

    feedforward_output = deit_feedforward(
        config,
        layernorm_after_output,
        attention_output,
        parameters=parameters,
    )

    return feedforward_output


def deit_encoder(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    encoder_input = hidden_states
    encoder_output = None
    for encoder_parameters in parameters.layer:
        encoder_output = deit_layer(
            config,
            encoder_input,
            attention_mask,
            parameters=encoder_parameters,
        )
        encoder_input = encoder_output
    return encoder_output


def deit(
    config,
    pixel_values,
    attention_mask,
    *,
    parameters,
):
    hidden_states = deit_embeddings(config=config, parameters=parameters.embeddings, pixel_values=pixel_values)

    hidden_states = deit_encoder(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.encoder,
    )

    output = torch.nn.functional.layer_norm(
        hidden_states,
        (config.hidden_size,),
        parameters.layernorm.weight,
        parameters.layernorm.bias,
        eps=config.layer_norm_eps,
    )

    return output


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.deit.modeling_deit.DeiTPatchEmbeddings):
        weight = torch_model.projection.weight
        bias = torch_model.projection.bias

        three_times_hidden_size, c, _, _ = weight.shape

        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(
            preprocessed_weight, (int(three_times_hidden_size * (4 / c)), three_times_hidden_size)
        )

        parameters = {"projection": {}}
        parameters["projection"]["weight"] = preprocessed_weight
        parameters["projection"]["bias"] = bias

    return parameters
