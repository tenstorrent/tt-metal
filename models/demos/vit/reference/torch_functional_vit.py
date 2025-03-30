# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers

# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def fold_torch(input_tensor, stride_h, stride_w):
    N, H, W, C = input_tensor.shape
    reshaped = input_tensor.reshape(N, H // stride_h, stride_h, W // stride_w, stride_w, C)
    transposed = reshaped.permute(0, 1, 3, 2, 4, 5)
    return transposed.reshape(N, H // stride_h, W // stride_w, C * stride_h * stride_w)


def vit_patch_embeddings(
    pixel_values,
    *,
    parameters,
):
    batch_size, img_c, img_h, img_w = pixel_values.shape
    patch_size = 16
    patch_count = img_h // patch_size  # 14
    patch_size_sq = int(patch_size * patch_size)  # 256
    patch_size_sq_trpl = int(patch_size_sq * img_c)  # 768
    patch_count_sq = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    pixel_values = torch.permute(pixel_values, (0, 2, 3, 1))
    pixel_values = torch.nn.functional.pad(pixel_values, (0, 4 - pixel_values.shape[3], 0, 0, 0, 0))
    pixel_values = pixel_values.reshape(
        pixel_values.shape[0],
        pixel_values.shape[1],
        pixel_values.shape[2] // patch_size,
        pixel_values.shape[3] * patch_size,
    )

    pixel_values = fold_torch(pixel_values, stride_h, stride_w)
    pixel_values = pixel_values.reshape(1, 1, -1, pixel_values.shape[-1])

    patch_embedding_output = pixel_values @ parameters.projection.weight
    patch_embedding_output = patch_embedding_output + parameters.projection.bias

    patch_embedding_output = patch_embedding_output.reshape(batch_size, patch_count_sq, patch_size_sq_trpl)

    return patch_embedding_output


def vit_embeddings(
    config,
    pixel_values,
    position_embeddings,
    cls_tokens,
    *,
    parameters,
):
    batch_size, img_c, img_h, img_w = pixel_values.shape
    patch_size = 16
    patch_count = img_h // patch_size  # 14

    patch_embeddings = vit_patch_embeddings(pixel_values, parameters=parameters.patch_embeddings)
    cls_tokens = cls_tokens.expand(batch_size, -1, -1)
    patch_embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)
    embedding_output = patch_embeddings + position_embeddings

    return embedding_output


def vit_layernorm_before(
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


def vit_layernorm_after(
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


def vit_attention(
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


def vit_intermediate(hidden_states, *, parameters):
    hidden_states = hidden_states @ parameters.dense.weight
    hidden_states = hidden_states + parameters.dense.bias
    hidden_states = torch.nn.functional.gelu(hidden_states)
    return hidden_states


def vit_output(config, hidden_states, residual, *, parameters):
    output = hidden_states @ parameters.dense.weight
    output = output + parameters.dense.bias
    output = output + residual

    return output


def vit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = vit_intermediate(hidden_states, parameters=parameters.intermediate)
    hidden_states = vit_output(config, intermediate, attention_output, parameters=parameters.output)

    return hidden_states


def vit_layer(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    layernorm_before_output = vit_layernorm_before(
        config,
        hidden_states,
        parameters=parameters,
    )
    attention_output = vit_attention(
        config,
        layernorm_before_output,
        attention_mask,
        parameters=parameters.attention,
    )
    attention_output = attention_output + hidden_states

    layernorm_after_output = vit_layernorm_after(
        config,
        attention_output,
        parameters=parameters,
    )

    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        attention_output,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    encoder_input = hidden_states
    encoder_output = None
    for encoder_parameters in parameters.layer:
        encoder_output = vit_layer(
            config,
            encoder_input,
            attention_mask,
            parameters=encoder_parameters,
        )
        encoder_input = encoder_output
    return encoder_output


def vit(
    config,
    pixel_values,
    position_embeddings,
    cls_tokens,
    attention_mask,
    *,
    parameters,
):
    hidden_states = vit_embeddings(
        config, pixel_values, position_embeddings, cls_tokens, parameters=parameters.vit.embeddings
    )

    hidden_states = vit_encoder(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.vit.encoder,
    )

    # Final LayerNorm
    output = torch.nn.functional.layer_norm(
        hidden_states,
        (config.hidden_size,),
        parameters.vit.layernorm.weight,
        parameters.vit.layernorm.bias,
        eps=config.layer_norm_eps,
    )

    # Pooler
    pooler_output = output[0] @ parameters.classifier.weight
    pooler_output = pooler_output + parameters.classifier.bias
    # pooler_output = torch.tanh(pooler_output)

    return pooler_output


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.vit.modeling_vit.ViTPatchEmbeddings):
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
