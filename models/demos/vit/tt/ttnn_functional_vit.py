# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
import torch

import ttnn
from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_linear_bias,
)


# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def vit_patch_embeddings(config, pixel_values, *, parameters, unittest_check=False):
    # batch_size, img_c, img_h, img_w = pixel_values.shape # NCHW
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = 16
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = pixel_values @ parameters.projection.weight
    patch_embedding_output = patch_embedding_output + parameters.projection.bias

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, patch_size_sq_trpl))

    return patch_embedding_output


def vit_embeddings(
    config,
    pixel_values,
    cls_token,
    position_embeddings,
    *,
    parameters,
):
    parameters = parameters.vit.embeddings

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    embedding_output = ttnn.concat((cls_token, patch_embeddings), dim=1)
    embedding_output = embedding_output + position_embeddings

    return embedding_output


def vit_layernorm_before(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
    )

    return attention_output


def vit_layernorm_after(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
    )

    return attention_output


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
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = hidden_states @ parameters.attention.key.weight
    key = key + parameters.attention.key.bias
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters.attention.value.weight
    value = value + parameters.attention.value.bias
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
    value = ttnn.permute(value, (0, 2, 1, 3))

    attention_scores = query @ key
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    self_output = context_layer
    self_output = self_output @ parameters.output.dense.weight
    self_output = self_output + parameters.output.dense.bias

    return self_output


def vit_intermediate(
    hidden_states,
    *,
    parameters,
):
    output = hidden_states @ parameters.dense.weight
    output = output + parameters.dense.bias
    output = ttnn.gelu(output)
    return output


def vit_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
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
    print("hhhh", hidden_states.shape)
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
    attention_mask,
    cls_token,
    position_embeddings,
    *,
    parameters,
):
    embeddings_output = vit_embeddings(config, pixel_values, cls_token, position_embeddings, parameters=parameters)

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        attention_mask=None,
        parameters=parameters.vit.encoder,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.vit.layernorm.weight,
        bias=parameters.vit.layernorm.bias,
    )

    # Classifier
    classifier_output = output @ parameters.classifier.weight
    classifier_output = classifier_output + parameters.classifier.bias

    return classifier_output


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.vit.modeling_vit.ViTEmbeddings):
        weight = torch_model.patch_embeddings.projection.weight
        bias = torch_model.patch_embeddings.projection.bias

        three_times_hidden_size, c, _, _ = weight.shape
        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(
            preprocessed_weight, (int(three_times_hidden_size * (4 / c)), three_times_hidden_size)
        )

        parameters = {"patch_embeddings": {}}
        parameters["patch_embeddings"] = {"projection": {}}
        parameters["patch_embeddings"]["projection"]["weight"] = ttnn.from_torch(
            preprocessed_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["patch_embeddings"]["projection"]["bias"] = ttnn.from_torch(
            bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        parameters["cls_token"] = ttnn.from_torch(torch_model.cls_token, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters["position_embeddings"] = ttnn.from_torch(
            torch_model.position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    return parameters
