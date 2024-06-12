# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
import torch
import ttnn


def vit_patch_embeddings(config, pixel_values, *, parameters, unittest_check=False):
    # batch_size, img_c, img_h, img_w = pixel_values.shape # NCHW
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = 16
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    folded_pixel_values = ttnn.experimental.tensor.fold(pixel_values, stride_h, stride_w)  # 1568, 1024
    folded_pixel_values = ttnn.to_memory_config(
        folded_pixel_values, memory_config=ttnn.L1_MEMORY_CONFIG
    )  # Convert back to interleaved or otherwise to_layout will fail
    folded_pixel_values = ttnn.to_layout(folded_pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.linear(
        folded_pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        dtype=ttnn.bfloat16,
    )

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
    device=None,
):
    parameters = parameters.embeddings

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)

    embedding_output = ttnn.concat([cls_token, patch_embeddings], -2)

    embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)

    embedding_output = ttnn.add(
        embedding_output, position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    return embedding_output


def vit_attention(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
    device=None,
):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = hidden_states @ parameters.attention.query.weight
    query = ttnn.from_device(query)
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.to_device(query, device)
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = hidden_states @ parameters.attention.key.weight
    key = ttnn.from_device(key)
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = ttnn.to_device(key, device)
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters.attention.value.weight
    value = ttnn.from_device(value)
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
    value = ttnn.to_device(value, device)
    value = ttnn.permute(value, (0, 2, 1, 3))

    query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
    key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)

    attention_scores = query @ key
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.from_device(context_layer)
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    self_output = context_layer
    self_output = ttnn.to_device(self_output, device)
    self_output = self_output @ parameters.output.dense.weight
    self_output = self_output + parameters.output.dense.bias

    return self_output


def vit_intermediate(
    config,
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        dtype=ttnn.bfloat16,
    )
    output = ttnn.gelu(output)
    ttnn.deallocate(hidden_states)

    return output


def vit_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(hidden_states)

    output = ttnn.add(output, residual, dtype=ttnn.bfloat16)
    ttnn.deallocate(residual)

    return output


def vit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = vit_intermediate(config, hidden_states, parameters=parameters.intermediate)
    hidden_states = vit_output(config, intermediate, attention_output, parameters=parameters.output)
    return hidden_states


def vit_layer(
    config,
    hidden_states,
    attention_mask,
    parameters,
    index=None,
    device=None,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        epsilon=config.layer_norm_eps,
    )

    multi_head_attention_output = vit_attention(
        config,
        layernorm_before_output,
        attention_mask=attention_mask,
        parameters=parameters.attention,
        device=device,
    )

    multi_head_attention_output = ttnn.add(
        multi_head_attention_output,
        hidden_states,
        dtype=ttnn.bfloat16,
    )

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        epsilon=config.layer_norm_eps,
    )

    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        multi_head_attention_output,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    embeddings,
    head_masks,
    parameters,
    device=None,
):
    encoder_input = embeddings

    for index, encoder_parameters in enumerate(parameters.layer):
        encoder_output = vit_layer(
            config,
            encoder_input,
            head_masks[index],
            encoder_parameters,
            index=index,
            device=device,
        )
        encoder_input = encoder_output

    return encoder_output


def vit_pooler(
    hidden_states,
    *,
    device,
    parameters,
):
    hidden_states = ttnn.to_torch(hidden_states)
    first_token_tensor = hidden_states[:, 0]
    first_token_tensor = ttnn.from_torch(
        first_token_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    pooled_output = ttnn.linear(first_token_tensor, parameters.dense.weight, bias=parameters.dense.bias)
    pooled_output = ttnn.tanh(pooled_output)
    return pooled_output


def vit(
    config,
    pixel_values,
    attention_mask,
    cls_token,
    position_embeddings,
    parameters,
    device=None,
):
    embeddings_output = vit_embeddings(
        config, pixel_values, cls_token, position_embeddings, parameters=parameters, device=device
    )
    hidden_states = vit_encoder(
        config,
        embeddings_output,
        attention_mask,
        parameters=parameters.encoder,
        device=device,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=config.layer_norm_eps,
    )

    pooled_output = vit_pooler(
        output,
        parameters=parameters.pooler,
        device=device,
    )

    return output, pooled_output


def preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    device,
):
    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 0, 0, 0, 0, 0, 0, batch_size - 1))
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    return input_ids, token_type_ids, attention_mask


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
            bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        parameters["cls_token"] = ttnn.from_torch(torch_model.cls_token, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters["position_embeddings"] = ttnn.from_torch(
            torch_model.position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    return parameters
