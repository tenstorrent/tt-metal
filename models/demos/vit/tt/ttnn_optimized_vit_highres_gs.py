# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import transformers
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn

core_grid = ttnn.CoreGrid(y=8, x=12)

# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def vit_patch_embeddings_weight_vars(
    config,
    pixel_values,
    proj_weight,
    proj_bias,
    patch_size=16,
):
    # batch_size, img_c, img_h, img_w = pixel_values.shape # NCHW
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_count = img_h // patch_size
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    ## Needed only when running the standalone module pytest test_vit_patch_embeddings
    ## Please comment out when running the pytest on parent module like test_vit_embeddings or test_vit
    # parameters = parameters.vit.embeddings.patch_embeddings
    patch_embedding_output = ttnn.linear(
        pixel_values,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
    )
    ttnn.deallocate(pixel_values)

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, -1))

    return patch_embedding_output


def vit_patch_embeddings(
    config,
    pixel_values,
    *,
    parameters,
):
    return vit_patch_embeddings_weight_vars(
        config,
        pixel_values,
        parameters.projection.weight,
        parameters.projection.bias,
    )


def siglip_patch_embeddings(
    pixel_values,
    *,
    parameters,
):
    # batch_size, img_c, img_h, img_w = pixel_values.shape # NCHW
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = 14
    patch_count = img_h // patch_size  # 16
    patch_count_all = int(patch_count * patch_count)  # 256
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    ## Needed only when running the standalone module pytest test_vit_patch_embeddings
    ## Please comment out when running the pytest on parent module like test_vit_embeddings or test_vit
    # parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.linear(
        pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.deallocate(pixel_values)

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, -1))

    return patch_embedding_output


def vit_embeddings(
    config,
    pixel_values,
    position_embeddings_interpolated,
    *,
    parameters,
):
    parameters = parameters.vit.embeddings

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    embedding_output = ttnn.concat((parameters.cls_token, patch_embeddings), dim=1)

    embedding_output = ttnn.add(
        embedding_output, position_embeddings_interpolated, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    # padding
    # 1024 / 16 = 64
    # 64*64 + 32 = 4128 (from cls_token concat)
    # 4352 = (4128 + 224)
    # 4352 / 8 = 136

    # embedding_output = ttnn.pad(embedding_output, ((0, 0), (0, 224), (0, 0)), 0)

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
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["layernorm_program_config"],
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
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["layernorm_program_config"],
    )

    return attention_output


def vit_attention_experimental(
    config,
    hidden_states,
    attention_mask,
    parameters,
):
    num_heads = config.num_attention_heads
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.attention.query_key_value.weight,
        bias=parameters.attention.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["query_key_value_matmul_program_config"],
    )
    ttnn.reallocate(hidden_states)
    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)
    ttnn.reallocate(value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["query_by_key_matmul_program_config"],
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)
    """
    attention_probs = ttnn.transformer.attention_softmax_(
        attention_scores,
        attention_mask=attention_mask,
        head_size=head_size,
        # program_config=program_configs["softmax_program_config"],
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)

    return self_output
    """
    return attention_scores


def vit_attention(
    config,
    hidden_states,
    attention_mask,
    parameters,
):
    num_heads = config.num_attention_heads
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.attention.query_key_value.weight,
        bias=parameters.attention.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["query_key_value_matmul_program_config"],
    )
    ttnn.reallocate(hidden_states)
    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)
    value = ttnn.reallocate(value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["query_by_key_matmul_program_config"],
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probs = ttnn.transformer.attention_softmax_(
        attention_scores,
        attention_mask=attention_mask,
        head_size=head_size,
        # program_config=program_configs["softmax_program_config"],
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)

    return self_output


def siglip_attention(
    hidden_states,
    attention_mask,
    parameters,
):
    num_heads = 16
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.query_key_value.weight,
        bias=parameters.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["query_key_value_matmul_program_config"],
    )
    ttnn.reallocate(hidden_states)
    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)
    value = ttnn.reallocate(value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["query_by_key_matmul_program_config"],
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probs = ttnn.transformer.attention_softmax_(
        attention_scores,
        attention_mask=attention_mask,
        head_size=head_size,
        # program_config=program_configs["softmax_program_config"],
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.proj.weight,
        bias=parameters.proj.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)

    return self_output


def vit_intermediate(
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        # program_config=program_configs["ff1_matmul_program_config"],
        core_grid=ttnn.CoreGrid(y=8, x=12),
        activation="gelu",
    )
    # ttnn.deallocate(hidden_states)

    return output


def siglip_intermediate(
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        # program_config=program_configs["ff1_matmul_program_config"],
        core_grid=ttnn.CoreGrid(y=8, x=8),
        activation="gelu",
    )
    # ttnn.deallocate(hidden_states)

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
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    output = ttnn.add(output, residual, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

    # ttnn.deallocate(residual)

    return output


def siglip_output(
    hidden_states,
    residual,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    output = ttnn.add(output, residual, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

    # ttnn.deallocate(residual)

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


def siglip_feedforward(
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = siglip_intermediate(hidden_states, parameters=parameters.mlp)
    hidden_states = siglip_output(intermediate, attention_output, parameters=parameters.mlp)
    return hidden_states


def vit_layer(
    config,
    hidden_states,
    attention_mask,
    parameters,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        # program_config=program_configs["layernorm_program_config"],
    )

    multi_head_attention_output = vit_attention(
        config,
        layernorm_before_output,
        attention_mask=attention_mask,
        parameters=parameters.attention,
    )

    multi_head_attention_output = ttnn.add(
        multi_head_attention_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        # program_config=program_configs["layernorm_program_config"],
    )

    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        multi_head_attention_output,
        parameters=parameters,
    )

    return feedforward_output


def siglip_layer(
    hidden_states,
    attention_mask,
    parameters,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.norm1.weight,
        bias=parameters.norm1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        # program_config=program_configs["layernorm_program_config"],
    )

    multi_head_attention_output = siglip_attention(
        layernorm_before_output,
        attention_mask=attention_mask,
        parameters=parameters.attn,
    )

    multi_head_attention_output = ttnn.add(
        multi_head_attention_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.norm2.weight,
        bias=parameters.norm2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        # program_config=program_configs["layernorm_program_config"],
    )

    feedforward_output = siglip_feedforward(
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
):
    # encoder_input = ttnn.to_memory_config(
    #     embeddings,
    #     memory_config=ttnn.create_sharded_memory_config(
    #         embeddings.shape,
    #         core_grid=core_grid,
    #         strategy=ttnn.ShardStrategy.BLOCK,
    #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #     ),
    #     dtype=ttnn.bfloat8_b,
    # )
    # ttnn.deallocate(embeddings)
    encoder_input = embeddings

    encoder_output = None
    for index, encoder_parameters in enumerate(parameters.layer):
        encoder_output = vit_layer(
            config,
            encoder_input,
            head_masks[index],
            encoder_parameters,
        )
        encoder_input = encoder_output

    return encoder_output


def siglip_encoder(
    embeddings,
    head_masks,
    parameters,
    layer_end_index=None,
):
    encoder_input = embeddings
    if layer_end_index is None:
        layer_end_index = len(parameters)
    params = [parameters[index] for index in parameters]
    encoder_output = None
    for index, param in enumerate(params[:layer_end_index]):
        encoder_output = siglip_layer(
            encoder_input,
            head_masks[index],
            param,
        )
        encoder_input = encoder_output

    return encoder_output


def vit(
    config,
    pixel_values,
    attention_mask,
    position_embeddings_interpolated,
    parameters,
):
    embeddings_output = vit_embeddings(config, pixel_values, position_embeddings_interpolated, parameters=parameters)

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        attention_mask,
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
            memory_config=ttnn.L1_MEMORY_CONFIG,
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
            preprocessed_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        parameters["patch_embeddings"]["projection"]["bias"] = ttnn.from_torch(
            bias.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

        parameters["cls_token"] = ttnn.from_torch(torch_model.cls_token, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        parameters["position_embeddings"] = ttnn.from_torch(
            torch_model.position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        qkv_weight = torch.cat(
            [
                torch_model.query.weight,
                torch_model.key.weight,
                torch_model.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [torch_model.query.bias, torch_model.key.bias, torch_model.value.bias],
            dim=0,
        )

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)

    elif isinstance(torch_model, torch.nn.Linear):
        parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat8_b)
        parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat8_b)

    return parameters


def upchannel_attn_weight_bias(qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads):
    qkv = 3
    is_padding_required = (qkv_weight.shape[0] // (num_heads * qkv)) % 32 != 0
    if is_padding_required:
        padded_val = int(np.ceil(qkv_weight.shape[0] / (num_heads * qkv * 32)) * (num_heads * qkv * 32))
        new_qkv_weight = torch.zeros((padded_val, qkv_weight.shape[1]), dtype=qkv_weight.dtype).reshape(
            qkv, num_heads, -1, qkv_weight.shape[1]
        )
        reshaped_qkv_weight = qkv_weight.reshape(qkv, num_heads, -1, qkv_weight.shape[1])
        new_qkv_weight[:, :, : reshaped_qkv_weight.shape[2], :] = reshaped_qkv_weight
        new_qkv_weight = new_qkv_weight.reshape(padded_val, qkv_weight.shape[1])
        new_qkv_bias = torch.zeros((padded_val), dtype=qkv_bias.dtype).reshape(qkv, num_heads, -1)
        reshaped_qkv_bias = qkv_bias.reshape(qkv, num_heads, -1)
        new_qkv_bias[:, :, : reshaped_qkv_bias.shape[2]] = reshaped_qkv_bias
        new_qkv_bias = new_qkv_bias.reshape((-1,))
        new_proj_weight = torch.zeros((proj_weight.shape[0], padded_val // qkv), dtype=proj_weight.dtype).reshape(
            proj_weight.shape[0], num_heads, -1
        )
        reshaped_proj_head = proj_weight.reshape(proj_weight.shape[0], num_heads, -1)
        new_proj_weight[:, :, : reshaped_proj_head.shape[2]] = reshaped_proj_head
        new_proj_weight = new_proj_weight.reshape((proj_weight.shape[0], padded_val // qkv))
        qkv_weight, qkv_bias, proj_weight = new_qkv_weight, new_qkv_bias, new_proj_weight
    return qkv_weight, qkv_bias, proj_weight, proj_bias


def prepare_dinov2_embedding_constants(tensors, device):
    assert len(tensors) == 2
    proj_weight = tensors[0]
    proj_bias = tensors[1]
    three_times_hidden_size, c, _, _ = proj_weight.shape
    pad_value = 4 - c
    preprocessed_weight = torch.nn.functional.pad(proj_weight, (0, 0, 0, 0, 0, pad_value))
    preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
    preprocessed_weight = torch.reshape(preprocessed_weight, (-1, three_times_hidden_size))

    tensors[0] = ttnn.from_torch(preprocessed_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[1] = ttnn.from_torch(proj_bias.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    return [tensors[0], tensors[1]]


def dinov2_embedding(var0, *args):
    var2 = vit_patch_embeddings_weight_vars(None, var0, args[0], args[1], patch_size=14)
    var5 = ttnn.add(var2, args[2])
    var6 = ttnn.concat([args[3], args[4], var5], dim=1)
    return var6


def prepare_dinov2_attention_constants(tensors, device):
    assert len(tensors) == 7
    tensors[0] = ttnn.from_torch(tensors[0].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[1] = ttnn.from_torch(tensors[1].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[2] = ttnn.from_torch(tensors[2].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[3] = ttnn.from_torch(tensors[3].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[4] = ttnn.from_torch(tensors[4].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[5] = ttnn.from_torch(tensors[5].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[6] = ttnn.from_torch(tensors[6].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tensors


def dinov2_attention(var0, *args, num_heads=16):
    hidden_states = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads
    query_key_value = ttnn.linear(
        hidden_states,
        args[3],
        bias=args[2],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["query_key_value_matmul_program_config"],
    )
    ttnn.reallocate(hidden_states)
    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)
    scale = 1.0 / (head_size**0.5)
    query = ttnn.mul_(
        query,
        scale,
    )
    value = ttnn.reallocate(value)
    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["query_by_key_matmul_program_config"],
    )

    ttnn.deallocate(query)
    ttnn.deallocate(key)
    attention_probs = ttnn.softmax_in_place(attention_scores, numeric_stable=True)

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)
    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    self_output = ttnn.linear(
        context_layer,
        args[5],
        bias=args[4],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)
    var19 = ttnn.mul(self_output, args[6])
    var20 = ttnn.add(var0, var19)
    return var20


def prepare_dinov2_feedforward_constants(tensors, device):
    assert len(tensors) == 7
    tensors[0] = ttnn.from_torch(tensors[0].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[1] = ttnn.from_torch(tensors[1].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[2] = ttnn.from_torch(tensors[2].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[3] = ttnn.from_torch(tensors[3].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[4] = ttnn.from_torch(tensors[4].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[5] = ttnn.from_torch(tensors[5].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[6] = ttnn.from_torch(tensors[6].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tensors


def dinov2_feedforward(var0, *args):
    hidden_states = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    hidden_states = ttnn.linear(
        hidden_states,
        args[3],
        bias=args[2],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # program_config=program_configs["ff1_matmul_program_config"],
        core_grid=ttnn.CoreGrid(y=8, x=8),
        activation="gelu",
    )
    hidden_states = ttnn.linear(
        hidden_states,
        args[5],
        bias=args[4],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        # program_config=program_configs["ff2_matmul_program_config"],
    )
    hidden_states = ttnn.mul(hidden_states, args[6])
    var11 = ttnn.add(var0, hidden_states)
    return var11


def prepare_dinov2_head_constants(tensors, device):
    assert len(tensors) == 2
    tensors[0] = ttnn.from_torch(tensors[0].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[1] = ttnn.from_torch(tensors[1].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tensors


def dinov2_head(var0, *args):
    var1 = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return var1[:, 0, :]


def custom_preprocessor_siglip(torch_model, name):
    import timm

    attention_class = None
    try:
        attention_class = timm.layers.attention.Attention
    except:
        try:
            attention_class = timm.models.vision_transformer.Attention
        except:
            attention_class = None
    assert (
        attention_class is not None
    ), f"Could not find Attention Class in timm library. Please check version timm={timm.__version__}."
    parameters = {}
    if isinstance(torch_model, timm.layers.patch_embed.PatchEmbed):
        weight = torch_model.proj.weight
        bias = torch_model.proj.bias

        three_times_hidden_size, c, _, _ = weight.shape
        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(preprocessed_weight, (-1, three_times_hidden_size))

        parameters = {"patch_embeddings": {}}
        parameters["patch_embeddings"] = {"projection": {}}
        parameters["patch_embeddings"]["projection"]["weight"] = ttnn.from_torch(
            preprocessed_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        parameters["patch_embeddings"]["projection"]["bias"] = ttnn.from_torch(
            bias.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
    elif isinstance(torch_model, attention_class):
        num_heads = 16
        qkv_weight, qkv_bias, proj_weight, proj_bias = (
            torch_model.qkv.weight,
            torch_model.qkv.bias,
            torch_model.proj.weight,
            torch_model.proj.bias,
        )
        qkv_weight, qkv_bias, proj_weight, proj_bias = upchannel_attn_weight_bias(
            qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads
        )
        parameters = {"query_key_value": {}, "proj": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)
        parameters["proj"]["weight"] = preprocess_linear_weight(proj_weight, dtype=ttnn.bfloat8_b)
        parameters["proj"]["bias"] = preprocess_linear_bias(proj_bias, dtype=ttnn.bfloat8_b)
    elif isinstance(torch_model, torch.nn.Linear):
        parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat8_b)
        parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat8_b)

    return parameters
