# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from torch import nn
from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask


def transpose_for_scores(config, x, device, permute_tensor: bool):
    new_x_shape = (x.shape[0], config.num_attention_heads, config.attention_head_size, x.shape[-1])
    x = ttnn.from_device(x)
    x = ttnn.reshape(x, new_x_shape)
    x = ttnn.to_device(x, device)
    if permute_tensor:
        x = ttnn.permute(x, (0, 1, 3, 2))

    return x


def transpose_output(config, x, device):
    all_head_size = config.num_attention_heads * config.attention_head_size
    if len(x.shape) == 4:
        x = ttnn.permute(x, (0, 1, 3, 2))

    new_x_shape = (x.shape[0], all_head_size, x.shape[3])
    x = ttnn.reshape(x, new_x_shape)

    return x


def squeezebert_conv_layernorm(
    config,
    hidden_states,
    input_tensor,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    cin,
    cout,
    groups,
):
    torch_hidden_states = ttnn.to_torch(hidden_states).to(torch.float32)

    self_output_conv1d_ = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
    self_output_conv1d_.weight = nn.Parameter(state_dict[f"{base_addr}conv1d.weight"])
    self_output_conv1d_.bias = nn.Parameter(state_dict[f"{base_addr}conv1d.bias"])

    torch_self_output = self_output_conv1d_(torch_hidden_states)
    self_output = ttnn.from_torch(torch_self_output, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    self_output_layernorm = ttnn.add(self_output, input_tensor)

    bs, *_ = self_output_layernorm.shape
    self_output_layernorm = ttnn.permute(self_output_layernorm, (0, 2, 1))
    self_output_layernorm = ttnn.reshape(
        self_output_layernorm, (bs, self_output_layernorm.shape[-2], self_output_layernorm.shape[-1])
    )
    attention_output = ttnn.layer_norm(
        self_output_layernorm,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=config.layer_norm_eps,
    )
    bs, *_ = attention_output.shape
    attention_output = ttnn.permute(attention_output, (0, 2, 1))
    attention_output = ttnn.reshape(attention_output, (bs, attention_output.shape[-2], attention_output.shape[-1]))
    return attention_output


def squeezebert_attention(
    config,
    hidden_states,
    attention_mask,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    reader_patterns_cache,
    num_cores_x=12,
):
    num_heads = config.num_attention_heads
    batch_size, hidden_size, _ = hidden_states.shape
    head_size = hidden_size // num_heads
    config.attention_head_size = head_size

    torch_hidden_states = ttnn.to_torch(hidden_states).to(torch.float32)

    query_layer = nn.Conv1d(
        in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1, groups=config.q_groups
    )
    query_layer.weight = nn.Parameter(state_dict[f"{base_addr}query.weight"])
    query_layer.bias = nn.Parameter(state_dict[f"{base_addr}query.bias"])

    key_layer = nn.Conv1d(
        in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1, groups=config.k_groups
    )
    key_layer.weight = nn.Parameter(state_dict[f"{base_addr}key.weight"])
    key_layer.bias = nn.Parameter(state_dict[f"{base_addr}key.bias"])

    value_layer = nn.Conv1d(
        in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1, groups=config.v_groups
    )
    value_layer.weight = nn.Parameter(state_dict[f"{base_addr}value.weight"])
    value_layer.bias = nn.Parameter(state_dict[f"{base_addr}value.bias"])

    mixed_query_layer = query_layer(torch_hidden_states)
    mixed_key_layer = key_layer(torch_hidden_states)
    mixed_value_layer = value_layer(torch_hidden_states)

    mixed_query_layer = ttnn.from_torch(mixed_query_layer, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    mixed_key_layer = ttnn.from_torch(mixed_key_layer, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    mixed_value_layer = ttnn.from_torch(mixed_value_layer, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    query = transpose_for_scores(config, mixed_query_layer, device, True)
    key = transpose_for_scores(config, mixed_key_layer, device, False)
    value = transpose_for_scores(config, mixed_value_layer, device, True)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probs = ttnn.transformer.attention_softmax_(
        attention_scores, attention_mask=attention_mask, head_size=head_size
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    context_layer = transpose_output(config, context_layer, device)

    return context_layer


def squeezebert_intermediate(
    config,
    hidden_states,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    num_cores_x=12,
):
    torch_hidden_states = ttnn.to_torch(hidden_states).to(torch.float32)

    torch_conv_ = nn.Conv1d(
        in_channels=config.hidden_size,
        out_channels=config.intermediate_size,
        kernel_size=1,
        groups=config.intermediate_groups,
    )
    torch_conv_.weight = nn.Parameter(state_dict[f"{base_addr}conv1d.weight"])
    torch_conv_.bias = nn.Parameter(state_dict[f"{base_addr}conv1d.bias"])

    torch_conv_output = torch_conv_(torch_hidden_states)
    ttnn_conv_output = ttnn.from_torch(torch_conv_output, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.gelu(ttnn_conv_output)
    return output


def squeezebert_layer(
    config,
    hidden_states,
    attention_mask,
    state_dict,
    base_addr,
    parameters,
    device,
    reader_patterns_cache,
):
    multi_head_attention_output = squeezebert_attention(
        config,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        state_dict=state_dict,
        base_addr=f"{base_addr}attention.",
        parameters=parameters.attention,
        device=device,
        reader_patterns_cache=reader_patterns_cache,
    )

    attention_output = squeezebert_conv_layernorm(
        config,
        hidden_states=multi_head_attention_output,
        input_tensor=hidden_states,
        state_dict=state_dict,
        base_addr=f"{base_addr}post_attention.",
        parameters=parameters.post_attention,
        device=device,
        cin=config.hidden_size,
        cout=config.hidden_size,
        groups=config.post_attention_groups,
    )

    intermediate = squeezebert_intermediate(
        config,
        attention_output,
        state_dict=state_dict,
        base_addr=f"{base_addr}intermediate.",
        parameters=parameters.intermediate,
        device=device,
    )

    output = squeezebert_conv_layernorm(
        config,
        hidden_states=intermediate,
        input_tensor=attention_output,
        state_dict=state_dict,
        base_addr=f"{base_addr}output.",
        parameters=parameters.output,
        device=device,
        cin=config.intermediate_size,
        cout=config.hidden_size,
        groups=config.output_groups,
    )

    return output


def squeezebert_encoder(
    config,
    hidden_states,
    attention_mask,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    reader_patterns_cache,
):
    bs, *_ = hidden_states.shape
    hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
    hidden_states = ttnn.reshape(hidden_states, (bs, hidden_states.shape[-2], hidden_states.shape[-1]))

    encoder_output = None

    for layer_idx, encoder_parameters in enumerate(parameters.layers):
        encoder_output = squeezebert_layer(
            config,
            hidden_states,
            attention_mask,
            state_dict,
            base_addr=f"{base_addr}layers.{layer_idx}.",
            parameters=encoder_parameters,
            device=device,
            reader_patterns_cache=reader_patterns_cache,
        )
        encoder_output = ttnn.reallocate(encoder_output)
        hidden_states = encoder_output

    bs, *_ = hidden_states.shape
    hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
    hidden_states = ttnn.reshape(hidden_states, (bs, hidden_states.shape[-2], hidden_states.shape[-1]))

    return hidden_states


def squeezebert(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    state_dict,
    base_addr,
    parameters,
    device,
    reader_patterns_cache,
):
    word_embeddings = ttnn.embedding(
        input_ids,
        parameters.embeddings.word_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
        pad_token=config.pad_token_id,
    )
    ttnn.deallocate(input_ids)

    token_type_embeddings = ttnn.embedding(
        token_type_ids,
        parameters.embeddings.token_type_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.deallocate(token_type_ids)

    word_plus_token_type_embeddings = word_embeddings + token_type_embeddings
    ttnn.deallocate(word_embeddings)
    ttnn.deallocate(token_type_embeddings)

    position_embeddings = ttnn.embedding(
        position_ids,
        parameters.embeddings.position_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.deallocate(position_ids)

    embeddings = word_plus_token_type_embeddings + position_embeddings
    ttnn.deallocate(word_plus_token_type_embeddings)
    ttnn.deallocate(position_embeddings)

    encoder_input = ttnn.layer_norm(
        embeddings,
        weight=parameters.embeddings.LayerNorm.weight,
        bias=parameters.embeddings.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(embeddings)

    encoder_output = squeezebert_encoder(
        config=config,
        hidden_states=encoder_input,
        attention_mask=attention_mask,
        state_dict=state_dict,
        base_addr=f"{base_addr}encoder.",
        parameters=parameters.encoder,
        device=device,
        reader_patterns_cache=reader_patterns_cache,
    )

    return encoder_output


def squeezebert_for_question_answering(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    reader_patterns_cache,
    name="transformer",
):
    squeezebert_output = squeezebert(
        config,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        state_dict,
        base_addr,
        parameters=parameters[name],
        device=device,
        reader_patterns_cache=reader_patterns_cache,
    )
    qa_outputs = ttnn.linear(
        squeezebert_output,
        parameters.qa_outputs.weight,
        bias=parameters.qa_outputs.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return qa_outputs


def preprocess_inputs(
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device,
):
    import torch

    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    if attention_mask is not None:
        attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, torch.float32)
        attention_mask = attention_mask.expand((batch_size, -1, -1, -1))
        attention_mask = torch.clamp(attention_mask, min=-100000)
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    return input_ids, token_type_ids, position_ids, attention_mask


def preprocess_conv_parameters(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(torch_model, name, ttnn_module_args):
    from torch import nn
    from ttnn.model_preprocessing import (
        preprocess_linear_bias,
        preprocess_linear_weight,
    )

    parameters = {}

    if isinstance(torch_model, nn.Conv1d):
        parameters["weight"] = preprocess_conv_parameters(torch_model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_conv_parameters(torch_model.bias, dtype=ttnn.bfloat16)

    if isinstance(torch_model, nn.Linear):
        parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat16)

    return parameters
