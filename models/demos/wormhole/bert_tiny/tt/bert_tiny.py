# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask


def bert_attention(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = ttnn.linear(
        hidden_states,
        parameters.self.query.weight,
        bias=parameters.self.query.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    query = ttnn.from_device(query)
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)
    query = ttnn.to_device(query, device)
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = ttnn.linear(
        hidden_states,
        parameters.self.key.weight,
        bias=parameters.self.key.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = ttnn.from_device(key)
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = ttnn.to_device(key, device)
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = ttnn.linear(
        hidden_states,
        parameters.self.value.weight,
        bias=parameters.self.value.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.from_device(value)
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
    value = ttnn.to_device(value, device)
    value = ttnn.permute(value, (0, 2, 1, 3))

    attention_scores = ttnn.matmul(query, key)
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = ttnn.to_layout(attention_scores, ttnn.TILE_LAYOUT)
        attention_scores = ttnn.to_device(attention_scores, device)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT)
        attention_scores = attention_scores + attention_mask
        value = ttnn.to_device(value, device)

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.from_device(context_layer)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_device(context_layer, device)
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    self_output = context_layer
    self_output = ttnn.linear(
        self_output,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    attention_output = ttnn.layer_norm(
        hidden_states + self_output,
        weight=parameters.output.LayerNorm.weight,
        bias=parameters.output.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return attention_output


def bert_intermediate(
    hidden_states,
    device=None,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        activation="gelu",
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    return output


def bert_output(
    config,
    hidden_states,
    residual,
    device=None,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    output = ttnn.layer_norm(
        output + residual,
        weight=parameters.LayerNorm.weight,
        bias=parameters.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return output


def bert_feedforward(
    config,
    hidden_states,
    device=None,
    *,
    parameters,
):
    intermediate = bert_intermediate(hidden_states, parameters=parameters.intermediate, device=device)
    hidden_states = bert_output(config, intermediate, hidden_states, parameters=parameters.output, device=device)
    return hidden_states


def bert_layer(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    attention_output = bert_attention(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.attention,
        device=device,
    )

    feedforward_output = bert_feedforward(
        config,
        attention_output,
        parameters=parameters,
        device=device,
    )

    return feedforward_output


def bert_encoder(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    encoder_input = hidden_states
    encoder_output = None
    for encoder_parameters in parameters.layer:
        encoder_output = bert_layer(
            config,
            encoder_input,
            attention_mask,
            parameters=encoder_parameters,
            device=device,
        )
        encoder_input = encoder_output
    return encoder_output


def bert(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device=None,
    *,
    parameters,
):
    word_embeddings = ttnn.embedding(input_ids, parameters.embeddings.word_embeddings.weight)
    token_type_embeddings = ttnn.embedding(token_type_ids, parameters.embeddings.token_type_embeddings.weight)
    position_embeddings = ttnn.embedding(position_ids, parameters.embeddings.position_embeddings.weight)
    word_embeddings = ttnn.to_layout(word_embeddings, ttnn.TILE_LAYOUT)
    token_type_embeddings = ttnn.to_layout(token_type_embeddings, ttnn.TILE_LAYOUT)
    position_embeddings = ttnn.to_layout(position_embeddings, ttnn.TILE_LAYOUT)

    embeddings = word_embeddings + token_type_embeddings + position_embeddings

    hidden_states = ttnn.layer_norm(
        embeddings,
        weight=parameters.embeddings.LayerNorm.weight,
        bias=parameters.embeddings.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    hidden_states = bert_encoder(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.encoder,
        device=device,
    )

    return hidden_states


def bert_for_question_answering(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device=None,
    *,
    parameters,
    name="bert",
):
    bert_output = bert(
        config,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        device=device,
        parameters=parameters[name],
    )

    qa_outputs = bert_output
    qa_outputs = ttnn.linear(
        qa_outputs,
        parameters.qa_outputs.weight,
        bias=parameters.qa_outputs.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    return qa_outputs


def preprocess_inputs(
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    mesh_device,
    inputs_mesh_mapper,
):
    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(
        input_ids,
        dtype=ttnn.uint32,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
        device=mesh_device,
    )
    token_type_ids = ttnn.from_torch(
        token_type_ids,
        dtype=ttnn.uint32,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
        device=mesh_device,
    )
    position_ids = ttnn.from_torch(
        position_ids,
        dtype=ttnn.uint32,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
        device=mesh_device,
    )

    if attention_mask is not None:
        attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, torch.float32)
        attention_mask = attention_mask.expand((batch_size, -1, -1, -1))
        attention_mask = torch.clamp(attention_mask, min=-100000)
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=inputs_mesh_mapper,
            device=mesh_device,
        )

    return input_ids, token_type_ids, position_ids, attention_mask


def custom_preprocessor(torch_model, name):
    return {}
