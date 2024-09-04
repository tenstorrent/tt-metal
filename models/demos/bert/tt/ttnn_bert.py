# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask


def bert_attention(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    fallback_reshape = ttnn.get_fallback_function(ttnn.reshape)

    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = hidden_states @ parameters.self.query.weight
    query = query + parameters.self.query.bias
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    query = fallback_reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = hidden_states @ parameters.self.key.weight
    key = key + parameters.self.key.bias
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = fallback_reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters.self.value.weight
    value = value + parameters.self.value.bias
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = fallback_reshape(value, (batch_size, sequence_size, num_heads, head_size))
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
    context_layer = fallback_reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    self_output = context_layer
    self_output = self_output @ parameters.output.dense.weight
    self_output = self_output + parameters.output.dense.bias

    attention_output = ttnn.layer_norm(
        hidden_states + self_output,
        weight=parameters.output.LayerNorm.weight,
        bias=parameters.output.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
    )

    return attention_output


def bert_intermediate(
    hidden_states,
    *,
    parameters,
):
    output = hidden_states @ parameters.dense.weight
    output = output + parameters.dense.bias
    output = ttnn.gelu(output)
    return output


def bert_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    output = hidden_states @ parameters.dense.weight
    output = output + parameters.dense.bias

    output = ttnn.layer_norm(
        output + residual,
        weight=parameters.LayerNorm.weight,
        bias=parameters.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
    )

    return output


def bert_feedforward(
    config,
    hidden_states,
    *,
    parameters,
):
    intermediate = bert_intermediate(hidden_states, parameters=parameters.intermediate)
    hidden_states = bert_output(config, intermediate, hidden_states, parameters=parameters.output)
    return hidden_states


def bert_layer(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    attention_output = bert_attention(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.attention,
    )

    feedforward_output = bert_feedforward(
        config,
        attention_output,
        parameters=parameters,
    )

    return feedforward_output


def bert_encoder(
    config,
    hidden_states,
    attention_mask,
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
        )
        encoder_input = encoder_output
    return encoder_output


def bert(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    *,
    parameters,
):
    word_embeddings = ttnn.embedding(input_ids, parameters.embeddings.word_embeddings.weight, layout=ttnn.TILE_LAYOUT)
    token_type_embeddings = ttnn.embedding(
        token_type_ids, parameters.embeddings.token_type_embeddings.weight, layout=ttnn.TILE_LAYOUT
    )
    position_embeddings = ttnn.embedding(
        position_ids, parameters.embeddings.position_embeddings.weight, layout=ttnn.TILE_LAYOUT
    )
    embeddings = word_embeddings + token_type_embeddings + position_embeddings

    hidden_states = ttnn.layer_norm(
        embeddings,
        weight=parameters.embeddings.LayerNorm.weight,
        bias=parameters.embeddings.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
    )

    hidden_states = bert_encoder(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.encoder,
    )

    return hidden_states


def bert_for_question_answering(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
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
        parameters=parameters[name],
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters.qa_outputs.weight
    qa_outputs = qa_outputs + parameters.qa_outputs.bias

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


def custom_preprocessor(torch_model, name):
    return {}
