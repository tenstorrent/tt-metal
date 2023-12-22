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
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = hidden_states @ parameters.self.query.weight
    query = query + parameters.self.query.bias
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = hidden_states @ parameters.self.key.weight
    key = key + parameters.self.key.bias
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters.self.value.weight
    value = value + parameters.self.value.bias
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.permute(value, (0, 2, 1, 3))

    attention_scores = query @ key
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))

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
    attention_mask,
    *,
    parameters,
):
    word_embeddings = ttnn.embedding(input_ids, parameters.embeddings.word_embeddings.weight, layout=ttnn.TILE_LAYOUT)
    token_type_embeddings = ttnn.embedding(
        token_type_ids, parameters.embeddings.token_type_embeddings.weight, layout=ttnn.TILE_LAYOUT
    )
    hidden_states = word_embeddings + token_type_embeddings

    hidden_states = ttnn.layer_norm(
        hidden_states,
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
    attention_mask,
    *,
    parameters,
):
    bert_output = bert(
        config,
        input_ids,
        token_type_ids,
        attention_mask,
        parameters=parameters.bert,
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters.qa_outputs.weight
    qa_outputs = qa_outputs + parameters.qa_outputs.bias

    return qa_outputs


def preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    device,
):
    import torch

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32)
    input_ids = ttnn.to_device(input_ids, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(token_type_ids, dtype=ttnn.uint32)
    token_type_ids = ttnn.to_device(token_type_ids, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    if attention_mask is not None:
        attention_mask = torch.zeros(input_ids.shape, dtype=torch.bfloat16)
        attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape)

        attention_mask = torch.nn.functional.pad(attention_mask, (0, 0, 0, 31))
        attention_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT)
        attention_mask = ttnn.to_device(attention_mask, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    return input_ids, token_type_ids, attention_mask


def custom_preprocessor(torch_model, name):
    return {}
