# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

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
    query = torch.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = torch.permute(query, (0, 2, 1, 3))

    key = hidden_states @ parameters.self.key.weight
    key = key + parameters.self.key.bias
    key = torch.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = torch.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters.self.value.weight
    value = value + parameters.self.value.bias
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

    attention_output = torch.nn.functional.layer_norm(
        hidden_states + self_output,
        (hidden_size,),
        parameters.output.LayerNorm.weight,
        parameters.output.LayerNorm.bias,
        eps=config.layer_norm_eps,
    )

    return attention_output


def bert_intermediate(hidden_states, *, parameters):
    hidden_states = hidden_states @ parameters.dense.weight
    hidden_states = hidden_states + parameters.dense.bias
    hidden_states = torch.nn.functional.gelu(hidden_states)
    return hidden_states


def bert_output(config, hidden_states, residual, *, parameters):
    output = hidden_states @ parameters.dense.weight
    output = output + parameters.dense.bias

    *_, hidden_size = residual.shape
    output = torch.nn.functional.layer_norm(
        output + residual,
        (hidden_size,),
        parameters.LayerNorm.weight,
        parameters.LayerNorm.bias,
        eps=config.layer_norm_eps,
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
    input_shape = input_ids.size()
    if attention_mask is not None:
        attention_mask = get_extended_attention_mask(attention_mask, input_shape)

    word_embeddings = torch.nn.functional.embedding(input_ids, parameters.embeddings.word_embeddings.weight)
    token_type_embeddings = torch.nn.functional.embedding(
        token_type_ids, parameters.embeddings.token_type_embeddings.weight
    )
    position_embeddings = torch.nn.functional.embedding(position_ids, parameters.embeddings.position_embeddings.weight)
    hidden_states = word_embeddings + token_type_embeddings + position_embeddings

    *_, hidden_size = hidden_states.shape
    hidden_states = torch.nn.functional.layer_norm(
        hidden_states,
        (hidden_size,),
        parameters.embeddings.LayerNorm.weight,
        parameters.embeddings.LayerNorm.bias,
        eps=config.layer_norm_eps,
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
