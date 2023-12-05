# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def torch_multi_head_attention(
    hidden_states,
    attention_mask,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    output_weight,
    output_bias,
    *,
    head_size,
):
    batch_size, sequence_size, hidden_size = hidden_states.shape
    num_heads = hidden_size // head_size

    query = hidden_states @ query_weight
    query = query + query_bias
    query = torch.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = torch.permute(query, (0, 2, 1, 3))

    key = hidden_states @ key_weight
    key = key + key_bias
    key = torch.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = torch.permute(key, (0, 2, 3, 1))

    value = hidden_states @ value_weight
    value = value + value_bias
    value = torch.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = torch.permute(value, (0, 2, 1, 3))

    attention_scores = query @ key
    attention_scores = attention_scores / (head_size**0.5)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = F.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value
    context_layer = torch.permute(context_layer, (0, 2, 1, 3))
    context_layer = torch.reshape(context_layer, (batch_size, sequence_size, hidden_size))

    self_output = context_layer
    self_output = self_output @ output_weight
    self_output = self_output + output_bias

    return self_output


def torch_feedforward(hidden_states, intermediate_weight, intermediate_bias, output_weight, output_bias):
    hidden_states = hidden_states @ intermediate_weight
    hidden_states = hidden_states + intermediate_bias
    hidden_states = F.gelu(hidden_states)
    hidden_states = hidden_states @ output_weight
    hidden_states = hidden_states + output_bias
    return hidden_states


def torch_bert_encoder(
    hidden_states,
    attention_mask,
    parameters,
    *,
    head_size,
):
    *_, hidden_size = hidden_states.shape
    multi_head_attention_output = torch_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters.attention.self.query.weight,
        parameters.attention.self.query.bias,
        parameters.attention.self.key.weight,
        parameters.attention.self.key.bias,
        parameters.attention.self.value.weight,
        parameters.attention.self.value.bias,
        parameters.attention.output.dense.weight,
        parameters.attention.output.dense.bias,
        head_size=head_size,
    )

    multi_head_attention_add_and_layer_norm_output = F.layer_norm(
        hidden_states + multi_head_attention_output,
        (hidden_size,),
        parameters.attention.output.LayerNorm.weight,
        parameters.attention.output.LayerNorm.bias,
    )

    feedforward_output = torch_feedforward(
        multi_head_attention_add_and_layer_norm_output,
        parameters.intermediate.dense.weight,
        parameters.intermediate.dense.bias,
        parameters.output.dense.weight,
        parameters.output.dense.bias,
    )

    feedforward_add_and_layer_norm_output = F.layer_norm(
        multi_head_attention_add_and_layer_norm_output + feedforward_output,
        (hidden_size,),
        parameters.output.LayerNorm.weight,
        parameters.output.LayerNorm.bias,
    )

    return feedforward_add_and_layer_norm_output


def torch_bert(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    head_size,
):
    word_embeddings = F.embedding(input_ids, parameters.bert.embeddings.word_embeddings.weight)
    token_type_embeddings = F.embedding(token_type_ids, parameters.bert.embeddings.token_type_embeddings.weight)
    encoder_input = word_embeddings + token_type_embeddings

    *_, hidden_size = encoder_input.shape
    encoder_input = F.layer_norm(
        encoder_input,
        (hidden_size,),
        parameters.bert.embeddings.LayerNorm.weight,
        parameters.bert.embeddings.LayerNorm.bias,
    )

    encoder_output = None
    for encoder_parameters in parameters.bert.encoder.layer:
        encoder_output = torch_bert_encoder(
            encoder_input,
            attention_mask,
            encoder_parameters,
            head_size=head_size,
        )
        encoder_input = encoder_output
    return encoder_output


def torch_bert_for_question_answering(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    head_size,
):
    bert_output = torch_bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        head_size=head_size,
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters.qa_outputs.weight
    qa_outputs = qa_outputs + parameters.qa_outputs.bias
    return qa_outputs
