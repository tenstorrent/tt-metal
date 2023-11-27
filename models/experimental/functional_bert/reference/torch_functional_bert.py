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
    encoder_index,
    head_size,
):
    *_, hidden_size = hidden_states.shape
    multi_head_attention_output = torch_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.weight"].T,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.weight"].T,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.weight"].T,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight"].T,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias"],
        head_size=head_size,
    )

    multi_head_attention_add_and_layer_norm_output = F.layer_norm(
        hidden_states + multi_head_attention_output,
        (hidden_size,),
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
    )

    feedforward_output = torch_feedforward(
        multi_head_attention_add_and_layer_norm_output,
        parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight"].T,
        parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.dense.weight"].T,
        parameters[f"bert.encoder.layer.{encoder_index}.output.dense.bias"],
    )

    feedforward_add_and_layer_norm_output = F.layer_norm(
        multi_head_attention_add_and_layer_norm_output + feedforward_output,
        (hidden_size,),
        parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias"],
    )

    return feedforward_add_and_layer_norm_output


def torch_bert(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
):
    word_embeddings = F.embedding(input_ids, parameters["bert.embeddings.word_embeddings.weight"])
    token_type_embeddings = F.embedding(token_type_ids, parameters["bert.embeddings.token_type_embeddings.weight"])
    encoder_input = word_embeddings + token_type_embeddings

    *_, hidden_size = encoder_input.shape
    encoder_input = F.layer_norm(
        encoder_input,
        (hidden_size,),
        parameters["bert.embeddings.LayerNorm.weight"],
        parameters["bert.embeddings.LayerNorm.bias"],
    )

    encoder_output = None
    for encoder_index in range(num_encoders):
        encoder_output = torch_bert_encoder(
            encoder_input,
            attention_mask,
            parameters,
            encoder_index=encoder_index,
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
    num_encoders,
    head_size,
):
    bert_output = torch_bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        num_encoders=num_encoders,
        head_size=head_size,
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters["qa_outputs.weight"].T
    qa_outputs = qa_outputs + parameters["qa_outputs.bias"]
    return qa_outputs
