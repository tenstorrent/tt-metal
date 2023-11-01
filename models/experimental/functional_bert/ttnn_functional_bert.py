# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def ttnn_multi_head_attention(
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
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = hidden_states @ key_weight
    key = key + key_bias
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = hidden_states @ value_weight
    value = value + value_bias
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
    self_output = self_output @ output_weight
    self_output = self_output + output_bias

    return self_output


def ttnn_feedforward(
    hidden_states,
    ff1_weight,
    ff1_bias,
    ff2_weight,
    ff2_bias,
):
    hidden_states = hidden_states @ ff1_weight
    hidden_states = hidden_states + ff1_bias
    hidden_states = ttnn.experimental.gelu(hidden_states)
    hidden_states = hidden_states @ ff2_weight
    hidden_states = hidden_states + ff2_bias
    return hidden_states


def ttnn_bert_encoder(
    hidden_states,
    attention_mask,
    parameters,
    *,
    encoder_index,
    head_size,
):
    multi_head_attention_output = ttnn_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias"],
        head_size=head_size,
    )

    hidden_states = ttnn.experimental.layer_norm(
        hidden_states + multi_head_attention_output,
        weight=parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
        bias=parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
    )

    feedforward_output = ttnn_feedforward(
        hidden_states,
        parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.dense.bias"],
    )

    hidden_states = ttnn.experimental.layer_norm(
        hidden_states + feedforward_output,
        weight=parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight"],
        bias=parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias"],
    )

    return hidden_states


def ttnn_bert(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
):
    word_embeddings = ttnn.embedding(input_ids, parameters["bert.embeddings.word_embeddings.weight"])
    token_type_embeddings = ttnn.embedding(token_type_ids, parameters["bert.embeddings.token_type_embeddings.weight"])
    encoder_input = word_embeddings + token_type_embeddings

    encoder_input = ttnn.experimental.layer_norm(
        encoder_input,
        weight=parameters[f"bert.embeddings.LayerNorm.weight"],
        bias=parameters[f"bert.embeddings.LayerNorm.bias"],
    )

    encoder_output = None
    for encoder_index in range(num_encoders):
        encoder_output = ttnn_bert_encoder(
            encoder_input,
            attention_mask,
            parameters,
            encoder_index=encoder_index,
            head_size=head_size,
        )
        encoder_input = encoder_output
    return encoder_output


def ttnn_bert_for_question_answering(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
):
    bert_output = ttnn_bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        num_encoders=num_encoders,
        head_size=head_size,
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters["qa_outputs.weight"]
    qa_outputs = qa_outputs + parameters["qa_outputs.bias"]

    return qa_outputs
