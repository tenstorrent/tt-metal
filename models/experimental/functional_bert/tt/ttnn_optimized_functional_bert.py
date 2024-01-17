# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def bert_attention(
    config,
    hidden_states,
    attention_mask,
    query_key_value_weight,
    query_key_value_bias,
    self_output_weight,
    self_output_bias,
    *,
    num_cores_x=12,
):
    num_heads = config.num_attention_heads
    batch_size, _, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value_output = ttnn.linear(
        hidden_states,
        query_key_value_weight,
        bias=query_key_value_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=(batch_size, num_cores_x),
    )

    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value_output)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=(batch_size, num_cores_x),
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
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        self_output_weight,
        bias=self_output_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.deallocate(context_layer)

    return self_output


def bert_feedforward(hidden_states, ff1_weight, ff1_bias, ff2_weight, ff2_bias, num_cores_x=12):
    batch_size, *_ = hidden_states.shape

    num_cores_x = 12
    ff1_output = ttnn.linear(
        hidden_states,
        ff1_weight,
        bias=ff1_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=(batch_size, num_cores_x),
        activation="gelu",
    )

    ff2_output = ttnn.linear(
        ff1_output,
        ff2_weight,
        bias=ff2_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.deallocate(ff1_output)

    return ff2_output


def bert_encoder(
    config,
    hidden_states,
    attention_mask,
    parameters,
):
    multi_head_attention_output = bert_attention(
        config,
        hidden_states,
        attention_mask,
        parameters.attention.self.query_key_value.weight,
        parameters.attention.self.query_key_value.bias,
        parameters.attention.output.dense.weight,
        parameters.attention.output.dense.bias,
    )

    multi_head_attention_add_and_layer_norm_output = ttnn.layer_norm(
        hidden_states,
        residual_input_tensor=multi_head_attention_output,
        weight=parameters.attention.output.LayerNorm.weight,
        bias=parameters.attention.output.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(hidden_states)
    ttnn.deallocate(multi_head_attention_output)

    feedforward_output = bert_feedforward(
        multi_head_attention_add_and_layer_norm_output,
        parameters.intermediate.dense.weight,
        parameters.intermediate.dense.bias,
        parameters.output.dense.weight,
        parameters.output.dense.bias,
    )

    feedforward_add_and_layer_norm_output = ttnn.layer_norm(
        multi_head_attention_add_and_layer_norm_output,
        residual_input_tensor=feedforward_output,
        weight=parameters.output.LayerNorm.weight,
        bias=parameters.output.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(multi_head_attention_add_and_layer_norm_output)
    ttnn.deallocate(feedforward_output)

    return feedforward_add_and_layer_norm_output


def bert(
    config,
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
):
    word_embeddings = ttnn.embedding(
        input_ids,
        parameters.embeddings.word_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.deallocate(input_ids)

    token_type_embeddings = ttnn.embedding(
        token_type_ids,
        parameters.embeddings.token_type_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.deallocate(token_type_ids)

    embeddings = word_embeddings + token_type_embeddings
    ttnn.deallocate(word_embeddings)
    ttnn.deallocate(token_type_embeddings)

    encoder_input = ttnn.layer_norm(
        embeddings,
        weight=parameters.embeddings.LayerNorm.weight,
        bias=parameters.embeddings.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(embeddings)

    encoder_output = None
    for encoder_parameters in parameters.encoder.layer:
        encoder_output = bert_encoder(
            config,
            encoder_input,
            attention_mask,
            encoder_parameters,
        )
        encoder_output = ttnn.reallocate(encoder_output)
        encoder_input = encoder_output

    return encoder_output


def bert_for_question_answering(
    config,
    input_ids,
    token_type_ids,
    attention_mask,
    *,
    parameters,
    name="bert",
):
    bert_output = bert(
        config,
        input_ids,
        token_type_ids,
        attention_mask,
        parameters[name],
    )

    qa_outputs = ttnn.linear(
        bert_output,
        parameters.qa_outputs.weight,
        bias=parameters.qa_outputs.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return qa_outputs


def preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    device,
):
    import torch

    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32)
    input_ids = ttnn.to_device(input_ids, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(token_type_ids, dtype=ttnn.uint32)
    token_type_ids = ttnn.to_device(token_type_ids, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 0, 0, 31, 0, 0, 0, batch_size - 1))
        attention_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT)
        attention_mask = ttnn.to_device(attention_mask, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    return input_ids, token_type_ids, attention_mask


def custom_preprocessor(torch_model, name):
    import torch
    import transformers
    from ttnn.model_preprocessing import (
        preprocess_linear_bias,
        preprocess_linear_weight,
    )

    parameters = {}
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
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16)
    return parameters
