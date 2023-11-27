# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def ttnn_optimized_multi_head_attention(
    hidden_states,
    attention_mask,
    fused_qkv_weight,
    fused_qkv_bias,
    self_output_weight,
    self_output_bias,
    *,
    head_size,
    num_cores_x=12,
):
    batch_size, *_ = hidden_states.shape

    fused_qkv_output = ttnn.matmul(
        hidden_states,
        fused_qkv_weight,
        bias=fused_qkv_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=(batch_size, num_cores_x),
    )

    (
        query,
        key,
        value,
    ) = ttnn.nlp.split_fused_qkv_and_split_heads(
        fused_qkv_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.free(fused_qkv_output)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.free(query)
    ttnn.free(key)

    attention_probs = ttnn.nlp.attention_softmax_(attention_scores, attention_mask=attention_mask, head_size=head_size)

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.free(attention_probs)

    context_layer = ttnn.nlp.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=(batch_size, num_cores_x),
    )

    self_output = ttnn.matmul(
        context_layer,
        self_output_weight,
        bias=self_output_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.free(context_layer)

    return self_output


def ttnn_optimized_feedforward(hidden_states, ff1_weight, ff1_bias, ff2_weight, ff2_bias):
    import tt_lib as ttl

    batch_size, sequence_size, hidden_size = hidden_states.shape

    hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, sequence_size, hidden_size))
    ff1_weight = ttnn.reshape(ff1_weight, (1, 1, hidden_size, hidden_size * 4))
    ff1_bias = ttnn.reshape(ff1_bias, (1, 1, 32, hidden_size * 4))
    ff2_weight = ttnn.reshape(ff2_weight, (1, 1, hidden_size * 4, hidden_size))
    ff2_bias = ttnn.reshape(ff2_bias, (1, 1, 32, hidden_size))

    hidden_states = hidden_states._tensor
    ff1_weight = ff1_weight._tensor
    ff1_bias = ff1_bias._tensor
    ff2_weight = ff2_weight._tensor
    ff2_bias = ff2_bias._tensor

    ff1_output = ttl.operations.primary.matmul(
        hidden_states,
        ff1_weight,
        bias=ff1_bias,
        program_config=(
            ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(12, batch_size),
                in0_block_w=4,
                out_subblock_h=6,
                out_subblock_w=1,
                per_core_M=12,
                per_core_N=11,
                transpose_mcast=False,
                fused_activation=(ttl.tensor.FusibleActivation.GELU, True),
            )
        ),
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
        output_dtype=ttnn.bfloat8_b,
    )

    ff2_output = ttl.operations.primary.matmul(
        ff1_output,
        ff2_weight,
        bias=ff2_bias,
        program_config=(
            ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(12, batch_size),
                in0_block_w=4,
                out_subblock_h=6,
                out_subblock_w=1,
                per_core_M=12,
                per_core_N=3,
                transpose_mcast=False,
                fused_activation=None,
            )
        ),
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
        output_dtype=ttnn.bfloat16,
    )
    ff1_output.deallocate()

    ff2_output = ttnn.Tensor(ff2_output)
    ff2_output = ttnn.reshape(ff2_output, (batch_size, sequence_size, hidden_size))

    return ff2_output


def ttnn_optimized_bert_encoder(
    hidden_states,
    attention_mask,
    parameters,
    *,
    encoder_index,
    head_size,
):
    multi_head_attention_output = ttnn_optimized_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.fused_qkv.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.self.fused_qkv.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias"],
        head_size=head_size,
    )

    multi_head_attention_add_and_layer_norm_output = ttnn.experimental.layer_norm(
        hidden_states,
        residual_input=multi_head_attention_output,
        weight=parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
        bias=parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.free(hidden_states)
    ttnn.free(multi_head_attention_output)

    feedforward_output = ttnn_optimized_feedforward(
        multi_head_attention_add_and_layer_norm_output,
        parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.dense.bias"],
    )

    feedforward_add_and_layer_norm_output = ttnn.experimental.layer_norm(
        multi_head_attention_add_and_layer_norm_output,
        residual_input=feedforward_output,
        weight=parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight"],
        bias=parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.free(multi_head_attention_add_and_layer_norm_output)
    ttnn.free(feedforward_output)

    return feedforward_add_and_layer_norm_output


def ttnn_optimized_bert(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
):
    import tt_lib as ttl

    word_embeddings = ttnn.embedding(
        input_ids, parameters["bert.embeddings.word_embeddings.weight"], layout=ttnn.TILE_LAYOUT
    )
    ttnn.free(input_ids)

    token_type_embeddings = ttnn.embedding(
        token_type_ids, parameters["bert.embeddings.token_type_embeddings.weight"], layout=ttnn.TILE_LAYOUT
    )
    ttnn.free(token_type_ids)

    embeddings = word_embeddings + token_type_embeddings
    ttnn.free(word_embeddings)
    ttnn.free(token_type_embeddings)

    encoder_input = ttnn.experimental.layer_norm(
        embeddings,
        weight=parameters[f"bert.embeddings.LayerNorm.weight"],
        bias=parameters[f"bert.embeddings.LayerNorm.bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.free(embeddings)

    encoder_output = None
    for encoder_index in range(num_encoders):
        encoder_input = ttnn.Tensor(ttl.tensor.move(encoder_input._tensor))
        encoder_output = ttnn_optimized_bert_encoder(
            encoder_input,
            attention_mask,
            parameters,
            encoder_index=encoder_index,
            head_size=head_size,
        )
        encoder_input = encoder_output

    encoder_output = ttnn.Tensor(ttl.tensor.move(encoder_output._tensor))
    return encoder_output


def ttnn_optimized_bert_for_question_answering(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
):
    import tt_lib as ttl

    bert_output = ttnn_optimized_bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        num_encoders=num_encoders,
        head_size=head_size,
    )

    qa_outputs = bert_output
    qa_outputs_weight = parameters["qa_outputs.weight"]
    qa_outputs_bias = parameters["qa_outputs.bias"]

    batch_size, sequence_size, hidden_size = qa_outputs.shape
    qa_outputs = ttnn.reshape(qa_outputs, (batch_size, 1, sequence_size, hidden_size))
    qa_outputs_weight = ttnn.reshape(qa_outputs_weight, (1, 1, hidden_size, 32))
    qa_outputs_bias = ttnn.reshape(qa_outputs_bias, (1, 1, 32, 32))

    qa_outputs = qa_outputs._tensor
    qa_outputs_weight = qa_outputs_weight._tensor
    qa_outputs_bias = qa_outputs_bias._tensor

    qa_outputs = ttl.operations.primary.matmul(
        qa_outputs,
        qa_outputs_weight,
        bias=qa_outputs_bias,
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
    )

    qa_outputs = ttnn.Tensor(qa_outputs)
    qa_outputs = ttnn.reshape(qa_outputs, (batch_size, sequence_size, 32))

    return qa_outputs
