# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from ttnn.dot_access import DotAccessDict

from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask

"""
from models.demos.metal_BERT_large_11.tt.model_config import get_model_config
model_config = get_model_config(12, ttnn.CoreGrid(y=8, x=12), "BFLOAT8_B-SHARDED")
for key, value in model_config.items():
    print(f"{key}: {value}")
"""


def update_model_config(config, batch_size):
    core_grid = ttnn.CoreGrid(y=8, x=batch_size)

    program_configs = {
        "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=6,
            per_core_M=12,
            per_core_N=12,
            transpose_mcast=True,
            fused_activation=None,
        ),
        "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            per_core_M=24,
            per_core_N=12,
        ),
        "attention_probabilities_by_value_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=12,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=24,
            per_core_N=2,
        ),
        "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=4,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=12,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=12,
            per_core_N=16,
            transpose_mcast=True,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=16,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=12,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        "layernorm_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=4,
            block_h=12,
            block_w=4,
            inplace=True,
        ),
        "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=6,
            block_h=24,
            block_w=12,
        ),
    }

    return DotAccessDict(dict(**config.to_dict(), core_grid=core_grid, program_configs=program_configs))


def bert_attention(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    num_heads = config.num_attention_heads
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.self.query_key_value.weight,
        bias=parameters.self.query_key_value.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["query_key_value_matmul_program_config"],
    )

    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["query_by_key_matmul_program_config"],
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probabilities = ttnn.transformer.attention_softmax_(
        attention_scores,
        attention_mask=attention_mask,
        head_size=head_size,
        program_config=config.program_configs["softmax_program_config"],
    )

    context_layer = ttnn.matmul(
        attention_probabilities,
        value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probabilities)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)

    attention_output = ttnn.layer_norm(
        hidden_states,
        residual_input_tensor=self_output,
        weight=parameters.output.LayerNorm.weight,
        bias=parameters.output.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )
    ttnn.deallocate(self_output)

    return attention_output


def bert_intermediate(
    config,
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff1_matmul_program_config"],
    )
    return output


def bert_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    normalized_output = ttnn.layer_norm(
        output,
        residual_input_tensor=residual,
        weight=parameters.LayerNorm.weight,
        bias=parameters.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )
    ttnn.deallocate(residual)

    return normalized_output


def bert_feedforward(
    config,
    hidden_states,
    *,
    parameters,
):
    intermediate = bert_intermediate(config, hidden_states, parameters=parameters.intermediate)
    hidden_states = bert_output(config, intermediate, hidden_states, parameters=parameters.output)
    return hidden_states


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
        parameters=parameters.attention,
    )

    feedforward_output = bert_feedforward(
        config,
        multi_head_attention_output,
        parameters=parameters,
    )

    return feedforward_output


def bert(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    parameters,
):
    word_embeddings = ttnn.embedding(
        input_ids,
        parameters.embeddings.word_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(input_ids)

    token_type_embeddings = ttnn.embedding(
        token_type_ids,
        parameters.embeddings.token_type_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(token_type_ids)

    word_plus_token_type_embeddings = ttnn.add(
        word_embeddings, token_type_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.deallocate(word_embeddings)
    ttnn.deallocate(token_type_embeddings)

    position_embeddings = ttnn.embedding(
        position_ids,
        parameters.embeddings.position_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(position_ids)

    embeddings = ttnn.layer_norm(
        word_plus_token_type_embeddings,
        residual_input_tensor=position_embeddings,
        weight=parameters.embeddings.LayerNorm.weight,
        bias=parameters.embeddings.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(word_plus_token_type_embeddings)
    ttnn.deallocate(position_embeddings)

    encoder_input = ttnn.to_memory_config(
        embeddings,
        memory_config=ttnn.create_sharded_memory_config(
            embeddings.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
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
        encoder_input = encoder_output

    return encoder_output


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
        parameters[name],
    )

    bert_output = ttnn.to_memory_config(bert_output, memory_config=ttnn.L1_MEMORY_CONFIG)

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
        attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape)
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
    import torch
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
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)
    elif isinstance(torch_model, torch.nn.Linear):
        parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat8_b)
        parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat8_b)
    return parameters
