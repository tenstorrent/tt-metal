# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import ttnn.torch_tracer
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn


def get_head_mask(
    head_mask: Optional[ttnn.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
):
    head_mask = [
        None,
    ] * num_hidden_layers
    return head_mask


def attention(
    config,
    hidden_states,
    mask,
    head_mask=None,
    output_attentions=None,
    device=None,
    base_address=None,
    parameters=None,
    num_cores_x=12,
    min_val_tensor=None,
    negative_val_tensor=None,
    mesh_mapper=None,
):
    batch_size, q_length, dim = hidden_states.shape
    k_length = hidden_states.shape[1]
    dim_per_head = config.dim // config.n_heads

    query_key_value_output = ttnn.linear(
        hidden_states,
        parameters.query_key_value.weight,
        bias=parameters.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=config.n_heads,
    )
    ttnn.deallocate(query_key_value_output)

    query = query * (1 / (dim_per_head) ** 0.5)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)
    score_list = []

    if batch_size <= 2:
        inter_scores = attention_scores * negative_val_tensor
        inter_scores = inter_scores + attention_scores
        scores = inter_scores + min_val_tensor
    else:
        for i in range(2, batch_size + 1, 2):
            inter_scores = attention_scores[i - 2 : i, :, :, :] * negative_val_tensor[i - 2 : i, :, :, :]
            inter_scores = inter_scores + attention_scores[i - 2 : i, :, :, :]
            score = inter_scores + min_val_tensor[i - 2 : i, :, :, :]
            score = ttnn.permute(score, (1, 0, 2, 3))

            score_list.append(score)
            ttnn.deallocate(inter_scores)

        scores = ttnn.concat(score_list, dim=1)
        scores = ttnn.permute(scores, (1, 0, 2, 3))

    weights = ttnn.transformer.attention_softmax(scores, head_size=1)
    ttnn.deallocate(scores)
    context_layer = ttnn.matmul(
        weights,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    ttnn.deallocate(weights)
    ttnn.deallocate(value)

    context_layer = ttnn.permute(context_layer, [0, 1, 3, 2])
    context_layer = ttnn.reshape(context_layer, (batch_size, config.n_heads * dim_per_head, -1))

    context_layer = ttnn.permute(context_layer, (0, 2, 1))

    self_output = ttnn.linear(
        context_layer,
        parameters.out_lin.weight,
        bias=parameters.out_lin.bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(context_layer)
    return self_output


def ffn(configs, hidden_state, device, base_address, parameters, num_cores_x=12, mesh_mapper=None):
    batch_size, *_ = hidden_state.shape

    output = ttnn.linear(
        hidden_state,
        parameters.lin1.weight,
        bias=parameters.lin1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        activation="gelu",
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    output = ttnn.linear(
        output,
        parameters.lin2.weight,
        bias=parameters.lin2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )
    return output


def transformer_block(
    config,
    x,
    attention_mask=None,
    head_mask=None,
    output_attentions: bool = False,
    base_address=None,
    parameters=None,
    device=None,
    min_val_tensor=None,
    negative_val_tensor=None,
    mesh_mapper=None,
):
    sa_output = attention(
        config,
        x,
        attention_mask,
        head_mask,
        output_attentions,
        device=device,
        base_address=base_address,
        parameters=parameters.attention,
        min_val_tensor=min_val_tensor,
        negative_val_tensor=negative_val_tensor,
    )

    sa_output = ttnn.layer_norm(
        x + sa_output,
        weight=parameters.sa_layer_norm.weight,
        bias=parameters.sa_layer_norm.bias,
        epsilon=1e-12,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    ttnn.deallocate(x)

    ffn_output = ffn(config, sa_output, device=device, base_address=base_address, parameters=parameters.ffn)

    ffn_output = ttnn.layer_norm(
        ffn_output + sa_output,
        weight=parameters.output_layer_norm.weight,
        bias=parameters.output_layer_norm.bias,
        epsilon=1e-12,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    ttnn.deallocate(sa_output)
    return ffn_output


def transformer(
    config,
    x,
    attention_mask=None,
    head_mask=None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    base_address=None,
    parameters=None,
    device=None,
    min_val_tensor=None,
    negative_val_tensor=None,
    mesh_mapper=None,
):
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    hidden_state = x

    for params in parameters.layer:
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        layer_outputs = transformer_block(
            config=config,
            x=hidden_state,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            base_address=f"{base_address}.layer",
            parameters=params,
            device=device,
            min_val_tensor=min_val_tensor,
            negative_val_tensor=negative_val_tensor,
        )
        hidden_state = layer_outputs

    return hidden_state


def distilbert(
    config,
    input_ids=None,
    attention_mask=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    position_ids=None,
    min_val_tensor=None,
    negative_val_tensor=None,
    *,
    base_address,
    parameters,
    device,
    mesh_mapper=None,
    ip_mesh_mapper=None,
):
    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.shape
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    head_mask = get_head_mask(head_mask, config.num_hidden_layers)

    if input_ids is not None:
        word_embeddings = ttnn.embedding(
            input_ids,
            parameters.distilbert.embeddings.word_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    seq_length = word_embeddings.shape[1]

    if position_ids is not None:
        position_ids = position_ids[:, :seq_length]

    position_embeddings = ttnn.embedding(
        position_ids,
        parameters.distilbert.embeddings.position_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    transpose = False
    if word_embeddings.shape[0] > 1:
        word_embeddings = ttnn.permute(word_embeddings, (1, 2, 0))
        position_embeddings = ttnn.permute(position_embeddings, (1, 2, 0))
        transpose = True

    embeddings = word_embeddings + position_embeddings

    ttnn.deallocate(word_embeddings)

    if transpose:
        embeddings = ttnn.permute(embeddings, (2, 0, 1))

    embeddings = ttnn.layer_norm(
        embeddings,
        epsilon=1e-12,
        weight=parameters.distilbert.embeddings.LayerNorm.weight,
        bias=parameters.distilbert.embeddings.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return transformer(
        config,
        embeddings,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        base_address=f"distilbert.transformer",
        parameters=parameters.distilbert.transformer,
        device=device,
        min_val_tensor=min_val_tensor,
        negative_val_tensor=negative_val_tensor,
    )


def distilbert_for_question_answering(
    config,
    input_ids,
    attention_mask,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    start_positions=None,
    end_positions=None,
    output_attentions=None,
    output_hidden_states=None,
    min_val_tensor=None,
    negative_val_tensor=None,
    *,
    parameters,
    device,
    base_address="",
    mesh_mapper=None,
    ip_mesh_mapper=None,
):
    distilbert_output = distilbert(
        config,
        input_ids,
        attention_mask,
        head_mask,
        inputs_embeds,
        output_attentions,
        output_hidden_states,
        position_ids=position_ids,
        device=device,
        base_address=f"",
        parameters=parameters,
        min_val_tensor=min_val_tensor,
        negative_val_tensor=negative_val_tensor,
        ip_mesh_mapper=ip_mesh_mapper,
    )

    qa_outputs = ttnn.linear(
        distilbert_output,
        parameters.qa_outputs.weight,
        bias=parameters.qa_outputs.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    return qa_outputs


def preprocess_inputs(
    input_ids,
    position_ids,
    attention_mask,
    device,
    mesh_mapper,
):
    input_ids = ttnn.from_torch(input_ids, mesh_mapper=mesh_mapper, device=device)
    if position_ids is not None:
        position_ids = ttnn.from_torch(position_ids, mesh_mapper=mesh_mapper, device=device)
    attention_mask = ttnn.from_torch(attention_mask, mesh_mapper=mesh_mapper, device=device)
    return (input_ids, position_ids, attention_mask)


def custom_preprocessor(torch_model, name):
    parameters = {}

    if hasattr(torch_model, "q_lin") and hasattr(torch_model, "k_lin") and hasattr(torch_model, "v_lin"):
        qkv_weight = torch.cat(
            [
                torch_model.q_lin.weight,
                torch_model.k_lin.weight,
                torch_model.v_lin.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [torch_model.q_lin.bias, torch_model.k_lin.bias, torch_model.v_lin.bias],
            dim=0,
        )
        output_weight = torch_model.out_lin.weight
        output_bias = torch_model.out_lin.bias
        parameters = {"query_key_value": {}, "out_lin": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16)
        parameters["out_lin"]["weight"] = preprocess_linear_weight(output_weight, dtype=ttnn.bfloat16)
        parameters["out_lin"]["bias"] = preprocess_linear_bias(output_bias, dtype=ttnn.bfloat16)
    return parameters
