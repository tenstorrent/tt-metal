# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

import ttnn
from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask
from models.utility_functions import is_grayskull
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


def transpose_for_scores(config, x, device, permute_tensor: bool):
    new_x_shape = (x.shape[0], config.num_attention_heads, config.attention_head_size, x.shape[-1])
    x = ttnn.from_device(x)
    x = ttnn.reshape(x, new_x_shape)
    x = ttnn.to_device(x, device)

    if permute_tensor:
        x = ttnn.permute(x, (0, 1, 3, 2))

    return x


def transpose_output(config, x, device):
    all_head_size = config.num_attention_heads * config.attention_head_size
    if len(x.shape) == 4:
        x = ttnn.permute(x, (0, 1, 3, 2))

    new_x_shape = (x.shape[0], all_head_size, x.shape[3])
    x = ttnn.reshape(x, new_x_shape)

    return x


def permute_reshape(hidden_states, shape=(0, 2, 1), reshape=True):
    bs, *_ = hidden_states.shape
    hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
    if reshape:
        hidden_states = ttnn.reshape(hidden_states, (bs, hidden_states.shape[-2], hidden_states.shape[-1]))

    return hidden_states


def ttnn_conv1d(
    device,
    tt_input_tensor,
    weights,
    conv_params,
    bias,
    *,
    output_dtype=ttnn.bfloat16,
    weights_dtype=ttnn.bfloat8_b,
    math_fidelity=ttnn.MathFidelity.LoFi,
    deallocate_activation=False,
    act_block_h=None,
    height_sharding=True,
    fp32_accum=False,
    packer_l1_acc=False,
    groups=4,
    math_approx=True,
    activation="",
    reallocate_halo=False,
    reshard=False,
):
    weights = ttnn.from_torch(weights, dtype=ttnn.float32)
    bias = ttnn.from_torch(bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.float32)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat8_b,
        activation=activation,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=reallocate_halo,
        act_block_h_override=32,
        reshard_if_not_optimal=reshard,
        shard_layout=(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        ),
        core_grid=get_shard_grid_from_num_cores(56, device),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )

    [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.conv1d(
        input_tensor=tt_input_tensor,
        weight_tensor=weights,
        in_channels=tt_input_tensor.shape[-1],
        out_channels=weights.shape[0],
        device=device,
        bias_tensor=bias,
        kernel_size=1,
        stride=1,
        padding=0,
        batch_size=tt_input_tensor.shape[0],
        input_length=tt_input_tensor.shape[1],
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_output_tensor_on_device = ttnn.squeeze(tt_output_tensor_on_device, 0)
    tt_output_tensor_on_device = ttnn.reshape(
        tt_output_tensor_on_device, (tt_input_tensor.shape[0], out_length, tt_output_tensor_on_device.shape[-1])
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)

    return tt_output_tensor


def squeezebert_conv_layernorm(
    config,
    hidden_states,
    input_tensor,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    cin,
    cout,
    groups,
):
    torch_hidden_states = ttnn.to_torch(hidden_states).to(torch.float32)
    self_output_conv1d_ = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
    self_output_conv1d_.weight = nn.Parameter(state_dict[f"{base_addr}conv1d.weight"])
    self_output_conv1d_.bias = nn.Parameter(state_dict[f"{base_addr}conv1d.bias"])

    torch_self_output = self_output_conv1d_(torch_hidden_states)
    self_output = ttnn.from_torch(torch_self_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    self_output_layernorm = ttnn.add(self_output, input_tensor)
    self_output_layernorm = permute_reshape(self_output_layernorm)

    attention_output = ttnn.layer_norm(
        self_output_layernorm,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=config.layer_norm_eps,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    ttnn.deallocate(self_output_layernorm)
    attention_output = permute_reshape(attention_output)

    return attention_output


def squeezebert_attention(
    config,
    hidden_states,
    attention_mask,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    num_cores_x=12,
):
    num_heads = config.num_attention_heads
    batch_size, hidden_size, _ = hidden_states.shape
    head_size = hidden_size // num_heads
    config.attention_head_size = head_size

    hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = permute_reshape(hidden_states)
    hidden_states = ttnn.from_device(hidden_states)
    mixed_query_layer = ttnn_conv1d(
        device,
        hidden_states,
        nn.Parameter(state_dict[f"{base_addr}query.weight"]),
        conv_params=[1, 0],
        bias=nn.Parameter(state_dict[f"{base_addr}query.bias"]),
    )
    mixed_query_layer = ttnn.to_device(mixed_query_layer, device)
    mixed_query_layer = ttnn.permute(mixed_query_layer, (0, 2, 1))

    mixed_key_layer = ttnn_conv1d(
        device,
        hidden_states,
        nn.Parameter(state_dict[f"{base_addr}key.weight"]),
        conv_params=[1, 0],
        bias=nn.Parameter(state_dict[f"{base_addr}key.bias"]),
    )
    mixed_key_layer = ttnn.to_device(mixed_key_layer, device)
    mixed_key_layer = ttnn.permute(mixed_key_layer, (0, 2, 1))

    mixed_value_layer = ttnn_conv1d(
        device,
        hidden_states,
        nn.Parameter(state_dict[f"{base_addr}value.weight"]),
        conv_params=[1, 0],
        bias=nn.Parameter(state_dict[f"{base_addr}value.bias"]),
    )
    mixed_value_layer = ttnn.to_device(mixed_value_layer, device)
    mixed_value_layer = ttnn.permute(mixed_value_layer, (0, 2, 1))

    query = transpose_for_scores(config, mixed_query_layer, device, True)
    key = transpose_for_scores(config, mixed_key_layer, device, False)
    value = transpose_for_scores(config, mixed_value_layer, device, True)

    ttnn.deallocate(mixed_query_layer)
    ttnn.deallocate(mixed_key_layer)
    ttnn.deallocate(mixed_value_layer)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
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
        core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    context_layer = transpose_output(config, context_layer, device)

    return context_layer


def squeezebert_intermediate(
    config,
    hidden_states,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    num_cores_x=12,
):
    torch_hidden_states = ttnn.to_torch(hidden_states).to(torch.float32)

    torch_conv_ = nn.Conv1d(
        in_channels=config.hidden_size,
        out_channels=config.intermediate_size,
        kernel_size=1,
        groups=config.intermediate_groups,
    )
    torch_conv_.weight = nn.Parameter(state_dict[f"{base_addr}conv1d.weight"])
    torch_conv_.bias = nn.Parameter(state_dict[f"{base_addr}conv1d.bias"])

    torch_conv_output = torch_conv_(torch_hidden_states)
    ttnn_conv_output = ttnn.from_torch(torch_conv_output, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.gelu(ttnn_conv_output)
    return output


def squeezebert_layer(
    config,
    hidden_states,
    attention_mask,
    state_dict,
    base_addr,
    parameters,
    device,
):
    multi_head_attention_output = squeezebert_attention(
        config,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        state_dict=state_dict,
        base_addr=f"{base_addr}attention.",
        parameters=parameters.attention,
        device=device,
    )

    attention_output = squeezebert_conv_layernorm(
        config,
        hidden_states=multi_head_attention_output,
        input_tensor=hidden_states,
        state_dict=state_dict,
        base_addr=f"{base_addr}post_attention.",
        parameters=parameters.post_attention,
        device=device,
        cin=config.hidden_size,
        cout=config.hidden_size,
        groups=config.post_attention_groups,
    )
    ttnn.deallocate(hidden_states)
    ttnn.deallocate(multi_head_attention_output)

    intermediate = squeezebert_intermediate(
        config,
        attention_output,
        state_dict=state_dict,
        base_addr=f"{base_addr}intermediate.",
        parameters=parameters.intermediate,
        device=device,
    )

    output = squeezebert_conv_layernorm(
        config,
        hidden_states=intermediate,
        input_tensor=attention_output,
        state_dict=state_dict,
        base_addr=f"{base_addr}output.",
        parameters=parameters.output,
        device=device,
        cin=config.intermediate_size,
        cout=config.hidden_size,
        groups=config.output_groups,
    )

    return output


def squeezebert_encoder(
    config,
    hidden_states,
    attention_mask,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
):
    hidden_states = permute_reshape(hidden_states)
    encoder_output = None

    for layer_idx, encoder_parameters in enumerate(parameters.layers):
        encoder_output = squeezebert_layer(
            config,
            hidden_states,
            attention_mask,
            state_dict,
            base_addr=f"{base_addr}layers.{layer_idx}.",
            parameters=encoder_parameters,
            device=device,
        )
        encoder_output = ttnn.reallocate(encoder_output)
        hidden_states = encoder_output

    hidden_states = permute_reshape(hidden_states)

    return hidden_states


def squeezebert(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    state_dict,
    base_addr,
    parameters,
    device,
):
    word_embeddings = ttnn.embedding(
        input_ids,
        parameters.embeddings.word_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
        padding_idx=config.pad_token_id,
    )
    ttnn.deallocate(input_ids)

    token_type_embeddings = ttnn.embedding(
        token_type_ids,
        parameters.embeddings.token_type_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.deallocate(token_type_ids)

    word_plus_token_type_embeddings = word_embeddings + token_type_embeddings
    ttnn.deallocate(word_embeddings)
    ttnn.deallocate(token_type_embeddings)

    position_embeddings = ttnn.embedding(
        position_ids,
        parameters.embeddings.position_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.deallocate(position_ids)

    embeddings = word_plus_token_type_embeddings + position_embeddings
    ttnn.deallocate(word_plus_token_type_embeddings)
    ttnn.deallocate(position_embeddings)

    encoder_input = ttnn.layer_norm(
        embeddings,
        weight=parameters.embeddings.LayerNorm.weight,
        bias=parameters.embeddings.LayerNorm.bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull() else ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    ttnn.deallocate(embeddings)

    encoder_output = squeezebert_encoder(
        config=config,
        hidden_states=encoder_input,
        attention_mask=attention_mask,
        state_dict=state_dict,
        base_addr=f"{base_addr}encoder.",
        parameters=parameters.encoder,
        device=device,
    )
    ttnn.deallocate(encoder_input)

    return encoder_output


def squeezebert_for_question_answering(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    *,
    state_dict,
    base_addr,
    parameters,
    device,
    name="transformer",
):
    squeezebert_output = squeezebert(
        config,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        state_dict,
        base_addr,
        parameters=parameters.transformer,
        device=device,
    )
    qa_outputs = ttnn.linear(
        squeezebert_output,
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


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv1d):
        weight = model.weight
        bias = model.bias

        while bias.dim() < 4:
            bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.float32)
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.float32)

    return parameters
