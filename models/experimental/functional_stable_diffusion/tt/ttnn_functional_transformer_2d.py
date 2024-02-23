# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional, Dict

from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_basic_transformer_block import (
    basic_transformer_block,
)
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
    post_process_output,
)


def permute_conv_parameters(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


def transformer_2d_model(
    hidden_states,
    parameters,
    config,
    encoder_hidden_states=None,
    timestep=None,
    class_labels=None,
    cross_attention_kwargs=None,
    return_dict=True,
    num_attention_heads=16,
    attention_head_dim=None,
    in_channels=None,
    out_channels=None,
    num_layers=1,
    norm_num_groups=32,
    cross_attention_dim=None,
    attention_bias=False,
    num_vector_embeds=None,
    patch_size=None,
    num_embeds_ada_norm=None,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    norm_type="layer_norm",
    eps=1e-5,
    device=None,
    norm_elementwise_affine: bool = True,
    reader_patterns_cache: Optional[Dict] = None,
):
    conv_on_device = reader_patterns_cache is not None
    inner_dim = num_attention_heads * attention_head_dim

    is_input_continuous = (in_channels is not None) and (patch_size is None)
    is_input_vectorized = num_vector_embeds is not None
    is_input_patches = in_channels is not None and patch_size is not None
    assert (
        is_input_continuous and (not is_input_patches) and (not is_input_vectorized)
    ), "we only support continuous input."
    if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
        deprecation_message = (
            f"The configuration file of this model: transformer_2d_model is outdated. `norm_type` is either not set or"
            " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
            " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
            " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
            " would be very nice if you could open a Pull request for the `transformer/config.json` file"
        )
        deprecate(
            "norm_type!=num_embeds_ada_norm",
            "1.0.0",
            deprecation_message,
            standard_warn=False,
        )
        norm_type = "ada_norm"

    # 1. Input
    if is_input_continuous:
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = ttnn.group_norm(
            input_tensor=hidden_states,
            num_groups=norm_num_groups,
            epsilon=eps,
            weight=parameters.norm.weight,
            bias=parameters.norm.bias,
        )

        if not use_linear_projection:
            parameters.proj_in.weight, parameters.proj_in.bias = permute_conv_parameters(
                parameters.proj_in.weight, parameters.proj_in.bias
            )
            if conv_on_device:
                batch_size = hidden_states.shape[0]
                input_height = hidden_states.shape[2]
                input_width = hidden_states.shape[3]
                parameters.proj_in.bias = torch.reshape(
                    parameters.proj_in.bias, (1, 1, 1, parameters.proj_in.bias.shape[-1])
                )
                tt_weight_tensor = ttnn.from_torch(parameters.proj_in.weight, ttnn.float32)
                tt_bias_tensor = ttnn.from_torch(parameters.proj_in.bias, ttnn.float32)
                out_channels = parameters.proj_in.weight.shape[0]
                in_channels = parameters.proj_in.weight.shape[1]
                proj_in = ttnn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    dtype=ttnn.bfloat8_b,
                    device=device,
                    use_1d_systolic_array=False,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    reader_patterns_cache=reader_patterns_cache,
                    weight=tt_weight_tensor,
                    bias=tt_bias_tensor,
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    weights_dtype=ttnn.bfloat8_b,
                    conv_blocking_and_parallelization_config_override={},
                    use_shallow_conv_variant=False,
                    enable_auto_formatting=True,
                )
                hidden_states = run_ttnn_conv_with_pre_and_post_tensor_formatting(
                    device, proj_in, hidden_states, batch_size, input_height, input_width, out_channels
                )
            else:
                hidden_states = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)))
                hidden_states = torch_to_tt_tensor_rm(hidden_states, device)
                parameters.proj_in.weight = torch_to_tt_tensor_rm(
                    parameters.proj_in.weight, device, put_on_device=False
                )
                parameters.proj_in.bias = torch_to_tt_tensor_rm(parameters.proj_in.bias, device, put_on_device=False)
                # assert False

                proj_in = fallback_ops.Conv2d(
                    weights=parameters.proj_in.weight,
                    biases=parameters.proj_in.bias,
                    in_channels=in_channels,
                    out_channels=inner_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                hidden_states = proj_in(hidden_states)

                hidden_states = tt_to_torch_tensor(hidden_states)
                hidden_states = ttnn.to_layout(
                    ttnn.to_device(ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16), device),
                    layout=ttnn.TILE_LAYOUT,
                )

            inner_dim = hidden_states.shape[1]

            hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))

            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (1, batch, height * width, inner_dim))

        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))

            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (1, batch, height * width, inner_dim))

            hidden_states = ttnn.to_device(hidden_states, device)
            hidden_states = ttnn.matmul(hidden_states, parameters.proj_in.weight)
            hidden_states = ttnn.add(hidden_states, parameters.proj_in.bias)

    # 2. Blocks
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
    for d in range(num_layers):
        hidden_states = basic_transformer_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
            attention_head_dim=attention_head_dim,
            config=config,
            parameters=parameters.transformer_blocks[d],
            device=device,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
        )

    # 3. Output
    out_channels = in_channels if out_channels is None else out_channels
    if is_input_continuous:
        if not use_linear_projection:
            parameters.proj_out.weight, parameters.proj_out.bias = permute_conv_parameters(
                parameters.proj_out.weight, parameters.proj_out.bias
            )
            if conv_on_device:
                batch_size = batch
                input_height = height
                input_width = width
                parameters.proj_out.bias = torch.reshape(
                    parameters.proj_out.bias, (1, 1, 1, parameters.proj_out.bias.shape[-1])
                )
                tt_weight_tensor = ttnn.from_torch(parameters.proj_out.weight, ttnn.float32)
                tt_bias_tensor = ttnn.from_torch(parameters.proj_out.bias, ttnn.float32)
                out_channels = parameters.proj_out.weight.shape[0]
                in_channels = parameters.proj_out.weight.shape[1]
                proj_out = ttnn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    dtype=ttnn.bfloat8_b,
                    device=device,
                    use_1d_systolic_array=False,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    reader_patterns_cache=reader_patterns_cache,
                    weight=tt_weight_tensor,
                    bias=tt_bias_tensor,
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    weights_dtype=ttnn.bfloat8_b,
                    conv_blocking_and_parallelization_config_override={},
                    use_shallow_conv_variant=False,
                    enable_auto_formatting=True,
                )
                hidden_states = proj_out(hidden_states)
                hidden_states = post_process_output(
                    device, hidden_states, batch_size, input_height, input_width, out_channels
                )
            else:
                hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
                hidden_states = ttnn.reshape(hidden_states, (batch, height, width, inner_dim))

                hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

                hidden_states = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)))
                hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

                parameters.proj_out.weight = torch_to_tt_tensor_rm(
                    parameters.proj_out.weight, device, put_on_device=False
                )
                parameters.proj_out.bias = torch_to_tt_tensor_rm(parameters.proj_out.bias, device, put_on_device=False)

                proj_out = fallback_ops.Conv2d(
                    weights=parameters.proj_out.weight,
                    biases=parameters.proj_out.bias,
                    in_channels=inner_dim,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                hidden_states = proj_out(hidden_states)

                hidden_states = tt_to_torch_tensor(hidden_states)
                hidden_states = ttnn.to_layout(
                    ttnn.to_device(ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16), device),
                    layout=ttnn.TILE_LAYOUT,
                )

        else:
            hidden_states = ttnn.to_device(hidden_states, device)
            hidden_states = ttnn.matmul(hidden_states, parameters.proj_out.weight)
            hidden_states = ttnn.add(hidden_states, parameters.proj_out.bias)

            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (batch, height, width, inner_dim))

            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

        output = ttnn.add(
            hidden_states,
            residual,
        )

    if not return_dict:
        return (output,)
    return output
