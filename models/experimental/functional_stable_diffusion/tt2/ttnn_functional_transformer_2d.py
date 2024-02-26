# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional, Dict
import os
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_basic_transformer_block import (
    basic_transformer_block,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
    pre_process_input_new,
    pad_encoder_hidden_states,
    pad_group_norm_weight,
    permute_conv_parameters,
    update_gn_expected_input_sharded_memory_config_and_grid_size,
)


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


class transformer_2d_model:
    def __init__(self, device, parameters, reader_patterns_cache, batch_size, input_height, input_width):
        self.device = device

        parameters.proj_in.weight, parameters.proj_in.bias = permute_conv_parameters(
            parameters.proj_in.weight, parameters.proj_in.bias
        )
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        parameters.proj_in.bias = torch.reshape(parameters.proj_in.bias, (1, 1, 1, parameters.proj_in.bias.shape[-1]))
        tt_weight_tensor = ttnn.from_torch(parameters.proj_in.weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(parameters.proj_in.bias, ttnn.float32)
        out_channels = parameters.proj_in.weight.shape[0]
        in_channels = parameters.proj_in.weight.shape[1]

        self.fallback_on_groupnorm = os.environ.get("FALLBACK_ON_GROUPNORM", "0") == "1"
        if not self.fallback_on_groupnorm:
            parameters.norm.weight = pad_group_norm_weight(parameters.norm.weight, 32, in_channels)
            parameters.norm.bias = pad_group_norm_weight(parameters.norm.bias, 32, in_channels)

        self.proj_in = ttnn.Conv2d(
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
            deallocate_activation=True,
        )

        (
            self.gn_expected_input_sharded_memory_config,
            self.group_norm_core_grid,
        ) = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=self.device,
            num_channels=in_channels,
            num_groups=32,
            input_nhw=batch_size * input_height * input_width,
            is_height_sharded=False,
        )

        parameters.proj_out.weight, parameters.proj_out.bias = permute_conv_parameters(
            parameters.proj_out.weight, parameters.proj_out.bias
        )
        parameters.proj_out.bias = torch.reshape(
            parameters.proj_out.bias, (1, 1, 1, parameters.proj_out.bias.shape[-1])
        )
        tt_weight_tensor = ttnn.from_torch(parameters.proj_out.weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(parameters.proj_out.bias, ttnn.float32)
        out_channels = parameters.proj_out.weight.shape[0]
        in_channels = parameters.proj_out.weight.shape[1]
        self.proj_out = ttnn.Conv2d(
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
            deallocate_activation=True,
        )

        self.output_height = self.proj_out.output_height
        self.output_width = self.proj_out.output_width

        self.blocks = [basic_transformer_block(device, block) for block in parameters.transformer_blocks]
        self.parameters = parameters

    def __call__(
        self,
        hidden_states,
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
        norm_elementwise_affine: bool = True,
        output_bfloat16: bool = False,
    ):
        inner_dim = num_attention_heads * attention_head_dim
        assert norm_num_groups == 32
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

        nhw = hidden_states.shape[2]
        if hidden_states.shape[0] == 1:
            assert nhw == self.batch_size * self.input_height * self.input_width
        batch = self.batch_size
        height = self.input_height
        width = self.input_width

        if ttnn.get_memory_config(hidden_states) != self.proj_in.conv.input_sharded_memory_config:
            hidden_states = ttnn.to_memory_config(hidden_states, self.proj_in.conv.input_sharded_memory_config)
        residual = hidden_states

        hidden_states = ttnn.to_layout(
            hidden_states,
            ttnn.ROW_MAJOR_LAYOUT,
            output_memory_config=self.proj_in.conv.input_sharded_memory_config,
            use_multicore=True,
        )

        if self.fallback_on_groupnorm:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.reshape(
                hidden_states,
                (self.proj_in.batch_size, self.proj_in.input_height, self.proj_in.input_width, in_channels),
            )
            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = ttnn.operations.normalization._fallback_group_norm(
                hidden_states,
                num_groups=norm_num_groups,
                weight=self.parameters.norm.weight,
                bias=self.parameters.norm.bias,
                epsilon=eps,
            )

            hidden_states = pre_process_input(self.device, hidden_states)

        else:
            hidden_states = ttnn.reshape(
                hidden_states, (self.batch_size, 1, self.input_height * self.input_width, in_channels)
            )
            if ttnn.get_memory_config(hidden_states) != self.gn_expected_input_sharded_memory_config:
                hidden_states = ttnn.to_memory_config(hidden_states, self.gn_expected_input_sharded_memory_config)
            print(f"Transformer GN: memory_config={ttnn.get_memory_config(hidden_states)}")
            print(f"Hidden states shape: {hidden_states.shape}")
            hidden_states = ttnn.group_norm(
                input_tensor=hidden_states,
                num_groups=norm_num_groups,
                epsilon=eps,
                weight=self.parameters.norm.weight,
                bias=self.parameters.norm.bias,
                memory_config=ttnn.get_memory_config(hidden_states),
                core_grid=self.group_norm_core_grid,
            )
        hidden_states = ttnn.to_memory_config(
            hidden_states, ttnn.L1_MEMORY_CONFIG
        )  # sharded to interleaved since we can't tilize block sharded
        hidden_states = ttnn.reshape(
            hidden_states, (1, 1, self.batch_size * self.input_height * self.input_width, in_channels)
        )
        hidden_states = ttnn.to_layout(
            hidden_states,
            ttnn.TILE_LAYOUT,
            output_memory_config=self.proj_in.conv.input_sharded_memory_config,
            use_multicore=True,
        )  # tilize

        hidden_states = self.proj_in(hidden_states)

        inner_dim = hidden_states.shape[-1]
        # hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (1, batch, height * width, inner_dim))

        # hidden_states = ttnn.to_memory_config(
        #     hidden_states, ttnn.L1_MEMORY_CONFIG
        # )  # sharded to interleaved since we can't tilize block sharded
        # hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT, use_multicore=True)
        # hidden_states = ttnn.to_memory_config(
        #     hidden_states, self.proj_in.conv.input_sharded_memory_config
        # )  # interleaved to sharded

        # 2. Blocks
        # hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                attention_head_dim=attention_head_dim,
                config=config,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
            )

        # 3. Output
        out_channels = in_channels if out_channels is None else out_channels
        if is_input_continuous:
            if not use_linear_projection:
                hidden_states = ttnn.to_memory_config(hidden_states, self.proj_out.conv.input_sharded_memory_config)
                hidden_states = self.proj_out(hidden_states)
                hidden_states = ttnn.reshape(hidden_states, (1, 1, batch * height * width, inner_dim))
                # hidden_states = ttnn.to_memory_config(
                #     hidden_states, ttnn.L1_MEMORY_CONFIG
                # )  # sharded to interleaved since we can't tilize block sharded
                # hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT, use_multicore=True)
                # hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

                if output_bfloat16:
                    hidden_states = ttnn.add(
                        hidden_states,
                        residual,
                        dtype=ttnn.bfloat16,
                    )
                else:
                    hidden_states = ttnn.add(
                        hidden_states,
                        residual,
                    )
            else:
                hidden_states = ttnn.to_device(hidden_states, self.device)
                hidden_states = ttnn.matmul(hidden_states, self.parameters.proj_out.weight)
                hidden_states = ttnn.add(hidden_states, self.parameters.proj_out.bias)

                hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
                hidden_states = ttnn.reshape(hidden_states, (batch, height, width, inner_dim))

                hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

                hidden_states = ttnn.add(
                    hidden_states,
                    residual,
                )

        if not return_dict:
            return (hidden_states,)
        return hidden_states
