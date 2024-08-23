# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional, Dict
import os
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_basic_transformer_block import (
    basic_transformer_block,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    pre_process_input,
    pad_group_norm_weight,
    permute_conv_parameters,
    dealloc_input,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import conv_cache

from loguru import logger


def ttnn_to_torch(input):
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


class transformer_2d_model:
    def __init__(
        self, device, parameters, reader_patterns_cache, batch_size, input_height, input_width, compute_kernel_config
    ):
        self.device = device
        self.compute_kernel_config = compute_kernel_config
        parameters.proj_in.weight, parameters.proj_in.bias = permute_conv_parameters(
            parameters.proj_in.weight, parameters.proj_in.bias
        )
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        parameters.proj_in.bias = torch.reshape(parameters.proj_in.bias, (1, 1, 1, parameters.proj_in.bias.shape[-1]))
        self.proj_in_conv_weights = ttnn.from_torch(parameters.proj_in.weight, ttnn.float32)
        self.proj_in_conv_bias = ttnn.from_torch(parameters.proj_in.bias, ttnn.float32)
        out_channels = parameters.proj_in.weight.shape[0]
        in_channels = parameters.proj_in.weight.shape[1]

        self.fallback_on_groupnorm = os.environ.get("FALLBACK_ON_GROUPNORM", "0") == "1"

        self.proj_in_in_channels = in_channels
        self.proj_in_out_channels = out_channels
        norm_num_groups = 32
        (
            self.gn_expected_input_sharded_memory_config,
            self.group_norm_core_grid,
        ) = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=self.device,
            num_channels=in_channels,
            num_groups=norm_num_groups,
            input_nhw=batch_size * input_height * input_width,
            is_height_sharded=False,
        )

        if not self.fallback_on_groupnorm:
            if (
                self.gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
            ):
                num_cores_across_channel = self.group_norm_core_grid.y
            elif (
                self.gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED
            ):
                num_cores_across_channel = 1
            else:
                num_cores_across_channel = int(self.group_norm_core_grid.x * self.group_norm_core_grid.y)

            parameters.norm.weight = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(parameters.norm.weight), in_channels, num_cores_across_channel
            )
            parameters.norm.bias = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(parameters.norm.bias), in_channels, num_cores_across_channel
            )
            parameters.norm.weight = ttnn.from_torch(
                parameters.norm.weight,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            parameters.norm.bias = ttnn.from_torch(
                parameters.norm.bias,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self.norm_input_mask_torch_tensor = ttnn.create_group_norm_input_mask(
                in_channels, norm_num_groups, num_cores_across_channel
            )
            self.norm_input_mask = ttnn.from_torch(
                self.norm_input_mask_torch_tensor,
                dtype=ttnn.DataType.BFLOAT8_B,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        parameters.proj_out.weight, parameters.proj_out.bias = permute_conv_parameters(
            parameters.proj_out.weight, parameters.proj_out.bias
        )
        parameters.proj_out.bias = torch.reshape(
            parameters.proj_out.bias, (1, 1, 1, parameters.proj_out.bias.shape[-1])
        )
        self.proj_out_conv_weights = ttnn.from_torch(parameters.proj_out.weight, ttnn.float32)
        self.proj_out_conv_bias = ttnn.from_torch(parameters.proj_out.bias, ttnn.float32)
        out_channels = parameters.proj_out.weight.shape[0]
        in_channels = parameters.proj_out.weight.shape[1]

        self.output_height = ttnn.get_conv_output_dim(input_height, 1, 1, 0)
        self.output_width = ttnn.get_conv_output_dim(input_width, 1, 1, 0)
        self.proj_out_in_channels = in_channels
        self.proj_out_out_channels = out_channels
        self.blocks = [
            basic_transformer_block(device, block, seq_len=input_height * input_width)
            for block in parameters.transformer_blocks
        ]
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

        residual = hidden_states
        spilled_residual = False
        if spilled_residual:
            residual = ttnn.to_memory_config(residual, ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = ttnn.to_layout(
            hidden_states,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=hidden_states.memory_config(),
        )

        if self.fallback_on_groupnorm:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.reshape(
                hidden_states,
                (self.batch_size, self.input_height, self.input_width, in_channels),
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
                # hidden_states = ttnn.reshard(hidden_states, self.gn_expected_input_sharded_memory_config)
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
                hidden_states = ttnn.to_memory_config(hidden_states, self.gn_expected_input_sharded_memory_config)
            hidden_states = ttnn.group_norm(
                input_tensor=hidden_states,
                num_groups=norm_num_groups,
                epsilon=eps,
                input_mask=self.norm_input_mask,
                weight=self.parameters.norm.weight,
                bias=self.parameters.norm.bias,
                memory_config=hidden_states.memory_config(),
                core_grid=self.group_norm_core_grid,
            )
        hidden_states = ttnn.reshape(
            hidden_states, (1, 1, self.batch_size * self.input_height * self.input_width, in_channels)
        )
        hidden_states = ttnn.to_memory_config(
            hidden_states, ttnn.L1_MEMORY_CONFIG
        )  # sharded to interleaved since we can't tilize block sharded
        hidden_states = ttnn.tilize(
            hidden_states,
            memory_config=hidden_states.memory_config(),
            dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
        )

        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation="",
            height_sharding=False,
            input_channels_alignment=32,
            fp32_dest_acc_enabled=self.compute_kernel_config.fp32_dest_acc_en,
            transpose_shards=False,
        )
        [hidden_states, _out_height, _out_width, self.proj_in_conv_weights, self.proj_in_conv_bias] = ttnn.conv2d(
            input_tensor=hidden_states,
            in_channels=self.proj_in_in_channels,
            out_channels=self.proj_in_out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            device=self.device,
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            weight_tensor=self.proj_in_conv_weights,
            bias_tensor=self.proj_in_conv_bias,
            conv_config=conv_config,
            conv_op_cache=conv_cache,
        )

        inner_dim = hidden_states.shape[-1]
        # hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        # hidden_states = ttnn.reshape(hidden_states, (1, batch, height * width, inner_dim))

        # 2. Blocks
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
                # hidden_states = ttnn.to_memory_config(hidden_states, self.proj_out.conv.input_sharded_memory_config)
                [
                    hidden_states,
                    _out_height,
                    _out_width,
                    self.proj_out_conv_weights,
                    self.proj_out_conv_bias,
                ] = ttnn.conv2d(
                    input_tensor=hidden_states,
                    in_channels=self.proj_out_in_channels,
                    out_channels=self.proj_out_out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    device=self.device,
                    batch_size=self.batch_size,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    weight_tensor=self.proj_out_conv_weights,
                    bias_tensor=self.proj_out_conv_bias,
                    conv_config=conv_config,
                    conv_op_cache=conv_cache,
                )

                if output_bfloat16:
                    hidden_states = dealloc_input(
                        ttnn.add,
                        hidden_states,
                        residual,
                        dtype=ttnn.bfloat16,
                        memory_config=hidden_states.memory_config(),
                    )
                else:
                    hidden_states = dealloc_input(
                        ttnn.add,
                        hidden_states,
                        residual,
                        memory_config=hidden_states.memory_config(),
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
        hidden_states = ttnn.reallocate(hidden_states)
        return hidden_states
