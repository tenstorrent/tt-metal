# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    get_default_compute_config,
    permute_conv_parameters,
    weight_to_bfp8,
)

config_override = {
    (320, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 32, 32): {"act_block_h": 64},
    (640, 1920, 32, 32): {"act_block_h": 32},
    (640, 1280, 32, 32): {"act_block_h": 32},
    (1280, 1920, 16, 16): {"act_block_h": 32},
    (1280, 1280, 32, 32): {"act_block_h": 32},
    (1280, 1280, 16, 16): {"act_block_h": 32},
    (320, 960, 64, 64): {"act_block_h": 32},
    (640, 960, 32, 32): {"act_block_h": 32},
    (320, 640, 64, 64): {"act_block_h": 32},
    (640, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 64, 64): {"act_block_h": 32},
}

split_chunks = {
    (320, 960, 64, 64): 2,
    (640, 1920, 32, 32): 2,
    # (640, 1280, 32, 32): 2,
    # (640, 960, 32, 32): 2,
    (1280, 1920, 16, 16): 2,
    # (1280, 2560, 8, 8): 2,
    (1280, 2560, 16, 16): 2,
}


class resnetBlock2D:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        input_height,
        input_width,
        compute_kernel_config,
        group_norm_on_device=True,
    ):
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.device = device
        self.parameters = parameters
        self.conv1s = []
        self.conv1s_weights = []
        self.conv1s_bias = []
        # TODO: instead of hardcoding grid size in many components, configure it in a single place
        self.grid_size = (8, 8)

        self.fallback_on_groupnorm = os.environ.get("FALLBACK_ON_GROUPNORM", "0") == "1"
        parameters.conv1.weight, parameters.conv1.bias = permute_conv_parameters(
            parameters.conv1.weight, parameters.conv1.bias
        )
        out_channels = parameters.conv1.bias.shape[-1]
        in_channels = parameters.conv1.weight.shape[1]

        parameters.conv1.bias = torch.reshape(parameters.conv1.bias, (1, 1, 1, out_channels))
        conv1_split_chunks = 1
        if (out_channels, in_channels, input_height, input_width) in split_chunks:
            conv1_split_chunks = split_chunks[(out_channels, in_channels, input_height, input_width)]
        split_input_channels = in_channels // conv1_split_chunks
        if conv1_split_chunks > 1:
            logger.info(
                f"Splitting: {(out_channels, in_channels, input_height, input_width)} into: {conv1_split_chunks}"
            )
        if conv1_split_chunks == 1:
            split_weight_tensors = [parameters.conv1.weight]
        else:
            split_weight_tensors = torch.split(parameters.conv1.weight, split_input_channels, 1)

        self.conv1_input_height = input_height
        self.conv1_input_width = input_width
        self.conv1_in_channels = split_input_channels
        self.conv1_out_channels = out_channels
        self.conv1_output_height = ttnn.get_conv_output_dim(self.conv1_input_height, 3, 1, 1)
        self.conv1_output_width = ttnn.get_conv_output_dim(self.conv1_input_width, 3, 1, 1)
        self.conv1_shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        self.conv1_input_memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            tensor_shape=ttnn.Shape(
                [
                    1,
                    1,
                    self.batch_size * self.conv1_input_height * self.conv1_input_width,
                    self.conv1_in_channels,
                ]
            ),
            parallel_config=ttnn._ttnn.operations.conv.determine_parallel_config(
                shard_layout=self.conv1_shard_layout,
                batch_size=self.batch_size,
                input_channels=self.conv1_in_channels,
                output_height=self.conv1_output_height,
                output_width=self.conv1_output_width,
                output_channels=self.conv1_out_channels,
                compute_grid_size=self.grid_size,
                block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                enable_channels_padding=False,
            ),
            tile_size=32,
        )

        for i in range(conv1_split_chunks):
            self.conv1s_weights.append(ttnn.from_torch(split_weight_tensors[i], ttnn.float32))
            if i == 0:
                self.conv1s_bias.append(ttnn.from_torch(parameters.conv1.bias, ttnn.float32))
            else:
                # TODO: fix no bias in conv error
                torch_bias_zeros_tensor = torch.zeros(parameters.conv1.bias.shape, dtype=torch.bfloat16).float()
                self.conv1s_bias.append(ttnn.from_torch(torch_bias_zeros_tensor, ttnn.float32))
            self.conv1_config_override = {}
            if (out_channels, in_channels, input_height, input_width) in config_override:
                self.conv1_config_override = config_override[(out_channels, in_channels, input_height, input_width)]
            self.conv1s.append(
                {
                    "output_height": input_height,
                    "output_width": input_width,
                }
            )

        use_in_shortcut = True if "conv_shortcut" in parameters else False
        if use_in_shortcut:
            parameters.conv_shortcut.weight, parameters.conv_shortcut.bias = permute_conv_parameters(
                parameters.conv_shortcut.weight, parameters.conv_shortcut.bias
            )

            convs_input_height = input_height
            convs_input_width = input_width
            parameters.conv_shortcut.bias = torch.reshape(parameters.conv_shortcut.bias, (1, 1, 1, out_channels))
            self.conv_shortcut_weights = ttnn.from_torch(parameters.conv_shortcut.weight, ttnn.float32)
            self.conv_shortcut_bias = ttnn.from_torch(parameters.conv_shortcut.bias, ttnn.float32)
            self.conv_shortcut_in_channels = parameters.conv_shortcut.weight.shape[1]
            self.conv_shortcut_out_channels = parameters.conv_shortcut.weight.shape[0]
            self.conv_shortcut_input_height = convs_input_height
            self.conv_shortcut_input_width = convs_input_width

            self.output_height = self.conv_shortcut_input_height
            self.output_width = self.conv_shortcut_input_width

        conv2_input_height = input_height
        conv2_input_width = input_width
        parameters.conv2.weight, parameters.conv2.bias = permute_conv_parameters(
            parameters.conv2.weight, parameters.conv2.bias
        )
        parameters.conv2.bias = torch.reshape(parameters.conv2.bias, (1, 1, 1, out_channels))
        self.conv2_weights = ttnn.from_torch(parameters.conv2.weight, ttnn.float32)
        self.conv2_bias = ttnn.from_torch(parameters.conv2.bias, ttnn.float32)
        self.conv2_config_override = {}
        if (out_channels, out_channels, input_height, input_width) in config_override:
            self.conv2_config_override = config_override[(out_channels, out_channels, input_height, input_width)]

        self.conv2_input_height = conv2_input_height
        self.conv2_input_width = conv2_input_width
        self.conv2_in_channels = parameters.conv2.weight.shape[1]
        self.conv2_out_channels = parameters.conv2.weight.shape[0]
        #     self.conv2_config_override = config_override[(out_channels, out_channels, input_height, input_width)]
        self.conv2_input_memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            tensor_shape=ttnn.Shape(
                [
                    1,
                    1,
                    self.batch_size * self.conv2_input_height * self.conv2_input_width,
                    out_channels,
                ]
            ),
            parallel_config=ttnn._ttnn.operations.conv.determine_parallel_config(
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                batch_size=self.batch_size,
                input_channels=self.conv2_in_channels,
                output_height=self.conv2_input_height,
                output_width=self.conv2_input_width,
                output_channels=self.conv2_out_channels,
                compute_grid_size=self.grid_size,
                block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                enable_channels_padding=False,
            ),
            tile_size=32,
        )

        self.groups = 32
        # if use_in_shortcut:
        #     assert self.conv2.conv.output_sharded_memory_config == self.conv_shortcut.conv.output_sharded_memory_config

        (
            self.first_gn_expected_input_sharded_memory_config,
            self.first_group_norm_core_grid,
        ) = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=self.device,
            num_channels=in_channels,
            num_groups=self.groups,
            input_nhw=batch_size * input_height * input_width,
            is_height_sharded=False,
        )
        (
            self.second_gn_expected_input_sharded_memory_config,
            self.second_group_norm_core_grid,
        ) = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=self.device,
            num_channels=out_channels,
            num_groups=self.groups,
            input_nhw=batch_size * input_height * input_width,
            is_height_sharded=False,
        )

        self.output_height = self.conv2_input_height
        self.output_width = self.conv2_input_width
        assert self.input_height == self.output_height
        assert self.input_width == self.output_width
        out_channels = parameters.conv1.bias.shape[-1]
        in_channels = parameters.conv1.weight.shape[1]

        if not self.fallback_on_groupnorm:
            if (
                self.first_gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.TensorMemoryLayout.BLOCK_SHARDED
            ):
                num_cores_across_channel = self.first_group_norm_core_grid.y
            elif (
                self.first_gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            ):
                num_cores_across_channel = 1
            else:
                num_cores_across_channel = int(self.first_group_norm_core_grid.x * self.first_group_norm_core_grid.y)

            self.parameters.norm1.weight = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(self.parameters.norm1.weight), in_channels, num_cores_across_channel
            )
            self.parameters.norm1.bias = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(self.parameters.norm1.bias), in_channels, num_cores_across_channel
            )

            self.norm1_input_mask_torch_tensor = ttnn.create_group_norm_input_mask(
                in_channels, self.groups, num_cores_across_channel
            )

            self.norm1_input_mask = ttnn.from_torch(
                self.norm1_input_mask_torch_tensor,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self.parameters.norm1.weight = ttnn.from_torch(
                self.parameters.norm1.weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.parameters.norm1.bias = ttnn.from_torch(
                self.parameters.norm1.bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if (
                self.second_gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.TensorMemoryLayout.BLOCK_SHARDED
            ):
                num_cores_across_channel = self.second_group_norm_core_grid.y
            elif (
                self.second_gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            ):
                num_cores_across_channel = 1
            else:
                num_cores_across_channel = int(self.second_group_norm_core_grid.x * self.second_group_norm_core_grid.y)

            self.parameters.norm2.weight = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(self.parameters.norm2.weight), out_channels, num_cores_across_channel
            )
            self.parameters.norm2.bias = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(self.parameters.norm2.bias), out_channels, num_cores_across_channel
            )
            self.parameters.norm2.weight = ttnn.from_torch(
                self.parameters.norm2.weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.parameters.norm2.bias = ttnn.from_torch(
                self.parameters.norm2.bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self.norm2_input_mask_torch_tensor = ttnn.create_group_norm_input_mask(
                out_channels, self.groups, num_cores_across_channel
            )
            self.norm2_input_mask = ttnn.from_torch(
                self.norm2_input_mask_torch_tensor,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.parameters.time_emb_proj.weight = weight_to_bfp8(self.parameters.time_emb_proj.weight)
            self.parameters.time_emb_proj.bias = weight_to_bfp8(self.parameters.time_emb_proj.bias)

    def reshard_to(self, tensor, grid_size, layout):
        if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            shard_spec = [tensor.volume() // tensor.shape[-1] // grid_size[0], tensor.shape[-1] // grid_size[1]]
        elif layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            num_cores = grid_size[0] * grid_size[1]
            shard_spec = [tensor.volume() // tensor.shape[-1] // num_cores, tensor.shape[-1]]
        output_shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        output_shard_spec = ttnn.ShardSpec(
            output_shard_grid,
            shard_spec,
            ttnn.ShardOrientation.COL_MAJOR,
        )
        output_mem_config = ttnn.MemoryConfig(
            layout,
            ttnn.BufferType.L1,
            output_shard_spec,
        )
        if tensor.is_sharded():
            tensor = ttnn.reshard(
                tensor,
                output_mem_config,
            )
        else:
            tensor = ttnn.interleaved_to_sharded(
                tensor,
                grid_size,
                shard_spec,
                layout,
                ttnn.ShardOrientation.COL_MAJOR,
            )
        return tensor

    def __call__(
        self,
        input_tensor,
        *,
        temb,
        in_channels,
        temb_channels=1280,
        groups: int = 32,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        out_channels: Optional[int] = None,
        non_linearity="silu",
        pre_norm=True,
        eps=1e-5,
        up=False,
        down=False,
        use_in_shortcut: Optional[bool] = None,
        dtype: Optional[ttnn.DataType] = None,
        index=-1,
    ):
        assert groups == self.groups
        if non_linearity == "mish":
            assert False, "Mish is not implemented!"
        else:
            nonlinearity = ttnn.silu

        out_channels = in_channels if out_channels is None else out_channels
        hidden_states = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        if ttnn.get_memory_config(hidden_states) != self.first_gn_expected_input_sharded_memory_config:
            hidden_states = ttnn.to_memory_config(hidden_states, self.first_gn_expected_input_sharded_memory_config)

        hidden_states = ttnn.reshape(
            hidden_states, (self.batch_size, 1, self.conv2_input_height * self.conv2_input_width, in_channels)
        )
        hidden_states = ttnn.reallocate(hidden_states)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=self.norm1_input_mask,
            weight=self.parameters.norm1.weight,
            bias=self.parameters.norm1.bias,
            epsilon=eps,
            memory_config=ttnn.get_memory_config(hidden_states),
            core_grid=self.first_group_norm_core_grid,
            dtype=ttnn.bfloat8_b,
        )
        hidden_states = ttnn.reshape(
            hidden_states,
            (1, 1, self.batch_size * self.conv2_input_height * self.conv2_input_width, in_channels),
        )
        if up:
            assert False, "Up block within residual block is not implemented!"
        elif down:
            assert False, "Down block within residual block is not implemented"

        conv1_split_chunks = len(self.conv1s)
        if conv1_split_chunks == 1:
            # Once https://github.com/tenstorrent/tt-metal/issues/7071 is in convert to reshard
            # hidden_states = ttnn.interleaved_to_sharded(
            #     hidden_states, self.conv1s[0].conv.input_sharded_memory_config, hidden_states.dtype
            # )
            hidden_states = nonlinearity(hidden_states, memory_config=ttnn.get_memory_config(hidden_states))
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG, hidden_states.dtype)
            hidden_states = ttnn.reallocate(hidden_states)
            # hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

            # hidden_states = self.conv1s[0](hidden_states)

            # hidden_states = nonlinearity(hidden_states, memory_config=ttnn.get_memory_config(hidden_states))
            # hidden_states = self.conv1s[0](hidden_states)

            hidden_states = ttnn.to_memory_config(hidden_states, self.conv1_input_memory_config)

            conv_config = ttnn.Conv2dConfig(
                dtype=ttnn.bfloat8_b,
                weights_dtype=ttnn.bfloat8_b,
                activation="",
                shard_layout=self.conv1_shard_layout,
                transpose_shards=False,
                reshard_if_not_optimal=False,
            )
            compute_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            if self.conv1_config_override and "act_block_h" in self.conv2_config_override:
                conv_config.act_block_h_override = self.conv1_config_override["act_block_h"]

            conv_kwargs_1 = {
                "in_channels": self.conv1_in_channels,
                "out_channels": self.conv1_out_channels,
                "batch_size": self.batch_size,
                "input_height": self.conv1_input_height,
                "input_width": self.conv1_input_width,
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": (1, 1),
                "dilation": (1, 1),
                "groups": 1,
                "device": self.device,
                "conv_config": conv_config,
            }

            if not ttnn.is_tensor_storage_on_device(self.conv1s_weights[0]):
                tt_weight = self.conv1s_weights[0]
                tt_bias = self.conv1s_bias[0]
                self.conv1s_weights[0] = ttnn.prepare_conv_weights(
                    weight_tensor=self.conv1s_weights[0],
                    weights_format="OIHW",
                    input_memory_config=hidden_states.memory_config(),
                    input_layout=hidden_states.get_layout(),
                    has_bias=True,
                    **conv_kwargs_1,
                )
                self.conv1s_bias[0] = ttnn.prepare_conv_bias(
                    bias_tensor=self.conv1s_bias[0],
                    input_memory_config=hidden_states.memory_config(),
                    input_layout=hidden_states.get_layout(),
                    **conv_kwargs_1,
                )
                self.conv1s_weights[0] = ttnn.to_device(self.conv1s_weights[0], self.device)
                self.conv1s_bias[0] = ttnn.to_device(self.conv1s_bias[0], self.device)

            hidden_states = ttnn.conv2d(
                input_tensor=hidden_states,
                weight_tensor=self.conv1s_weights[0],
                bias_tensor=self.conv1s_bias[0],
                **conv_kwargs_1,
                compute_config=compute_config,
            )

        else:
            split_hidden_states = []
            output_tensor_start_width_dim = 0
            in_channels = self.parameters.conv1.weight.shape[1]
            split_input_channels = in_channels // conv1_split_chunks

            # unpad sharded causes output mismatch
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG, hidden_states.dtype)
            output_tensor_end_width_dim = split_input_channels
            for i in range(conv1_split_chunks):
                # TODO: Can we replace this with interleaved_to_sharded_partial
                split_hidden_states.append(
                    ttnn.slice(
                        hidden_states,
                        [0, 0, 0, output_tensor_start_width_dim],
                        [
                            hidden_states.shape[0],
                            hidden_states.shape[1],
                            hidden_states.shape[2],
                            output_tensor_end_width_dim,
                        ],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                )
                output_tensor_start_width_dim += split_input_channels
                output_tensor_end_width_dim += split_input_channels

                split_hidden_states[i] = ttnn.to_layout(split_hidden_states[i], ttnn.TILE_LAYOUT)
                # split_hidden_states[i] = ttnn.interleaved_to_sharded(
                #     split_hidden_states[i],
                #     self.conv1s[i].conv.input_sharded_memory_config,
                #     split_hidden_states[i].dtype,
                # )
                split_hidden_states[i] = nonlinearity(
                    split_hidden_states[i], memory_config=ttnn.get_memory_config(split_hidden_states[i])
                )
                # split_hidden_states[i] = self.conv1s[i](split_hidden_states[i])

                conv_config = ttnn.Conv2dConfig(
                    dtype=ttnn.bfloat8_b,
                    weights_dtype=ttnn.bfloat8_b,
                    activation="",
                    shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    transpose_shards=False,
                    reshard_if_not_optimal=False,
                )
                compute_config = ttnn.init_device_compute_kernel_config(
                    self.device.arch(),
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=True,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )
                if self.conv1_config_override and "act_block_h" in self.conv2_config_override:
                    conv_config.act_block_h_override = self.conv1_config_override["act_block_h"]

                conv_kwargs_2 = {
                    "in_channels": self.conv1_in_channels,
                    "out_channels": self.conv1_out_channels,
                    "batch_size": self.batch_size,
                    "input_height": self.conv1_input_height,
                    "input_width": self.conv1_input_width,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "device": self.device,
                    "conv_config": conv_config,
                }

                if not ttnn.is_tensor_storage_on_device(self.conv1s_weights[i]):
                    self.conv1s_weights[i] = ttnn.prepare_conv_weights(
                        weight_tensor=self.conv1s_weights[i],
                        weights_format="OIHW",
                        input_memory_config=split_hidden_states[i].memory_config(),
                        input_layout=split_hidden_states[i].get_layout(),
                        has_bias=True,
                        **conv_kwargs_2,
                    )
                    self.conv1s_bias[i] = ttnn.prepare_conv_bias(
                        bias_tensor=self.conv1s_bias[i],
                        input_memory_config=split_hidden_states[i].memory_config(),
                        input_layout=split_hidden_states[i].get_layout(),
                        **conv_kwargs_2,
                    )

                    self.conv1s_weights[i] = ttnn.to_device(self.conv1s_weights[i], self.device)
                    self.conv1s_bias[i] = ttnn.to_device(self.conv1s_bias[i], self.device)

                [
                    split_hidden_states[i],
                    [_out_height, _out_width],
                ] = ttnn.conv2d(
                    input_tensor=split_hidden_states[i],
                    weight_tensor=self.conv1s_weights[i],
                    bias_tensor=self.conv1s_bias[i],
                    **conv_kwargs_2,
                    compute_config=compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                )
                if i != 0:
                    split_hidden_states[i] = ttnn.add(
                        split_hidden_states[i],
                        split_hidden_states[i - 1],
                        # memory_config=self.conv1s[i].conv.output_sharded_memory_config,
                    )
                    ttnn.deallocate(split_hidden_states[i - 1])
            hidden_states = split_hidden_states[-1]
            split_hidden_states = []

        if temb is not None:
            grid_size = (2, 5)  # 5 is the Magic Number!
            # num_cores = grid_size[0] * grid_size[1]
            # temb = self.reshard_to(temb, grid_size, ttnn.TensorMemoryLayout.BLOCK_SHARDED)
            temb = nonlinearity(temb, memory_config=temb.memory_config())
            if temb_channels is not None:
                if time_embedding_norm == "default":
                    time_emb_proj_out_channels = out_channels
                elif time_embedding_norm == "scale_shift":
                    time_emb_proj_out_channels = out_channels * 2
                else:
                    raise ValueError(f"unknown time_embedding_norm : {time_embedding_norm} ")
                program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=temb.shape[-1] // grid_size[1] // 32,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=self.parameters.time_emb_proj.weight.shape[-1] // grid_size[1] // 32,
                    transpose_mcast=True,
                    fused_activation=None,
                )
                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=True,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )
                l1_memory_config = ttnn.L1_MEMORY_CONFIG
                temb = ttnn.linear(
                    temb,
                    self.parameters.time_emb_proj.weight,
                    bias=self.parameters.time_emb_proj.bias,
                    program_config=program_config,
                    memory_config=l1_memory_config,
                    dtype=ttnn.bfloat8_b,
                    compute_kernel_config=compute_kernel_config,
                )

        if temb is not None and time_embedding_norm == "default":
            hidden_states = ttnn.bcast(
                hidden_states,
                temb,
                ttnn.BcastOpMath.ADD,
                ttnn.BcastOpDim.H,
                memory_config=hidden_states.memory_config(),
            )

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.to_memory_config(hidden_states, self.second_gn_expected_input_sharded_memory_config)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=self.norm2_input_mask,
            weight=self.parameters.norm2.weight,
            bias=self.parameters.norm2.bias,
            epsilon=eps,
            memory_config=self.second_gn_expected_input_sharded_memory_config,
            core_grid=self.second_group_norm_core_grid,
            dtype=ttnn.bfloat8_b,
        )
        hidden_states = ttnn.reshape(
            hidden_states,
            (1, 1, self.batch_size * self.conv2_input_height * self.conv2_input_width, out_channels),
        )

        # hidden_states = ttnn.sharded_to_interleaved(
        #     hidden_states, ttnn.L1_MEMORY_CONFIG, hidden_states.dtype
        # )
        # hidden_states = ttnn.interleaved_to_sharded(
        #     hidden_states, self.conv2.conv.input_sharded_memory_config, hidden_states.dtype
        # )

        hidden_states = nonlinearity(hidden_states, memory_config=ttnn.get_memory_config(hidden_states))
        # hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG, hidden_states.dtype)

        hidden_states = ttnn.to_memory_config(hidden_states, self.conv2_input_memory_config)
        # hidden_states = self.conv2(hidden_states)

        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            transpose_shards=False,
            reshard_if_not_optimal=False,
        )
        compute_config = get_default_compute_config(self.device)
        if self.conv2_config_override and "act_block_h" in self.conv2_config_override:
            conv_config.act_block_h_override = self.conv2_config_override["act_block_h"]

        conv_kwargs_3 = {
            "in_channels": self.conv2_in_channels,
            "out_channels": self.conv2_out_channels,
            "batch_size": self.batch_size,
            "input_height": self.conv2_input_height,
            "input_width": self.conv2_input_width,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": conv_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.conv2_weights):
            self.conv2_weights = ttnn.prepare_conv_weights(
                weight_tensor=self.conv2_weights,
                weights_format="OIHW",
                input_memory_config=hidden_states.memory_config(),
                input_layout=hidden_states.get_layout(),
                has_bias=True,
                **conv_kwargs_3,
            )
            self.conv2_bias = ttnn.prepare_conv_bias(
                bias_tensor=self.conv2_bias,
                input_memory_config=hidden_states.memory_config(),
                input_layout=hidden_states.get_layout(),
                **conv_kwargs_3,
            )

            self.conv2_weights = ttnn.to_device(self.conv2_weights, self.device)
            self.conv2_bias = ttnn.to_device(self.conv2_bias, self.device)

        [hidden_states, [_out_height, _out_width]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.conv2_weights,
            bias_tensor=self.conv2_bias,
            **conv_kwargs_3,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=False,
        )
        use_in_shortcut = in_channels != out_channels if use_in_shortcut is None else use_in_shortcut

        if use_in_shortcut:
            # if ttnn.get_memory_config(input_tensor) != self.conv_shortcut.conv.input_sharded_memory_config:
            #     # TODO: Once reshard fix is in, store input tensor in sharded
            #     if input_tensor.memory_config().is_sharded():
            #         input_tensor = ttnn.sharded_to_interleaved(
            #             input_tensor, ttnn.L1_MEMORY_CONFIG, hidden_states.dtype
            #         )
            #     input_tensor = ttnn.interleaved_to_sharded(
            #         input_tensor, self.conv_shortcut.conv.input_sharded_memory_config, hidden_states.dtype
            #     )
            # input_tensor = self.conv_shortcut(input_tensor)
            conv_config = ttnn.Conv2dConfig(
                dtype=ttnn.bfloat8_b,
                weights_dtype=ttnn.bfloat8_b,
                activation="",
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                transpose_shards=False,
                reshard_if_not_optimal=False,
            )
            compute_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            conv_kwargs_4 = {
                "in_channels": self.conv_shortcut_in_channels,
                "out_channels": self.conv_shortcut_out_channels,
                "batch_size": self.batch_size,
                "input_height": self.conv_shortcut_input_height,
                "input_width": self.conv_shortcut_input_width,
                "kernel_size": (1, 1),
                "stride": (1, 1),
                "padding": (0, 0),
                "dilation": (1, 1),
                "groups": 1,
                "device": self.device,
                "conv_config": conv_config,
            }
            if not ttnn.is_tensor_storage_on_device(self.conv_shortcut_weights):
                self.conv_shortcut_weights = ttnn.prepare_conv_weights(
                    weight_tensor=self.conv_shortcut_weights,
                    weights_format="OIHW",
                    input_memory_config=input_tensor.memory_config(),
                    input_layout=input_tensor.get_layout(),
                    has_bias=True,
                    **conv_kwargs_4,
                )
                self.conv_shortcut_bias = ttnn.prepare_conv_bias(
                    bias_tensor=self.conv_shortcut_bias,
                    input_memory_config=input_tensor.memory_config(),
                    input_layout=input_tensor.get_layout(),
                    **conv_kwargs_4,
                )
                self.conv_shortcut_weights = ttnn.to_device(self.conv_shortcut_weights, self.device)
                self.conv_shortcut_bias = ttnn.to_device(self.conv_shortcut_bias, self.device)

            [
                input_tensor,
                [_out_height, _out_width],
            ] = ttnn.conv2d(
                input_tensor=input_tensor,
                weight_tensor=self.conv_shortcut_weights,
                bias_tensor=self.conv_shortcut_bias,
                **conv_kwargs_4,
                compute_config=compute_config,
                return_output_dim=True,
                return_weights_and_bias=False,
            )

        if ttnn.get_memory_config(input_tensor) != ttnn.get_memory_config(hidden_states):
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.get_memory_config(hidden_states))
        output_tensor = ttnn.add(input_tensor, hidden_states, memory_config=hidden_states.memory_config())

        ttnn.deallocate(hidden_states)
        output_tensor = ttnn.reallocate(output_tensor)
        return output_tensor
