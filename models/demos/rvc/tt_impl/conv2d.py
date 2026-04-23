# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from .utils import ConvConfiguration, _normalize_conv2d_activation, get_shard_strategy_for_conv, resolve_padding_2d


class Conv2d:
    """Stateful Conv2d wrapper around `ttnn.conv2d`."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_channels,
        out_channels,
        kernel_size,
        stride: int = 1,
        padding: PaddingType = 0,
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        activation: str | tuple[str, dict] | None = None,
        deallocate_input: bool = False,
        bias: bool = True,
    ) -> None:
        self.device = device

        padding_final = resolve_padding_2d(
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.is_biased = bias
        TILE_WIDTH = 32
        if out_channels % TILE_WIDTH == 0:
            output_layout = ttnn.TILE_LAYOUT
        elif (TILE_WIDTH - out_channels % TILE_WIDTH) / out_channels < 1.2:
            output_layout = ttnn.TILE_LAYOUT
        else:
            output_layout = ttnn.ROW_MAJOR_LAYOUT

        self.configuration = ConvConfiguration(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_final,
            dilation=dilation,
            groups=groups,
            activation=_normalize_conv2d_activation(activation),
            dtype=ttnn.bfloat16,
            output_layout=output_layout,
            deallocate_input=deallocate_input,
        )

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        key: str,
        module_prefix: str | None = None,
    ) -> None:
        if module_prefix is None:
            module_prefix = ""
        base_key = f"{module_prefix}{key}"
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        reshaped_weight = state_dict[weight_key].reshape(
            self.configuration.out_channels,
            self.configuration.in_channels // self.configuration.groups,
            self.configuration.kernel_size[0],
            self.configuration.kernel_size[1],
        )
        self.weight = ttnn.from_torch(
            reshaped_weight, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.is_biased:
            bias = state_dict[bias_key]
            self.bias = ttnn.from_torch(
                torch.reshape(bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bias = None

    def __call__(self, input: ttnn.Tensor) -> ttnn.Tensor:
        if input.is_sharded():
            input = ttnn.sharded_to_interleaved(input, ttnn.L1_MEMORY_CONFIG)
        batch_size, input_height, input_width, _ = input.shape
        shard_layout = get_shard_strategy_for_conv(input.shape)
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.configuration.weights_dtype,
            activation=self.configuration.activation,
            shard_layout=shard_layout,
            # reshard_if_not_optimal=True,
            act_block_h_override=32,
            # deallocate_activation=True,
        )
        output, [output_height, output_width], [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=input,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.configuration.in_channels,
            out_channels=self.configuration.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=self.configuration.kernel_size,
            stride=self.configuration.stride,
            padding=self.configuration.padding,
            dilation=self.configuration.dilation,
            groups=self.configuration.groups,
            device=self.device,
            conv_config=conv_config,
            return_output_dim=True,
            # slice_config=slice_config,
            return_weights_and_bias=True,
        )
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (batch_size, output_height, output_width, self.configuration.out_channels))
        return output

    def deallocate(self) -> None:
        if self.weight is not None:
            ttnn.deallocate(self.weight)
            self.weight = None
        if self.bias is not None:
            ttnn.deallocate(self.bias)
            self.bias = None
