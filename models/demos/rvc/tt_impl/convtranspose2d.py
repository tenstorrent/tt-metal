# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from .utils import ConvConfiguration, get_shard_strategy_for_conv, resolve_padding_2d


class ConvTranspose2d:
    """Stateful ConvTranspose2d wrapper built on top of `ttnn.conv_transpose2d`."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        dtype: ttnn.DataType | None = None,
        deallocate_input: bool = False,
        bias: bool = True,
    ) -> None:
        self.device = device
        self.output_padding = output_padding
        if isinstance(padding, str):
            raise ValueError("String padding mode is not supported for ConvTranspose1d")
        tile_width = 32
        if out_channels % tile_width == 0:
            output_layout = ttnn.TILE_LAYOUT
        elif (tile_width - out_channels % tile_width) / out_channels < 1.2:
            output_layout = ttnn.TILE_LAYOUT
        else:
            output_layout = ttnn.ROW_MAJOR_LAYOUT

        padding_final = resolve_padding_2d(
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.configuration = ConvConfiguration(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_final,
            dilation=dilation,
            groups=groups,
            dtype=dtype or ttnn.bfloat16,
            output_layout=output_layout,
            deallocate_input=False,  # deallocate_input,
        )
        self.is_biased = bias

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], key: str, module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        base_key = f"{module_prefix}{key}"
        bias_key = f"{base_key}.bias"
        reshaped_weight_torch = state_dict[f"{base_key}.weight"]
        reshaped_weight = reshaped_weight_torch.reshape(
            self.configuration.in_channels,
            self.configuration.out_channels // self.configuration.groups,
            self.configuration.kernel_size[0],
            self.configuration.kernel_size[1],
        )
        self.weight = ttnn.from_torch(
            reshaped_weight,
            dtype=ttnn.bfloat16,
            # device=self.device,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if self.is_biased:
            self.bias = ttnn.from_torch(
                state_dict[bias_key].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                # device=self.device,
                # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bias = None

    def __call__(self, input: ttnn.Tensor) -> ttnn.Tensor:
        input = ttnn.to_memory_config(input, ttnn.DRAM_MEMORY_CONFIG)
        batch_size, input_height, input_width, _ = input.shape
        shard_layout = get_shard_strategy_for_conv(input.shape)
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            # output_layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_layout=shard_layout,
            # reshard_if_not_optimal=True,
        )
        output, [output_height, output_width], [self.weight, self.bias] = ttnn.conv_transpose2d(
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
            output_padding=self.output_padding,
            dilation=self.configuration.dilation,
            groups=self.configuration.groups,
            device=self.device,
            conv_config=conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            # mirror_kernel=True,
        )
        output = ttnn.reshape(output, (batch_size, output_height, output_width, self.configuration.out_channels))
        return output
