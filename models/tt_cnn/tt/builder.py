# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from ttnn.model_preprocessing import Conv2dArgs

import ttnn


@dataclass(frozen=True)
class ShardedStrategyConfiguration:
    def get_tensor_memory_layout(self):
        ...

    @classmethod
    def get_core_grid_from_num_cores(cls, num_cores: int, grid_rows: int, grid_cols: int):
        rows = num_cores // grid_cols
        assert rows <= grid_rows, "Not enough cores for specified core grid"
        ranges = []
        if rows != 0:
            ranges.append(
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_rows - 1, rows - 1),
                )
            )
        remainder = num_cores % grid_rows
        if remainder != 0:
            assert rows + 1 <= grid_rows, "Not enough cores for specified core grid"
            ranges.append(
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, rows),
                    ttnn.CoreCoord(remainder - 1, rows),
                )
            )
        return ttnn.CoreRangeSet({*ranges})


@dataclass(frozen=True)
class AutoShardedStrategyConfiguration(ShardedStrategyConfiguration):
    def get_tensor_memory_layout(self):
        return None


@dataclass(frozen=True)
class HeightShardedStrategyConfiguration(ShardedStrategyConfiguration):
    reshard_if_not_optimal: bool = False
    override_core_grid: Optional[ttnn.CoreRangeSet] = None
    act_block_h_override: int = 0

    def get_tensor_memory_layout(self):
        return ttnn.TensorMemoryLayout.HEIGHT_SHARDED


@dataclass(frozen=True)
class WidthShardedStrategyConfiguration(ShardedStrategyConfiguration):
    reshard_if_not_optimal: bool = False
    override_core_grid: Optional[ttnn.CoreRangeSet] = None
    act_block_w_div: int = 1

    def get_tensor_memory_layout(self):
        return ttnn.TensorMemoryLayout.WIDTH_SHARDED


@dataclass(frozen=True)
class BlockShardedStrategyConfiguration(ShardedStrategyConfiguration):
    reshard_if_not_optimal: bool = False
    override_core_grid: Optional[ttnn.CoreRangeSet] = None

    def get_tensor_memory_layout(self):
        return ttnn.TensorMemoryLayout.BLOCK_SHARDED


ShardingStrategy = Union[
    HeightShardedStrategyConfiguration,
    WidthShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    AutoShardedStrategyConfiguration,
]


@dataclass(frozen=True)
class Conv2dConfiguration:
    input_height: int
    input_width: int
    in_channels: int
    out_channels: int
    batch_size: int
    kernel_size: Tuple[int, int]
    weight: torch.Tensor
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    groups: int = 1
    dilation: Tuple[int, int] = (1, 1)
    activation: Optional[ttnn.UnaryWithParam] = None
    bias: Optional[torch.Tensor] = None

    activation_dtype: ttnn.DataType = ttnn.bfloat16
    weights_dtype: ttnn.DataType = ttnn.bfloat16
    output_dtype: ttnn.DataType = ttnn.bfloat16
    output_layout: ttnn.Layout = ttnn.TILE_LAYOUT

    sharding_strategy: ShardedStrategyConfiguration = AutoShardedStrategyConfiguration()

    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False

    enable_act_double_buffer: bool = False
    enable_weights_double_buffer: bool = True

    deallocate_activation: bool = False
    reallocate_halo_output: bool = True

    @classmethod
    def convert_torch_weight_and_bias_to_ttnn(cls, weight, bias=None, mesh_mapper=None):
        weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        if bias is not None:
            bias = torch.reshape(bias, (1, 1, 1, -1))
            bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        return weight, bias

    @classmethod
    def with_random_weights(
        cls,
        input_height: int,
        input_width: int,
        in_channels: int,
        out_channels: int,
        batch_size: int,
        kernel_size: Tuple[int, int],
        **kwargs,
    ):
        weight_shape = (
            out_channels,
            in_channels // kwargs.get("groups", 1),
            kernel_size[0],
            kernel_size[1],
        )
        weight = torch.randn(weight_shape, dtype=torch.bfloat16).float()
        bias = torch.randn(out_channels, dtype=torch.bfloat16).float()

        weight, bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            weight, bias, mesh_mapper=kwargs.get("mesh_mapper", None)
        )

        return cls(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=kernel_size,
            weight=weight,
            bias=bias,
            **kwargs,
        )

    @classmethod
    def from_torch(
        cls,
        torch_layer: torch.nn.Conv2d,
        input_height: int,
        input_width: int,
        batch_size: int,
        mesh_mapper=None,
        **kwargs,
    ):
        weight, bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            torch_layer.weight.data,
            bias=(torch_layer.bias.data if torch_layer.bias is not None else None),
            mesh_mapper=mesh_mapper,
        )
        return cls(
            input_height=input_height,
            input_width=input_width,
            in_channels=torch_layer.in_channels,
            out_channels=torch_layer.out_channels,
            batch_size=batch_size,
            kernel_size=torch_layer.kernel_size,
            stride=torch_layer.stride,
            padding=torch_layer.padding,
            groups=torch_layer.groups,
            dilation=torch_layer.dilation,
            weight=weight,
            bias=bias,
            **kwargs,
        )

    @classmethod
    def from_model_args(
        cls, conv2d_args: Conv2dArgs, weights: ttnn.Tensor, bias: Optional[ttnn.Tensor] = None, **kwargs
    ):
        return cls(
            input_height=conv2d_args.input_height,
            input_width=conv2d_args.input_width,
            in_channels=conv2d_args.in_channels,
            out_channels=conv2d_args.out_channels,
            batch_size=conv2d_args.batch_size,
            kernel_size=conv2d_args.kernel_size,
            stride=conv2d_args.stride,
            padding=conv2d_args.padding,
            groups=conv2d_args.groups,
            dilation=conv2d_args.dilation,
            output_dtype=conv2d_args.dtype,
            weight=weights,
            bias=bias,
            **kwargs,
        )

    def validate_weights(self):
        """Validate that weight and bias tensor shapes match layer configuration.

        Ensures weight tensor dimensions align with the specified convolution
        parameters.
        """
        if not isinstance(self.weight, ttnn.Tensor):
            raise ValueError(f"Expected weights to be of type ttnn.Tensor")
        if self.bias is not None and not isinstance(self.bias, ttnn.Tensor):
            raise ValueError(f"Expected bias to be of type ttnn.Tensor")

        expected_weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        if self.weight.shape != expected_weight_shape:
            raise ValueError(
                f"Weight shape {self.weight.shape} doesn't match expected shape "
                f"{expected_weight_shape} for configuration parameters"
            )

        if self.bias is not None and self.bias.shape != (
            1,
            1,
            1,
            self.out_channels,
        ):
            raise ValueError(
                f"Bias shape {self.bias.shape} doesn't match expected shape "
                f"({1, 1, 1, self.out_channels}) for out_channels={self.out_channels}"
            )

    def __post_init__(self):
        self.validate_weights()

    def __str__(self):
        weight_shape = tuple(self.weight.shape) if hasattr(self.weight, "shape") else "N/A"
        bias_shape = tuple(self.bias.shape) if self.bias is not None and hasattr(self.bias, "shape") else "None"

        return (
            f"Conv2dConfiguration(\n"
            f"  input_shape: ({self.batch_size}, {self.input_height}, {self.input_width}, {self.in_channels})\n"
            f"  kernel: {self.kernel_size}, stride: {self.stride}, padding: {self.padding}\n"
            f"  groups: {self.groups}, dilation: {self.dilation}\n"
            f"  weight_shape: {weight_shape}, bias_shape: {bias_shape}\n"
            f"  dtypes: act={self.activation_dtype}, weights={self.weights_dtype}, out={self.output_dtype}\n"
            f"  sharding: {self.sharding_strategy}\n"
            f"  math_fidelity: {self.math_fidelity}\n"
            f")"
        )


@dataclass(frozen=True)
class MaxPool2dConfiguration:
    input_height: int
    input_width: int
    channels: int
    batch_size: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    dilation: Tuple[int, int] = (1, 1)
    ceil_mode: bool = False

    in_place: bool = False
    deallocate_input: bool = False
    reallocate_halo_output: bool = True

    @classmethod
    def from_torch(
        cls,
        torch_layer: torch.nn.MaxPool2d,
        input_height: int,
        input_width: int,
        batch_size: int,
        channels: int,
        **kwargs,
    ):
        kernel_size = (torch_layer.kernel_size, torch_layer.kernel_size)
        stride = (torch_layer.stride, torch_layer.stride)
        padding = (torch_layer.padding, torch_layer.padding)
        dilation = (torch_layer.dilation, torch_layer.dilation)
        return cls(
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            **kwargs,
        )


LayerConfiguration = Union[Conv2dConfiguration, MaxPool2dConfiguration]


def sharding_strategy_to_conv2d_config(sharding_strategy: ShardingStrategy):
    output = dict()

    if isinstance(sharding_strategy, AutoShardedStrategyConfiguration):
        return output

    if sharding_strategy.override_core_grid is not None:
        core_grid = sharding_strategy.override_core_grid
        override_sharding_config = True
    else:
        core_grid = None
        override_sharding_config = False
    output["core_grid"] = core_grid
    output["override_sharding_config"] = override_sharding_config

    if isinstance(sharding_strategy, HeightShardedStrategyConfiguration):
        output["act_block_h_override"] = sharding_strategy.act_block_h_override
    elif isinstance(sharding_strategy, WidthShardedStrategyConfiguration):
        output["act_block_w_div"] = sharding_strategy.act_block_w_div
    elif isinstance(sharding_strategy, BlockShardedStrategyConfiguration):
        ...
    else:
        raise ValueError(f"Invalid sharding ShardedStrategyConfiguration was encountered: {sharding_strategy}")

    return output


def to_conv2d_config(configuration: Conv2dConfiguration):
    parameters_from_sharding_configuration = sharding_strategy_to_conv2d_config(configuration.sharding_strategy)
    return ttnn.Conv2dConfig(
        weights_dtype=configuration.weights_dtype,
        shard_layout=configuration.sharding_strategy.get_tensor_memory_layout(),
        deallocate_activation=configuration.deallocate_activation,
        enable_act_double_buffer=configuration.enable_act_double_buffer,
        activation=configuration.activation,
        output_layout=configuration.output_layout,
        reshard_if_not_optimal=(
            configuration.sharding_strategy.reshard_if_not_optimal
            if not isinstance(configuration.sharding_strategy, AutoShardedStrategyConfiguration)
            else None
        ),
        reallocate_halo_output=configuration.reallocate_halo_output,
        enable_weights_double_buffer=configuration.enable_weights_double_buffer,
        **parameters_from_sharding_configuration,
    )


def to_compute_config(configuration: Conv2dConfiguration, device: ttnn.Device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=configuration.math_fidelity,
        fp32_dest_acc_en=configuration.fp32_dest_acc_en,
        packer_l1_acc=configuration.packer_l1_acc,
    )


class TtConv2d:
    def __init__(
        self,
        configuration: Conv2dConfiguration,
        device: ttnn.Device,
    ):
        self.configuration = configuration
        self.conv2d_config = to_conv2d_config(configuration)
        self.compute_config = to_compute_config(configuration, device)

        self.device = device

        self.weight = configuration.weight
        self.bias = configuration.bias

        if not isinstance(self.weight, ttnn.Tensor):
            raise ValueError(f"Weight tensor should be of type ttnn.Tensor (was {type(self.weight)}")

        if self.bias is not None and not isinstance(self.bias, ttnn.Tensor):
            raise ValueError(f"Bias tensor should be of type ttnn.Tensor (was {type(self.bias)}")

    def get_conv2d_kwargs(self):
        return {
            "input_height": self.configuration.input_height,
            "input_width": self.configuration.input_width,
            "in_channels": self.configuration.in_channels,
            "out_channels": self.configuration.out_channels,
            "batch_size": self.configuration.batch_size,
            "kernel_size": self.configuration.kernel_size,
            "stride": self.configuration.stride,
            "padding": self.configuration.padding,
            "dilation": self.configuration.dilation,
            "groups": self.configuration.groups,
            "dtype": self.configuration.output_dtype,
            "device": self.device,
            "conv_config": self.conv2d_config,
        }

    def __call__(self, x):
        print(
            f"running conv: shape={x.shape} - memory_config={x.memory_config()} - input H/W {self.configuration.input_height}, {self.configuration.input_width}"
        )
        print(f"{self.configuration}")
        x, [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            return_output_dim=False,
            return_weights_and_bias=True,
            compute_config=self.compute_config,
            **self.get_conv2d_kwargs(),
        )
        return x


class TtMaxPool2d:
    def __init__(self, configuration: MaxPool2dConfiguration, device: ttnn.Device):
        self.configuration = configuration
        self.device = device

    def __call__(self, x):
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.configuration.batch_size,
            input_h=self.configuration.input_height,
            input_w=self.configuration.input_width,
            channels=self.configuration.channels,
            kernel_size=self.configuration.kernel_size,
            stride=self.configuration.stride,
            padding=self.configuration.padding,
            dilation=self.configuration.dilation,
            ceil_mode=self.configuration.ceil_mode,
            in_place_halo=self.configuration.in_place,
            deallocate_input=self.configuration.deallocate_input,
            reallocate_halo_output=self.configuration.reallocate_halo_output,
        )
        return x
