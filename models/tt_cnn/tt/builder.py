# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from ttnn.model_preprocessing import Conv2dArgs

import ttnn


@dataclass
class SliceStrategyConfiguration:
    num_slices: int = 0

    def get_slice_type(self):
        ...

    def get_num_slices(self):
        return self.num_slices


# Currently, channel slicing is not natively supported by the Conv2D operation and must be done manually
@dataclass
class HeightSliceStrategyConfiguration(SliceStrategyConfiguration):
    def get_slice_type(self):
        return ttnn.Conv2dDRAMSliceHeight


@dataclass
class WidthSliceStrategyConfiguration(SliceStrategyConfiguration):
    def get_slice_type(self):
        return ttnn.Conv2dDRAMSliceWidth


@dataclass
class ChannelSliceStrategyConfiguration(SliceStrategyConfiguration):
    def get_slice_type(self):
        return ttnn.Conv2dL1Full

    def __post_init__(self):
        if self.num_slices <= 1:
            # for height and width slicing passing 0 will result in auto-slice, but for channel slice it's not implemented
            raise ValueError(f"Channel slicing requires num_slices > 1")


# If slicing is None, DRAM is assumed; Need explicit Strategy for L1
@dataclass
class L1FullSliceStrategyConfiguration(SliceStrategyConfiguration):
    def get_slice_type(self):
        return ttnn.Conv2dL1Full


SliceStrategy = Union[
    HeightSliceStrategyConfiguration,
    WidthSliceStrategyConfiguration,
    ChannelSliceStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
]


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
    act_block_h_override: int = 0
    act_block_w_div: int = 1
    override_output_sharding_config: bool = False

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
    slice_strategy: Optional[SliceStrategy] = None

    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False

    enable_act_double_buffer: bool = False
    enable_weights_double_buffer: bool = True

    deallocate_activation: bool = False
    reallocate_halo_output: bool = True

    config_tensors_in_dram: bool = False

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

    dtype: ttnn.DataType = ttnn.bfloat16
    output_layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT

    slice_strategy: Optional[SliceStrategy] = None

    def __post_init__(self):
        # Validate that only channel slicing is supported for MaxPool2d
        if self.slice_strategy is not None and isinstance(
            self.slice_strategy, (HeightSliceStrategyConfiguration, WidthSliceStrategyConfiguration)
        ):
            raise ValueError(
                "Height and Width slicing are not supported for MaxPool2d. Only channel slicing is supported."
            )

        # Validate channel slicing configuration
        if self.slice_strategy is not None and isinstance(self.slice_strategy, ChannelSliceStrategyConfiguration):
            if self.channels % self.slice_strategy.get_num_slices() != 0:
                raise ValueError(
                    f"Number of channels ({self.channels}) must be divisible by number of slices ({self.slice_strategy.get_num_slices()})"
                )

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


@dataclass(frozen=True)
class UpsampleConfiguration:
    input_height: int
    input_width: int
    channels: int
    batch_size: int
    scale_factor: Union[int, Tuple[int, int]]
    mode: str = "nearest"

    slice_strategy: Optional[SliceStrategy] = None

    def __post_init__(self):
        # Validate that only channel slicing is supported for Upsample
        if self.slice_strategy is not None and isinstance(
            self.slice_strategy, (HeightSliceStrategyConfiguration, WidthSliceStrategyConfiguration)
        ):
            raise ValueError(
                "Height and Width slicing are not supported for Upsample. Only channel slicing is supported."
            )

        # Validate channel slicing configuration
        if self.slice_strategy is not None and isinstance(self.slice_strategy, ChannelSliceStrategyConfiguration):
            if self.channels % self.slice_strategy.get_num_slices() != 0:
                raise ValueError(
                    f"Number of channels ({self.channels}) must be divisible by number of slices ({self.slice_strategy.get_num_slices()})"
                )

        # Validate mode
        supported_modes = {"nearest", "bilinear"}
        if self.mode not in supported_modes:
            raise ValueError(f"Mode must be one of {supported_modes}, got '{self.mode}'")

    @classmethod
    def from_torch(
        cls,
        torch_layer: torch.nn.Upsample,
        input_height: int,
        input_width: int,
        batch_size: int,
        channels: int,
        **kwargs,
    ):
        scale_factor = torch_layer.scale_factor
        if isinstance(scale_factor, (list, tuple)):
            scale_factor = tuple(scale_factor)
        elif isinstance(scale_factor, (int, float)):
            # Normalize single number to tuple for consistency
            scale_factor = (scale_factor, scale_factor)
        mode = torch_layer.mode
        return cls(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            batch_size=batch_size,
            scale_factor=scale_factor,
            mode=mode,
            **kwargs,
        )


LayerConfiguration = Union[Conv2dConfiguration, MaxPool2dConfiguration, UpsampleConfiguration]


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
        output["act_block_h_override"] = sharding_strategy.act_block_h_override
        output["act_block_w_div"] = sharding_strategy.act_block_w_div
        output["override_output_sharding_config"] = sharding_strategy.override_output_sharding_config
    else:
        raise ValueError(f"Invalid sharding ShardedStrategyConfiguration was encountered: {sharding_strategy}")

    return output


def to_conv2d_config(configuration: Conv2dConfiguration):
    parameters_from_sharding_configuration = sharding_strategy_to_conv2d_config(configuration.sharding_strategy)
    return ttnn.Conv2dConfig(
        weights_dtype=configuration.weights_dtype,
        activation=configuration.activation,
        deallocate_activation=configuration.deallocate_activation,
        reallocate_halo_output=configuration.reallocate_halo_output,
        config_tensors_in_dram=configuration.config_tensors_in_dram,
        reshard_if_not_optimal=(
            configuration.sharding_strategy.reshard_if_not_optimal
            if not isinstance(configuration.sharding_strategy, AutoShardedStrategyConfiguration)
            else False
        ),
        shard_layout=configuration.sharding_strategy.get_tensor_memory_layout(),
        enable_act_double_buffer=configuration.enable_act_double_buffer,
        output_layout=configuration.output_layout,
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


def to_slice_config(slice_strategy: Optional[SliceStrategy]):
    if slice_strategy is None:
        return None
    # Channel slicing uses the predefined Conv2dL1FullSliceConfig
    if isinstance(slice_strategy, ChannelSliceStrategyConfiguration):
        return ttnn.Conv2dL1FullSliceConfig
    return ttnn.Conv2dSliceConfig(
        slice_type=slice_strategy.get_slice_type(),
        num_slices=slice_strategy.get_num_slices(),
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
        self.slice_config = to_slice_config(configuration.slice_strategy)

        self.device = device

        self.weight = configuration.weight
        self.bias = configuration.bias

        # Initialize weight_slices as empty list
        self.weight_slices = []

        # Check for channel slicing
        if (
            configuration.slice_strategy is not None
            and isinstance(configuration.slice_strategy, ChannelSliceStrategyConfiguration)
            and configuration.slice_strategy.get_num_slices() > 0
        ):
            split_in_channels = configuration.in_channels // configuration.slice_strategy.get_num_slices()

            # slice weights - this should only run on first inference
            if ttnn.is_tensor_storage_on_device(self.weight):
                # Weights are on device - use ttnn.slice
                for i in range(configuration.slice_strategy.get_num_slices()):
                    start_idx = i * split_in_channels
                    end_idx = (i + 1) * split_in_channels
                    weight_slice = ttnn.slice(
                        self.weight,
                        [0, start_idx, 0, 0],
                        [
                            configuration.out_channels,
                            end_idx,
                            configuration.kernel_size[0],
                            configuration.kernel_size[1],
                        ],
                    )
                    self.weight_slices.append(weight_slice)
            else:
                # Weights are on host - convert to torch, slice, then convert back to TTNN
                torch_weight = ttnn.to_torch(self.weight)
                for i in range(configuration.slice_strategy.get_num_slices()):
                    start_idx = i * split_in_channels
                    end_idx = (i + 1) * split_in_channels
                    torch_slice = torch_weight[:, start_idx:end_idx, :, :]
                    # This TTNN tensor will remain on host until pulled by Conv2D
                    weight_slice = ttnn.from_torch(torch_slice, dtype=self.weight.dtype, layout=self.weight.layout)
                    self.weight_slices.append(weight_slice)

            # bias needs to be on device for channel slicing due to ttnn.add requirements
            if not ttnn.is_tensor_storage_on_device(self.bias):
                self.bias = ttnn.to_device(self.bias, self.device)

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
            "slice_config": self.slice_config,
        }

    def _apply_channel_slicing(self, x):
        """Apply channel slicing to the input tensor and return the result."""
        # check for flattened input tensor
        is_flattened = (
            x.shape[0] == 1
            and x.shape[1] == 1
            and x.shape[2]
            == self.configuration.batch_size * self.configuration.input_height * self.configuration.input_width
        )
        # slice input
        input_slices = []
        for i in range(self.configuration.slice_strategy.get_num_slices()):
            input_slices.append(
                ttnn.slice(
                    x,
                    [
                        0,
                        0,
                        0,
                        i * self.configuration.in_channels // self.configuration.slice_strategy.get_num_slices(),
                    ],
                    [
                        self.configuration.batch_size if not is_flattened else 1,
                        self.configuration.input_height if not is_flattened else 1,
                        self.configuration.input_width
                        if not is_flattened
                        else self.configuration.batch_size
                        * self.configuration.input_height
                        * self.configuration.input_width,
                        (i + 1) * self.configuration.in_channels // self.configuration.slice_strategy.get_num_slices(),
                    ],
                )
            )

        # perform conv2d on each slice
        accumulated_output = None
        channels_per_slice = self.configuration.in_channels // self.configuration.slice_strategy.get_num_slices()

        for i in range(self.configuration.slice_strategy.get_num_slices()):
            # Create kwargs with correct in_channels for this slice
            slice_kwargs = self.get_conv2d_kwargs()
            slice_kwargs["in_channels"] = channels_per_slice

            output_slice, (h_out, w_out), (weight, bias) = ttnn.conv2d(
                input_tensor=input_slices[i],
                weight_tensor=self.weight_slices[i],
                bias_tensor=None,
                return_output_dim=True,
                return_weights_and_bias=True,
                compute_config=self.compute_config,
                **slice_kwargs,
            )
            # Store only the weight, not the tuple
            self.weight_slices[i] = weight
            # Without this, some edge case convs OOM
            output_slice = ttnn.move(output_slice)
            if i == 0:
                accumulated_output = ttnn.to_memory_config(output_slice, ttnn.DRAM_MEMORY_CONFIG)
            else:
                accumulated_output = ttnn.add(output_slice, accumulated_output, output_tensor=accumulated_output)
            output_slice.deallocate(True)

        # Apply bias
        if self.bias is not None:
            # ttnn.add will fail if layout is not tile or dtype doesn't match
            # this should only run on first inference
            if self.bias.layout != ttnn.TILE_LAYOUT or self.bias.dtype != accumulated_output.dtype:
                self.bias = ttnn.to_layout(self.bias, ttnn.TILE_LAYOUT, dtype=accumulated_output.dtype)

            accumulated_output = ttnn.add(accumulated_output, self.bias, output_tensor=accumulated_output)

        return accumulated_output, (h_out, w_out)

    def __call__(self, x, return_output_dim: bool = False):
        if not self.weight_slices:
            # No slicing
            x, [h_out, w_out], [self.weight, self.bias] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.weight,
                bias_tensor=self.bias,
                return_output_dim=True,
                return_weights_and_bias=True,
                compute_config=self.compute_config,
                **self.get_conv2d_kwargs(),
            )
        else:
            x, (h_out, w_out) = self._apply_channel_slicing(x)

        if return_output_dim:
            return x, (h_out, w_out)

        return x


class TtMaxPool2d:
    def __init__(self, configuration: MaxPool2dConfiguration, device: ttnn.Device):
        self.configuration = configuration
        self.device = device

        # Check for channel slicing
        self.use_channel_slicing = configuration.slice_strategy is not None and isinstance(
            configuration.slice_strategy, ChannelSliceStrategyConfiguration
        )

        if self.use_channel_slicing:
            self.num_slices = configuration.slice_strategy.get_num_slices()
            self.channels_per_slice = configuration.channels // self.num_slices

    def get_maxpool2d_kwargs(self):
        return {
            "batch_size": self.configuration.batch_size,
            "input_h": self.configuration.input_height,
            "input_w": self.configuration.input_width,
            "kernel_size": self.configuration.kernel_size,
            "stride": self.configuration.stride,
            "padding": self.configuration.padding,
            "dilation": self.configuration.dilation,
            "ceil_mode": self.configuration.ceil_mode,
            "deallocate_input": self.configuration.deallocate_input,
            "reallocate_halo_output": self.configuration.reallocate_halo_output,
            "dtype": self.configuration.dtype,
            "output_layout": self.configuration.output_layout,
        }

    def _apply_channel_slicing(self, x):
        """Apply channel slicing to the input tensor and return the result."""
        # Slice input tensor along channel dimension
        input_slices = []
        for i in range(self.num_slices):
            start_channel = i * self.channels_per_slice
            end_channel = (i + 1) * self.channels_per_slice

            input_slice = ttnn.slice(
                x,
                [0, 0, 0, start_channel],
                [
                    1,
                    1,
                    self.configuration.batch_size * self.configuration.input_height * self.configuration.input_width,
                    end_channel,
                ],
            )
            input_slices.append(input_slice)

        # Perform max pooling on each slice
        output_slices = []
        for i in range(self.num_slices):
            output_slice = ttnn.max_pool2d(
                input_tensor=input_slices[i],
                channels=self.channels_per_slice,
                **self.get_maxpool2d_kwargs(),
            )
            # Output slice to DRAM
            output_slice = ttnn.to_memory_config(output_slice, ttnn.DRAM_MEMORY_CONFIG)
            output_slices.append(output_slice)

        # Concatenate output slices along channel dimension
        x = ttnn.concat(output_slices, dim=3)

        # Clean up intermediate tensors
        for slice_tensor in input_slices + output_slices:
            slice_tensor.deallocate(True)

        return x

    def __call__(self, x):
        if not self.use_channel_slicing:
            # No slicing
            x = ttnn.max_pool2d(
                input_tensor=x,
                channels=self.configuration.channels,
                **self.get_maxpool2d_kwargs(),
            )
        else:
            x = self._apply_channel_slicing(x)

        return x


class TtUpsample:
    def __init__(self, configuration: UpsampleConfiguration, device: ttnn.Device):
        self.configuration = configuration
        self.device = device

        # Check for channel slicing
        self.use_channel_slicing = configuration.slice_strategy is not None and isinstance(
            configuration.slice_strategy, ChannelSliceStrategyConfiguration
        )

        if self.use_channel_slicing:
            self.num_slices = configuration.slice_strategy.get_num_slices()
            self.channels_per_slice = configuration.channels // self.num_slices

    def get_upsample_kwargs(self):
        # Convert scale_factor to the format expected by ttnn.upsample
        scale_factor = self.configuration.scale_factor
        if isinstance(scale_factor, tuple) and len(scale_factor) == 2:
            # Convert to list of integers as expected by ttnn
            scale_factor = [int(scale_factor[0]), int(scale_factor[1])]
        elif isinstance(scale_factor, (int, float)):
            scale_factor = int(scale_factor)

        return {
            "scale_factor": scale_factor,
            "mode": self.configuration.mode,
        }

    def _apply_channel_slicing(self, x):
        """Apply channel slicing to the input tensor and return the result."""
        # check for flattened input tensor
        is_flattened = (
            x.shape[0] == 1
            and x.shape[1] == 1
            and x.shape[2]
            == self.configuration.batch_size * self.configuration.input_height * self.configuration.input_width
        )
        # Slice input tensor along channel dimension
        input_slices = []
        for i in range(self.num_slices):
            start_channel = i * self.channels_per_slice
            end_channel = (i + 1) * self.channels_per_slice

            input_slice = ttnn.slice(
                x,
                [0, 0, 0, start_channel],
                [
                    self.configuration.batch_size if not is_flattened else 1,
                    self.configuration.input_height if not is_flattened else 1,
                    self.configuration.input_width
                    if not is_flattened
                    else self.configuration.batch_size
                    * self.configuration.input_height
                    * self.configuration.input_width,
                    end_channel,
                ],
            )
            input_slices.append(input_slice)

        # Perform upsampling on each slice
        output_slices = []
        for i in range(self.num_slices):
            output_slice = ttnn.upsample(
                input_tensor=input_slices[i],
                **self.get_upsample_kwargs(),
            )
            # Output slice to DRAM
            output_slice = ttnn.to_memory_config(output_slice, ttnn.DRAM_MEMORY_CONFIG)
            output_slices.append(output_slice)

        # Concatenate output slices along channel dimension
        x = ttnn.concat(output_slices, dim=3)

        # Clean up intermediate tensors
        for slice_tensor in input_slices + output_slices:
            slice_tensor.deallocate(True)

        return x

    def __call__(self, x):
        if not self.use_channel_slicing:
            # No slicing
            x = ttnn.upsample(
                input_tensor=x,
                **self.get_upsample_kwargs(),
            )
        else:
            x = self._apply_channel_slicing(x)

        return x
