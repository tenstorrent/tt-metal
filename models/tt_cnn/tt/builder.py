from dataclasses import dataclass
from typing import Optional, Tuple, Union

# TODO: This shouldn't depend on torch
import torch

import ttnn


@dataclass(frozen=True)
class ShardedStrategyConfiguration:
    def get_tensor_memory_layout(self):
        ...


@dataclass(frozen=True)
class AutoShardedStrategyConfiguration(ShardedStrategyConfiguration):
    def get_tensor_memory_layout(self):
        return None


@dataclass(frozen=True)
class HeightShardedStrategyConfiguration(ShardedStrategyConfiguration):
    reshard_if_not_optimal: bool = False
    override_core_grid: Optional[int] = None
    act_block_h_override: int = 0

    def get_tensor_memory_layout(self):
        return ttnn.TensorMemoryLayout.HEIGHT_SHARDED


@dataclass(frozen=True)
class WidthShardedStrategyConfiguration(ShardedStrategyConfiguration):
    reshard_if_not_optimal: bool = False
    override_core_grid: Optional[int] = None
    act_block_w_div: int = 1

    def get_tensor_memory_layout(self):
        return ttnn.TensorMemoryLayout.WIDTH_SHARDED


@dataclass(frozen=True)
class BlockShardedStrategyConfiguration(ShardedStrategyConfiguration):
    reshard_if_not_optimal: bool = False
    override_core_grid: Optional[int] = None

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
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    groups: int = 1
    dilation: Tuple[int, int] = (1, 1)
    activation: str = ""

    activation_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat16
    output_dtype = ttnn.bfloat16

    output_layout = ttnn.TILE_LAYOUT

    sharding_strategy: ShardedStrategyConfiguration = AutoShardedStrategyConfiguration()

    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False

    enable_act_double_buffer: bool = False
    enable_weights_double_buffer: bool = True
    enable_split_reader: bool = False

    deallocate_activation: bool = True
    reallocate_halo_output: bool = True


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int, grid_cols: int):
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


class DeviceDescriptor:
    def __init__(self, device: ttnn.Device, grid_size: Tuple[int, int]):
        self.device = device
        self.grid_size = grid_size

    def get_grid_size(self):
        return self.grid_size


def sharding_strategy_to_conv2d_config(sharding_strategy: ShardingStrategy, device_descriptor: DeviceDescriptor):
    output = dict()

    if isinstance(sharding_strategy, AutoShardedStrategyConfiguration):
        return output

    if sharding_strategy.override_core_grid is not None:
        grid_size = device_descriptor.get_grid_size()
        core_grid = get_core_grid_from_num_cores(
            sharding_strategy.override_core_grid,
            grid_rows=grid_size[0],  # TODO: Fetch this from the device automatically?
            grid_cols=grid_size[1],
        )
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


def to_conv2d_config(configuration: Conv2dConfiguration, device_descriptor: DeviceDescriptor):
    parameters_from_sharding_configuration = sharding_strategy_to_conv2d_config(
        configuration.sharding_strategy, device_descriptor
    )
    return ttnn.Conv2dConfig(
        weights_dtype=configuration.weights_dtype,
        shard_layout=configuration.sharding_strategy.get_tensor_memory_layout(),
        deallocate_activation=configuration.deallocate_activation,
        enable_act_double_buffer=configuration.enable_act_double_buffer,
        enable_split_reader=configuration.enable_split_reader,
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


def to_compute_config(configuration: Conv2dConfiguration, device_descriptor: DeviceDescriptor):
    return ttnn.init_device_compute_kernel_config(
        device_descriptor.device.arch(),
        math_fidelity=configuration.math_fidelity,
        fp32_dest_acc_en=configuration.fp32_dest_acc_en,
        packer_l1_acc=configuration.packer_l1_acc,
    )


class TtConv2d:
    def __init__(
        self,
        configuration: Conv2dConfiguration,
        weight,
        bias,
        device_descriptor,
        mesh_mapper=None,
    ):
        self.configuration = configuration
        self.conv2d_config = to_conv2d_config(configuration, device_descriptor)
        self.compute_config = to_compute_config(configuration, device_descriptor)

        self.device = device_descriptor.device

        # TODO: Here we should check the weight shapes to see if they are correct
        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

        if bias is not None:
            bias = torch.reshape(bias, (1, 1, 1, -1))
            self.bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

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
        print(self.compute_config, self.get_conv2d_kwargs())
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
