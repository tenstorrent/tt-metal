# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

InputLayout = Literal["NLC", "NHWC"]
OutputLayout = Literal["NLC", "NHWC"]
PaddingType = int | tuple[int, int] | Literal["same"]


def input1d_to_2d(
    input_tensor: ttnn.Tensor,
) -> ttnn.Tensor:
    batch_size, input_length, input_channel = input_tensor.shape
    return ttnn.reshape(input_tensor, (batch_size, 1, input_length, input_channel))


def input_shape_to_memory_config(
    input_shape, output_length, in_channels, kernel_size, device: ttnn.MeshDevice
) -> ttnn.MemoryConfig:
    batch_size, input_height, input_width, input_channels = input_shape
    memory_cost = (
        batch_size * input_height * input_width * input_channels * 2
    )  # assuming bfloat16, so 2 bytes per element
    if (output_length, in_channels, kernel_size) in dims_to_num_slices:
        return ttnn.DRAM_MEMORY_CONFIG

    # Keep tiny-channel inputs interleaved to avoid expensive/invalid sharding setups.

    if memory_cost > 64 * 1_400_000:  # if input is larger than 1.4MB, use DRAM to avoid L1 thrashing
        return ttnn.DRAM_MEMORY_CONFIG
    if input_channels < 16:
        return ttnn.DRAM_MEMORY_CONFIG

    nhw = batch_size * input_height * input_width
    c = input_channels

    # Use best sharding strategy based on NHW-to-C ratio:
    # - HEIGHT_SHARDED if NHW >> C
    # - WIDTH_SHARDED if C >> NHW
    # - BLOCK_SHARDED if NHW ~= C
    if nhw >= 4 * c:
        strategy = ttnn.ShardStrategy.HEIGHT
    elif c >= 4 * nhw:
        strategy = ttnn.ShardStrategy.WIDTH
    else:
        strategy = ttnn.ShardStrategy.BLOCK

    grid_size = device.compute_with_storage_grid_size()
    candidate_grids = [
        ttnn.CoreGrid(y=grid_size.y, x=grid_size.x),
        ttnn.CoreGrid(y=grid_size.y, x=1),
        ttnn.CoreGrid(y=1, x=grid_size.x),
        ttnn.CoreGrid(y=1, x=1),
    ]

    for core_grid in candidate_grids:
        try:
            return ttnn.create_sharded_memory_config_(
                input_shape,
                core_grid=core_grid,
                strategy=strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
        except RuntimeError:
            continue

    return ttnn.DRAM_MEMORY_CONFIG


@dataclass(frozen=True)
class Conv1dConfiguration:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: tuple[int, int] = (0, 0)  # (padding_left, padding_right)
    dilation: int = 1
    groups: int = 1
    activation: Optional[ttnn.UnaryWithParam] = None
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    weights_dtype: ttnn.DataType = ttnn.bfloat16
    dtype: ttnn.DataType = ttnn.bfloat16
    output_layout: ttnn.Layout = ttnn.TILE_LAYOUT
    # slice_strategy: Optional[SliceStrategy] = None
    # math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    # fp32_dest_acc_en: bool = False
    # packer_l1_acc: bool = False
    # enable_act_double_buffer: bool = False
    # enable_weights_double_buffer: bool = False
    # deallocate_activation: bool = True
    # reallocate_halo_output: bool = True
    # config_tensors_in_dram: bool = True


def _normalize_conv2d_activation(activation: str | tuple[str, dict] | None) -> ttnn.UnaryWithParam | None:
    if activation is None:
        return None
    if isinstance(activation, tuple):
        activation_name, kwargs = activation
    else:
        activation_name = activation.strip().lower()
    activation_aliases = {
        "swish": "silu",
    }
    activation_name = activation_aliases.get(activation_name, activation_name)

    if activation_name == "relu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    if activation_name == "silu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    if activation_name == "gelu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)
    if activation_name == "sigmoid":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
    if activation_name == "tanh":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH)
    if activation_name == "leaky_relu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, kwargs["negative_slope"])

    supported = "relu, silu (alias swish), gelu, sigmoid, tanh, leaky_relu"
    raise ValueError(f"Unsupported conv activation '{activation}'. Supported activations: {supported}")


def resolve_padding(
    padding: PaddingType,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    if isinstance(padding, str):
        if padding != "same":
            raise ValueError(f"Unsupported padding mode: {padding}")
        if stride != 1:
            raise ValueError("Only stride=1 is supported for 'same' padding")
        padding_needed = dilation * (kernel_size - 1)
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        return (left_padding, right_padding)

    if isinstance(padding, tuple):
        if len(padding) != 2:
            raise ValueError(f"padding tuple must contain 2 values, got {len(padding)}")
        return (padding[0], padding[1])

    return (padding, padding)


def output_length_from_input_length(input_length, conv1d_config: Conv1dConfiguration):
    padding_left, padding_right = conv1d_config.padding[0], conv1d_config.padding[1]
    return (
        input_length + padding_left + padding_right - conv1d_config.dilation * (conv1d_config.kernel_size - 1) - 1
    ) // conv1d_config.stride + 1


dims_to_config_values = {
    # (output_length, in_channels, out_channels, kernel_size): (num_slices, act_block_h_override, act_block_w_div)
    (1780, 768, 768, 128): (56, 0, 1),
    (113986, 1, 512, 10): (2, 32, 1),
    ((113986 // 128) * 128, 1, 512, 10): (2, 128, 1),
    (56992, 512, 512, 3): (3, 32 * 3, 1),
    (28495, 512, 512, 3): (4, 32 * 0, 1),
    (14247, 512, 512, 3): (1, 32 * 24, 1),
    (35600, 1, 256, 96): (1, 32, 1),
    (35600, 256, 256, 3): (1, 32 * 24, 1),
    (35600, 256, 256, 7): (1, 32 * 24, 1),
    (35600, 256, 256, 11): (1, 32 * 16, 1),
    (213600, 1, 128, 16): (2, 32, 1),
    (213600, 128, 128, 3): (2, 32, 1),
    (213600, 128, 128, 7): (2, 32, 1),
    (213600, 128, 128, 11): (4, 32, 1),
    (427200, 1, 64, 8): (1, 32, 1),
    (427200, 64, 64, 3): (2, 32, 1),
    (427200, 64, 64, 7): (2, 32, 1),
    (427200, 64, 64, 11): (2, 32, 1),
    (854400, 1, 32, 4): (1, 32 * 4, 1),
    (854400, 32, 32, 3): (2, 32 * 24, 1),
    (854400, 32, 32, 7): (2, 32 * 16, 1),
    (854400, 32, 32, 11): (32, 32 * 0, 1),
    (1708800, 16, 16, 3): (32, 32 * 0, 1),
    (1708800, 16, 16, 7): (32, 32, 1),
    (1708800, 16, 16, 11): (32, 32 * 0, 1),
    (1708800, 16, 1, 7): (16, 32 * 0, 1),
}


def get_conv_configs(
    input_length, conv1d_config: Conv1dConfiguration, device: ttnn.Device
) -> tuple[ttnn.Conv2dConfig, ttnn.Conv2dSliceConfig, ttnn.DeviceComputeKernelConfig]:
    output_length = output_length_from_input_length(input_length, conv1d_config)
    num_slices, act_block_h_override, act_block_w_div = dims_to_config_values.get(
        (output_length, conv1d_config.in_channels, conv1d_config.out_channels, conv1d_config.kernel_size),
        (1, 0, 1),
    )
    slice_config = ttnn.Conv2dSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)

    return (
        ttnn.Conv2dConfig(
            weights_dtype=conv1d_config.weights_dtype,
            # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            output_layout=conv1d_config.output_layout,
            # deallocate_activation=conv1d_config.deallocate_activation,
            # reallocate_halo_output=conv1d_config.reallocate_halo_output,
            # enable_act_double_buffer=conv1d_config.enable_act_double_buffer,
            # enable_weights_double_buffer=conv1d_config.enable_weights_double_buffer,
            # config_tensors_in_dram=conv1d_config.config_tensors_in_dram,
            # force_split_reader=True,
            config_tensors_in_dram=True,  # Force tensors in DRAM to avoid L1 thrashing for large activations
            act_block_h_override=act_block_h_override,
            act_block_w_div=act_block_w_div,
            # reshard_if_not_optimal=True,
            activation=conv1d_config.activation,
            # slice_config=slice_config,
        ),
        slice_config,
        ttnn.init_device_compute_kernel_config(
            device.arch(),
            # math_fidelity=conv1d_config.math_fidelity,
            # fp32_dest_acc_en=conv1d_config.fp32_dest_acc_en,
            # packer_l1_acc=conv1d_config.packer_l1_acc,
        ),
    )


class Conv1d:
    """Stateful Conv1d wrapper around `ttnn.conv1d`."""

    def __init__(
        self,
        # configuration: Conv1dConfiguration | None = None,
        device: ttnn.MeshDevice | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int | None = None,
        stride: int = 1,
        padding: PaddingType = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype: ttnn.DataType | None = None,
        activation: str | tuple[str, dict] | None = None,
    ) -> None:
        # if configuration is None:
        if device is None:
            raise ValueError("device is required")
        if in_channels is None or out_channels is None or kernel_size is None:
            raise ValueError(
                "in_channels, out_channels, and kernel_size are required when configuration is not provided"
            )
        if isinstance(padding, str) and padding != "same":
            raise ValueError(f"Unsupported padding mode: {padding}")
        padding_final = resolve_padding(
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        TILE_WIDTH = 32
        if out_channels % TILE_WIDTH == 0:
            output_layout = ttnn.TILE_LAYOUT
        elif (TILE_WIDTH - out_channels % TILE_WIDTH) / out_channels < 1.2:
            output_layout = ttnn.TILE_LAYOUT
        else:
            output_layout = ttnn.ROW_MAJOR_LAYOUT

        configuration = Conv1dConfiguration(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_final,
            dilation=dilation,
            groups=groups,
            activation=_normalize_conv2d_activation(activation),
            dtype=dtype or ttnn.bfloat16,
            output_layout=output_layout,
        )

        self.device = device
        self.configuration = configuration
        self.memory_config = ttnn.L1_MEMORY_CONFIG

    def load_parameters(
        self,
        parameters: dict[str, torch.Tensor],
        key: str,
        prefix: str = "",
    ) -> None:
        base_key = f"{prefix}{key}" if prefix else key
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        wt = parameters[weight_key].reshape(
            self.configuration.out_channels,
            self.configuration.in_channels // self.configuration.groups,
            1,
            self.configuration.kernel_size,
        )
        # Keep a torch-reference copy for internal F.conv1d parity check.
        self.torch_weight = parameters[weight_key].detach().to(torch.float32).contiguous()
        bias = parameters[bias_key] if bias_key in parameters and parameters[bias_key] is not None else None
        self.torch_bias = None if bias is None else bias.detach().to(torch.float32).contiguous()
        self.weight_tensor = ttnn.from_torch(wt, dtype=ttnn.bfloat16)
        self.bias_tensor = None
        if bias is not None:
            self.bias_tensor = ttnn.from_torch(
                torch.reshape(bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

    def __call__(
        self,
        input_tensor: ttnn.Tensor,
    ):
        input_2d = input1d_to_2d(input_tensor)
        batch_size = input_2d.shape[0]
        input_length = input_2d.shape[2]
        conv2d_config, slice_config, compute_config = get_conv_configs(input_length, self.configuration, self.device)
        conv_result, [self.weight_tensor, self.bias_tensor] = ttnn.conv2d(
            input_tensor=input_2d,
            weight_tensor=self.weight_tensor,
            return_output_dim=False,
            return_weights_and_bias=True,
            device=self.device,
            in_channels=self.configuration.in_channels,
            out_channels=self.configuration.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            kernel_size=[1, self.configuration.kernel_size],
            stride=[1, self.configuration.stride],
            padding=[0, 0, self.configuration.padding[0], self.configuration.padding[1]],
            dilation=[1, self.configuration.dilation],
            groups=self.configuration.groups,
            bias_tensor=self.bias_tensor,
            dtype=self.configuration.dtype,
            conv_config=conv2d_config,
            compute_config=compute_config,
            slice_config=slice_config,
        )
        output_shape = conv_result.shape
        x = ttnn.reshape(conv_result, (batch_size, output_shape[2], output_shape[3]))
        # self._check_against_torch(input_tensor, x)
        return x

    def _check_against_torch(self, input_tensor: ttnn.Tensor, tt_output: ttnn.Tensor) -> None:
        # Compare TT Conv1d output against torch.nn.functional.conv1d reference.
        torch_input = ttnn.to_torch(ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)).to(torch.float32)
        if not hasattr(self, "torch_weight"):
            raise ValueError("Conv1d torch reference weight is not initialized. Call load_parameters first.")
        torch_weight = self.torch_weight
        torch_bias = self.torch_bias

        # TT interface uses NLC, while torch conv1d expects NCL.
        torch_input_ncl = torch_input.permute(0, 2, 1).contiguous()
        pad_left, pad_right = self.configuration.padding
        if pad_left != 0 or pad_right != 0:
            torch_input_ncl = F.pad(torch_input_ncl, (pad_left, pad_right))
        torch_ref = F.conv1d(
            torch_input_ncl,
            torch_weight,
            bias=torch_bias,
            stride=self.configuration.stride,
            padding=0,
            dilation=self.configuration.dilation,
            groups=self.configuration.groups,
        )
        torch_ref_nlc = torch_ref.permute(0, 2, 1).contiguous()
        tt_output_torch = ttnn.to_torch(ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)).to(torch.float32)
        assert_with_pcc(torch_ref_nlc, tt_output_torch, pcc=0.99)

    def deallocate(self) -> None:
        if self.weight_tensor is not None:
            ttnn.deallocate(self.weight_tensor)
            self.weight_tensor = None
        if self.bias_tensor is not None:
            ttnn.deallocate(self.bias_tensor)
            self.bias_tensor = None
