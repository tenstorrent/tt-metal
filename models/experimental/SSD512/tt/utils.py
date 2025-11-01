# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from math import sqrt
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from models.common.utility_functions import torch_to_tt_tensor_rm

conv_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


class TtConv2d:
    """TTNN implementation of 2D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        memory_config=None,
        conv_config=None,
    ):
        """Initialize conv2d layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolving kernel
            stride: Stride of convolution
            padding: Zero-padding added to both sides of input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output channels
            bias: If True, adds a learnable bias to output
            device: Device to place ops on
            memory_config: Memory configuration for Conv operations
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.device = device
        # Use L1 for activations and DRAM for weights
        self.activation_memory_config = memory_config if memory_config else ttnn.L1_MEMORY_CONFIG
        self.weight_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Initialize weights and bias
        k = 1 / (in_channels * kernel_size**2)
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = torch.zeros(weight_shape).uniform_(-sqrt(k), sqrt(k))

        if bias:
            self.bias = torch.zeros(out_channels).uniform_(-sqrt(k), sqrt(k))
        else:
            self.bias = None

        # Create Conv instance - weights and biases stored in DRAM
        parameters = {
            "weight": torch_to_tt_tensor_rm(self.weight, device, memory_config=self.weight_memory_config),
            "bias": torch_to_tt_tensor_rm(self.bias, device, memory_config=self.weight_memory_config) if bias else None,
        }

        self.conv = Conv(
            stride=stride,
            padding=padding,
            dilation=dilation,
            parameters=parameters,
            groups=groups,
            kernel_fidelity=conv_config,
            memory_config=self.activation_memory_config,
            reallocate_halo_output=True,  # Force reallocation
            deallocate_activation=True,  # Free up memory after use
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, height, width, in_channels]

        Returns:
            Output tensor [batch_size, out_height, out_width, out_channels]
        """
        return self.conv(self.device, x)


class Conv:
    def __init__(
        self,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        parameters: dict | None = None,
        kernel_fidelity=conv_config,
        *,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        act_block_h=None,
        act_block_w=None,
        deallocate_activation=False,
        reallocate_halo_output=True,
        shard_layout=None,
        activation=None,
        groups=1,
        num_cores_nhw=None,
        is_reshape=False,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        math_approx_mode=False,
        input_channels_alignment=32,
        reshard_if_not_optimal=False,
        slice_config=None,
        dtype=None,
        weights_dtype=None,
        math_fidelity=None,
        width_sharding=False,
        height_sharding=True,
    ) -> None:
        self.kernel_size = (parameters["weight"].shape[2], parameters["weight"].shape[3])

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            ValueError("Invalid config")
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            ValueError("Invalid config")
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        elif isinstance(dilation, tuple):
            self.dilation = dilation
        else:
            ValueError("Invalid config")
        if width_sharding == True:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )

        self.kernel_fidelity = conv_config
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.deallocate_activation = deallocate_activation
        self.reallocate_halo_output = reallocate_halo_output
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.math_approx_mode = math_approx_mode
        self.input_channels_alignment = input_channels_alignment
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.act_block_w = act_block_w
        self.groups = groups
        self.activation = activation
        self.memory_config = memory_config
        self.shard_layout = shard_layout
        self.slice_config = slice_config
        self.num_cores_nhw = num_cores_nhw
        self.is_reshape = is_reshape
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_weights_double_buffer = enable_weights_double_buffer
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.kernel_fidelity["ACTIVATIONS_DTYPE"]
        if weights_dtype is not None:
            self.weights_dtype = weights_dtype
        else:
            self.weights_dtype = self.kernel_fidelity["WEIGHTS_DTYPE"]
        if math_fidelity is not None:
            self.math_fidelity = math_fidelity
        else:
            self.math_fidelity = self.kernel_fidelity["MATH_FIDELITY"]

    def __call__(self, device, input_tensor):
        shape = input_tensor.shape

        # Create optimized conv config to minimize memory usage
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            deallocate_activation=True,  # Always deallocate to save memory
            reallocate_halo_output=True,  # Always reallocate
            config_tensors_in_dram=True,  # Force DRAM usage
            enable_act_double_buffer=False,  # Disable double buffering
            enable_weights_double_buffer=False,
            in_place=True,  # Enable in-place operations
            full_inner_dim=False,  # Disable full inner dimension to save memory
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.kernel_fidelity["MATH_FIDELITY"],
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            math_approx_mode=self.math_approx_mode,
        )
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h
        if self.act_block_w is not None:
            conv_config.act_block_w_div = self.act_block_w

        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=shape[-1],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=shape[-4],
            input_height=shape[-3],
            input_width=shape[-2],
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=self.slice_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.dtype,
            memory_config=self.memory_config,
        )

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, self.memory_config)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        output_tensor = ttnn.reshape(output_tensor, (1, _out_height, _out_width, output_tensor.shape[3]))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config)
        del _out_height, _out_width

        return output_tensor
