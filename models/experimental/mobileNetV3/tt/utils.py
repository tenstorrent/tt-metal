# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)

conv_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


def post_conv_reshape(x):
    """Convert sharded conv output to [N,1,1,C] tile layout for SE block."""
    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (x.shape[0], 1, 1, x.shape[3]))
    return ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)


def create_se_conv_config(input_shape, parameters, stride=(1, 1), padding=(0, 0)):
    """
    Create Conv2dConfiguration from parameters dict for SqueezeExcitation.
    """
    return Conv2dConfiguration(
        input_height=input_shape[-3],
        input_width=input_shape[-2],
        in_channels=input_shape[-1],
        out_channels=parameters["weight"].shape[0],
        batch_size=input_shape[-4],
        kernel_size=(parameters["weight"].shape[2], parameters["weight"].shape[3]),
        stride=stride,
        padding=padding,
        weight=parameters["weight"],
        bias=parameters["bias"],
        activation_dtype=conv_config["ACTIVATIONS_DTYPE"],
        weights_dtype=conv_config["WEIGHTS_DTYPE"],
        output_dtype=conv_config["ACTIVATIONS_DTYPE"],
        math_fidelity=conv_config["MATH_FIDELITY"],
        sharding_strategy=AutoShardedStrategyConfiguration(),
        slice_strategy=L1FullSliceStrategyConfiguration(),
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        deallocate_activation=False,
        reallocate_halo_output=True,
    )


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
        slice_config=ttnn.Conv2dL1FullSliceConfig,
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

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=self.reallocate_halo_output,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
            shard_layout=self.shard_layout,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
            # in_place=True,
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

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        output_tensor = ttnn.reshape(output_tensor, (1, _out_height, _out_width, output_tensor.shape[3]))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        del _out_height, _out_width

        return output_tensor
