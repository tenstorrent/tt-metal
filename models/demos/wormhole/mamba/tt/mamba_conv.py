# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import ttnn


@dataclass
class MambaConvConfig:
    input_channels: int = 5120
    output_channels: int = 5120
    groups: int = 5120
    input_length: int = 1027
    kernel_size: int = 4
    stride: int = 1
    padding: int = 0
    weights_dtype: ttnn.DataType = ttnn.bfloat8_b
    activations_dtype: ttnn.DataType = ttnn.bfloat16
    output_dtype: ttnn.DataType = ttnn.bfloat8_b
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    channels_split_factor: int = 2
    weight_name = "mixer.conv1d.weight"


class MambaConv:
    def __init__(self, device, load_fn, config: MambaConvConfig):
        self.device = device
        self.config = config
        self.prepare_conv_config()
        self.prepare_weights(load_fn)

    def prepare_weights(self, load_fn):
        torch_conv1d_weights = load_fn(self.config.weight_name, return_as_torch=True)  # (2E, 1, 4)
        initial_weight_dtype = (
            self.config.weights_dtype if self.config.weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
        split_size = self.config.input_channels // self.config.channels_split_factor
        self.tt_weight_tensor_splits = [
            ttnn.from_torch(torch_conv1d_weights[i * split_size : (i + 1) * split_size, :, :], initial_weight_dtype)
            for i in range(self.config.channels_split_factor)
        ]

    def prepare_conv_config(self):
        if self.config.channels_split_factor == 1:
            # We run of out memory for input_channels = 5120 with bfloat16
            self.config.weights_dtype = ttnn.bfloat8_b
            self.config.output_dtype = ttnn.bfloat8_b

        self.conv1d_config = ttnn.Conv1dConfig(
            dtype=self.config.output_dtype,
            weights_dtype=self.config.weights_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
        )
        self.conv1d_compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=self.config.math_fidelity,
        )

    def prepare_input(self, input_tensor):
        # input_tensor (1, 1, B, 2E)
        # typecast to ttnn.bfloat16 for RM layout
        if input_tensor.dtype != self.config.activations_dtype:
            input_tensor = ttnn.typecast(input_tensor, self.config.activations_dtype)
        # convert from tile layout to row major layout
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        # reshape to (B, H, W, C)
        input_tensor = ttnn.reshape(input_tensor, [1, self.config.input_length, 1, self.config.input_channels])
        if self.config.channels_split_factor == 1:
            return [input_tensor]
        else:
            input_tensor_splits = []
            split_size = self.config.input_channels // self.config.channels_split_factor
            for i in range(self.config.channels_split_factor):
                slice_start = (0, 0, 0, i * split_size)
                slice_end = (1, self.config.input_length, 1, (i + 1) * split_size)
                input_tensor_splits.append(
                    ttnn.slice(input_tensor, starts=slice_start, ends=slice_end, steps=(1, 1, 1, 1))
                )
            ttnn.deallocate(input_tensor)
            return input_tensor_splits

    def __call__(self, input_tensor):
        input_tensor_splits = self.prepare_input(input_tensor)
        output_tensor_splits = []
        for i in range(self.config.channels_split_factor):
            [tt_output_tensor_on_device, out_length, [weights_device, _]] = ttnn.conv1d(
                input_tensor=input_tensor_splits[i],
                weight_tensor=self.tt_weight_tensor_splits[i],
                in_channels=self.config.input_channels // self.config.channels_split_factor,
                out_channels=self.config.output_channels // self.config.channels_split_factor,
                device=self.device,
                bias_tensor=None,
                kernel_size=self.config.kernel_size,
                stride=self.config.stride,
                padding=self.config.padding,
                batch_size=1,
                input_length=self.config.input_length,
                conv_config=self.conv1d_config,
                compute_config=self.conv1d_compute_config,
                groups=self.config.groups // self.config.channels_split_factor,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            self.tt_weight_tensor_splits[i] = weights_device
            output_tensor_splits.append(ttnn.sharded_to_interleaved(tt_output_tensor_on_device))
        if self.config.channels_split_factor == 1:
            return output_tensor_splits[0]
        else:
            # Concatenate the output tensor splits
            tt_output_tensor = ttnn.concat(output_tensor_splits, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
            return tt_output_tensor
