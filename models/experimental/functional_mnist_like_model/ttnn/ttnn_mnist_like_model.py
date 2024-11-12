# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from ttnn.model_preprocessing import ParameterDict, fold_batch_norm2d_into_conv2d
from torch import nn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class Mnist_like_model_Conv2D:
    def __init__(
        self,
        conv,
        bn=None,
        device=None,
        cache={},
        activation="relu",
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        num_cores_nhw=None,
    ):
        self.device = device
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = True
        self.cache = cache

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            activation=activation,
            shard_layout=shard_layout,
            input_channels_alignment=16 if conv.in_channels < 16 else 32,
            deallocate_activation=self.deallocate_activation,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(num_cores_nhw, device)

            self.conv_config.core_grid = shard_grid
            self.conv_config.override_sharding_config = True

        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]
        if bn is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(conv.module, bn.module)
        else:
            weight, bias = conv.module.weight, conv.module.bias
        weight = weight
        if bias is not None:
            bias = torch.reshape(bias, (1, 1, 1, -1))
            self.bias = ttnn.from_torch(bias, dtype=ttnn.float32)
        else:
            self.bias = None
        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32)

    def __call__(self, x):
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            batch_size=self.batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            conv_op_cache=self.cache,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x


class Mnist_like_model:
    def __init__(self, device, parameters: ParameterDict):
        self.device = device
        self.parameters = parameters
        self.conv1 = Mnist_like_model_Conv2D(parameters.conv1, device=device, num_cores_nhw=56)
        self.conv2 = Mnist_like_model_Conv2D(parameters.conv2, device=device, num_cores_nhw=48)
        self.conv3 = Mnist_like_model_Conv2D(parameters.conv3, device=device, num_cores_nhw=16)
        self.conv4 = Mnist_like_model_Conv2D(parameters.conv4, device=device)
        self.fc1_weights = parameters.fc1.module.weight
        self.fc2_weights = parameters.fc2.module.weight
        self.fc1_bias = parameters.fc1.module.bias
        self.fc2_bias = parameters.fc2.module.bias

        self.pool3 = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(3 * 3 * 64, 1024)  # 100 x 100 region
        self.fc1.weight = parameters.fc1.module.weight
        self.fc1.bias = parameters.fc1.module.bias
        self.fc2 = nn.Linear(1024, 11)
        self.fc2.weight = parameters.fc2.module.weight
        self.fc2.bias = parameters.fc2.module.bias

    def __call__(self, x):
        x = self.conv1(x)

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.parameters.pool3.batch_size,
            input_h=self.parameters.pool3.input_height,
            input_w=self.parameters.pool3.input_width,
            channels=self.parameters.conv2.out_channels,
            kernel_size=[self.parameters.pool3.kernel_size, self.parameters.pool3.kernel_size],
            stride=[self.parameters.pool3.stride, self.parameters.pool3.stride],
            padding=[self.parameters.pool3.padding, self.parameters.pool3.padding],
            dilation=[self.parameters.pool3.dilation, self.parameters.pool3.dilation],
        )

        x = self.conv2(x)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.parameters.pool2.batch_size,
            input_h=self.parameters.pool2.input_height,
            input_w=self.parameters.pool2.input_width,
            channels=self.parameters.conv2.out_channels,
            kernel_size=[self.parameters.pool2.kernel_size, self.parameters.pool2.kernel_size],
            stride=[self.parameters.pool2.stride, self.parameters.pool2.stride],
            padding=[self.parameters.pool2.padding, self.parameters.pool2.padding],
            dilation=[self.parameters.pool2.dilation, self.parameters.pool2.dilation],
        )

        x = self.conv3(x)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.parameters.pool4.batch_size,
            input_h=self.parameters.pool4.input_height,
            input_w=self.parameters.pool4.input_width,
            channels=self.parameters.conv3.out_channels,
            kernel_size=[self.parameters.pool4.kernel_size, self.parameters.pool4.kernel_size],
            stride=[self.parameters.pool4.stride, self.parameters.pool4.stride],
            padding=[self.parameters.pool4.padding, self.parameters.pool4.padding],
            dilation=[self.parameters.pool4.dilation, self.parameters.pool4.dilation],
        )
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.conv4(x)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.parameters.pool5.batch_size,
            input_h=self.parameters.pool5.input_height,
            input_w=self.parameters.pool5.input_width,
            channels=self.parameters.conv4.out_channels,
            kernel_size=[self.parameters.pool5.kernel_size, self.parameters.pool5.kernel_size],
            stride=[self.parameters.pool5.stride, self.parameters.pool5.stride],
            padding=[self.parameters.pool5.padding, self.parameters.pool5.padding],
            dilation=[self.parameters.pool5.dilation, self.parameters.pool5.dilation],
        )

        pool5_ouput_height = int(
            (
                (
                    self.parameters.pool5.input_height
                    + ((-(self.parameters.pool5.dilation) * (self.parameters.pool5.kernel_size - 1)) - 1)
                )
                / self.parameters.pool5.stride
            )
            + 1
        )
        pool5_ouput_width = int(
            (
                (
                    self.parameters.pool5.input_width
                    + ((-(self.parameters.pool5.dilation) * (self.parameters.pool5.kernel_size - 1)) - 1)
                )
                / self.parameters.pool5.stride
            )
            + 1
        )

        x = ttnn.reshape(
            x,
            [
                self.parameters.pool5.batch_size,
                pool5_ouput_height,
                pool5_ouput_width,
                self.parameters.conv4.out_channels,
            ],
        )

        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.reshape(x, (2, -1))

        x = ttnn.to_torch(x)
        x = x.to(torch.float32)
        x = self.fc1(x)

        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)

        x = ttnn.relu(x)

        x = ttnn.to_torch(x).to(torch.float32)

        x = self.fc2(x)

        return x
