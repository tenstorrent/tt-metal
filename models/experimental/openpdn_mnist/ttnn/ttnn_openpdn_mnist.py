# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ttnn.model_preprocessing import ParameterDict
from torch import nn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class OpenPDNMnistConv2D:
    def __init__(
        self,
        conv,
        conv_pth,
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
        self.conf_output_dtype = activation_dtype
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            activation=activation,
            shard_layout=shard_layout,
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

        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, tt_tensor):
        [tt_tensor, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=tt_tensor,
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
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conf_output_dtype,
        )
        return tt_tensor


class TtOpenPDNMnist:
    def __init__(self, device, parameters: ParameterDict):
        self.device = device
        self.parameters = parameters.conv_args
        self.parameter_pth = parameters
        self.conv1 = OpenPDNMnistConv2D(parameters.conv_args.conv1, parameters.conv1, device=device, num_cores_nhw=56)
        self.conv2 = OpenPDNMnistConv2D(parameters.conv_args.conv2, parameters.conv2, device=device, num_cores_nhw=48)
        self.conv3 = OpenPDNMnistConv2D(parameters.conv_args.conv3, parameters.conv3, device=device, num_cores_nhw=16)
        self.conv4 = OpenPDNMnistConv2D(parameters.conv_args.conv4, parameters.conv4, device=device)
        self.fc1_weights = parameters.conv_args.fc1.module.weight
        self.fc2_weights = parameters.conv_args.fc2.module.weight
        self.fc1_bias = parameters.conv_args.fc1.module.bias
        self.fc2_bias = parameters.conv_args.fc2.module.bias

        self.pool3 = nn.MaxPool2d(3, 3)

    def __call__(self, input):
        conv1_out = self.conv1(input)

        maxpool1_out = ttnn.max_pool2d(
            input_tensor=conv1_out,
            batch_size=self.parameters.pool3.batch_size,
            input_h=self.parameters.pool3.input_height,
            input_w=self.parameters.pool3.input_width,
            channels=self.parameters.conv2.out_channels,
            kernel_size=[self.parameters.pool3.kernel_size, self.parameters.pool3.kernel_size],
            stride=[self.parameters.pool3.stride, self.parameters.pool3.stride],
            padding=[self.parameters.pool3.padding, self.parameters.pool3.padding],
            dilation=[self.parameters.pool3.dilation, self.parameters.pool3.dilation],
        )
        ttnn.deallocate(conv1_out)

        conv2_out = self.conv2(maxpool1_out)
        ttnn.deallocate(maxpool1_out)
        maxpool2_out = ttnn.max_pool2d(
            input_tensor=conv2_out,
            batch_size=self.parameters.pool2.batch_size,
            input_h=self.parameters.pool2.input_height,
            input_w=self.parameters.pool2.input_width,
            channels=self.parameters.conv2.out_channels,
            kernel_size=[self.parameters.pool2.kernel_size, self.parameters.pool2.kernel_size],
            stride=[self.parameters.pool2.stride, self.parameters.pool2.stride],
            padding=[self.parameters.pool2.padding, self.parameters.pool2.padding],
            dilation=[self.parameters.pool2.dilation, self.parameters.pool2.dilation],
        )
        ttnn.deallocate(conv2_out)

        conv3 = self.conv3(maxpool2_out)
        ttnn.deallocate(maxpool2_out)
        maxpool3_out = ttnn.max_pool2d(
            input_tensor=conv3,
            batch_size=self.parameters.pool4.batch_size,
            input_h=self.parameters.pool4.input_height,
            input_w=self.parameters.pool4.input_width,
            channels=self.parameters.conv3.out_channels,
            kernel_size=[self.parameters.pool4.kernel_size, self.parameters.pool4.kernel_size],
            stride=[self.parameters.pool4.stride, self.parameters.pool4.stride],
            padding=[self.parameters.pool4.padding, self.parameters.pool4.padding],
            dilation=[self.parameters.pool4.dilation, self.parameters.pool4.dilation],
        )
        ttnn.deallocate(conv3)
        memory_config = ttnn.create_sharded_memory_config(
            [
                32,
                64,
            ],
            core_grid=ttnn.CoreGrid(y=1, x=3),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        maxpool3_out = ttnn.to_memory_config(maxpool3_out, memory_config)
        conv4_out = self.conv4(maxpool3_out)
        ttnn.deallocate(maxpool3_out)

        maxpool4_out = ttnn.max_pool2d(
            input_tensor=conv4_out,
            batch_size=self.parameters.pool5.batch_size,
            input_h=self.parameters.pool5.input_height,
            input_w=self.parameters.pool5.input_width,
            channels=self.parameters.conv4.out_channels,
            kernel_size=[self.parameters.pool5.kernel_size, self.parameters.pool5.kernel_size],
            stride=[self.parameters.pool5.stride, self.parameters.pool5.stride],
            padding=[self.parameters.pool5.padding, self.parameters.pool5.padding],
            dilation=[self.parameters.pool5.dilation, self.parameters.pool5.dilation],
        )
        ttnn.deallocate(conv4_out)

        pool5_output_height = int(
            (
                (
                    self.parameters.pool5.input_height
                    + ((-(self.parameters.pool5.dilation) * (self.parameters.pool5.kernel_size - 1)) - 1)
                )
                / self.parameters.pool5.stride
            )
            + 1
        )
        pool5_output_width = int(
            (
                (
                    self.parameters.pool5.input_width
                    + ((-(self.parameters.pool5.dilation) * (self.parameters.pool5.kernel_size - 1)) - 1)
                )
                / self.parameters.pool5.stride
            )
            + 1
        )

        maxpool4_out = ttnn.reshape(
            maxpool4_out,
            [
                self.parameters.pool5.batch_size,
                pool5_output_height,
                pool5_output_width,
                self.parameters.conv4.out_channels,
            ],
        )

        maxpool4_out = ttnn.sharded_to_interleaved(maxpool4_out, ttnn.L1_MEMORY_CONFIG)
        maxpool4_out = ttnn.permute(maxpool4_out, (0, 3, 1, 2))
        maxpool4_out = ttnn.reshape(
            maxpool4_out,
            (1, maxpool4_out.shape[0] * maxpool4_out.shape[1] * maxpool4_out.shape[2], maxpool4_out.shape[3], 1),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        maxpool4_out = ttnn.reshape(
            maxpool4_out, (1, self.parameters.pool5.batch_size, -1, 1), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        maxpool4_out = ttnn.permute(maxpool4_out, (0, 3, 1, 2))
        maxpool4_out = ttnn.reshape(
            maxpool4_out, (self.parameters.pool5.batch_size, -1), memory_config=ttnn.L1_MEMORY_CONFIG
        )

        maxpool4_out = ttnn.to_layout(maxpool4_out, ttnn.TILE_LAYOUT)

        grid_size = (8, 8)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        shard_spec = ttnn.ShardSpec(shard_grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR)
        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )
        maxpool4_out = ttnn.to_memory_config(maxpool4_out, width_sharded_mem_config)

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        Linear_out = ttnn.linear(
            maxpool4_out,
            self.parameter_pth.fc1.weight,
            bias=self.parameter_pth.fc1.bias,
            memory_config=width_sharded_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.deallocate(maxpool4_out)
        act_out = ttnn.relu(Linear_out)
        ttnn.deallocate(Linear_out)

        grid_size = (8, 8)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        shard_spec = ttnn.ShardSpec(shard_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )
        act_out = ttnn.to_memory_config(act_out, width_sharded_mem_config)

        output = ttnn.linear(
            act_out,
            self.parameter_pth.fc2.weight,
            bias=self.parameter_pth.fc2.bias,
            memory_config=width_sharded_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.deallocate(act_out)
        return output
