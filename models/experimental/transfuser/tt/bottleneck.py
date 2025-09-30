# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.experimental.transfuser.tt.utils import TTConv2D


class TTRegNetBottleneck:
    def __init__(
        self,
        parameters,
        model_config,
        stride=1,
        downsample=False,
        groups=1,
    ):
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.model_config = model_config
        # print(parameters)

        # print(parameters)
        # conv1: 1x1 convolution
        self.conv1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            # parameters=parameters["conv1"],
            parameters=parameters["conv1"]["conv"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            is_reshape=False,
        )

        # conv2: 3x3 grouped convolution
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            # parameters=parameters["conv2"],
            parameters=parameters["conv2"]["conv"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            groups=groups,
            act_block_h=32,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            # is_reshape=True,
        )

        # SE Module
        print(parameters["se"]["fc1"])
        self.se_fc1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["se"]["fc1"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
        )

        self.se_fc2 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["se"]["fc2"],
            kernel_fidelity=model_config,
            activation=None,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
        )

        # conv3: 1x1 convolution (no activation)
        self.conv3 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            # parameters=parameters["conv3"],
            parameters=parameters["conv3"]["conv"],
            kernel_fidelity=model_config,
            activation=None,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
        )

        # Downsample layer if needed
        if downsample:
            self.downsample_layer = TTConv2D(
                kernel_size=1,
                stride=stride,
                padding=0,
                # parameters=parameters["downsample"],
                parameters=parameters["downsample"][0],
                kernel_fidelity=model_config,
                activation=None,
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                deallocate_activation=True,
                reallocate_halo_output=True,
                reshard_if_not_optimal=True,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
                dtype=ttnn.bfloat16,
            )
        else:
            self.downsample_layer = None

    def __call__(self, x, device):
        identity = x
        logger.info(f"conv1- 1x1 convolution")
        logger.info(f"{x.shape=}")
        # conv1: 1x1 expansion
        out, shape_ = self.conv1(device, x, x.shape)

        logger.info(f"conv2- 3x3 grouped convolution")
        logger.info(f"{out.shape=}")
        # conv2: 3x3 grouped convolution
        out, shape_ = self.conv2(device, out, shape_)

        # SE Module
        logger.info(f"SE module")
        logger.info(f"reduce mean")
        # Global average pooling
        out = ttnn.reshape(out, shape_)
        se_out = ttnn.mean(out, dim=[1, 2], keepdim=True)
        shape_ = se_out.shape

        logger.info(f"SE fc1")
        se_out, shape_ = self.se_fc1(device, se_out, shape_)

        logger.info(f"SE fc2")
        se_out, shape_ = self.se_fc2(device, se_out, shape_)
        se_out = ttnn.sigmoid(se_out)

        # Apply SE scaling
        out = ttnn.multiply(out, se_out)
        shape_ = out.shape

        # conv3: 1x1 projection
        out, shape_ = self.conv3(device, out, shape_)

        # Handle downsample
        if self.downsample_layer is not None:
            print(f"{identity.shape=}")
            identity, shape_ = self.downsample_layer(device, identity, identity.shape)

        # Residual connection
        if identity.shape != shape_:
            identity = ttnn.reshape(identity, shape_)
        out = ttnn.reshape(out, shape_)

        out = ttnn.add(out, identity)
        out = ttnn.relu(out)  # Final ReLU activation

        return out
