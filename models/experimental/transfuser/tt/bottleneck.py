# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.transfuser.tt.utils import TTConv2D


class RegNetBottleneck:
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

        # conv1: 1x1 convolution
        self.conv1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["conv1"],
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

        # conv2: 3x3 grouped convolution
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            parameters=parameters["conv2"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            groups=groups,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
        )

        # SE Module
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
            parameters=parameters["conv3"],
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
                parameters=parameters["downsample"],
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

        # conv1: 1x1 expansion
        out, _ = self.conv1(device, x, x.shape)

        # conv2: 3x3 grouped convolution
        out, _ = self.conv2(device, out, out.shape)

        # SE Module
        # Global average pooling
        se_out = ttnn.global_avg_pool2d(out)
        se_out, _ = self.se_fc1(device, se_out, se_out.shape)
        se_out, _ = self.se_fc2(device, se_out, se_out.shape)
        se_out = ttnn.sigmoid(se_out)

        # Apply SE scaling
        out = ttnn.multiply(out, se_out)

        # conv3: 1x1 projection
        out, _ = self.conv3(device, out, out.shape)

        # Handle downsample
        if self.downsample_layer is not None:
            identity, _ = self.downsample_layer(device, identity, identity.shape)

        # Residual connection
        if identity.shape != out.shape:
            identity = ttnn.reshape(identity, out.shape)

        out = ttnn.add(out, identity)
        out = ttnn.relu(out)  # Final ReLU activation

        return out
