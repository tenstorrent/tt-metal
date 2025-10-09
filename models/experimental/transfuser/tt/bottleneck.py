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
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            shard_layout=shard_layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=False,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            is_reshape=False,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
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
            act_block_h=32,
            shard_layout=shard_layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            is_reshape=False,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
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
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
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
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
        )

        # conv3: 1x1 convolution (no activation)
        self.conv3 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["conv3"],
            kernel_fidelity=model_config,
            activation=None,
            shard_layout=shard_layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
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
                shard_layout=shard_layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                deallocate_activation=True,
                reallocate_halo_output=True,
                reshard_if_not_optimal=True,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
                dtype=ttnn.bfloat16,
                fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
                packer_l1_acc=model_config.get("packer_l1_acc", True),
                math_approx_mode=model_config.get("math_approx_mode", False),
            )
        else:
            self.downsample_layer = None

    def __call__(self, x, device):
        identity = x
        logger.info(f"conv1- 1x1 convolution")
        logger.info(f"x.shape{x.shape =}")
        # conv1: 1x1 expansion
        # import pdb; pdb.set_trace()
        out, shape_ = self.conv1(device, x, x.shape)

        logger.info(f"conv2- 3x3 grouped convolution")
        # conv2: 3x3 grouped convolution
        out, shape_ = self.conv2(device, out, shape_)

        # SE Module
        logger.info(f"SE module")
        logger.info(f"reduce mean")
        # Global average pooling
        out = ttnn.reshape(out, shape_)
        print("""""" """""" """""" "")
        print(out.shape)
        se_out = ttnn.mean(out, dim=[1, 2], keepdim=True)
        shape_ = se_out.shape

        logger.info(f"SE fc1")
        se_out, shape_ = self.se_fc1(device, se_out, shape_)

        logger.info(f"SE fc2")
        se_out, shape_ = self.se_fc2(device, se_out, shape_)
        se_out = ttnn.sigmoid(se_out)
        print("""""" """""" """""" "")
        print(shape_)
        print(se_out.shape)

        # Apply SE scaling
        out = ttnn.multiply(out, se_out)
        shape_ = out.shape

        # conv3: 1x1 projection
        out, shape_ = self.conv3(device, out, shape_)
        logger.info(f"after conv 1x1{out.shape =}")
        out = ttnn.reshape(out, shape_)
        logger.info(f"reshape shape{out.shape =}")
        # Handle downsample
        if self.downsample_layer is not None:
            logger.info(f"downsample")
            logger.info(f"identity shape{identity.shape =}")
            logger.info(f" shape{shape_ =}")
            identity, _ = self.downsample_layer(device, identity, identity.shape)
            identity = ttnn.reshape(identity, shape_)

        logger.info(f"Add")
        logger.info(f"Add in l shape{out.shape =}")
        logger.info(f"Add in r shape{identity.shape =}")
        out = ttnn.add(out, identity)
        logger.info(f"Add out shape{out.shape =}")
        # out = ttnn.reshape(out, shape_)
        out = ttnn.relu(out)  # Final ReLU activation

        return out
