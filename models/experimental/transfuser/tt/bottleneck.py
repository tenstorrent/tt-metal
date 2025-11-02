# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger
from models.experimental.transfuser.tt.utils import TTConv2D


class TTRegNetBottleneck:
    def __init__(
        self,
        parameters,
        model_config,
        layer_config,
        stride=1,
        downsample=False,
        groups=1,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        torch_model=None,
        use_fallback=False,
        block_name=None,
        stage_name=None,
    ):
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.model_config = model_config
        # Extract config for each convolution
        conv1_config = layer_config.get("conv1", {})
        conv2_config = layer_config.get("conv2", {})
        se_fc1_config = layer_config.get("se_fc1", {})
        se_fc2_config = layer_config.get("se_fc2", {})
        conv3_config = layer_config.get("conv3", {})
        downsample_config = layer_config.get("downsample", {})

        self.torch_model = torch_model
        self.use_fallback = use_fallback
        self.block_name = block_name
        self.stage_name = stage_name

        # conv1: 1x1 convolution
        self.conv1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["conv1"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=conv1_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=conv1_config.get("act_block_h", None),
            enable_act_double_buffer=conv1_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=conv1_config.get("enable_weights_double_buffer", False),
            memory_config=conv1_config.get("memory_config", ttnn.L1_MEMORY_CONFIG),
            deallocate_activation=False,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
            is_reshape=False,
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
            shard_layout=conv2_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=conv2_config.get("act_block_h", None),
            enable_act_double_buffer=conv2_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=conv2_config.get("enable_weights_double_buffer", False),
            memory_config=conv2_config.get("memory_config", ttnn.L1_MEMORY_CONFIG),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
            is_reshape=False,
        )

        # SE Module
        self.se_fc1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["se"]["fc1"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=se_fc1_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=se_fc1_config.get("act_block_h", None),
            enable_act_double_buffer=se_fc1_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=se_fc1_config.get("enable_weights_double_buffer", False),
            memory_config=se_fc1_config.get("memory_config", ttnn.L1_MEMORY_CONFIG),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
            is_reshape=False,
        )

        self.se_fc2 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["se"]["fc2"],
            kernel_fidelity=model_config,
            activation=None,
            shard_layout=se_fc2_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=se_fc2_config.get("act_block_h", None),
            enable_act_double_buffer=se_fc2_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=se_fc2_config.get("enable_weights_double_buffer", False),
            memory_config=se_fc2_config.get("memory_config", ttnn.L1_MEMORY_CONFIG),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
            is_reshape=False,
        )

        # conv3: 1x1 convolution (no activation)
        self.conv3 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["conv3"],
            kernel_fidelity=model_config,
            activation=None,
            shard_layout=conv3_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=conv3_config.get("act_block_h", None),
            enable_act_double_buffer=conv3_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=conv3_config.get("enable_weights_double_buffer", False),
            memory_config=conv3_config.get("memory_config", ttnn.L1_MEMORY_CONFIG),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
            is_reshape=False,
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
                shard_layout=downsample_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
                act_block_h=downsample_config.get("act_block_h", None),
                enable_act_double_buffer=downsample_config.get("enable_act_double_buffer", False),
                enable_weights_double_buffer=downsample_config.get("enable_weights_double_buffer", False),
                memory_config=downsample_config.get("memory_config", ttnn.L1_MEMORY_CONFIG),
                deallocate_activation=True,
                reallocate_halo_output=True,
                reshard_if_not_optimal=True,
                dtype=ttnn.bfloat16,
                fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
                packer_l1_acc=model_config.get("packer_l1_acc", True),
                math_approx_mode=model_config.get("math_approx_mode", False),
            )
        else:
            self.downsample_layer = None

    def __call__(self, x, device, input_shape=None):
        if input_shape is None:
            input_shape = x.shape
        identity = x
        identity_shape = input_shape

        logger.info(f"conv1- 1x1 convolution")
        out, shape_ = self.conv1(device, x, input_shape)

        logger.info(f"conv2- 3x3 grouped convolution")
        out, shape_ = self.conv2(device, out, shape_)

        # SE Module
        logger.info(f"SE module")
        logger.info(f"reduce mean")

        out1 = ttnn.reallocate(out)
        # Reshape to 4D for mean operation
        out_4d = ttnn.reshape(out, shape_)
        se_out = ttnn.mean(out_4d, dim=[1, 2], keepdim=True)
        if self.use_fallback and self.torch_model is not None:
            logger.info(f"Falling Back SE module for {self.stage_name} {self.block_name}")
            se_out_torch = ttnn.to_torch(
                se_out,
                device=device,
            )
            se_out_torch = torch.permute(se_out_torch, (0, 3, 1, 2))
            se_out_torch = se_out_torch.to(torch.float32)
            se_out_torch = self.torch_model.fallback(
                se_out_torch, block_name=self.block_name, stage_name=self.stage_name
            )
            se_out = ttnn.from_torch(
                se_out_torch,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                device=device,
            )
            se_out = ttnn.permute(se_out, (0, 2, 3, 1))

        else:
            logger.info(f"SE fc1")
            se_out, se_shape = self.se_fc1(device, se_out, se_out.shape)

            logger.info(f"SE fc2")
            se_out, se_shape = self.se_fc2(device, se_out, se_shape)
            se_out = ttnn.sigmoid(se_out)

        # return out_4d, se_shape
        out_4d = ttnn.multiply(out1, se_out)
        # Flatten back to match identity format
        batch, height, width, channels = shape_
        out = ttnn.reshape(out_4d, (1, 1, batch * height * width, channels))

        # conv3: 1x1 projection - now in flattened format
        out, shape_ = self.conv3(device, out, (batch, height, width, channels))

        # Handle downsample - identity is already in flattened format
        if self.downsample_layer is not None:
            logger.info(f"downsample")
            identity, _ = self.downsample_layer(device, identity, identity_shape)

        logger.info(f"Add")
        # Both tensors are now in flattened format
        out = ttnn.add(out, identity)
        out = ttnn.relu(out)

        return out, shape_
