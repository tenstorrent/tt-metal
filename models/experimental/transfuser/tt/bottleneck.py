# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
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

        # Extract per-layer override dicts
        conv1_cfg = layer_config.get("conv1", {})
        conv2_cfg = layer_config.get("conv2", {})
        se_fc1_cfg = layer_config.get("se_fc1", {})
        se_fc2_cfg = layer_config.get("se_fc2", {})
        conv3_cfg = layer_config.get("conv3", {})
        downsample_cfg = layer_config.get("downsample", {})

        self.torch_model = torch_model
        self.use_fallback = use_fallback
        self.block_name = block_name
        self.stage_name = stage_name

        def make_conv2d(
            params_key,
            *,
            kernel_size,
            stride,
            padding,
            activation,
            cfg_overrides,
            groups=None,
            is_reshape=False,
        ):
            return TTConv2D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                parameters=parameters[params_key],
                kernel_fidelity=model_config,
                activation=activation,
                groups=(groups if groups is not None else 1),
                shard_layout=cfg_overrides.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
                act_block_h=cfg_overrides.get("act_block_h", None),
                memory_config=cfg_overrides.get("memory_config", ttnn.L1_MEMORY_CONFIG),
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
                deallocate_activation=True
                if kernel_size != 1 or params_key in ("conv2", "se_fc1", "se_fc2", "conv3", "downsample")
                else False,
                reallocate_halo_output=True,
                reshard_if_not_optimal=True,
                dtype=ttnn.bfloat16,
                fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
                packer_l1_acc=model_config.get("packer_l1_acc", True),
                math_approx_mode=model_config.get("math_approx_mode", False),
                is_reshape=is_reshape,
            )

        # ------------------------- conv1: 1x1 + ReLU -------------------------
        self.conv1 = make_conv2d(
            "conv1",
            kernel_size=1,
            stride=1,
            padding=0,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            cfg_overrides=conv1_cfg,
            groups=1,
            is_reshape=False,
        )

        # --------------------- conv2: 3x3 grouped + ReLU ---------------------
        self.conv2 = make_conv2d(
            "conv2",
            kernel_size=3,
            stride=stride,
            padding=1,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            cfg_overrides=conv2_cfg,
            groups=groups,
            is_reshape=False,
        )

        # --------------------------- SE: fc1 (1x1 + ReLU) --------------------
        self.se_fc1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["se"]["fc1"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=se_fc1_cfg.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=se_fc1_cfg.get("act_block_h", None),
            memory_config=se_fc1_cfg.get("memory_config", ttnn.L1_MEMORY_CONFIG),
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
            is_reshape=False,
        )

        # --------------------------- SE: fc2 (1x1, no act) -------------------
        self.se_fc2 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["se"]["fc2"],
            kernel_fidelity=model_config,
            activation=None,
            shard_layout=se_fc2_cfg.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=se_fc2_cfg.get("act_block_h", None),
            memory_config=se_fc1_cfg.get("memory_config", ttnn.L1_MEMORY_CONFIG),
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
            is_reshape=False,
        )

        # ----------------------- conv3: 1x1 projection (no act) --------------
        self.conv3 = make_conv2d(
            "conv3",
            kernel_size=1,
            stride=1,
            padding=0,
            activation=None,
            cfg_overrides=conv3_cfg,
            groups=1,
            is_reshape=False,
        )

        # ------------------------------ optional downsample -------------------
        if downsample:
            self.downsample_layer = TTConv2D(
                kernel_size=1,
                stride=stride,
                padding=0,
                parameters=parameters["downsample"],
                kernel_fidelity=model_config,
                activation=None,
                shard_layout=downsample_cfg.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
                act_block_h=downsample_cfg.get("act_block_h", None),
                memory_config=se_fc1_cfg.get("memory_config", ttnn.L1_MEMORY_CONFIG),
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
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

        # conv1- 1x1 convolution
        out, shape_ = self.conv1(device, x, input_shape)

        # conv2- 3x3 grouped convolution
        out, shape_ = self.conv2(device, out, shape_)

        # SE module
        # reduce mean
        out1 = ttnn.reallocate(out)
        # Reshape to 4D for mean operation
        out_4d = ttnn.reshape(out, shape_)
        se_out = ttnn.mean(out_4d, dim=[1, 2], keepdim=True)
        if self.use_fallback and self.torch_model is not None:
            # Falling Back SE module
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
            # SE fc1
            se_out, se_shape = self.se_fc1(device, se_out, se_out.shape)

            # SE fc2
            se_out, se_shape = self.se_fc2(device, se_out, se_shape)
            se_out = ttnn.sigmoid(se_out)

        out_4d = ttnn.multiply(out1, se_out)
        # Flatten back to match identity format
        batch, height, width, channels = shape_
        out = ttnn.reshape(out_4d, (1, 1, batch * height * width, channels))

        # conv3: 1x1 projection - now in flattened format
        out, shape_ = self.conv3(device, out, (batch, height, width, channels))

        # Handle downsample - identity is already in flattened format
        if self.downsample_layer is not None:
            # downsample
            identity, _ = self.downsample_layer(device, identity, identity_shape)

        # Add
        # Both tensors are now in flattened format
        out = ttnn.add(out, identity)
        out = ttnn.relu(out)

        return out, shape_
