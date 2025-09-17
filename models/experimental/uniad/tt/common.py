# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtnnConv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        activation=None,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk=False,
        dealloc_act=False,
        act_block_h=None,
        is_fpn=False,
        is_wdth=False,
        config_override=None,
    ):
        if is_wdth:
            shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        if is_blk:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        self.conv = conv
        self.conv_pth = conv_pth
        self.is_fpn = is_fpn
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.activation_dtype = activation_dtype
        if self.is_fpn:
            fp32_dest_acc_en = True
        else:
            fp32_dest_acc_en = False
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=dealloc_act,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            activation=activation,
        )
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]
        elif act_block_h is not None:
            self.conv_config.act_block_h_override = act_block_h

        if conv_pth.bias is not None:
            self.bias = conv_pth.bias
        else:
            self.bias = None

        self.weight = conv_pth.weight

    def __call__(self, x):
        if self.is_fpn:
            input_height = self.conv_pth["height"]
            input_width = self.conv_pth["width"]
            batch_size = self.conv_pth["batch"]
        else:
            input_height = self.conv.input_height
            input_width = self.conv.input_width
            batch_size = self.conv.batch_size
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x, output_height, output_width
