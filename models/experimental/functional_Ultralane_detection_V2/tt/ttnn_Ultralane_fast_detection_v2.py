# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def p(x, b="x"):
    print(f"{b}'s shape is {x.shape}")
    print(f"{b}'s layout is {x.layout}")
    print(f"{b}'s dtype is {x.dtype}")
    print(f"{b}'s config is {x.memory_config()}")


class ttnn_UFLD_V2_Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = False
        self.cache = cache
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            math_approx_mode=True,
        )
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            activation=activation,
            input_channels_alignment=16,
        )
        config_override = None
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]
        # if conv_pth.bias is not None:
        #     bias = ttnn.from_device(conv_pth.bias)
        #     self.bias = bias
        # else:
        self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        input_height = x.shape[1]
        input_width = x.shape[2]
        batch_size = x.shape[0]
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
            conv_op_cache=self.cache,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            # memory_config = ttnn.DRAM_MEMORY_CONFIG
        )
        return x


class ttnn_Basic_Block:
    def __init__(self, conv_args, conv_pth, device):
        self.conv1 = ttnn_UFLD_V2_Conv2D(conv_args.conv1, conv_pth.conv1, device=device, activation="relu")
        self.conv2 = ttnn_UFLD_V2_Conv2D(conv_args.conv2, conv_pth.conv2, device=device, activation="")

    def __call__(self, device, x):
        x = self.conv1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (1, 80, 200, 64))  # RESHAPING FROM (1,1,NHW,C) TO (N,H,W,C) TO AVOID OOM
        x = self.conv2(x)
        return x


class ttnn_Resnet_34:
    def __init__(self, conv_args, conv_pth, device):
        self.conv1 = ttnn_UFLD_V2_Conv2D(conv_args.conv1, conv_pth.conv1, device=device, activation="relu")

    def __call__(self, device, x):  # [1, 320, 800, 3]
        x = self.conv1(x)  # [1, 1, 64000, 64] #0.99974
        # torch.save(ttnn.to_torch(x).permute(0,3,1,2).reshape(1,64,160,400),"/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_Ultralane_detection_V2/dumps/ttnn_out.pth")
        p(x, "x")
        x = ttnn.max_pool2d(
            x,
            batch_size=1,
            input_h=160,
            input_w=400,
            channels=64,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )
        print("after maxpool")
        p(x, "x")
