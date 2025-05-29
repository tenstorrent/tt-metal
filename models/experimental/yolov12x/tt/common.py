# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import math

from models.experimental.yolo_common.yolo_utils import determine_num_cores, get_core_grid_from_num_cores


class Yolov12x_Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_detect=False,
        is_dfl=False,
        config_override=None,
        deallocate_activation=False,
    ):
        self.is_detect = is_detect
        self.is_dfl = is_dfl
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

        if hasattr(self.padding, "__len__"):
            if len(self.padding) == 2:
                self.padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
            elif len(padding) == 4:
                self.padding = (self.padding[0], self.padding[1], self.padding[2], self.padding[3])
            else:
                raise ValueError("Padding should be a scalar or a list of 2 or 4 elements")
        else:
            self.padding = (self.padding, self.padding, self.padding, self.padding)

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
            # input_channels_alignment=32,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        self.weight = ttnn.from_device(conv_pth.weight)
        self.bias = ttnn.from_device(conv_pth.bias) if "bias" in conv_pth else None

    def __call__(self, x):
        if self.is_detect:
            input_height = int(math.sqrt(x.shape[2]))
            input_width = int(math.sqrt(x.shape[2]))
            batch_size = x.shape[0]
        elif self.is_dfl:
            input_height = x.shape[1]
            input_width = x.shape[2]
            batch_size = x.shape[0]
        else:
            batch_size = self.conv.batch_size
            input_height = self.conv.input_height
            input_width = self.conv.input_width

        print(f"=======================================================================================")
        print(f"input_tensor={x.shape}")
        print(f"weight_tensor={self.weight.shape}")
        print(f"bias_tensor={self.bias.shape}")
        print(f"batch_size={batch_size}")
        print(f"out_channels={self.out_channels}")
        print(f"in_channels={self.in_channels}")
        print(f"input_height={input_height}")
        print(f"input_width={input_width}")
        print(f"kernel_size={self.kernel_size}")
        print(f"stride={self.stride}")
        print(f"padding={self.padding}")
        print(f"dilation= ")
        print(f"groups={self.groups}")
        print(f"use_1d_systolic_array={self.use_1d_systolic_array}")
        print(f"use_shallow_conv_variant= ")
        print(f"memory_config={x.memory_config()}")
        print(f"----------------------------------------------------------------------------")
        print(f"conv_config")
        print(f"dtype={self.conv_config.dtype}")
        print(f"weights_dtype={self.conv_config.weights_dtype}")
        print(f"activation={self.conv_config.activation}")
        # print(f"input_channels_alignment={self.conv_config.input_channels_alignment}")
        print(f"deallocate_activation={self.conv_config.deallocate_activation}")
        print(f"reallocate_halo_output={self.conv_config.reallocate_halo_output}")
        print(f"act_block_h_override={self.conv_config.act_block_h_override}")
        print(f"act_block_w_div={self.conv_config.act_block_w_div}")
        print(f"reshard_if_not_optimal={self.conv_config.reshard_if_not_optimal}")
        print(f"override_sharding_config={self.conv_config.override_sharding_config}")
        print(f"core_grid={self.conv_config.core_grid}")
        print(f"transpose_shards={self.conv_config.transpose_shards}")
        print(f"output_layout={self.conv_config.output_layout}")
        print(f"preprocess_weights_on_device={self.conv_config.preprocess_weights_on_device}")
        print(f"always_preprocess_weights={self.conv_config.always_preprocess_weights}")
        print(f"enable_act_double_buffer={self.conv_config.enable_act_double_buffer}")
        print(f"enable_weights_double_buffer={self.conv_config.enable_weights_double_buffer}")
        print(f"enable_split_reader={self.conv_config.enable_split_reader}")
        print(f"enable_subblock_padding={self.conv_config.enable_subblock_padding}")
        print(f"=======================================================================================")

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
        hw = output_height * output_width
        if x.shape[2] != hw:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = x[:, :, :hw, :]
        return x


class Bottleneck:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = Yolov12x_Conv2D(conv=parameter.cv1.conv, conv_pth=conv_pt.cv1.conv, device=device, activation="silu")
        self.cv2 = Yolov12x_Conv2D(conv=parameter.cv2.conv, conv_pth=conv_pt.cv2.conv, device=device, activation="silu")

    def __call__(self, x):
        input = x
        x = self.cv1(x)
        x = self.cv2(x)
        return input + x


def interleaved_to_sharded(x, num_cores=None):
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    nhw = x.shape[0] * x.shape[1] * x.shape[2]
    if num_cores == None:
        num_cores = determine_num_cores(nhw, int(math.sqrt(x.shape[2])))
        num_cores = 56 if num_cores == 1 else num_cores
    core_grid = get_core_grid_from_num_cores(num_cores)
    shardspec = ttnn.create_sharded_memory_config_(
        x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.reshard(x, shardspec) if x.is_sharded() else ttnn.interleaved_to_sharded(x, shardspec)
