# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.experimental.yolo_common.yolo_utils import determine_num_cores, get_core_grid_from_num_cores


def interleaved_to_sharded(x):
    if x.get_layout() == ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
    nhw = x.shape[0] * x.shape[1] * x.shape[2]
    num_cores = determine_num_cores(nhw, x.shape[2])
    core_grid = get_core_grid_from_num_cores(num_cores)
    shardspec = ttnn.create_sharded_memory_config_(
        x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    return ttnn.reshard(x, shardspec) if x.is_sharded() else ttnn.interleaved_to_sharded(x, shardspec)


class TtYolov10Conv2D:
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
        auto_shard=False,
        deallocate_activation=False,
        act_block_h_override=0,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
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
        self.deallocate_activation = deallocate_activation
        self.auto_shard = auto_shard
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        if shard_layout is None and not self.auto_shard:
            shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if use_1d_systolic_array
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.conv_output_dtype = activation_dtype
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
            enable_subblock_padding=False,
            output_layout=ttnn.TILE_LAYOUT,
            act_block_h_override=act_block_h_override,
            enable_act_double_buffer=enable_act_double_buffer,
            enable_weights_double_buffer=enable_weights_double_buffer,
        )
        if auto_shard:
            self.conv_config.shard_layout = None

        config_override = None
        config_override = {"act_block_h": 64} if conv.in_channels == 3 else None
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth:
            self.bias = conv_pth.bias
        else:
            self.bias = None

        self.weight = conv_pth.weight

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
            dtype=self.conv_output_dtype,
        )
        return x


class Conv:
    def __init__(
        self,
        device,
        parameter,
        conv_pt,
        enable_act=True,
        is_detect=False,
        enable_identity=False,
        is_dfl=False,
        config_override=None,
        use_1d_systolic_array=True,
        auto_shard=False,
        shard_layout=None,
        activation="",
        deallocate_activation=False,
        activation_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
    ):
        self.enable_identity = enable_identity
        self.enable_act = enable_act
        if not self.enable_identity:
            activation = "silu"
        self.conv = TtYolov10Conv2D(
            parameter.conv,
            conv_pt.conv,
            device=device,
            is_detect=is_detect,
            is_dfl=is_dfl,
            config_override=config_override,
            use_1d_systolic_array=use_1d_systolic_array,
            auto_shard=auto_shard,
            shard_layout=shard_layout,
            activation=activation,
            deallocate_activation=deallocate_activation,
            activation_dtype=activation_dtype,
            enable_act_double_buffer=enable_act_double_buffer,
            enable_weights_double_buffer=enable_weights_double_buffer,
        )

    def __call__(self, x):
        x = self.conv(x)
        return x


def deallocate_tensors(*tensors):
    for t in tensors:
        ttnn.deallocate(t)
