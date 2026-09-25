# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn

# Width slices for convs whose activation is kept in DRAM, capped per conv by what the
# output width allows. Sized for the strided 1x1 conv that opens stage 2 at 928x1600:
# each slice's halo buffer holds 1/num_slices of its 6 x 232 x 400 x 256 bf16 input,
# 3.25 MB per L1 bank in total against 576 KB free, so it needs at least 6 slices. Its
# 200-wide output caps it at 7.
DRAM_CONV_SLICES = 8


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
        dram_activation=False,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
    ):
        """``input_dtype`` and ``input_layout`` describe the interleaved DRAM tensor every call
        receives. conv2d lays its weights out for the input it will see, so they are prepared
        here against that description and a forward does no host work; a call with a
        different input fails instead of running on mismatched weights."""
        self.dram_activation = dram_activation
        self.input_dtype = input_dtype
        self.input_layout = input_layout
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
        # Blackhole's LoFi accumulation is less accurate than Wormhole's; UniAD's
        # ResNet-101 backbone PCC drops to ~0.16 with LoFi on BH. Use HiFi2 on BH
        # to recover PCC ≥ 0.99. Wormhole keeps LoFi for performance.
        math_fidelity = ttnn.MathFidelity.HiFi2 if ttnn.get_arch_name() == "blackhole" else ttnn.MathFidelity.LoFi
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
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

        self.activation = activation
        self.weights_dtype = weights_dtype
        self.is_pointwise = (
            tuple(self.kernel_size) == (1, 1)
            and tuple(self.stride) == (1, 1)
            and tuple(self.padding) == (0, 0)
            and self.groups == 1
        )
        if self.is_fpn:
            self.input_height = conv_pth["height"]
            self.input_width = conv_pth["width"]
            self.batch_size = conv_pth["batch"]
        else:
            self.input_height = conv.input_height
            self.input_width = conv.input_width
            self.batch_size = conv.batch_size

        self.slice_config = None
        if self.dram_activation and not self.is_pointwise:
            # An interleaved DRAM input sends conv2d down its DRAM path, which computes the
            # output in slices that each fit in L1. Its automatic slice count overflows L1
            # on the strided 1x1 conv that opens stage 2, so the count is fixed here. Width
            # slices of a TILE output are whole tiles wide, so a conv whose output is W wide
            # takes at most ceil(W / TILE_SIZE) of them.
            output_width = (self.input_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
            num_slices = min(DRAM_CONV_SLICES, math.ceil(output_width / ttnn.TILE_SIZE))
            self.slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=num_slices)

        self.linear_weight = None
        self.linear_bias = None
        self.linear_activation = None
        if self.dram_activation and self.is_pointwise and activation is not None:
            assert (
                activation.op_type == ttnn.UnaryOpType.RELU
            ), f"only RELU is supported after the pointwise linear, got {activation.op_type}"
            self.linear_activation = "relu"
        if not (self.dram_activation and self.is_pointwise):
            self._prepare_conv_weights()
        else:
            weight = ttnn.to_torch(self.weight).reshape(self.out_channels, self.in_channels).t().contiguous()
            self.linear_weight = ttnn.from_torch(
                weight, dtype=self.weights_dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            if self.bias is not None:
                bias = ttnn.to_torch(self.bias).reshape(1, self.out_channels)
                self.linear_bias = ttnn.from_torch(
                    bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )

    def _prepare_conv_weights(self):
        # conv2d lays its weights out for the input it sees and prepares them on the host on
        # its first call. Making that call here, on zeros shaped like every later input, puts
        # the host work in the constructor and uses conv2d's own preparation, which
        # ttnn.prepare_conv_weights does not reproduce for interleaved DRAM inputs.
        warmup_input = ttnn.zeros(
            (1, 1, self.batch_size * self.input_height * self.input_width, self.in_channels),
            dtype=self.input_dtype,
            layout=self.input_layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        warmup_output, _, _ = self._conv2d(warmup_input)
        ttnn.deallocate(warmup_output)

    def _pointwise_as_linear(self, x):
        # conv2d lowers a 1x1/stride-1 conv to a matmul that always shards its activation
        # in L1 and never slices it, so an activation larger than L1 cannot run as a conv.
        # A 1x1 conv over NHWC is a matmul over channels, and ttnn.linear streams
        # interleaved DRAM operands of any height.
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        # conv2d returns (1, 1, N*H*W, C) whatever the input rank; match it so the
        # residual add sees the same shape on both branches.
        rows = math.prod(list(x.shape)[:-1])
        x = ttnn.reshape(x, (1, 1, rows, x.shape[-1]))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            x,
            self.linear_weight,
            bias=self.linear_bias,
            activation=self.linear_activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )

    def __call__(self, x):
        if self.dram_activation and self.is_pointwise:
            return self._pointwise_as_linear(x), self.input_height, self.input_width

        memory_config = x.memory_config()
        assert (
            memory_config.buffer_type == ttnn.BufferType.DRAM
            and not memory_config.is_sharded()
            and x.dtype == self.input_dtype
            and x.layout == self.input_layout
        ), (
            f"conv {self.in_channels}->{self.out_channels} k={tuple(self.kernel_size)}: weights prepared for an "
            f"interleaved DRAM {self.input_dtype} {self.input_layout} input, got {memory_config.buffer_type} "
            f"sharded={memory_config.is_sharded()} {x.dtype} {x.layout}"
        )
        return self._conv2d(x)

    def _conv2d(self, x):
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
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
            slice_config=self.slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x, output_height, output_width
