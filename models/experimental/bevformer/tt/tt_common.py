# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Conv shared by the ResNet101-DCN backbone and the FPN.

Weights are prepared once, in the constructor, for one interleaved DRAM input dtype
and layout. A later call with a different input fails. A 1x1 stride-1 conv whose
activation lives in DRAM runs as a matmul; a spatial conv there runs in a fixed
number of width slices.
"""

import math

import ttnn


def layer_norm(x, params, residual=None):
    """``ttnn.layer_norm(x + residual)`` with the eps of the module ``params`` were preprocessed from.

    ttnn defaults to epsilon=1e-12; ``preprocess_layer_norm_parameters`` records the
    module's own (1e-5 for nn.LayerNorm). The residual add runs inside the norm's kernel.
    """
    return ttnn.layer_norm(
        x, weight=params.weight, bias=params.bias, epsilon=params.eps, residual_input_tensor=residual
    )


class TtnnConv2D:
    """``conv`` carries the conv's geometry and its input's batch, height and width.

    Most convs run through conv2d. ``input_dtype`` and ``input_layout`` describe the
    interleaved DRAM tensor every call receives; conv2d lays its weights out for that
    input, so they are prepared here against it and a forward does no host work, and a
    call with a different input fails instead of running on mismatched weights.
    With ``dram_activation``, a 1x1 stride-1 conv runs as ``ttnn.linear`` instead, which
    takes any layout, sharding or dtype and ignores ``input_dtype`` and ``input_layout``;
    a spatial conv keeps conv2d and runs in ``dram_conv_slices`` width slices.

    ``dealloc_act`` lets conv2d free the L1-sharded copy it makes of a DRAM input once the
    halo has read it; the DRAM input itself is never freed.

    ``output_dtype`` is the dtype the conv emits, None for its input's. A spatial conv, on
    conv2d's own path with packer_l1_acc off, keeps its partial sums between reduction blocks
    in the output dtype; a 1x1 stride-1 conv runs as a matmul (conv2d lowers it, or
    ``ttnn.linear`` above), which keeps them in fp32 with ``fp32_dest_acc_en``.
    """

    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        activation=None,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk=False,
        dealloc_act=False,
        act_block_h=None,
        fp32_dest_acc_en=False,
        dram_activation=False,
        dram_conv_slices=None,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        output_dtype=None,
    ):
        self.dram_activation = dram_activation
        self.input_dtype = input_dtype
        self.input_layout = input_layout
        self.output_dtype = output_dtype
        if is_blk:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        # conv2d gets no dilation, nor does the slice count below account for one.
        assert tuple(conv.dilation) == (1, 1), f"TtnnConv2D supports dilation (1, 1) only, got {conv.dilation}"
        self.groups = conv.groups
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
        if act_block_h is not None:
            self.conv_config.act_block_h_override = act_block_h

        self.bias = conv_pth.bias

        self.weight = conv_pth.weight

        self.weights_dtype = weights_dtype
        self.is_pointwise = (
            tuple(self.kernel_size) == (1, 1)
            and tuple(self.stride) == (1, 1)
            and tuple(self.padding) == (0, 0)
            and self.groups == 1
        )
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.batch_size = conv.batch_size

        self.slice_config = None
        if self.dram_activation and not self.is_pointwise:
            # conv2d's automatic slice count can overflow L1 on large activations, so the
            # count is fixed here. Width slices of a TILE output are whole tiles wide, so a
            # conv whose output is W wide takes at most ceil(W / TILE_SIZE) of them.
            assert dram_conv_slices is not None, "a spatial conv with dram_activation needs dram_conv_slices"
            output_width = (self.input_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
            num_slices = min(dram_conv_slices, math.ceil(output_width / ttnn.TILE_SIZE))
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
            dtype=self.output_dtype,
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
            dtype=self.output_dtype,
            compute_config=self.compute_config,
            slice_config=self.slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x, output_height, output_width
