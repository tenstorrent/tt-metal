# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn

# One op's L1 working set may take at most 1/L1_BUDGET_DIVISOR of the grid's total L1; the rest is
# left to the other tensors live at the time. Past it, an activation goes to DRAM.
L1_BUDGET_DIVISOR = 8
# Outputs that stay alive until a module returns share L1 with everything after them, so each may take
# only 1/KEPT_BUDGET_DIVISOR of it.
KEPT_BUDGET_DIVISOR = 64
# A sharded conv spreads its input and output over every core, so it fits L1 until they take about
# half of the grid's L1.
CONV_BUDGET_DIVISOR = 2
# A band leaves room beside its input and output for the conv's own circular buffers.
BAND_BUDGET_DIVISOR = 4
BF16_BYTES = 2
# The full grid's block-sharded convs split their input channels over this many core columns, and
# accumulate them in as many blocks.
FULL_GRID_CHANNEL_BLOCKS = 8


def _pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def l1_budget_bytes(device, divisor=L1_BUDGET_DIVISOR) -> int:
    l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    return l1.num_banks * l1.total_bytes_per_bank // divisor


def exceeds_l1_budget(device, num_elements: int, divisor=L1_BUDGET_DIVISOR) -> bool:
    """Whether bf16 data of this size is too large for the device's L1 and has to live in DRAM.

    The full grid keeps these activations in L1; a reduced grid has too little L1 left for them.
    """
    return num_elements * BF16_BYTES > l1_budget_bytes(device, divisor)


def activation_memory_config(device, tensor, width=None, divisor=L1_BUDGET_DIVISOR) -> ttnn.MemoryConfig:
    """interleaved_memory_config for an output with tensor's rows and width columns, its own by default."""
    shape = list(tensor.padded_shape)
    width = width or shape[-1]
    if tensor.layout == ttnn.TILE_LAYOUT:
        width = ttnn.core.roundup(width, ttnn.TILE_SIZE)
    return interleaved_memory_config(device, math.prod(shape[:-1]) * width, divisor)


def kept_memory_config(device, tensor, width=None) -> ttnn.MemoryConfig:
    """activation_memory_config for an output that stays alive until its module returns."""
    return activation_memory_config(device, tensor, width, KEPT_BUDGET_DIVISOR)


def interleaved_memory_config(device, num_elements: int, divisor=L1_BUDGET_DIVISOR) -> ttnn.MemoryConfig:
    """L1 for a bf16 interleaved intermediate that fits the L1 budget, DRAM otherwise."""
    return ttnn.DRAM_MEMORY_CONFIG if exceeds_l1_budget(device, num_elements, divisor) else ttnn.L1_MEMORY_CONFIG


class TtConv2D:
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
        fp32_dest_acc=True,
    ):
        if is_blk:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.activation_dtype = activation_dtype
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=fp32_dest_acc,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        # An L1 conv holds its whole input and output in L1 at once. Past its budget it runs over bands
        # of output rows instead, each through the same L1 conv with the input rows it reads and zero
        # padding only at the image's edges, so every band computes the rows the whole conv would.
        (kh, kw), (sh, sw), (ph, pw) = (_pair(conv.kernel_size), _pair(conv.stride), _pair(conv.padding))
        self.kernel_hw, self.stride_hw, self.padding_hw = (kh, kw), (sh, sw), (ph, pw)
        self.output_hw = ((conv.input_height + 2 * ph - kh) // sh + 1, (conv.input_width + 2 * pw - kw) // sw + 1)
        output_height, output_width = self.output_hw
        self.band_rows = None
        self.run_in_dram = False
        if exceeds_l1_budget(
            device,
            conv.batch_size * conv.input_height * conv.input_width * conv.in_channels
            + conv.batch_size * output_height * output_width * conv.out_channels,
            CONV_BUDGET_DIVISOR,
        ):
            l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1)
            strided_1x1 = (kh, kw) == (1, 1) and (sh, sw) != (1, 1)
            if not strided_1x1 and kh * kw * conv.in_channels * conv.out_channels > l1.total_bytes_per_bank // 2:
                # A deep stage's weights leave a band no room in a small grid's L1. Such a conv runs from
                # DRAM instead: sliced by height and block sharded, which splits the weights by output
                # channel, or for a 1x1 as a matmul from interleaved DRAM. ttnn cannot slice a strided
                # 1x1 conv, which stays in bands.
                self.run_in_dram = True
            else:
                fitting = [
                    rows
                    for rows in range(1, output_height + 1)
                    if output_height % rows == 0
                    and not exceeds_l1_budget(
                        device,
                        ((rows - 1) * sh + kh) * conv.input_width * conv.in_channels
                        + rows * output_width * conv.out_channels,
                        BAND_BUDGET_DIVISOR,
                    )
                ]
                assert fitting, f"no band of the {output_height}-row output fits L1"
                self.band_rows = max(fitting)
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=self._dram_shard_layout(shard_layout),
            deallocate_activation=dealloc_act,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            activation=activation,
            # Every band shape keeps its own config tensors; a small grid's L1_SMALL holds few.
            config_tensors_in_dram=self.band_rows is not None or self.run_in_dram,
            # A sliced conv's output layout depends on the grid unless pinned.
            **({"output_layout": ttnn.TILE_LAYOUT} if self.run_in_dram else {}),
        )
        if self.band_rows is not None and self.conv_config.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            # A block-sharded conv adds up its input channels in one block per core column, rounding the
            # partial sum to the output format between blocks. On a few cores the blocks are larger and
            # the sum rounds less often, which changes the result the rest of the model was matched to.
            # Width sharding takes the block width explicitly, so the bands keep the full grid's blocks.
            grid = device.compute_with_storage_grid_size()
            cores = grid.x * grid.y
            assert FULL_GRID_CHANNEL_BLOCKS % cores == 0 and conv.in_channels % (32 * FULL_GRID_CHANNEL_BLOCKS) == 0
            self.conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
            self.conv_config.act_block_w_div = FULL_GRID_CHANNEL_BLOCKS // cores
        if not self.run_in_dram:
            self.slice_config = ttnn.Conv2dL1FullSliceConfig
        elif self._is_matmul():
            self.slice_config = None
        else:
            self.slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight)
        # Sliced convs write to DRAM; only the 1x1 matmul path needs to be told.
        self.output_memory_config = ttnn.DRAM_MEMORY_CONFIG if self.run_in_dram and self._is_matmul() else None
        if act_block_h is not None:
            self.conv_config.act_block_h_override = act_block_h
        # conv2d prepares the raw weight and bias together only when both are on the host; a raw
        # device bias would be left unprepared next to the prepared weight.
        for tensor in (conv_pth.weight, conv_pth.bias):
            assert tensor is None or not ttnn.is_tensor_storage_on_device(
                tensor
            ), "raw conv weights and biases must stay on the host until conv2d prepares them"
        self.bias = conv_pth.bias
        self.weight = conv_pth.weight
        self.raw_weight, self.raw_bias = conv_pth.weight, conv_pth.bias
        self.band_weights = {}

    def _is_matmul(self):
        return self.kernel_hw == (1, 1) and self.stride_hw == (1, 1) and self.padding_hw == (0, 0)

    def _dram_shard_layout(self, shard_layout):
        if not self.run_in_dram:
            return shard_layout
        # A 1x1 conv runs its matmul straight from interleaved DRAM only when it picks its own
        # sharding; a fixed shard layout would shard the whole input into L1 first.
        return None if self._is_matmul() else ttnn.TensorMemoryLayout.BLOCK_SHARDED

    def __call__(self, x):
        if self.band_rows is not None:
            return self._call_in_bands(x)
        if self.run_in_dram:
            # A conv only slices through DRAM when its input is there.
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, output_height, output_width, _ = self._conv(
            x, self.conv.batch_size, self.conv.input_height, self.conv.input_width, self.padding
        )
        return x, output_height, output_width

    def _conv(self, x, batch_size, input_height, input_width, padding, weights=None):
        weight, bias = weights if weights is not None else (self.weight, self.bias)
        [x, [output_height, output_width], [weight, bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            conv_config=self.conv_config,
            slice_config=self.slice_config,
            memory_config=self.output_memory_config,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
        )
        if weights is None:
            self.weight, self.bias = weight, bias
        return x, output_height, output_width, (weight, bias)

    def _call_in_bands(self, x):
        batch_size, input_height, input_width = self.conv.batch_size, self.conv.input_height, self.conv.input_width
        (kh, _), (sh, _), (ph, pw) = self.kernel_hw, self.stride_hw, self.padding_hw
        output_height, output_width = self.output_hw
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Bands are row ranges of the flattened NHWC input.
        x = ttnn.reshape(x, (1, 1, batch_size * input_height * input_width, x.shape[-1]))
        outputs = []
        for image in range(batch_size):
            for top in range(0, output_height, self.band_rows):
                first = top * sh - ph
                last = (top + self.band_rows - 1) * sh - ph + kh - 1
                pad_top, pad_bottom = max(0, -first), max(0, last - (input_height - 1))
                first, last = max(first, 0), min(last, input_height - 1)
                rows = ttnn.slice(
                    x,
                    [0, 0, (image * input_height + first) * input_width, 0],
                    [1, 1, (image * input_height + last + 1) * input_width, x.shape[-1]],
                )
                # Each band shape gets its own prepared weights, from the raw ones on the host.
                key = (last - first + 1, pad_top, pad_bottom)
                weights = self.band_weights.get(key, (self.raw_weight, self.raw_bias))
                out, _, _, self.band_weights[key] = self._conv(
                    rows, 1, last - first + 1, input_width, (pad_top, pad_bottom, pw, pw), weights
                )
                outputs.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
        out = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out, output_height, output_width
