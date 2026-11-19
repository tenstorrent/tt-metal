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
# A conv fits a core when its input and output shards and its program's own buffers take at most this
# share of the core's L1. The rest is for tensors other modules keep alive and for the gaps L1 allocation
# leaves between buffers.
CONV_CORE_L1_SHARE = 0.85
# The per-core estimate in TtConv2D._core_l1_bytes mirrors the circular buffers ttnn allocates in
#   ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp (get_cb_info), for convs
#     other than 1x1 stride 1,
#   ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp,
#     for height-sharded 1x1 stride-1 convs, which run as matmuls,
#   ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp,
#     for block-sharded ones.
# Bytes of one 32x32 tile; a block-float tile carries 64 bytes of shared exponents.
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.float32: 4096}
# A conv that runs as a matmul with float32 accumulation keeps one output block of float32 partial sums
# beside its output; in the output's own format they share the output buffer instead.
PARTIALS_TILE_BYTES = TILE_BYTES[ttnn.float32]
# A block-sharded matmul double-buffers the activation and weight blocks it multicasts.
MCAST_BUFFERS = 2
# The halo in front of a non-matmul conv pads its row-major input sticks to 16 bytes, 8 bfloat16 channels.
HALO_CHANNEL_ALIGNMENT = 8


def _pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def _tiles(n):
    return ttnn.core.divup(n, ttnn.TILE_SIZE)


def _row_cores(row_tiles, max_cores, block_tiles):
    """The cores ttnn shards row_tiles rows of tiles over, and the tiles each gets.

    Like ttnn, takes the most cores whose padding stays under one core's share, every share a multiple
    of block_tiles.
    """
    cores = max_cores
    while True:
        padded = ttnn.core.roundup(row_tiles, cores * block_tiles)
        if cores == 1 or padded - row_tiles < padded // cores:
            return cores, padded // cores
        cores -= 1


def _channel_cores(channel_tiles, max_cores):
    """The core columns ttnn splits channel_tiles tiles of channels over, by the same padding rule."""
    cores = max_cores
    while True:
        padded = ttnn.core.roundup(channel_tiles, cores)
        if padded - channel_tiles < padded // cores:
            return cores
        cores -= 1


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
        self.weights_dtype = weights_dtype
        self.fp32_dest_acc = fp32_dest_acc
        self.has_bias = conv_pth.bias is not None
        self.grid = grid = device.compute_with_storage_grid_size()
        self.l1_bank_bytes = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
        cores = grid.x * grid.y
        # Banded block-sharded convs keep the full grid's channel blocks (see below), which needs the core
        # count to divide them. Other grids, such as Wormhole's 8x8, run such a conv whole in L1.
        bandable = shard_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED or FULL_GRID_CHANNEL_BLOCKS % cores == 0
        band_layout = (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED
            if shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
            else shard_layout
        )
        whole = (conv.batch_size, conv.input_height, conv.input_width, output_height, output_width)
        fits_whole = not exceeds_l1_budget(
            device,
            conv.batch_size * conv.input_height * conv.input_width * conv.in_channels
            + conv.batch_size * output_height * output_width * conv.out_channels,
            CONV_BUDGET_DIVISOR,
        ) and self._fits_core(shard_layout, whole, act_block_h)
        if not fits_whole:
            l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1)
            strided_1x1 = (kh, kw) == (1, 1) and (sh, sw) != (1, 1)
            if not strided_1x1 and kh * kw * conv.in_channels * conv.out_channels > l1.total_bytes_per_bank // 2:
                # A deep stage's weights leave a band no room in a small grid's L1. Such a conv runs from
                # DRAM instead: sliced by height and block sharded, which splits the weights by output
                # channel, or for a 1x1 as a matmul from interleaved DRAM. ttnn cannot slice a strided
                # 1x1 conv, which stays in bands.
                self.run_in_dram = True
            elif bandable:
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
                    and self._fits_core(
                        band_layout, (1, (rows - 1) * sh + kh, conv.input_width, rows, output_width), act_block_h
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
            assert conv.in_channels % (32 * FULL_GRID_CHANNEL_BLOCKS) == 0
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

    def _core_l1_bytes(self, layout, shape, act_block_h):
        """L1 one core needs to run this conv on a (batch, input h, input w, output h, output w) shape.

        Returns the bytes of the input shard, the output shard and the buffers the conv's program
        allocates beside them. None for width sharding, which only small grids' bands use and which this
        does not model.
        """
        batch, input_height, input_width, output_height, output_width = shape
        grid = self.grid
        kh, kw = self.kernel_hw
        block_tiles = act_block_h // ttnn.TILE_SIZE if act_block_h else 1
        in_tiles, out_tiles = _tiles(self.in_channels), _tiles(self.out_channels)
        out_rows = _tiles(batch * output_height * output_width)
        if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            row_cores, m = _row_cores(out_rows, grid.x * grid.y, block_tiles)
            k, n = in_tiles, out_tiles
        elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            row_cores, m = _row_cores(out_rows, grid.y, block_tiles)
            channel_cores = _channel_cores(in_tiles, grid.x)
            k, n = ttnn.core.divup(in_tiles, channel_cores), ttnn.core.divup(out_tiles, channel_cores)
        else:
            return None
        weight_tile = TILE_BYTES[self.weights_dtype]
        bf16_tile = TILE_BYTES[ttnn.bfloat16]
        out_tile = TILE_BYTES[self.activation_dtype]
        in_width = k * ttnn.TILE_SIZE
        if not self._is_matmul():
            in_width = min(in_width, ttnn.core.roundup(self.in_channels, HALO_CHANNEL_ALIGNMENT))
        input_bytes = ttnn.core.divup(batch * input_height * input_width, row_cores) * in_width * BF16_BYTES
        output_bytes = m * n * out_tile
        bias_bytes = n * weight_tile if self.has_bias else 0
        if self._is_matmul():
            partials_bytes = m * n * PARTIALS_TILE_BYTES if self.fp32_dest_acc else 0
            if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
                # Every core holds all the weights and reads its activations from the input shard.
                own_bytes = k * n * weight_tile
            else:
                own_bytes = MCAST_BUFFERS * (m * k * bf16_tile + k * n * weight_tile)
            return input_bytes, output_bytes, own_bytes + bias_bytes + partials_bytes
        act_rows = block_tiles if act_block_h else m
        act_width = _tiles(kw * in_width)
        if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            # The activation block as gathered from the halo and again tilized. The block spans one kernel
            # row; every kernel row's weights stay resident when a core computes more than one block.
            act_tile_bytes = bf16_tile + out_tile
            weight_rows = kh if m > act_rows else 1
        else:
            # The block as multicast and tilized, both in the output format, and gathered row-major when
            # that format is not the halo's bfloat16.
            act_tile_bytes = 2 * out_tile + (bf16_tile if self.activation_dtype != ttnn.bfloat16 else 0)
            weight_rows = 1
        own_bytes = act_rows * act_width * act_tile_bytes + act_width * n * weight_tile * weight_rows + bias_bytes
        return input_bytes, output_bytes, own_bytes

    def _fits_core(self, layout, shape, act_block_h):
        needed = self._core_l1_bytes(layout, shape, act_block_h)
        return needed is None or sum(needed) <= self.l1_bank_bytes * CONV_CORE_L1_SHARE

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
