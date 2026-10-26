# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Bottleneck ResNet backbone for BEVFormer: ResNet101 with a modulated deformable conv2 in
layer3 and layer4, or ResNet50 without one, as ``model_config`` selects.
Each BatchNorm is already folded into the conv before it.
"""

import math

import ttnn
from models.experimental.bevformer.tt.tt_common import TTConv2D, is_pointwise, signpost
from models.experimental.bevformer.tt.tt_modulated_deform_conv import TTModulatedDeformConv2dDevice


class TTModulatedDeformConv2dPack:
    """DCNv2 conv together with the conv that predicts its offsets and mask.

    The geometry (stride, padding, dilation, input shape) comes from ``conv_args.conv_offset``,
    which shares it with the deformable conv.
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        input_dtype=ttnn.bfloat16,
        relu=False,
        weights_dtype=ttnn.bfloat8_b,
        sharded_input=False,
    ):
        """``weights_dtype`` is the offset conv's; ``sharded_input`` gives the DCN and the offset
        conv one height-sharded L1 input."""
        offset_args = conv_args.conv_offset
        # __call__ reshapes the input to the output's (B, H, W), which holds only at stride 1.
        # The DCN variants' caffe-style ResNet puts every stride on conv1 or the downsample shortcut.
        assert tuple(offset_args.stride) == (1, 1), f"DCN supports stride 1 only, got {offset_args.stride}"
        self.batch_size = offset_args.batch_size

        kernel_size = offset_args.kernel_size[0]
        self.device_dcn = TTModulatedDeformConv2dDevice(
            weight=conv_pth.weight,
            bias=conv_pth.bias,
            kernel_size=kernel_size,
            stride=tuple(offset_args.stride),
            padding=tuple(offset_args.padding),
            dilation=tuple(offset_args.dilation),
            device=device,
            input_shape=(offset_args.batch_size, offset_args.input_height, offset_args.input_width),
            relu=relu,
        )

        # The DCN's input, height-sharded over the grid in whole 32-row blocks where that pays
        # (``sharded_input``): the op gives each core the output rows of its own shard, so
        # most bilinear corners are in its own L1 and the rest in a neighbour's, instead of spread
        # over every core's. conv_offset then reads that same shard, so the block input reaches L1
        # once, as TILE, and is untilized there. Otherwise the DCN reads an interleaved L1 copy and
        # conv_offset a ROW_MAJOR one in DRAM.
        self.input_memory_config = None
        conv_offset_memory_config = ttnn.DRAM_MEMORY_CONFIG
        if sharded_input:
            grid = device.compute_with_storage_grid_size()
            rows = offset_args.batch_size * offset_args.input_height * offset_args.input_width
            shard_rows = math.ceil(rows / (grid.x * grid.y) / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            self.input_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
                    [shard_rows, offset_args.in_channels],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            conv_offset_memory_config = self.input_memory_config

        # conv_offset emits DCNv2's offsets, in pixels and (y, x) order, then the mask logits, the
        # order the DCN op takes, and in ROW_MAJOR, the layout it reads. Its output is bfloat16
        # whatever the input dtype, since the offsets set the sampling positions. Offsets can
        # reach tens of pixels, and accumulating in a bfloat16 destination register errs by up to
        # a few of them, so the register is fp32. The partial sums conv2d writes back between
        # reduction blocks stay bfloat16 (packer_l1_acc is off): fp32 partial sums gain no
        # accuracy here. On the shared L1 input conv2d sizes its activation blocks to the whole
        # shard, and their buffers then overrun the L1 that input holds; one tile row at a time fits.
        self.conv_offset = TTConv2D(
            offset_args,
            conv_pth.conv_offset,
            device=device,
            weights_dtype=weights_dtype,
            fp32_dest_acc_en=True,
            input_dtype=input_dtype,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            input_memory_config=conv_offset_memory_config,
            act_block_h=None if self.input_memory_config is None else ttnn.TILE_SIZE,
            output_dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def __call__(self, x):
        """``x`` is a (1, 1, B*H*W, C_in) tensor in conv2d's layout. Returns the
        (1, 1, B*H_out*W_out, C_out) output, also in conv2d's layout, and its height and width."""
        if self.input_memory_config is not None:
            x = ttnn.to_memory_config(x, self.input_memory_config)
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x_dcn = x
        else:
            # One untilize feeds both conv_offset and the DCN: conv2d reshapes its input
            # between flat and NHWC, which in TILE copies whenever the width is not a tile multiple.
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            # The DCN's reads are many small scattered sticks, which L1 serves far faster than
            # DRAM; conv_offset's weights are prepared for a DRAM input, so only the DCN gets the
            # L1 copy.
            x_dcn = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        offset_mask, out_h, out_w = self.conv_offset(x)
        # conv2d's L1 path returns its output sharded, and the DCN op takes offset_mask only
        # interleaved; in L1, which serves the op's small per-row reads faster than DRAM.
        offset_mask = ttnn.sharded_to_interleaved(offset_mask, ttnn.L1_MEMORY_CONFIG)
        offset_mask = ttnn.reshape(offset_mask, (self.batch_size, out_h, out_w, offset_mask.shape[-1]))

        # At stride 1 the input has the output's height and width.
        x_nhwc = ttnn.reshape(x_dcn, (self.batch_size, out_h, out_w, x.shape[-1]))
        out = self.device_dcn(x_nhwc, offset_mask)
        ttnn.deallocate(x_nhwc)
        ttnn.deallocate(offset_mask)
        return out, out_h, out_w


class TTResLayer:
    """One ResNet layer (layer1 .. layer4), one bottleneck per block in ``conv_pth``.

    ``input_dtype``, ``input_layout`` and ``dram_input`` describe the layer input, which
    only the first block reads. The other arguments go to every block.

    With ``cameras``, the layer runs its blocks on that many cameras at a time with every activation
    in L1 (TTCameraBottleneck): each group's rows of the DRAM input go into L1, pass through every
    block there, and only the layer's output returns to DRAM. A strided first block runs on the whole
    batch first (``whole_first_block``) when its group input would take more than
    CAMERA_INPUT_MAX_SHARD_BYTES per core, as the subsample holds a second copy of that input, or
    when its stride is on the 3x3 conv2 (pytorch style), which TTCameraBottleneck does not take.
    Such a layer takes the weights of a DRAM layer.
    """

    CAMERA_INPUT_MAX_SHARD_BYTES = 512 * 1024

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        dram_activation=False,
        dram_input=False,
        dram_conv_slices=None,
        conv2_dram_conv_slices=None,
        block_sharded=False,
        fp32_acc=False,
        dcn_sharded_input=False,
        conv2_act_block_h=None,
        weights_dtype=ttnn.bfloat8_b,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        cameras=None,
    ):
        num_blocks = len(conv_pth)
        self.layer = []
        self.cameras = cameras
        if cameras is not None:
            assert input_layout == ttnn.TILE_LAYOUT, "a per-camera layer slices its cameras out of a TILE input"
            self.whole_first_block = None
            if self._first_block_too_wide(conv_args[0].conv1, cameras, device) or self._strided_conv2(conv_args[0]):
                self.whole_first_block = TTBottleneck(
                    conv_args[0],
                    conv_pth[0],
                    device,
                    dram_activation=dram_activation,
                    dram_input=dram_input,
                    dram_conv_slices=dram_conv_slices,
                    block_sharded=block_sharded,
                    fp32_acc=fp32_acc,
                    dcn_sharded_input=dcn_sharded_input,
                    conv2_act_block_h=conv2_act_block_h,
                    weights_dtype=weights_dtype,
                    input_dtype=input_dtype,
                    input_layout=input_layout,
                )
            start = 0 if self.whole_first_block is None else 1
            first = conv_args[start].conv1
            assert first.batch_size % cameras == 0, f"{first.batch_size} cameras do not split into groups of {cameras}"
            self.batch_size = first.batch_size
            self.camera_rows = first.input_height * first.input_width
            self.in_channels = first.in_channels
            for j in range(start, num_blocks):
                self.layer.append(
                    TTCameraBottleneck(
                        conv_args[j],
                        conv_pth[j],
                        device,
                        cameras=cameras,
                        fp32_acc=fp32_acc,
                        dcn_sharded_input=dcn_sharded_input,
                        conv2_act_block_h=conv2_act_block_h,
                        weights_dtype=weights_dtype,
                    )
                )
            self.output_dtype = ttnn.bfloat16
            return
        for j in range(num_blocks):
            first = j == 0
            self.layer.append(
                TTBottleneck(
                    conv_args[j],
                    conv_pth[j],
                    device,
                    dram_activation=dram_activation,
                    dram_input=dram_input and first,
                    dram_conv_slices=dram_conv_slices,
                    conv2_dram_conv_slices=conv2_dram_conv_slices,
                    block_sharded=block_sharded,
                    fp32_acc=fp32_acc,
                    dcn_sharded_input=dcn_sharded_input,
                    conv2_act_block_h=conv2_act_block_h,
                    weights_dtype=weights_dtype,
                    input_dtype=input_dtype if first else self.layer[-1].output_dtype,
                    input_layout=input_layout if first else ttnn.TILE_LAYOUT,
                )
            )
        self.output_dtype = self.layer[-1].output_dtype

    @classmethod
    def _first_block_too_wide(cls, conv1, cameras, device):
        """Whether a group's input to a strided first block with ``conv1`` exceeds
        CAMERA_INPUT_MAX_SHARD_BYTES per core, in bfloat16."""
        if tuple(conv1.stride) == (1, 1):
            return False
        grid = device.compute_with_storage_grid_size()
        rows = cameras * conv1.input_height * conv1.input_width
        shard_rows = math.ceil(math.ceil(rows / ttnn.TILE_SIZE) / (grid.x * grid.y)) * ttnn.TILE_SIZE
        return shard_rows * conv1.in_channels * 2 > cls.CAMERA_INPUT_MAX_SHARD_BYTES

    @staticmethod
    def _strided_conv2(block_args):
        """Whether a block's 3x3 conv2 carries its stride."""
        conv2 = block_args.conv2.conv_offset if "conv_offset" in block_args.conv2 else block_args.conv2
        return tuple(conv2.stride) != (1, 1)

    def __call__(self, x, subsample=None):
        """``subsample``: see TTBottleneck; it goes to the first block."""
        if self.cameras is not None:
            return self.run_per_camera([self], x, first_subsample=subsample)
        x = self.layer[0](x, subsample=subsample)
        for block in self.layer[1:]:
            x = block(x)
        return x

    @staticmethod
    def run_per_camera(layers, x, group_input=None, first_subsample=None, next_conv1=None):
        """Consecutive per-camera ``layers`` with one group size on the DRAM batch ``x``, a group of
        cameras at a time: its rows go into L1, pass through every block of every layer there, and
        the last layer's output is written back into one DRAM batch. Only that output crosses DRAM.
        ``group_input``, given a group's index, returns its input already in L1 instead.

        Each group writes its rows straight into the batch (``slice_write``) rather than the groups
        being concatenated afterwards, which would copy the whole batch once more. A group whose rows
        are whole tiles writes its TILE shards as they are. A tiled write must start on a tile row,
        so groups that do not write ROW_MAJOR instead, untilized in L1; that batch suits its
        consumers too, as a strided first block subsamples ROW_MAJOR and an FPN lateral tilizes its
        DRAM input itself.

        ``first_subsample`` goes to a whole first block (TTBottleneck). With ``next_conv1``, the
        strided 1x1 conv1 of the next layer's first block, each group also takes that block's
        subsample of its output while it is in L1, and the run returns ``(output, subsample)``, the
        subsample ROW_MAJOR in DRAM; taken from the output in DRAM, it would untilize and read the
        whole batch again."""
        first = layers[0]
        assert all(layer.cameras == first.cameras for layer in layers), "chained layers share a group size"
        assert all(layer.whole_first_block is None for layer in layers[1:]), "only a run's first layer reads DRAM"
        if first.whole_first_block is not None:
            x = first.whole_first_block(x, subsample=first_subsample)
        rows = first.cameras * first.camera_rows
        # A group that starts inside a tile row makes the slice untilize, slice and tilize again, with
        # temporaries the size of the group; those go through DRAM, and one copy brings the group
        # into L1.
        aligned = rows % ttnn.TILE_SIZE == 0

        def slice_group(group):
            y = ttnn.slice(
                x,
                [0, 0, group * rows, 0],
                [1, 1, (group + 1) * rows, first.in_channels],
                memory_config=first.layer[0].input_memory_config if aligned else ttnn.DRAM_MEMORY_CONFIG,
            )
            if aligned:
                return y
            staged = ttnn.to_memory_config(y, first.layer[0].input_memory_config)
            ttnn.deallocate(y)
            return staged

        num_groups = first.batch_size // first.cameras
        out = subsample = None
        for group in range(num_groups):
            y = (group_input or slice_group)(group)
            for layer in layers:
                for block in layer.layer:
                    y = block(y)
            group_rows, channels = y.shape[2], y.shape[3]
            tiled = group_rows % ttnn.TILE_SIZE == 0
            if not tiled:
                # Untilized from L1 interleaved, which handles rows that end inside a tile row.
                staged = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(y)
                y = ttnn.to_layout(staged, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(staged)
            if next_conv1 is not None:
                # Whole tiles untilize in their shards; the ROW_MAJOR rows are already in L1 otherwise.
                rows_major = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT) if tiled else y
                group_sub = TTResLayer._subsample_group(rows_major, next_conv1, first.cameras)
                if tiled:
                    ttnn.deallocate(rows_major)
                sub_rows = group_sub.shape[2]
                if subsample is None:
                    subsample = ttnn.empty(
                        (1, 1, num_groups * sub_rows, channels),
                        dtype=group_sub.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=group_sub.device(),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                ttnn.experimental.slice_write(
                    group_sub,
                    subsample,
                    [0, 0, group * sub_rows, 0],
                    [1, 1, (group + 1) * sub_rows, channels],
                    [1, 1, 1, 1],
                )
                ttnn.deallocate(group_sub)
            if out is None:
                out = ttnn.empty(
                    (1, 1, num_groups * group_rows, channels),
                    dtype=y.dtype,
                    layout=ttnn.TILE_LAYOUT if tiled else ttnn.ROW_MAJOR_LAYOUT,
                    device=y.device(),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            ttnn.experimental.slice_write(
                y, out, [0, 0, group * group_rows, 0], [1, 1, (group + 1) * group_rows, channels], [1, 1, 1, 1]
            )
            ttnn.deallocate(y)
        return out if next_conv1 is None else (out, subsample)

    @staticmethod
    def _subsample_group(rows_major, conv1, cameras):
        """``cameras`` cameras of ``conv1``'s input, ``(1, 1, cameras*H*W, C)`` ROW_MAJOR in L1, at
        its stride, in L1: ROW_MAJOR, where the strided slice copies whole channel rows."""
        nhwc = (cameras, conv1.input_height, conv1.input_width, rows_major.shape[-1])
        sub = ttnn.slice(
            ttnn.reshape(rows_major, nhwc),
            [0, 0, 0, 0],
            list(nhwc),
            [1, *conv1.stride, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.reshape(sub, (1, 1, cameras * conv1.output_height * conv1.output_width, nhwc[-1]))


def _for_cameras(args, cameras):
    """A copy of the conv args ``args`` with its batch set to ``cameras``."""
    group = type(args)(args)
    group["batch_size"] = cameras
    return group


class TTCameraBottleneck:
    """A bottleneck on ``cameras`` cameras at a time, every tensor in L1.

    The whole-batch block (TTBottleneck) moves each of its tensors through DRAM, several times over in
    its width slices and layout changes, which bounds it by DRAM bandwidth. A few cameras'
    activations fit in L1, so here they stay there: every tensor is height-sharded over the grid
    with the same rows per core, the 1x1 convs run as linears on those shards, conv2 as an L1
    conv2d or the DCN (TTModulatedDeformConv2dPack, built for the group), and the residual add
    writes into conv3's output. A strided block (a layer's first, whose 1x1 conv1 and downsample
    share the stride) takes that subsample of its input in L1 first. ``conv_pth`` holds a DRAM
    layer's weights (``create_resnet_parameters``): Linear weights for the 1x1 convs.
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        cameras=1,
        fp32_acc=False,
        dcn_sharded_input=False,
        conv2_act_block_h=None,
        weights_dtype=ttnn.bfloat8_b,
    ):
        conv1_args = conv_args.conv1
        self.with_dcn = "conv_offset" in conv_pth.conv2
        conv2_geometry = conv_args.conv2.conv_offset if self.with_dcn else conv_args.conv2
        self.stride = tuple(conv1_args.stride)
        assert tuple(conv2_geometry.stride) == (1, 1), f"conv2 runs at stride 1, got {conv2_geometry.stride}"
        if "downsample" in conv_pth:
            assert tuple(conv_args.downsample[0].stride) == self.stride, "conv1 and the downsample share a stride"
        else:
            assert self.stride == (1, 1), f"a block without a downsample runs at stride 1, got {self.stride}"
        self.device = device
        self.cameras = cameras
        self.in_channels = conv1_args.in_channels
        self.input_height, self.input_width = conv1_args.input_height, conv1_args.input_width
        self.height, self.width = conv2_geometry.input_height, conv2_geometry.input_width
        grid = device.compute_with_storage_grid_size()
        self.grid = grid
        self.input_shard_tiles = self._shard_tiles(cameras * self.input_height * self.input_width)
        self.shard_tiles = self._shard_tiles(cameras * self.height * self.width)
        self.fp32_acc = fp32_acc
        math_fidelity = ttnn.MathFidelity.HiFi2 if ttnn.get_arch_name() == "blackhole" else ttnn.MathFidelity.LoFi
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        relu = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        self.input_memory_config = self._sharded(self.in_channels, self.input_shard_tiles)
        self.conv1 = self._linear(conv_args.conv1, conv_pth.conv1, relu)
        self.conv3 = self._linear(conv_args.conv3, conv_pth.conv3, None)
        self.downsample = (
            self._linear(conv_args.downsample[0], conv_pth.downsample, None) if "downsample" in conv_pth else None
        )

        if self.with_dcn:
            dcn_args = type(conv_args.conv2)(conv_args.conv2)
            dcn_args["conv_offset"] = _for_cameras(conv2_geometry, cameras)
            # The BatchNorm is folded into the DCN weights; the ReLU runs inside the DCN op.
            self.dcn = TTModulatedDeformConv2dPack(
                dcn_args,
                conv_pth.conv2,
                device,
                input_dtype=ttnn.bfloat16,
                relu=True,
                weights_dtype=weights_dtype,
                sharded_input=dcn_sharded_input,
            )
            return

        self.conv2_channels = (conv2_geometry.in_channels, conv2_geometry.out_channels)
        self.conv2_weight, self.conv2_bias = conv_pth.conv2.weight, conv_pth.conv2.bias
        self.conv2_kernel, self.conv2_padding = tuple(conv2_geometry.kernel_size), tuple(conv2_geometry.padding)
        # The input shard is conv2d's to use as it is, and freed once the halo has read it.
        self.conv2_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            activation=relu,
        )
        if conv2_act_block_h is not None:
            self.conv2_config.act_block_h_override = conv2_act_block_h
        # conv2d prepares its weights on the host on its first call, for the input it sees; making
        # that call here keeps the forward free of host work.
        warmup = ttnn.to_memory_config(
            ttnn.zeros(
                (1, 1, cameras * self.height * self.width, self.conv2_channels[0]),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            self._sharded(self.conv2_channels[0]),
        )
        ttnn.deallocate(self._conv2(warmup))

    def _shard_tiles(self, rows):
        """Tile rows per core for ``rows`` rows height-sharded over the grid."""
        return math.ceil(math.ceil(rows / ttnn.TILE_SIZE) / (self.grid.x * self.grid.y))

    def _sharded(self, channels, shard_tiles=None):
        """The block's layout: the group's rows height-sharded over the grid, ``shard_tiles`` tile
        rows per core, the block's own by default."""
        shard_tiles = self.shard_tiles if shard_tiles is None else shard_tiles
        return ttnn.create_sharded_memory_config(
            shape=(shard_tiles * ttnn.TILE_SIZE, channels),
            core_grid=ttnn.CoreGrid(y=self.grid.y, x=self.grid.x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # Weight tiles a linear multicasts per inner block: every core holds that block, in0_block_w by
    # the whole output width, which for layer4's 2048-channel outputs would outgrow L1.
    LINEAR_WEIGHT_BLOCK_TILES = 128

    def _linear(self, conv, params, activation):
        """A 1x1 conv as a linear on the block's shards: each core multiplies its own rows, its inner
        dimension in the widest blocks whose weights fit LINEAR_WEIGHT_BLOCK_TILES, with the weights
        multicast."""
        in_t, out_t = conv.in_channels // ttnn.TILE_SIZE, conv.out_channels // ttnn.TILE_SIZE
        max_subblock = 4 if self.fp32_acc else 8
        in0_block_w = max(
            w for w in range(1, in_t + 1) if in_t % w == 0 and w * out_t <= self.LINEAR_WEIGHT_BLOCK_TILES
        )
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(self.grid.x, self.grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=max(w for w in range(1, max_subblock + 1) if out_t % w == 0),
            out_block_h=1,
            out_block_w=out_t,
            per_core_M=self.shard_tiles,
            per_core_N=out_t,
            fuse_batch=True,
            fused_activation=activation,
            mcast_in0=False,
        )
        weight = ttnn.to_device(params.weight, self.device)
        bias = None if params.bias is None else ttnn.to_device(params.bias, self.device)
        memory_config = self._sharded(conv.out_channels)
        return lambda x: ttnn.linear(
            x,
            weight,
            bias=bias,
            program_config=program_config,
            compute_kernel_config=self.compute_config,
            memory_config=memory_config,
        )

    def _conv2(self, x):
        if self.with_dcn:
            if self.dcn.input_memory_config is None:
                # Without a sharded input the DCN's offset conv reads conv1's output from DRAM.
                staged = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            else:
                # The DCN's circular buffers fill L1 from the bottom up to the lowest tensor in it,
                # so its ROW_MAJOR input moves to the top of L1 once the TILE copies are freed.
                tiled = ttnn.to_memory_config(x, self.dcn.input_memory_config)
                staged = ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(tiled)
                ttnn.deallocate(x)
                staged = ttnn.reallocate(staged)
            ttnn.deallocate(x)
            y, _, _ = self.dcn(staged)
            ttnn.deallocate(staged)
            return y
        y, [self.conv2_weight, self.conv2_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv2_weight,
            bias_tensor=self.conv2_bias,
            device=self.device,
            in_channels=self.conv2_channels[0],
            out_channels=self.conv2_channels[1],
            input_height=self.height,
            input_width=self.width,
            batch_size=self.cameras,
            kernel_size=self.conv2_kernel,
            stride=(1, 1),
            padding=self.conv2_padding,
            conv_config=self.conv2_config,
            compute_config=self.compute_config,
            memory_config=self._sharded(self.conv2_channels[1]),
            return_weights_and_bias=True,
        )
        return y

    def _subsample(self, x):
        """The block input at the block's stride, in the block's layout. ROW_MAJOR, where the
        strided slice copies whole channel rows."""
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x)
        flat = ttnn.sharded_to_interleaved(rm, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(rm)
        nhwc = [self.cameras, self.input_height, self.input_width, self.in_channels]
        sub = ttnn.slice(
            ttnn.reshape(flat, nhwc), [0, 0, 0, 0], nhwc, [1, *self.stride, 1], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(flat)
        rows = self.cameras * self.height * self.width
        tiled = ttnn.to_layout(ttnn.reshape(sub, (1, 1, rows, self.in_channels)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(sub)
        out = ttnn.to_memory_config(tiled, self._sharded(self.in_channels))
        ttnn.deallocate(tiled)
        return out

    def __call__(self, x):
        if self.stride != (1, 1):
            x = self._subsample(x)
        # The downsample first, while L1 holds only the block input: its output and conv3's are the
        # block's two wide tensors, and freeing a narrow one between them would split the room
        # the second needs.
        identity = x if self.downsample is None else self.downsample(x)
        y = self.conv1(x)
        if self.downsample is not None:
            ttnn.deallocate(x)
        y = self._conv2(y)
        out = self.conv3(y)
        ttnn.deallocate(y)
        ttnn.add(out, identity, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)], output_tensor=out)
        ttnn.deallocate(identity)
        # The output, the next block's input, sits wherever L1 had room among this block's
        # tensors; moved to the top of the now free L1, it leaves the next block's circular
        # buffers, which fill L1 from the bottom, room below it.
        return ttnn.reallocate(out)


class TTBottleneck:
    """One bottleneck block.

    Every conv's shape and stride come from ``conv_args``, recorded from one forward of
    the reference model. The block's conv2 is DCNv2 when ``conv_pth.conv2`` has an offset
    conv, and the block has a downsample shortcut when ``conv_pth`` has one.

    ``input_dtype`` and ``input_layout`` describe the block input; the dtype and layout
    each conv sees follow from them. ``dram_activation`` keeps the activations of conv1,
    a non-DCN conv2, conv3 and the downsample in DRAM, slicing the spatial convs into
    ``dram_conv_slices`` width slices, a non-DCN conv2 in ``conv2_dram_conv_slices`` when given;
    a DCN conv2 is unaffected. ``dram_input`` only
    covers the convs that read the block input, for a block fed by a DRAM layer whose own
    activations fit in L1. ``block_sharded`` block-shards the downsample and a non-DCN conv2.
    ``fp32_acc`` turns on fp32 destination accumulation (``fp32_dest_acc_en``) for conv1, a
    non-DCN conv2, conv3 and the downsample; a DCN conv2 always has it
    (TTModulatedDeformConv2dPack). ``conv2_act_block_h`` is a non-DCN conv2's act_block_h
    override and ``weights_dtype`` every conv's weight dtype but the DCN's own.
    """

    @staticmethod
    def dram_convs(dram_activation=False, dram_input=False):
        """The convs whose activations are in DRAM, so that a 1x1 one among them runs as a matmul
        and takes ``model_preprocessing``'s Linear weight. A DCN conv2 ignores it."""
        convs = {"conv1", "conv2", "conv3", "downsample"} if dram_activation else set()
        return convs | {"conv1", "downsample"} if dram_input else convs

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        dram_activation=False,
        dram_input=False,
        dram_conv_slices=None,
        conv2_dram_conv_slices=None,
        block_sharded=False,
        fp32_acc=False,
        dcn_sharded_input=False,
        conv2_act_block_h=None,
        weights_dtype=ttnn.bfloat8_b,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
    ):
        self.with_dcn = "conv_offset" in conv_pth.conv2
        dram_convs = self.dram_convs(dram_activation, dram_input)
        self.is_downsample = "downsample" in conv_pth
        # A layer's first block opens with a strided 1x1 conv1 and downsample on the block input.
        # In DRAM both run as linears over the input subsampled at their shared stride, so the
        # block takes that subsample once, and tilizes it once, for the two.
        self.shared_subsample = (
            self.is_downsample
            and {"conv1", "downsample"} <= dram_convs
            and is_pointwise(conv_args.conv1)
            and is_pointwise(conv_args.downsample[0])
            and tuple(conv_args.conv1.stride) == tuple(conv_args.downsample[0].stride) != (1, 1)
        )

        # A sliced DRAM conv2 reads and writes ROW_MAJOR, where conv2d's reshapes to NHWC and back
        # are free; in TILE both copy the tensor, as the widths are not tile multiples. conv3, a
        # 1x1 linear, tilizes its input itself.
        self.row_major_conv2 = not self.with_dcn and "conv2" in dram_convs

        # conv2d and ttnn.linear keep their input's dtype, and the DCN branch emits bfloat16.
        conv2_dtype = input_dtype
        conv3_dtype = ttnn.bfloat16 if self.with_dcn else input_dtype
        self.output_dtype = conv3_dtype

        self.conv1 = TTConv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            weights_dtype=weights_dtype,
            fp32_dest_acc_en=fp32_acc,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            dram_activation="conv1" in dram_convs,
            dram_conv_slices=dram_conv_slices,
            input_dtype=input_dtype,
            input_layout=input_layout,
            output_layout=ttnn.ROW_MAJOR_LAYOUT if self.row_major_conv2 else ttnn.TILE_LAYOUT,
        )

        if not self.with_dcn:
            self.conv2 = TTConv2D(
                conv_args.conv2,
                conv_pth.conv2,
                device=device,
                weights_dtype=weights_dtype,
                fp32_dest_acc_en=fp32_acc,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                is_blk=block_sharded,
                act_block_h=conv2_act_block_h,
                dealloc_act=True,
                dram_activation="conv2" in dram_convs,
                dram_conv_slices=conv2_dram_conv_slices or dram_conv_slices,
                input_dtype=conv2_dtype,
                input_layout=ttnn.ROW_MAJOR_LAYOUT if self.row_major_conv2 else ttnn.TILE_LAYOUT,
                output_layout=ttnn.ROW_MAJOR_LAYOUT if self.row_major_conv2 else ttnn.TILE_LAYOUT,
            )
        else:
            # The BatchNorm is folded into the DCN weights (create_resnet_parameters); the ReLU
            # runs inside the DCN's last matmul or bias add.
            self.conv2 = TTModulatedDeformConv2dPack(
                conv_args.conv2,
                conv_pth.conv2,
                device,
                input_dtype=conv2_dtype,
                relu=True,
                weights_dtype=weights_dtype,
                sharded_input=dcn_sharded_input,
            )

        self.conv3 = TTConv2D(
            conv_args.conv3,
            conv_pth.conv3,
            device=device,
            weights_dtype=weights_dtype,
            fp32_dest_acc_en=fp32_acc,
            activation=None,
            dealloc_act=True,
            dram_activation="conv3" in dram_convs,
            dram_conv_slices=dram_conv_slices,
            input_dtype=conv3_dtype,
            l1_output=True,
        )

        if self.is_downsample:
            self.downsample = TTConv2D(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                weights_dtype=weights_dtype,
                fp32_dest_acc_en=fp32_acc,
                activation=None,
                is_blk=block_sharded,
                dram_activation="downsample" in dram_convs,
                dram_conv_slices=dram_conv_slices,
                input_dtype=input_dtype,
                input_layout=input_layout,
            )

    def __call__(self, x_identity, subsample=None):
        """``subsample``, for a block with ``shared_subsample``: the block input already subsampled at
        conv1's stride, ROW_MAJOR in DRAM, which spares taking it from the whole input here."""
        if self.shared_subsample:
            if subsample is None:
                subsample = self.conv1.subsample(x_identity)
            x_identity = ttnn.to_layout(subsample, ttnn.TILE_LAYOUT)
            ttnn.deallocate(subsample)
        x, _, _ = self.conv1(x_identity, subsampled=self.shared_subsample)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if self.row_major_conv2 and x.layout != ttnn.ROW_MAJOR_LAYOUT:
            # A 1x1 conv1 runs as ttnn.linear, which emits TILE.
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x, _, _ = self.conv2(x)
        # Before conv3, whose output may stay in L1 for the add and would crowd the downsample's.
        if self.is_downsample:
            x_identity, _, _ = self.downsample(x_identity, subsampled=self.shared_subsample)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)
        x, _, _ = self.conv3(x)
        if not self.conv3.output_in_l1:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(
            x,
            x_identity,
            dtype=self.output_dtype,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        ttnn.deallocate(x_identity)
        return out


class TTResNet:
    """Bottleneck ResNet built from ``create_resnet_parameters``' ``params``.

    ``params.conv_args`` holds what one forward of the reference model records: the conv shapes and
    strides, the max pool, each layer's block count and which blocks are DCNv2. ``params.tt_kwargs``
    holds the memory and precision arguments below, the ones the weights were prepared for.

    ``out_indices`` and every ``*_stages`` argument index the four ResNet layers (0 is layer1).
    ``dram_activation_stages`` lists the layers
    whose activations are kept in DRAM, for layers whose convs do not fit in L1: their
    spatial convs run in ``dram_conv_slices`` width slices, their non-DCN conv2s in
    ``conv2_dram_conv_slices[i]`` where given, and their 1x1 convs as a DRAM matmul. A layer that follows one of them and is not listed itself reads its input from
    DRAM the same way. ``block_sharded_stages`` lists the layers whose downsample and non-DCN
    conv2s are block sharded, ``fp32_acc_stages`` the layers whose convs accumulate in an
    fp32 destination register (see TTBottleneck), and ``dcn_sharded_input_stages`` the layers
    whose DCNs take a height-sharded L1 input (TTModulatedDeformConv2dPack). The stem conv and the
    max pool read their DRAM inputs in ``stem_conv_height_slices`` and ``stem_maxpool_height_slices``
    height slices. ``stem_act_block_h`` and ``conv2_act_block_h`` are the stem conv's and the
    non-DCN conv2s' act_block_h overrides, None for conv2d's own, and
    ``weights_dtype`` the convs' weight dtype. ``model_config.tt_resnet_kwargs`` gives the
    configured variant's values, and ``model_config.RESNET_KWARGS["out_indices"]`` its
    ``out_indices``.
    """

    num_layers = 4

    def __init__(self, params, device, out_indices=(0, 1, 2, 3)):
        conv_args, conv_pth = params.conv_args, params["res_model"]
        stage_config = dict(params.tt_kwargs)
        stem_act_block_h = stage_config.pop("stem_act_block_h")
        stem_conv_height_slices = stage_config.pop("stem_conv_height_slices")
        stem_maxpool_height_slices = stage_config.pop("stem_maxpool_height_slices")
        weights_dtype = stage_config["weights_dtype"]
        self.out_indices = out_indices
        self.maxpool_args = conv_args.maxpool

        self.conv1 = TTConv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            weights_dtype=weights_dtype,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h=stem_act_block_h,
            dealloc_act=True,
            dram_activation=True,
            dram_conv_slices=stem_conv_height_slices,
            dram_height_slices=True,
            input_dtype=ttnn.bfloat16,
            input_layout=ttnn.TILE_LAYOUT,
        )
        self.maxpool_slice_config = ttnn.Op2DSliceConfig(
            slice_type=ttnn.Op2DDRAMSliceHeight, num_slices=stem_maxpool_height_slices
        )

        # The max pool and every layer emit bfloat16 TILE: layer1's first block feeds its input to
        # two linears, conv1 and the downsample, which would each tilize a ROW_MAJOR one.
        # Activations stay bfloat16: outlier channels lose their neighbours to bfloat8_b's shared
        # block exponent.
        layer_input_dtype = ttnn.bfloat16
        layer_input_layout = ttnn.TILE_LAYOUT
        self.output_dtypes = []
        self.res_layers = []
        for i in range(self.num_layers):
            res_layer = TTResLayer(
                conv_args=conv_args[f"layer{i+1}"],
                conv_pth=conv_pth[f"layer{i+1}"],
                device=device,
                **self.layer_kwargs(i, **stage_config),
                input_dtype=layer_input_dtype,
                input_layout=layer_input_layout,
            )
            self.res_layers.append(res_layer)
            layer_input_dtype = res_layer.output_dtype
            layer_input_layout = ttnn.TILE_LAYOUT
            if i in out_indices:
                self.output_dtypes.append(layer_input_dtype)

    @staticmethod
    def layer_kwargs(
        i,
        dram_activation_stages=(),
        dram_conv_slices=None,
        conv2_dram_conv_slices=None,
        block_sharded_stages=(),
        fp32_acc_stages=(),
        dcn_sharded_input_stages=(),
        conv2_act_block_h=None,
        weights_dtype=ttnn.bfloat8_b,
        camera_groups=None,
    ):
        """The per-layer arguments of layer ``i`` (0 is layer1); a layer after a DRAM layer reads
        its input from DRAM."""
        camera_groups = camera_groups or {}
        assert set(camera_groups) <= set(dram_activation_stages), "a per-camera layer takes a DRAM layer's weights"
        return dict(
            cameras=camera_groups.get(i),
            dram_activation=i in dram_activation_stages,
            dram_input=i not in dram_activation_stages and i - 1 in dram_activation_stages,
            dram_conv_slices=dram_conv_slices,
            conv2_dram_conv_slices=(conv2_dram_conv_slices or {}).get(i),
            block_sharded=i in block_sharded_stages,
            fp32_acc=i in fp32_acc_stages,
            dcn_sharded_input=i in dcn_sharded_input_stages,
            conv2_act_block_h=conv2_act_block_h,
            weights_dtype=weights_dtype,
        )

    @staticmethod
    def block_dram_convs(i, j, dram_activation_stages):
        """``TTBottleneck.dram_convs`` of block ``j`` of layer ``i`` for TTResNet's
        ``dram_activation_stages``; only a layer's first block reads the layer input."""
        layer = TTResNet.layer_kwargs(i, dram_activation_stages)
        return TTBottleneck.dram_convs(layer["dram_activation"], layer["dram_input"] and j == 0)

    def __call__(self, x):
        """``x`` the images, ``(N, 3, H, W)`` bfloat16 ROW_MAJOR in DRAM, as torch lays them out."""
        signpost(header="TTNN ResNet Forward Start")
        signpost(header="ResNet Stem Start")
        # The stem conv reads its input through conv2d's DRAM slicing, which fetches a ROW_MAJOR
        # tensor one pixel at a time, millions of reads whatever the channel count; a TILE input is
        # read in tiles. Tilizing NCHW, whose rows are image rows, is cheap; NCHW goes to NHWC
        # already tiled, and its (1, 1, N*H*W, 3) view is free as W is a tile multiple. Two
        # transposes, NHCW then NHWC, run in under half the time of the one permute.
        n, c, h, w = x.shape
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        nhcw = ttnn.transpose(x, 1, 2)
        ttnn.deallocate(x)
        nhwc = ttnn.transpose(nhcw, 2, 3)
        ttnn.deallocate(nhcw)
        x = ttnn.reshape(nhwc, (1, 1, n * h * w, c))
        x, out_h, out_w = self.conv1(x)
        first = self.res_layers[0]
        if first.cameras == 1 and first.whole_first_block is None:
            # layer1 runs per camera: each camera's conv output is pooled in L1 and goes straight
            # into layer1's first block, so the pooled batch never reaches DRAM.
            x = ttnn.reshape(x, (1, 1, n * out_h * out_w, x.shape[-1]))
            signpost(header="ResNet Stem End")
            return self._run_layers(x, group_input=lambda camera: self._pool_camera(x, camera, out_h, out_w))
        pool = self.maxpool_args
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv1.batch_size,
            input_h=out_h,
            input_w=out_w,
            channels=x.shape[3],
            kernel_size=[pool.kernel_size, pool.kernel_size],
            stride=[pool.stride, pool.stride],
            padding=[pool.padding, pool.padding],
            dilation=[pool.dilation, pool.dilation],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
            dram_slice_config=self.maxpool_slice_config,
            output_layout=ttnn.TILE_LAYOUT,
        )
        # The pool writes NHWC tiles with its 400-wide rows padded; flattening them copies, so it
        # happens once here rather than in both of layer1's first linears.
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]))
        signpost(header="ResNet Stem End")
        return self._run_layers(x)

    def _pool_camera(self, x, camera, height, width):
        """Camera ``camera`` of the stem conv's ``height x width`` DRAM output ``x``, max pooled in L1
        into layer1's first block's layout."""
        pool = self.maxpool_args
        channels = x.shape[-1]
        rows = height * width
        grid = self.conv1.device.compute_with_storage_grid_size()
        shard_rows = math.ceil(math.ceil(rows / ttnn.TILE_SIZE) / (grid.x * grid.y)) * ttnn.TILE_SIZE
        sharded = ttnn.create_sharded_memory_config(
            shape=(shard_rows, channels),
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        image = ttnn.slice(x, [0, 0, camera * rows, 0], [1, 1, (camera + 1) * rows, channels], memory_config=sharded)
        # The input is freed once the halo has read it, which leaves the pool room for its output.
        pooled = ttnn.max_pool2d(
            input_tensor=image,
            batch_size=1,
            input_h=height,
            input_w=width,
            channels=channels,
            kernel_size=[pool.kernel_size, pool.kernel_size],
            stride=[pool.stride, pool.stride],
            padding=[pool.padding, pool.padding],
            dilation=[pool.dilation, pool.dilation],
            ceil_mode=False,
            deallocate_input=True,
            output_layout=ttnn.TILE_LAYOUT,
        )
        layout = self.res_layers[0].layer[0].input_memory_config
        if pooled.memory_config() == layout:
            return pooled
        out = ttnn.to_memory_config(pooled, layout)
        ttnn.deallocate(pooled)
        return out

    def _shared_subsample_conv1(self, i):
        """The strided conv1 of layer ``i``'s first block when that block takes a shared subsample of
        its input (TTBottleneck.shared_subsample), else None."""
        if i >= self.num_layers:
            return None
        layer = self.res_layers[i]
        block = layer.whole_first_block if layer.cameras is not None else layer.layer[0]
        return block.conv1 if isinstance(block, TTBottleneck) and block.shared_subsample else None

    def _run_layers(self, x, group_input=None):
        """The layers on the stem's output ``x``; ``group_input`` feeds the first per-camera run
        instead (TTResLayer.run_per_camera)."""
        outs = []
        subsample = None
        i = 0
        while i < self.num_layers:
            # Consecutive per-camera layers with one group size run together, so the activations
            # between them stay in L1, up to the first whose output is a level.
            cameras = self.res_layers[i].cameras
            j = i
            while (
                cameras is not None
                and j not in self.out_indices
                and j + 1 < self.num_layers
                and self.res_layers[j + 1].cameras == cameras
                and self.res_layers[j + 1].whole_first_block is None
            ):
                j += 1
            # A run of chained layers is one region, named by its span.
            region = f"ResNet Layer{i + 1}" if i == j else f"ResNet Layer{i + 1}-{j + 1}"
            signpost(header=f"{region} Start")
            next_conv1 = self._shared_subsample_conv1(j + 1) if cameras is not None else None
            if cameras is not None:
                x = TTResLayer.run_per_camera(
                    self.res_layers[i : j + 1],
                    x,
                    group_input if i == 0 else None,
                    first_subsample=subsample,
                    next_conv1=next_conv1,
                )
            else:
                x = self.res_layers[i](x, subsample=subsample)
            subsample = None
            if next_conv1 is not None:
                x, subsample = x
            signpost(header=f"{region} End")
            if j in self.out_indices:
                outs.append(x)
            i = j + 1
        signpost(header="TTNN ResNet Forward End")
        return tuple(outs)
