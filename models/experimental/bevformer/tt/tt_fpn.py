# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""FPN neck for BEVFormer.

Lateral 1x1 convs, a top-down upsample-and-add, then the output convs. Extra
convs, when the reference FPN has them, run on the last (coarsest) output.
"""

import ttnn
from models.experimental.bevformer.tt.tt_common import TTConv2D, signpost


class TTConvModule:
    """One FPN convolution.

    Accumulates in an fp32 destination register. The call returns the output tensor only; the
    top-down pass reads a lateral conv's input height and width, which a 1x1 conv keeps.
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device=None,
        act_block_h=None,
        dealloc_act=True,
        dram_activation=False,
        dram_conv_slices=None,
        weights_dtype=ttnn.bfloat8_b,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        output_layout=ttnn.TILE_LAYOUT,
    ):
        self.conv = TTConv2D(
            conv_args.conv,
            conv_pth.conv,
            device=device,
            weights_dtype=weights_dtype,
            dealloc_act=dealloc_act,
            fp32_dest_acc_en=True,
            act_block_h=act_block_h,
            dram_activation=dram_activation,
            dram_conv_slices=dram_conv_slices,
            input_dtype=input_dtype,
            input_layout=input_layout,
            output_layout=output_layout,
        )

    def __call__(self, x):
        x = self.conv(x)
        return x[0]


class TTFPN:
    """Feature pyramid over the backbone levels, built from ``create_fpn_parameters``' ``params``,
    whose ``tt_kwargs`` holds the memory and precision arguments below, the ones the weights were
    prepared for.

    ``input_dtypes`` are the dtypes of the backbone levels the FPN reads, in order.
    ``dram_activation_levels`` lists the pyramid levels whose lateral and output convs
    keep their activations in DRAM, for levels too large for L1; their 3x3 convs run in
    ``dram_conv_slices`` width slices, or ``output_conv_dram_conv_slices[level]`` for a
    level's output conv, with act_block_h ``output_conv_act_block_h``. ``weights_dtype``
    is the convs' weight dtype.
    """

    def __init__(self, params, device, input_dtypes):
        conv_args, conv_pth = params.conv_args, params
        config = params.tt_kwargs
        dram_activation_levels = config["dram_activation_levels"]
        dram_conv_slices = config["dram_conv_slices"]
        output_conv_dram_conv_slices = config["output_conv_dram_conv_slices"]
        output_conv_act_block_h = config["output_conv_act_block_h"]
        weights_dtype = config["weights_dtype"]
        assert conv_args.add_extra_convs == "on_output", f"extra convs {conv_args.add_extra_convs!r} are not supported"
        self.lateral_convs = []
        self.fpn_convs = []
        self.relu_before_extra_convs = conv_args.relu_before_extra_convs
        num_levels = len(conv_args.lateral_convs)
        assert len(input_dtypes) == num_levels, f"{num_levels} levels, got {len(input_dtypes)} input dtypes"
        # Each level keeps its input's dtype through its lateral conv and the top-down add.
        # The top-down pass turns every level but the lowest to ROW_MAJOR for the upsample, and
        # the lowest goes to ROW_MAJOR for its output conv: the DRAM-sliced convs reshape their
        # input to NHWC and their output back, which in TILE copies a tensor whose width is not a
        # tile multiple. ROW_MAJOR also makes a level bfloat16, as bfloat8_b exists only in TILE.
        output_conv_layouts = [ttnn.ROW_MAJOR_LAYOUT] * num_levels
        output_conv_dtypes = [ttnn.bfloat16] * num_levels
        for i in range(num_levels):
            self.lateral_convs.append(
                TTConvModule(
                    conv_args.lateral_convs[i],
                    conv_pth.fpn.lateral_convs[str(i)],
                    device=device,
                    dram_activation=i in dram_activation_levels,
                    dram_conv_slices=dram_conv_slices,
                    weights_dtype=weights_dtype,
                    input_dtype=input_dtypes[i],
                )
            )
        for i in range(num_levels):
            in_dram = i in dram_activation_levels
            self.fpn_convs.append(
                TTConvModule(
                    conv_args.fpn_convs[i],
                    conv_pth.fpn.fpn_convs[str(i)],
                    device=device,
                    act_block_h=output_conv_act_block_h if in_dram else None,
                    dram_activation=in_dram,
                    dram_conv_slices=output_conv_dram_conv_slices.get(i, dram_conv_slices),
                    weights_dtype=weights_dtype,
                    input_dtype=output_conv_dtypes[i],
                    input_layout=output_conv_layouts[i],
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )
        for i in range(num_levels, len(conv_args.fpn_convs)):
            self.fpn_convs.append(
                TTConvModule(
                    conv_args.fpn_convs[i],
                    conv_pth.fpn.fpn_convs[str(i)],
                    device=device,
                    dealloc_act=False,
                    weights_dtype=weights_dtype,
                    input_dtype=output_conv_dtypes[-1],
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )

    def __call__(self, input_tensor):
        signpost(header="TTNN FPN Forward Start")
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            output = lateral_conv(input_tensor[i])
            ttnn.deallocate(input_tensor[i])
            laterals.append(output)

        outs = []
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i] = ttnn.to_layout(laterals[i], ttnn.ROW_MAJOR_LAYOUT)
            # A lateral conv is 1x1, so its output has its input level's size.
            level = self.lateral_convs[i].conv
            lower_level = self.lateral_convs[i - 1].conv
            laterals_reshaped = ttnn.reshape(
                laterals[i], (level.batch_size, level.input_height, level.input_width, laterals[i].shape[-1])
            )
            laterals_upsample = ttnn.upsample(laterals_reshaped, 2)
            # The reference interpolates to the lower level's exact size; a 2x upsample
            # overshoots it by one row or column wherever that level's size is odd.
            laterals_sliced = laterals_upsample[:, : lower_level.input_height, : lower_level.input_width, :]
            laterals_sliced = ttnn.reshape(
                laterals_sliced,
                [
                    1,
                    1,
                    laterals_sliced.shape[0] * laterals_sliced.shape[1] * laterals_sliced.shape[2],
                    laterals_sliced.shape[-1],
                ],
            )

            laterals_sliced = ttnn.to_memory_config(laterals_sliced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # Added in ROW_MAJOR, the layout the upsample emits and the output conv reads: the lower
            # lateral untilizes once, where a TILE add would tilize the upsample and untilize the sum.
            lower = ttnn.to_layout(laterals[i - 1], ttnn.ROW_MAJOR_LAYOUT)
            laterals[i - 1] = ttnn.add(lower, laterals_sliced)
        # Every level leaves L1 before the first output conv, whose circular buffers would clash
        # with a level still there.
        laterals = [ttnn.to_memory_config(lateral, memory_config=ttnn.DRAM_MEMORY_CONFIG) for lateral in laterals]
        for i in range(used_backbone_levels):
            lateral = ttnn.to_layout(laterals[i], ttnn.ROW_MAJOR_LAYOUT)
            # Every level comes out ROW_MAJOR: its consumer splits the cameras in ROW_MAJOR anyway
            # (TTPerceptionTransformer.camera_features), and the extra conv reshapes the last level
            # to NHWC, which in TILE copies it, as its width is not a tile multiple.
            out = self.fpn_convs[i](lateral)
            if out.is_sharded():
                # A level that fits L1 comes out in conv2d's shards. The levels live through the
                # whole detector, and one left in L1 shrinks what the encoder's attention can stage
                # there; the extra conv's weights are prepared for an interleaved DRAM input too.
                moved = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(out)
                out = moved
            outs.append(out)
        for i in range(used_backbone_levels, len(self.fpn_convs)):
            extra_source = outs[-1]
            if i > used_backbone_levels and self.relu_before_extra_convs:
                extra_source = ttnn.relu(extra_source)
            outs.append(self.fpn_convs[i](extra_source))
        signpost(header="TTNN FPN Forward End")
        return tuple(outs)
