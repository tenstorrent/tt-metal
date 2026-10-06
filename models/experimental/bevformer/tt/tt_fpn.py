# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""FPN neck for BEVFormer.

Lateral 1x1 convs, a top-down upsample-and-add, then the output convs. Extra
convs, when the reference FPN has them, run on the lowest output.
"""

import ttnn
from models.experimental.bevformer.tt.tt_common import TtnnConv2D


class TtConvModule:
    """One FPN convolution.

    Accumulates in an fp32 destination register. The call returns the output tensor;
    output height and width stay on the conv, which the top-down pass reads.
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device=None,
        is_blk=False,
        act_block_h=None,
        dealloc_act=True,
        dram_activation=False,
        dram_conv_slices=None,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
    ):
        self.conv = TtnnConv2D(
            conv_args.conv,
            conv_pth.conv,
            device=device,
            dealloc_act=dealloc_act,
            fp32_dest_acc_en=True,
            is_blk=is_blk,
            act_block_h=act_block_h,
            dram_activation=dram_activation,
            dram_conv_slices=dram_conv_slices,
            input_dtype=input_dtype,
            input_layout=input_layout,
        )

    def __call__(self, x):
        x = self.conv(x)
        return x[0]


class TtFPN:
    """Feature pyramid over the backbone levels.

    ``input_dtypes`` are the dtypes of the backbone levels the FPN reads, in order.
    ``dram_activation_levels`` lists the pyramid levels whose lateral and output convs
    keep their activations in DRAM, for levels too large for L1; their 3x3 convs run in
    ``dram_conv_slices`` width slices. ``block_sharded_levels`` lists the levels whose
    output conv is block sharded with act_block_h 128.
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        input_dtypes,
        dram_activation_levels=(),
        dram_conv_slices=None,
        block_sharded_levels=(),
    ):
        assert conv_args.add_extra_convs == "on_output", f"extra convs {conv_args.add_extra_convs!r} are not supported"
        self.lateral_convs = []
        self.fpn_convs = []
        self.relu_before_extra_convs = conv_args.relu_before_extra_convs
        num_levels = len(conv_args.lateral_convs)
        assert len(input_dtypes) == num_levels, f"{num_levels} levels, got {len(input_dtypes)} input dtypes"
        # Each level keeps its input's dtype through its lateral conv and the top-down add.
        # The top-down pass turns every level but the lowest to ROW_MAJOR for the upsample,
        # which also makes it bfloat16, since bfloat8_b exists only in TILE.
        output_conv_layouts = [ttnn.TILE_LAYOUT] + [ttnn.ROW_MAJOR_LAYOUT] * (num_levels - 1)
        output_conv_dtypes = [input_dtypes[0]] + [ttnn.bfloat16] * (num_levels - 1)
        for i in range(num_levels):
            self.lateral_convs.append(
                TtConvModule(
                    conv_args.lateral_convs[i],
                    conv_pth.fpn.lateral_convs[str(i)],
                    device=device,
                    dram_activation=i in dram_activation_levels,
                    dram_conv_slices=dram_conv_slices,
                    input_dtype=input_dtypes[i],
                )
            )
        for i in range(num_levels):
            block_sharded = i in block_sharded_levels
            self.fpn_convs.append(
                TtConvModule(
                    conv_args.fpn_convs[i],
                    conv_pth.fpn.fpn_convs[str(i)],
                    device=device,
                    is_blk=block_sharded,
                    # From the UniAD port, not re-tuned for 928x1600.
                    act_block_h=128 if block_sharded else None,
                    dram_activation=i in dram_activation_levels,
                    dram_conv_slices=dram_conv_slices,
                    input_dtype=output_conv_dtypes[i],
                    input_layout=output_conv_layouts[i],
                )
            )
        for i in range(num_levels, len(conv_args.fpn_convs)):
            self.fpn_convs.append(
                TtConvModule(
                    conv_args.fpn_convs[i],
                    conv_pth.fpn.fpn_convs[str(i)],
                    device=device,
                    dealloc_act=False,
                    input_dtype=output_conv_dtypes[-1],
                )
            )

    def __call__(self, input_tensor):
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
            laterals_sliced = ttnn.to_layout(laterals_sliced, ttnn.TILE_LAYOUT)
            laterals[i - 1] = ttnn.add(laterals[i - 1], laterals_sliced)
        for i in range(used_backbone_levels):
            laterals[i] = ttnn.to_memory_config(laterals[i], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            output = self.fpn_convs[i](laterals[i])
            outs.append(output)
        for i in range(used_backbone_levels, len(self.fpn_convs)):
            extra_source = outs[-1]
            if i > used_backbone_levels and self.relu_before_extra_convs:
                extra_source = ttnn.relu(extra_source)
            outs.append(self.fpn_convs[i](extra_source))
        return tuple(outs)
