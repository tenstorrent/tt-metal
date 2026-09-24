# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.bevformer.tt.tt_common import TtnnConv2D


class TtConvModule:
    def __init__(
        self,
        conv_args,
        conv_pth,
        device=None,
        is_blk=False,
        config_override=None,
        dealloc_act=True,
        dram_activation=False,
    ):
        self.device = device
        self.conv = TtnnConv2D(
            conv_args.conv,
            conv_pth.conv,
            device=self.device,
            dealloc_act=dealloc_act,
            is_fpn=True,
            is_blk=is_blk,
            config_override=config_override,
            dram_activation=dram_activation,
        )

    def __call__(self, x):
        x = self.conv(x)
        return x[0]


class TtFPN:
    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        dram_activation_levels=(),
    ):
        """``dram_activation_levels`` lists the pyramid levels whose lateral and output convs
        keep their activations in DRAM, for levels too large for L1."""
        assert conv_args.add_extra_convs == "on_output", f"extra convs {conv_args.add_extra_convs!r} are not supported"
        self.device = device
        self.start_level = 0
        self.lateral_convs = []
        self.fpn_convs = []
        self.conv_pth = conv_pth
        self.relu_before_extra_convs = conv_args.relu_before_extra_convs
        num_levels = len(conv_args.lateral_convs)
        for i in range(num_levels):
            self.lateral_convs.append(
                TtConvModule(
                    conv_args.lateral_convs[i],
                    conv_pth.fpn.lateral_convs[str(i)],
                    device=device,
                    dram_activation=i in dram_activation_levels,
                )
            )
        for i in range(num_levels):
            if i == 0 or i == 1:
                self.fpn_convs.append(
                    TtConvModule(
                        conv_args.fpn_convs[i],
                        conv_pth.fpn.fpn_convs[str(i)],
                        device=device,
                        is_blk=True,
                        config_override={"act_block_h": 128},
                        dram_activation=i in dram_activation_levels,
                    )
                )
            else:
                self.fpn_convs.append(
                    TtConvModule(
                        conv_args.fpn_convs[i],
                        conv_pth.fpn.fpn_convs[str(i)],
                        device=device,
                        dram_activation=i in dram_activation_levels,
                    )
                )
        for i in range(num_levels, len(conv_args.fpn_convs)):
            self.fpn_convs.append(
                TtConvModule(
                    conv_args.fpn_convs[i],
                    conv_pth.fpn.fpn_convs[str(i)],
                    device=device,
                    dealloc_act=False,
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
            inp_dim = self.conv_pth.fpn.lateral_convs[str(i)].conv
            prev_dim = self.conv_pth.fpn.lateral_convs[str(i - 1)].conv
            laterals_reshaped = ttnn.reshape(
                laterals[i], (inp_dim.batch, inp_dim.height, inp_dim.width, laterals[i].shape[-1])
            )
            laterals_upsample = ttnn.upsample(laterals_reshaped, 2)
            # The reference interpolates to the lower level's exact size; a 2x upsample
            # overshoots it by one row or column wherever that level's size is odd.
            laterals_sliced = laterals_upsample[:, : prev_dim.height, : prev_dim.width, :]
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
