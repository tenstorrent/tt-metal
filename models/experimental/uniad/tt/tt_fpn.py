# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.uniad.tt.common import TtnnConv2D


class TtConvModule:
    def __init__(self, conv_args, conv_pth, device=None, is_blk=False, config_override=None):
        self.device = device
        self.conv = TtnnConv2D(
            conv_args.conv,
            conv_pth.conv,
            device=self.device,
            dealloc_act=True,
            is_fpn=True,
            is_blk=is_blk,
            config_override=config_override,
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
    ):
        self.device = device
        self.start_level = 0
        self.lateral_convs = []
        self.fpn_convs = []
        self.conv_pth = conv_pth
        for i in range(3):
            self.lateral_convs.append(
                TtConvModule(conv_args.lateral_convs[i], conv_pth.fpn.lateral_convs[str(i)], device=device)
            )
        for i in range(3):
            if i == 0 or i == 1:
                self.fpn_convs.append(
                    TtConvModule(
                        conv_args.fpn_convs[i],
                        conv_pth.fpn.fpn_convs[str(i)],
                        device=device,
                        is_blk=True,
                        config_override={"act_block_h": 128},
                    )
                )
            else:
                self.fpn_convs.append(
                    TtConvModule(conv_args.fpn_convs[i], conv_pth.fpn.fpn_convs[str(i)], device=device)
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
            laterals[i] = ttnn.reshape(
                laterals[i], (inp_dim.batch, inp_dim.height, inp_dim.width, laterals[i].shape[-1])
            )
            laterals_upsample = ttnn.upsample(laterals[i], 2)
            laterals_sliced = laterals_upsample[:, :, : (laterals_upsample.shape[2] - 1), :]
            laterals[i] = ttnn.reshape(
                laterals_sliced,
                [
                    1,
                    1,
                    laterals_sliced.shape[0] * laterals_sliced.shape[1] * laterals_sliced.shape[2],
                    laterals_sliced.shape[-1],
                ],
            )
            ttnn.deallocate(laterals_upsample)
            ttnn.deallocate(laterals_sliced)
            if i != 1:
                laterals[i] = ttnn.sharded_to_interleaved(laterals[i], memory_config=ttnn.L1_MEMORY_CONFIG)
            if i == 1:
                laterals[i] = ttnn.sharded_to_interleaved(laterals[i], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            laterals[i] = ttnn.to_layout(laterals[i], ttnn.TILE_LAYOUT)
            laterals[i - 1] = ttnn.add(laterals[i - 1], laterals[i])

        for i in range(used_backbone_levels):
            output = self.fpn_convs[i](laterals[i])
            outs.append(output)

        return tuple(outs)
