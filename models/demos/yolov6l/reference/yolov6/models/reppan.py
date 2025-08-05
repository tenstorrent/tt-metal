# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from yolov6.layers.common import BepC3, BiFusion, BottleRep, ConvBNReLU


class CSPRepBiFPANNeck(nn.Module):
    """
    CSPRepBiFPANNeck module.
    """

    def __init__(
        self, channels_list=None, num_repeats=None, block=BottleRep, csp_e=float(1) / 2, stage_block_type="BepC3"
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], out_channels=channels_list[5], kernel_size=1, stride=1  # 1024  # 256
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]],  # 512, 256
            out_channels=channels_list[5],  # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=channels_list[5],  # 256
            out_channels=channels_list[5],  # 256
            n=num_repeats[5],
            e=csp_e,
            block=block,
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5], out_channels=channels_list[6], kernel_size=1, stride=1  # 256  # 128
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]],  # 256, 128
            out_channels=channels_list[6],  # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=channels_list[6],  # 128
            out_channels=channels_list[6],  # 128
            n=num_repeats[6],
            e=csp_e,
            block=block,
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6], out_channels=channels_list[7], kernel_size=3, stride=2  # 128  # 128
        )

        self.Rep_n3 = stage_block(
            in_channels=channels_list[6] + channels_list[7],  # 128 + 128
            out_channels=channels_list[8],  # 256
            n=num_repeats[7],
            e=csp_e,
            block=block,
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8], out_channels=channels_list[9], kernel_size=3, stride=2  # 256  # 256
        )

        self.Rep_n4 = stage_block(
            in_channels=channels_list[5] + channels_list[9],  # 256 + 256
            out_channels=channels_list[10],  # 512
            n=num_repeats[8],
            e=csp_e,
            block=block,
        )

    def forward(self, input):
        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs
