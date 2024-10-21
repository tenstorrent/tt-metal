# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_fadnetpp.reference.cunet import CUNet
from models.experimental.functional_fadnetpp.reference.extractnet import ExtractNet

import torch
import torch.nn as nn


class DispNetC(nn.Module):
    def __init__(self):
        super(DispNetC, self).__init__()
        self.resBlock = True
        self.maxdisp = 192
        self.input_channel = 6
        self.encoder_ratio = 16
        self.decoder_ratio = 16
        self.extractnet = ExtractNet(
            resBlock=self.resBlock,
            maxdisp=self.maxdisp,
            input_channel=self.input_channel,
            encoder_ratio=self.encoder_ratio,
            decoder_ratio=self.decoder_ratio,
        )
        self.cunet = CUNet(
            resBlock=self.resBlock,
            maxdisp=self.maxdisp,
            input_channel=self.input_channel,
            encoder_ratio=self.encoder_ratio,
            decoder_ratio=self.decoder_ratio,
        )

    def forward(self, input: torch.Tensor):
        conv1_l, conv2_l, conv3a_l, conv3a_r = self.extractnet(input)

        def build_corr(img_left, img_right, max_disp=40, zero_volume=None):
            B, C, H, W = img_left.shape
            if zero_volume is not None:
                tmp_zero_volume = zero_volume  # * 0.0
                volume = tmp_zero_volume
            else:
                volume = img_left.new_zeros([B, max_disp, H, W])
            for i in range(max_disp):
                if (i > 0) & (i < W):
                    volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, : W - i]).mean(dim=1)
                else:
                    volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

            volume = volume.contiguous()
            return volume

        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.maxdisp // 8 + 16)
        dispnetc_flows = self.cunet(input, conv1_l, conv2_l, conv3a_l, out_corr)
        return dispnetc_flows
