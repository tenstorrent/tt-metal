# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.panoptic_deeplab.reference.aspp import PanopticDeeplabASPPModel as ASPPModel
from models.experimental.panoptic_deeplab.reference.head import HeadModel
from models.experimental.panoptic_deeplab.reference.res_block import ResModel


class DecoderModel(torch.nn.Module):
    def __init__(self, in_channels, res3_intermediate_channels, res2_intermediate_channels, out_channels, name):
        super().__init__()
        self.name = name
        self.aspp = ASPPModel()
        if name == "Semantics_head":
            self.res3 = ResModel(512, 320, 256)
            self.res2 = ResModel(256, 288, 256)
            self.head_1 = HeadModel(256, 256, 19)
        else:
            self.res3 = ResModel(512, 320, 128)
            self.res2 = ResModel(256, 160, 128)
            self.head_1 = HeadModel(128, 32, 2)
            self.head_2 = HeadModel(128, 32, 1)

    def forward(self, x, res3, res2):
        y = self.aspp(x)
        y = self.res3(y, res3)
        y = self.res2(y, res2)
        out = self.head_1(y)

        if self.name == "instance_head":
            y_2 = self.head_2(y)
            return out, y_2
        return out, None
