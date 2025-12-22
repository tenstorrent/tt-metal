# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn


class TopDown(nn.Module):
    """
    Standalone Top-Down feature pyramid from TransfuserBackbone.
    """

    def __init__(self, perception_output_features=512, bev_features_chanels=64, bev_upsample_factor=2):
        super(TopDown, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=bev_upsample_factor, mode="bilinear", align_corners=False)

        # top down convs
        self.up_conv5 = nn.Conv2d(bev_features_chanels, bev_features_chanels, (1, 1))
        self.up_conv4 = nn.Conv2d(bev_features_chanels, bev_features_chanels, (1, 1))
        self.up_conv3 = nn.Conv2d(bev_features_chanels, bev_features_chanels, (1, 1))

        # lateral conv
        self.c5_conv = nn.Conv2d(perception_output_features, bev_features_chanels, (1, 1))

    def forward(self, x):
        p5 = self.relu(self.c5_conv(x))

        p4 = self.relu(self.up_conv5(self.upsample(p5)))

        p3 = self.relu(self.up_conv4(self.upsample(p4)))

        p2 = self.relu(self.up_conv3(self.upsample(p3)))

        return p2, p3, p4, p5
