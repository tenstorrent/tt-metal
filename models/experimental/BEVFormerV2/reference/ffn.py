# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

##########################################################################
# Adapted from BEVFormer (https://github.com/fundamentalvision/BEVFormer).
# Original work Copyright (c) OpenMMLab.
# Modified by Zhiqi Li.
# Licensed under the Apache License, Version 2.0.
##########################################################################

import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=512, ffn_dropout=0.0):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.activate = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(ffn_dropout)

        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(self.embed_dims, feedforward_channels), nn.ReLU(inplace=True)),
            nn.Linear(feedforward_channels, self.embed_dims),
        )

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        return identity + self.dropout(self.layers(x))
