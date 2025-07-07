# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, embed_dims=256):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.activate = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(self.embed_dims, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.1)),
            nn.Linear(512, self.embed_dims),
            nn.Dropout(p=0.1),
        )

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        return identity + self.layers(x)
