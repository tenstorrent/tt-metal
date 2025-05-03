# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            conv1 = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
            bn1 = nn.BatchNorm2d(ch)
            relu = nn.ReLU(inplace=True)
            conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
            bn2 = nn.BatchNorm2d(ch)
            resblock_one = nn.ModuleList([conv1, bn1, relu, conv2, bn2, relu])
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x
