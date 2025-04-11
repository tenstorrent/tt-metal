# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.yolov6l.tt.ttnn_bottlerep import TtBottleRep


class TtRepBlock:
    def __init__(self, device, parameters, model_params):
        self.parameters = parameters
        self.model_params = model_params
        self.conv1 = TtBottleRep(device, parameters.conv1, model_params.conv1)
        self.bottle_rep1 = TtBottleRep(device, parameters.block[0], model_params.block[0])
        self.bottle_rep2 = TtBottleRep(device, parameters.block[1], model_params.block[1])

    def __call__(self, x):
        conv1 = self.conv1(x)
        br1 = self.bottle_rep1(conv1)
        br2 = self.bottle_rep2(br1)
        return br2
