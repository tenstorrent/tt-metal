# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn
import torch

from models.experimental.functional_yolox_m.reference.dark2 import Focus, Dark2
from models.experimental.functional_yolox_m.reference.dark3 import Dark3
from models.experimental.functional_yolox_m.reference.dark4 import Dark4
from models.experimental.functional_yolox_m.reference.dark5 import Dark5


class CSPDarknet(nn.Module):
    def __init__(self):
        super(CSPDarknet, self).__init__()

        self.stem = Focus()
        self.dark2 = Dark2()
        self.dark3 = Dark3()
        self.dark4 = Dark4()
        self.dark5 = Dark5()

    def forward(self, x: torch.Tensor):
        outputs = {}
        d1 = self.stem(x)
        outputs["stem"] = d1
        d2 = self.dark2(d1)
        outputs["dark2"] = d2
        d3 = self.dark3(d2)
        outputs["dark3"] = d3
        d4 = self.dark4(d3)
        outputs["dark4"] = d4
        d5 = self.dark5(d4)
        outputs["dark5"] = d5

        return outputs
