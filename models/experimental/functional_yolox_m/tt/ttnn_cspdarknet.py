# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.functional_yolox_m.tt.ttnn_dark2 import TtDark2, TtFocus
from models.experimental.functional_yolox_m.tt.ttnn_dark3 import TtDark3
from models.experimental.functional_yolox_m.tt.ttnn_dark4 import TtDark4
from models.experimental.functional_yolox_m.tt.ttnn_dark5 import TtDark5


class TtCSPDarknet:
    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.focus = TtFocus(parameters["stem"])
        self.dark2 = TtDark2(parameters["dark2"])
        self.dark3 = TtDark3(parameters["dark3"])
        self.dark4 = TtDark4(parameters["dark4"])
        self.dark5 = TtDark5(device, parameters["dark5"])

    def __call__(self, device, input_tensor: ttnn.Tensor):
        d1 = self.focus(device, input_tensor)
        d2 = self.dark2(device, d1)
        d3 = self.dark3(device, d2)
        d4 = self.dark4(device, d3)
        d5 = self.dark5(device, d4)

        outputs = {}
        outputs["stem"] = d1
        outputs["dark2"] = d2
        outputs["dark3"] = d3
        outputs["dark4"] = d4
        outputs["dark5"] = d5

        return outputs
