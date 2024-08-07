# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.experimental.functional_yolox_m.tt.ttnn_yolopafpn import TtYOLOPAFPN
from models.experimental.functional_yolox_m.tt.ttnn_yolohead import TtYOLOXHead


class TtYOLOX:
    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.backbone = TtYOLOPAFPN(device, parameters["backbone"])
        self.head = TtYOLOXHead(parameters["head"])

    def __call__(self, device, input_tensor: ttnn.Tensor):
        fpn_outs = self.backbone(device, input_tensor)
        outputs = self.head(device, fpn_outs)

        return outputs
