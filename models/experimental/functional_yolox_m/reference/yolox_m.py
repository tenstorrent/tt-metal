# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn
import torch

from models.experimental.functional_yolox_m.reference.yolo_pafpn import YOLOPAFPN
from models.experimental.functional_yolox_m.reference.yolo_head import YOLOXHead


class YOLOX(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = YOLOPAFPN()
        self.head = YOLOXHead()

    def forward(self, x: torch.Tensor):
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)

        return outputs
