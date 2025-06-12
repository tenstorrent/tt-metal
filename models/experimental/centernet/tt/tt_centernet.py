# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from models.experimental.centernet.tt.tt_resnet import TtBasicBlock, TtResNet
from models.experimental.centernet.tt.tt_centernet_head import TtCTResNetHead
from models.experimental.centernet.tt.tt_centernet_neck import TtCTResNetNeck


class Ttcenternet:
    def __init__(self, parameters=None, device=None):
        self.parameters = parameters
        self.device = device
        self.backbone = TtResNet(
            TtBasicBlock, [2, 2, 2, 2], parameters=self.parameters, base_address="backbone", device=device
        )
        self.neck = TtCTResNetNeck(parameters=self.parameters, device=self.device)
        self.box_head = TtCTResNetHead(parameters=self.parameters, device=device)

    def forward(self, input):
        output = self.backbone.forward(input)
        output = self.neck.forward(output)
        output = self.box_head.forward(output[0])
        return output
