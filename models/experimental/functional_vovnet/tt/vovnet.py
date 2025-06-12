# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn

from models.experimental.functional_vovnet.tt.classifier_head import TtClassifierHead
from models.experimental.functional_vovnet.tt.conv_norm_act import TtConvNormAct
from models.experimental.functional_vovnet.tt.osa_stage import TtOsaStage
from models.experimental.functional_vovnet.tt.separable_conv_norm_act import TtSeparableConvNormAct


class TtVoVNet:
    def __init__(
        self,
        num_classes=1000,
        device=None,
        parameters=None,
        base_address=None,
    ):
        self.num_classes = num_classes

        self.device = device
        self.base_address = base_address

        self.stem = [
            TtConvNormAct(
                stride=2,
                base_address=f"stem.0",
                device=self.device,
                parameters=parameters,
            ),
            TtSeparableConvNormAct(
                base_address=f"stem.1",
                device=device,
                parameters=parameters,
                stride=1,
                padding=1,
            ),
            TtSeparableConvNormAct(
                base_address=f"stem.2",
                device=device,
                parameters=parameters,
                stride=2,
                padding=1,
            ),
        ]

        self.stages = []

        downsample = False
        for i in range(4):
            self.stages += [
                TtOsaStage(
                    block_per_stage=1,
                    downsample=downsample,
                    base_address=f"stages.{i}",
                    parameters=parameters,
                    device=self.device,
                )
            ]
            downsample = True

        self.head = TtClassifierHead(
            base_address=f"head",
            device=self.device,
            parameters=parameters,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i, module in enumerate(self.stem):
            x = module.forward(x)[0]
        for i, module in enumerate(self.stages):
            x = module.forward(x)

        x = self.head.forward(x)
        return x
