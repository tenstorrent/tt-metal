# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn

from models.experimental.vovnet_fused_conv.tt.classifier_head import TtClassifierHead
from models.experimental.vovnet_fused_conv.tt.conv_norm_act import TtConvNormAct
from models.experimental.vovnet_fused_conv.tt.osa_stage import TtOsaStage
from models.experimental.vovnet_fused_conv.tt.separable_conv_norm_act import TtSeparableConvNormAct


class TtVoVNet:
    def __init__(
        self,
        num_classes=1000,
        device=None,
        # torch_model=None,
        base_address=None,
        parameters=None,
    ):
        self.num_classes = num_classes

        self.device = device
        # self.torch_model = torch_model
        self.base_address = base_address

        self.stem = [
            TtConvNormAct(
                stride=2,
                base_address="stem.0",
                device=self.device,
                # torch_model=self.torch_model,
                parameters=parameters,
            ),
            TtSeparableConvNormAct(
                base_address=f"stem.1",
                device=device,
                # torch_model=self.torch_model,
                stride=1,
                padding=1,
                parameters=parameters,
            ),
            TtSeparableConvNormAct(
                base_address=f"stem.2",
                device=device,
                # torch_model=self.torch_model,
                stride=2,
                padding=1,
                parameters=parameters,
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
                    # torch_model=self.torch_model,
                    device=self.device,
                    parameters=parameters,
                )
            ]
            downsample = True

        self.head = TtClassifierHead(
            base_address=f"head",
            device=self.device,
            # torch_model=self.torch_model,
            parameters=parameters,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i, module in enumerate(self.stem):
            x = module.forward(x)[0]
        for i, module in enumerate(self.stages):
            x = module.forward(x)
            print(i)
            print(x)
            print("------------------")
        x = self.head.forward(x)
        return x
