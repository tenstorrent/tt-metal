# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn
import ttnn

from collections import OrderedDict
from typing import (
    Dict,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_features import (
    TtMobileNetV3Features,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_extract import (
    TtMobileNetV3extract,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_inverted_residual import (
    TtMobileNetV3InvertedResidual,
)


class TtSSDLiteFeatureExtractorMobileNet(nn.Module):
    def __init__(
        self,
        config,
        state_dict=None,
        base_address="",
        device=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.base_address = base_address

        in_channels = [480, 512, 256, 256]
        out_channels = [512, 256, 256, 128]
        expand_channels = [256, 128, 128, 64]

        self.backbone_features = nn.ModuleList()
        self.features_1 = TtMobileNetV3Features(self.config, state_dict, f"{base_address}.features", self.device)

        self.backbone_features.append(self.features_1)
        self.features_2 = TtMobileNetV3extract(self.config, state_dict, f"{base_address}.features", self.device)

        self.backbone_features.append(self.features_2)
        for i in range(4):
            self.block = TtMobileNetV3InvertedResidual(
                config,
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                expanded_channels=expand_channels[i],
                kernel_size=3,
                stride=2,
                padding=1,
                use_activation=True,
                dilation=1,
                state_dict=state_dict,
                base_address=f"{self.base_address}.extra.{i}",
                extractor=True,
            )

            self.backbone_features.append(self.block)

    def forward(self, x: ttnn.Tensor) -> Dict[str, ttnn.Tensor]:
        output = []
        for i, block in enumerate(self.backbone_features):
            x = block(x)
            output.append(x)
        return OrderedDict([(str(i), v) for i, v in enumerate(output)])
