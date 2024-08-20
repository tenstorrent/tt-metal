# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import (
    Dict,
    List,
)
import ttnn

from models.experimental.ssd.tt.ssd_classification_head import TtSSDclassificationhead
from models.experimental.ssd.tt.ssd_regression_head import TtSSDregressionhead


class TtSSDLiteHead(nn.Module):
    def __init__(
        self,
        config,
        in_channels: List[int],
        num_anchors: List[int],
        num_classes: int,
        num_columns: int,
        state_dict=None,
        base_address="",
        device=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_columns = num_columns
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        self.classification_head = TtSSDclassificationhead(
            config,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            state_dict=self.state_dict,
            base_address=f"head.classification_head.module_list",
            device=device,
        )

        self.regression_head = TtSSDregressionhead(
            config,
            in_channels=self.in_channels,
            num_anchors=self.num_anchors,
            num_columns=self.num_columns,
            state_dict=self.state_dict,
            base_address=f"head.regression_head.module_list",
            device=device,
        )

    def forward(self, x: List[ttnn.Tensor]) -> Dict[str, ttnn.Tensor]:
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }
