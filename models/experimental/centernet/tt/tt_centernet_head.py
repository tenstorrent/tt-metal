# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.centernet.tt.common import TtConv


class TtCTResNetHead:
    def __init__(self, parameters=None, device=None, init_cfg=None) -> None:
        self.parameters = parameters
        self.device = device
        self.heatmap_head = self._build_head(80, self.parameters, "bbox_head.heatmap_head", device=device)
        self.wh_head = self._build_head(2, self.parameters, "bbox_head.wh_head", device=device)
        self.offset_head = self._build_head(2, self.parameters, "bbox_head.offset_head", device=device)

    def _build_head(self, out_channels: int, parameters, base_address, device):
        """Build head for each branch."""
        conv1 = TtConv(
            device=device,
            parameters=parameters,
            path=f"{base_address}.0",
            conv_params=[1, 1, 1, 1],
            fused_op=True,
            activation="relu",
        )
        conv2 = TtConv(
            device=device,
            parameters=parameters,
            path=f"{base_address}.2",
            conv_params=[1, 1, 0, 0],
            fused_op=True,
            activation="",
        )
        layer = [conv1, conv2]
        return layer

    def forward(self, x):
        center_heatmap_pred = self.heatmap_head[0](x)
        center_heatmap_pred = self.heatmap_head[1](center_heatmap_pred)
        center_heatmap_pred = ttnn.sigmoid_accurate(center_heatmap_pred, memory_config=ttnn.L1_MEMORY_CONFIG)

        wh_pred = self.wh_head[0](x)
        wh_pred = self.wh_head[1](wh_pred)

        offset_pred = self.offset_head[0](x)
        offset_pred = self.offset_head[1](offset_pred)

        return center_heatmap_pred, wh_pred, offset_pred
