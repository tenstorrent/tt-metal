# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os
from models.experimental.vadv2.tt.tt_mlp import TtMLP

try:
    from tracy import signpost

    use_signpost = os.getenv("USE_SIGNPOST", "False").lower() in ("true", "1", "yes")
except ModuleNotFoundError:
    use_signpost = False


class TtLaneNet:
    def __init__(self, params, device, in_channels, hidden_unit, num_subgraph_layers):
        super(TtLaneNet, self).__init__()
        self.params = params
        self.device = device
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = []
        self.layer_seq = []
        for i in range(num_subgraph_layers):
            layer_name = f"lmlp_{i}"
            self.layer_seq.append(TtMLP(params[layer_name], device, in_channels, hidden_unit))
            in_channels = hidden_unit * 2  # Update if output is concatenated like in LaneNet

    def __call__(self, pts_lane_feats):
        if use_signpost:
            signpost(header="TtLaneNet_call_start")
        x = pts_lane_feats
        for layer in self.layer_seq:
            if isinstance(layer, TtMLP):
                x = layer(x)
                x_max = ttnn.max(x, -2)[0]
                x_max = ttnn.unsqueeze(x_max, 0)
                x_max = ttnn.unsqueeze(x_max, 2)
                x_max = ttnn.repeat(x_max, (1, 1, x.shape[2], 1))
                x = ttnn.concat([x, x_max], dim=-1)

        x_max = ttnn.max(x, -2)[0]
        if use_signpost:
            signpost(header="TtLaneNet_call_end")
        return x_max
