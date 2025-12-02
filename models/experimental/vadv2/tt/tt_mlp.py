# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os

try:
    from tracy import signpost

    use_signpost = os.getenv("USE_SIGNPOST", "False").lower() in ("true", "1", "yes")
except ModuleNotFoundError:
    use_signpost = False


class TtMLP:
    def __init__(self, params, device, in_channels, hidden_unit, verbose=False):
        super(TtMLP, self).__init__()
        self.params = params
        self.device = device

    def __call__(self, x):
        if use_signpost:
            signpost(header="TtMLP_call_start")
        x = ttnn.linear(x, self.params.linear.weight, bias=self.params.linear.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.layer_norm(
            x, weight=self.params.norm.weight, bias=self.params.norm.bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x = ttnn.relu(x)
        if use_signpost:
            signpost(header="TtMLP_call_end")
        return x
