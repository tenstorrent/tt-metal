# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tt_lib.fallback_ops import fallback_ops


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16, device=device, layout=layout)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


class TtPatchMerging:
    def __init__(self, device, parameters, dim):
        self.dim = dim
        self.device = device
        self.parameters = parameters

    def __call__(self, x):
        _, H, W, _ = x.get_legacy_shape()
        x = ttnn.pad(
            x, x.shape, [0, 0, 0, 0], 0
        )  # This is not needed(check on this) x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2)) , No difference in shape
        x = ttnn_to_torch(x)
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C Issue #8920
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C Issue #8920
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C Issue #8920
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C Issue #8920
        x = torch_to_ttnn(x, self.device)
        x0 = torch_to_ttnn(x0, self.device)
        x1 = torch_to_ttnn(x1, self.device)
        x2 = torch_to_ttnn(x2, self.device)
        x3 = torch_to_ttnn(x3, self.device)
        x = ttnn.concat([x0, x1, x2, x3], -1)
        x = ttnn.layer_norm(x, weight=self.parameters.norm["weight"], bias=self.parameters.norm["bias"])
        x = ttnn.linear(x, self.parameters.reduction["weight"], core_grid=ttnn.CoreGrid(y=8, x=8))
        return x
