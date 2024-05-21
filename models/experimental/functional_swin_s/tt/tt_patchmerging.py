# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtPatchMerging:
    def __init__(self, device, parameters, dim):
        self.dim = dim
        self.device = device
        self.parameters = parameters

    def __call__(self, x):
        x = x.to(self.device)
        _, H, W, _ = x.get_legacy_shape()
        x = fallback_ops.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = tt_to_torch_tensor(x)
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)
        x0 = torch_to_tt_tensor_rm(x0, self.device, put_on_device=True)
        x1 = torch_to_tt_tensor_rm(x1, self.device, put_on_device=True)
        x2 = torch_to_tt_tensor_rm(x2, self.device, put_on_device=True)
        x3 = torch_to_tt_tensor_rm(x3, self.device, put_on_device=True)
        x = ttnn.concat([x0, x1, x2, x3], -1)
        x = ttnn.layer_norm(x, weight=self.parameters.norm.weight, bias=self.parameters.norm.bias)
        x = ttnn.linear(x, self.parameters.reduction.weight)
        return ttnn.from_device(x)
