# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
import tt_lib as ttl
import ttnn


class TtDistributedLayernormDLNP1:
    def __init__(self):
        super().__init__()

    def __call__(self, xs: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
        num_devices = len(xs)

        counts = []
        total_count = 0
        meanxs = []

        # Each device computes local statistics mean(x) and mean(x^2)
        # meanx = torch.mean(xs, dim=-1, keepdim=True)
        for i in range(num_devices):
            count_local = xs[i].shape[-1]
            total_count += count_local
            counts.append(count_local)

            meanx_local = ttl.tensor.reduce(
                xs[i], ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, scaler=1.0 / counts[i]
            )
            meanxs.append(meanx_local)

        # meanx2 = torch.mean(torch.square(xs), dim=-1, keepdim=True)
        meanx2s = []
        for i in range(num_devices):
            x2_local = ttl.tensor.pow(xs[i], 2)
            meanx2_local = ttl.tensor.reduce(
                x2_local, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, scaler=1.0 / counts[i]
            )
            meanx2s.append(meanx2_local)

        # Weighted meanx to number of samples per device
        for i in range(num_devices):
            meanxs[i] = ttnn.multiply(meanxs[i], counts[i])

        # Weighted meanx2 to number of samples per device
        for i in range(num_devices):
            meanx2s[i] = ttnn.multiply(meanx2s[i], counts[i])

        output = []
        for i in range(num_devices):
            output.append(ttl.tensor.concat([meanxs[i], meanx2s[i]], 3))

        return output
