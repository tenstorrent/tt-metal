# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def feed_forward(config, x: ttnn.Tensor, *, parameters):
    silu_out = ttnn.silu(x @ parameters.w1.weight)
    x = silu_out * (x @ parameters.w3.weight)
    return x @ parameters.w2.weight
