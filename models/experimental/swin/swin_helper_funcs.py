# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor = None,
) -> ttnn.Tensor:
    weight = ttnn.transpose(weight, -2, -1)
    x = ttnn.matmul(x, weight)

    if bias is not None:
        x = ttnn.add(x, bias)
    return x
