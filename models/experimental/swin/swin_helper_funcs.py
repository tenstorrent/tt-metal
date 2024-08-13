# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn.deprecated
import ttnn


def linear(
    x: ttnn.experimental.tensor.Tensor,
    weight: ttnn.experimental.tensor.Tensor,
    bias: ttnn.experimental.tensor.Tensor = None,
) -> ttnn.experimental.tensor.Tensor:
    weight = ttnn.transpose(weight, -2, -1)
    x = ttnn.matmul(x, weight)

    if bias is not None:
        x = ttnn.experimental.tensor.bcast(
            x, bias, ttnn.experimental.tensor.BcastOpMath.ADD, ttnn.experimental.tensor.BcastOpDim.H
        )
    return x
