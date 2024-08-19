# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import ttnn


def linear(
    x: tt_lib.tensor.Tensor,
    weight: tt_lib.tensor.Tensor,
    bias: tt_lib.tensor.Tensor = None,
) -> tt_lib.tensor.Tensor:
    weight = ttnn.transpose(weight, -2, -1)
    x = ttnn.matmul(x, weight)

    if bias is not None:
        x = ttnn.add(x, bias)
    return x
