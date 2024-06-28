# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import ttnn


def linear(
    x: tt_lib.tensor.Tensor,
    weight: tt_lib.tensor.Tensor,
    bias: tt_lib.tensor.Tensor = None,
) -> tt_lib.tensor.Tensor:
    weight = tt_lib.tensor.transpose(weight, -2, -1)
    x = ttnn.matmul(x, weight)

    if bias is not None:
        x = tt_lib.tensor.bcast(x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H)
    return x
