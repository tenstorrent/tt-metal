# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib


def tt_linear(weight: tt_lib.tensor, bias: tt_lib.tensor, device):
    """Perform a linear operation on the input tensor using transposed weight and bias."""

    def linear_(activation):
        weight_T = tt_lib.tensor.transpose(weight, -2, -1)
        output = tt_lib.tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tt_lib.tensor.bcast(
                output, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
            )
            return output_plus_bias

        return output

    return linear_
