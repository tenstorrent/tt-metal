# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn.deprecated as ttm


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> ttm.tensor.Tensor:
    if beta != 1.0:
        input = ttm.tensor.mul(beta, input)

    tmp = ttm.tensor.bmm(batch1, batch2)

    if alpha != 1.0:
        tmp = ttm.tensor.mul(alpha, tmp)

    result = ttm.tensor.add(input, tmp)

    return result
