# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> ttnn.Tensor:
    if beta != 1.0:
        input = ttnn.mul(beta, input)

    tmp = ttnn.bmm(batch1, batch2)

    if alpha != 1.0:
        tmp = ttnn.mul(alpha, tmp)

    result = ttnn.add(input, tmp)

    return result
