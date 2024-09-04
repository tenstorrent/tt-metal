# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import models.experimental.bloom.bloom_utils as bloom_utils

mem_config = ttnn.L1_MEMORY_CONFIG


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> ttnn.Tensor:
    if beta != 1.0:
        input = ttnn.mul(beta, input, memory_config=mem_config)

    tmp = bloom_utils.tt_bmm(batch1, batch2, device)

    tmp = ttnn.mul(alpha, tmp, memory_config=mem_config)

    result = ttnn.add(input, tmp, memory_config=mem_config)

    return result
