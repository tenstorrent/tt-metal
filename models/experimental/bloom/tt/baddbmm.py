# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttnn.deprecated
import models.experimental.bloom.bloom_utils as bloom_utils

mem_config = ttnn.experimental.tensor.MemoryConfig(
    ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
)


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> ttnn.experimental.tensor.Tensor:
    if beta != 1.0:
        input = ttnn.mul(beta, input, memory_config=mem_config)

    tmp = bloom_utils.tt_bmm(batch1, batch2, device)

    tmp = ttnn.mul(alpha, tmp, memory_config=mem_config)

    result = ttnn.add(input, tmp, memory_config=mem_config)

    return result
