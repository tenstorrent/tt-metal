# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib
import models.experimental.bloom.bloom_utils as bloom_utils

mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1)


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> tt_lib.tensor.Tensor:
    if beta != 1.0:
        input = ttnn.mul(beta, input, memory_config=mem_config)

    tmp = bloom_utils.tt_bmm(batch1, batch2, device)

    tmp = ttnn.mul(alpha, tmp, memory_config=mem_config)

    result = ttnn.add(input, tmp, memory_config=mem_config)

    return result
