# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import ttnn.deprecated


@dataclass
class TtModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0
    FALLBACK_SOFTMAX: bool = False
    FALLBACK_EMPTY: bool = False
    FALLBACK_SCATTER: bool = True
    FALLBACK_DRAM: bool = True
    WEIGHTS_DTYPE = ttnn.experimental.tensor.DataType.BFLOAT16

    if FALLBACK_DRAM:
        out_mem_config = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
        )
    else:
        out_mem_config = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
        )
