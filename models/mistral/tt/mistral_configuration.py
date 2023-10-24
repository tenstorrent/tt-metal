# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass
class TtModelArgs:
    FALLBACK_EMPTY = False
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
    
