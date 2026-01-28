# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable

import torch

import ttnn


@dataclass
class WeightSpec:
    """Declarative specification for weight conversion.

    Compatible with cache.py for future integration.
    """

    name: str
    shard_dims: tuple[int | None, int | None] = (None, None)
    remove_dims: tuple[bool, bool] = (False, False)
    dtype: ttnn.DataType = ttnn.bfloat16
    layout: ttnn.Layout = ttnn.TILE_LAYOUT
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG

    preprocessor: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    postprocessor: Callable[[ttnn.Tensor], ttnn.Tensor] = lambda x: x
