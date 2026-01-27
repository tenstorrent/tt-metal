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

    torch_tensor: torch.Tensor
    shard_dims: tuple[int | None, int | None]
    remove_dims: tuple[bool, bool] | bool = False
    dtype: ttnn.DataType | None = None
    layout: ttnn.Layout | None = None
    memory_config: ttnn.MemoryConfig | None = None

    # For future cache.py integration
    preprocessor: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    postprocessor: Callable[[ttnn.Tensor], ttnn.Tensor] = lambda x: x
