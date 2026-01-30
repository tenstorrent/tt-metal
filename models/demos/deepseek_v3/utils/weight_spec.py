# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, Union

import torch

import ttnn
from models.demos.deepseek_v3.utils.cache import TensorCache


@dataclass
class WeightSpec:
    """Declarative specification for weight conversion."""

    name: str = ""
    dtype: ttnn.DataType = ttnn.bfloat16
    layout: ttnn.Layout = ttnn.TILE_LAYOUT
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG

    shard_dims: tuple[int | None, int | None] = (None, None)
    remove_dims: tuple[bool, bool] = (False, False)

    preprocessor: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    postprocessor: Callable[[ttnn.Tensor], ttnn.Tensor] = lambda x: x


ModuleWeightSpec = dict[str, Union[WeightSpec, "ModuleWeightSpec"]]


@dataclass(frozen=True)
class WeightSpecContext:
    resolver: Callable[[str], torch.Tensor]

    def with_prefix(self, prefix: str) -> "WeightSpecContext":
        def prefixed(name: str) -> torch.Tensor:
            key = f"{prefix}.{name}" if prefix else name
            return self.resolver(key)

        return WeightSpecContext(resolver=prefixed)

    def get_reference_tensor(self, name: str) -> torch.Tensor:
        return self.resolver(name)


def create_weight_config_from_weight_spec(
    module_weight_spec: ModuleWeightSpec, path: str, cache: TensorCache, delimiter: str = "."
):
    """
    Materialize a weight config from a weight spec.

    This will recursively materialize the weight config from the weight spec, but querying the cache for each weight spec with the fully qualified path (the key in the original state dict).
    """
    weight_config = {}
    for key, value in module_weight_spec.items():
        if isinstance(value, WeightSpec):
            # If its a weight spec we should load it from the cache using the fully qualified path (the key in the original state dict)
            name = path + delimiter + key
            tensor = cache.get_tensor(
                name,
                value.dtype,
                value.layout,
                preprocessor=value.preprocessor,
                postprocessor=value.postprocessor,
            )
            weight_config[key] = tensor
        else:
            weight_config[key] = create_weight_config_from_weight_spec(value, path + delimiter + key, cache)
    return weight_config
