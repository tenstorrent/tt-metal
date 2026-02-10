# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, Union

import torch

import ttnn
from models.demos.deepseek_v3.utils.cache import MeshMapper, TensorCache


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

    def get_mesh_mapper(self, mesh_device: ttnn.MeshDevice) -> MeshMapper:
        """Return the mesh mapper for this spec, derived from shard_dims (and mesh_device shape).

        Note: Matches the logic used in config_helpers._shard_device_impl / shard_and_save.
        """
        shard_dims = self.shard_dims
        if shard_dims[0] is None and shard_dims[1] is None:
            return ttnn.ReplicateTensorToMesh(mesh_device)
        if shard_dims[0] == shard_dims[1] and shard_dims[0] is not None:
            return ttnn.ShardTensorToMesh(mesh_device, dim=shard_dims[0])
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=shard_dims)


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
    module_weight_spec: ModuleWeightSpec,
    path: str,
    cache: TensorCache,
    device: ttnn.Device | None = None,
    delimiter: str = ".",
):
    """
    Materialize a weight config from a weight spec.

    This will recursively materialize the weight config from the weight spec, querying the cache for each weight spec with the fully qualified path (the key in the original state dict). When device is a MeshDevice, the mesh mapper for each tensor is derived from that weight spec's shard_dims (and remove_dims) via WeightSpec.get_mesh_mapper.
    """
    weight_config = {}
    for key, value in module_weight_spec.items():
        if isinstance(value, WeightSpec):
            # TODO: We should be able to shard on host if the device is None
            mesh_mapper = (
                value.get_mesh_mapper(device) if device is not None and isinstance(device, ttnn.MeshDevice) else None
            )

            name = path + delimiter + key
            tensor = cache.get_tensor(
                name,
                value.dtype,
                value.layout,
                preprocessor=value.preprocessor,
                postprocessor=value.postprocessor,
                device=device,
                mesh_mapper=mesh_mapper,
            )
            weight_config[key] = tensor
        else:
            weight_config[key] = create_weight_config_from_weight_spec(
                value, path + delimiter + key, cache, device=device
            )
    return weight_config
