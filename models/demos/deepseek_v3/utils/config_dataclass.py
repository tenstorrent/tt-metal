# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import itertools
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import NoneType
from typing import Any, Callable, Union

import ttnn

# Union type for all possible program configs used with ttnn.linear
ProgramConfig = Union[
    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig,
    ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig,
    None,
]


@dataclass(frozen=True)
class FromWeightConfig:
    """A stub in a model config that gets replaced with a real ttnn.Tensor when creating the RunConfig.
    Needs a matching entry in the WeightConfig"""


@dataclass(frozen=True, eq=True)
class MeshDeviceStub:
    """A stub that gets replaced with a real ttnn.MeshDevice when creating the RunConfig."""

    mesh_shape: tuple[int, int]

    def __init__(self, mesh_shape: tuple[int, int] | ttnn.MeshDevice):
        object.__setattr__(self, "mesh_shape", tuple(mesh_shape))


ConfigDevice = ttnn.MeshDevice | MeshDeviceStub
ConfigWeight = ttnn.Tensor | FromWeightConfig


@dataclass
class OpConfigBase:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        """Return only keys with non-None values for clean dictionary expansion."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name) is not None)

    def items(self):
        """Return only key-value pairs with non-None values."""
        return ((f.name, getattr(self, f.name)) for f in fields(self) if getattr(self, f.name) is not None)

    def all_keys(self):
        """Return all keys, including those with None values."""
        return tuple(f.name for f in fields(self))

    def get(self, key: str, default=None):
        """Get a field value with a default fallback."""
        return getattr(self, key, default)


@dataclass
class LinearConfig(OpConfigBase):
    """Common parameters for a ttnn.linear op, weights are in input_tensor_b"""

    input_tensor_b: ConfigWeight
    memory_config: ttnn.MemoryConfig | None = None
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    program_config: ProgramConfig = None


@dataclass
class EmbeddingConfig(OpConfigBase):
    """Common parameters for a ttnn.embedding op"""

    weight: ConfigWeight
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    layout: ttnn.Layout = ttnn.TILE_LAYOUT


@dataclass
class MulConfig(OpConfigBase):
    """Common parameters for a ttnn.mul op"""

    memory_config: ttnn.MemoryConfig | None = None
    input_tensor_a_activations: list[ttnn.UnaryOpType] | None = None


@dataclass
class AllReduceConfig(OpConfigBase):
    """Common parameters for a ttnn.all_reduce op"""

    cluster_axis: int
    dim: int
    num_reduce_scatter_links: int
    num_all_gather_links: int
    topology: ttnn.Topology
    dtype: ttnn.DataType
    use_composite: bool
    mesh_device: ConfigDevice


@dataclass
class AllGatherConfig(OpConfigBase):
    """Common parameters for a ttnn.all_gather op"""

    memory_config: ttnn.MemoryConfig
    mesh_device: ConfigDevice


@dataclass
class RMSNormConfig(OpConfigBase):
    """RMSNorm config"""

    epsilon: float
    weight: ConfigWeight
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    stats_memcfg: ttnn.MemoryConfig | None = None
    output_memcfg: ttnn.MemoryConfig | None = None
    output_dtype: ttnn.DataType = ttnn.bfloat16
    is_distributed: bool = False
    topology: ttnn.Topology = ttnn.Topology.Linear
    norm_category: str = None


ConfigWithOp = dict[str, Any] | OpConfigBase
ConfigWithoutOp = dict[str, Any]


def merge_config_containers(
    cfg_a: Any,
    cfg_b: Any,
    merge_config_specific_items: Callable[[Any, Any, ttnn.MeshDevice], Any],
    mesh_device: ttnn.MeshDevice,
) -> Any:
    """Helper function to merge two configs, where the first one may partially consist of OpConfigs."""
    if cfg_a is None and cfg_b is None:
        return None

    if is_op_config(cfg_a):
        op_config_dict = {f.name: getattr(cfg_a, f.name) for f in fields(cfg_a)}  # type: ignore
        return cfg_a.__class__(**merge_config_containers(op_config_dict, cfg_b, merge_config_specific_items, mesh_device))  # type: ignore

    # If both configs are lists/tuples of the same length or one of them is None, merge them as a list/tuple.
    if isinstance(cfg_a, (list, tuple, NoneType)) and isinstance(cfg_b, (list, tuple, NoneType)):
        if cfg_a is None or cfg_b is None or (len(cfg_a) == len(cfg_b) and type(cfg_a) == type(cfg_b)):
            container = type(cfg_a) if cfg_a is not None else type(cfg_b)
            cfg_a = cfg_a or (container([None]) * len(cfg_b))
            cfg_b = cfg_b or (container([None]) * len(cfg_a))
            return container(
                merge_config_containers(a, b, merge_config_specific_items, mesh_device)
                for a, b in zip(cfg_a, cfg_b, strict=True)
            )

    if isinstance(cfg_a, (dict, NoneType)) and isinstance(cfg_b, (dict, NoneType)):
        cfg_a = cfg_a or {}
        cfg_b = cfg_b or {}
        return {
            k: merge_config_containers(cfg_a.get(k, None), cfg_b.get(k, None), merge_config_specific_items, mesh_device)
            for k in itertools.chain(cfg_a.keys(), cfg_b.keys())
        }

    return merge_config_specific_items(cfg_a, cfg_b, mesh_device)


def is_op_config(obj: Any) -> bool:
    """Check if the object is an op config instance."""
    return issubclass(type(obj), OpConfigBase) and is_dataclass(obj)


WeightConfig = dict[str, "WeightConfig | str"]

_PRIMITIVE_COPYABLE_TYPES = bool | int | float | complex | str | bytes | None | Enum
# In general, we require ModelConfig to be deepcopyable
ModelPrefillConfig = dict[str, "ModelPrefillConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase
ModelDecodeConfig = dict[str, "ModelDecodeConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase

RunPrefillConfig = dict[str, "RunPrefillConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase
RunDecodeConfig = dict[str, "RunDecodeConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase
