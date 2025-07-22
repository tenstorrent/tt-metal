# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, fields
from typing import Any, Union

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
    Needs a matching entry in the WeightConfig. During the creation of the run config, mesh_device
    gets filled with the ttnn.Device to load the tensor on"""

    mesh_device: "ConfigDevice"


@dataclass(frozen=True, eq=True)
class MeshDeviceStub:
    """A stub that gets replaced with a real ttnn.MeshDevice when creating the RunConfig."""

    mesh_shape: tuple[int, int]

    def __init__(self, mesh_shape: tuple[int, int] | ttnn.MeshShape):
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
    program_config: ProgramConfig | None = None


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
class AllGatherAsyncConfig(OpConfigBase):
    """Common parameters for a ttnn.experimental.all_gather_async op"""

    mesh_device: ConfigDevice
    cluster_axis: int
    dim: int
    multi_device_global_semaphore: object
    num_links: int
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    topology: ttnn.Topology = ttnn.Topology.Linear


@dataclass
class ReduceScatterAsyncConfig(OpConfigBase):
    """Common parameters for a ttnn.experimental.reduce_scatter_async op"""

    mesh_device: ConfigDevice
    cluster_axis: int
    dim: int
    from_remote_multi_device_global_semaphore: object
    to_remote_multi_device_global_semaphore: object
    math_op: ttnn.ReduceType
    num_links: int
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    topology: ttnn.Topology = ttnn.Topology.Linear


@dataclass
class AllGatherConfig(OpConfigBase):
    dim: int
    mesh_device: ConfigDevice | None = None
    cluster_axis: int | None = None
    memory_config: ttnn.MemoryConfig | None = None
    num_workers: int | None = None
    num_buffers_per_channel: int | None = None
    topology: ttnn.Topology = ttnn.Topology.Ring
    num_links: int = 1


@dataclass
class ReduceScatterConfig(OpConfigBase):
    """Common parameters for a ttnn.reduce_scatter op"""

    dim: int
    math_op: ttnn.ReduceType
    mesh_device: ConfigDevice | None = None
    cluster_axis: int | None = None
    memory_config: ttnn.MemoryConfig = None
    topology: ttnn.Topology = ttnn.Topology.Ring
    num_links: int = 1


@dataclass
class ReshardConfig(OpConfigBase):
    """Common parameters for a ttnn.to_memory_config op"""

    memory_config: ttnn.MemoryConfig
    dtype: ttnn.DataType | None = None


@dataclass
class RMSNormConfig(OpConfigBase):
    """ttnn.rms_norm config"""

    epsilon: float = 1e-12
    weight: ConfigWeight | None = None
    bias: ConfigWeight | None = None
    residual_input_tensor: ConfigWeight | None = None
    memory_config: ttnn.MemoryConfig | None = None
    program_config: ttnn.LayerNormDefaultProgramConfig | ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    compute_kernel_config: ttnn.GrayskullComputeKernelConfig | ttnn.WormholeComputeKernelConfig | None = None


@dataclass
class RMSNormPreAllGatherConfig(OpConfigBase):
    """ttnn.rms_norm_pre_all_gather config"""

    dtype: ttnn.DataType = ttnn.bfloat16
    residual_input_tensor: ConfigWeight | None = None
    compute_kernel_config: ttnn.GrayskullComputeKernelConfig | ttnn.WormholeComputeKernelConfig | None = None
    program_config: ttnn.LayerNormDefaultProgramConfig | ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    memory_config: ttnn.MemoryConfig | None = None


@dataclass
class RMSNormPostAllGatherConfig(OpConfigBase):
    """ttnn.rms_norm_post_all_gather config"""

    epsilon: float = 1e-12
    weight: ConfigWeight | None = None
    bias: ConfigWeight | None = None
    memory_config: ttnn.MemoryConfig | None = None
    program_config: ttnn.LayerNormDefaultProgramConfig | ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    compute_kernel_config: ttnn.GrayskullComputeKernelConfig | ttnn.WormholeComputeKernelConfig | None = None
    dtype: ttnn.DataType | None = None
