# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, fields
from pathlib import Path
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


@dataclass
class SavedWeight:  # TODO: bring regular tensor saving back once Issue #26763 is resolved
    path: Path
    memory_config: ttnn.MemoryConfig | None = None


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

    dim: int | None = None
    cluster_axis: int | None = None
    mesh_device: ttnn._ttnn.multi_device.MeshDevice | None = None
    topology: ttnn._ttnn.operations.ccl.Topology | None = None
    multi_device_global_semaphore: ttnn._ttnn.operations.experimental.ccl_experimental.GlobalSemaphoreArg | None = None
    persistent_output_tensor: ttnn._ttnn.tensor.Tensor | None = None
    num_links: int | None = None
    memory_config: ttnn._ttnn.tensor.MemoryConfig | None = None
    subdevice_id: ttnn._ttnn.device.SubDeviceId | None = None
    use_optimal_ccl_for_llama: bool | None = None
    barrier_semaphore: ttnn._ttnn.global_semaphore.global_sempahore | None = None


@dataclass
class ReduceScatterAsyncConfig(OpConfigBase):
    """Common parameters for a ttnn.experimental.reduce_scatter_async op"""

    mesh_device: ConfigDevice | None = None
    cluster_axis: int | None = None
    dim: int | None = None
    from_remote_multi_device_global_semaphore: object | None = None
    to_remote_multi_device_global_semaphore: object | None = None
    math_op: ttnn.ReduceType | None = None
    num_links: int | None = None
    memory_config: ttnn.MemoryConfig | None = None
    topology: ttnn.Topology | None = None


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


@dataclass
class BinaryOpConfig(OpConfigBase):
    """Common parameters for a ttnn.add/sub/mul/div op, weights are in input_tensor_b"""

    input_tensor_b: ConfigWeight
    memory_config: ttnn.MemoryConfig | None = None
    dtype: ttnn.DataType | None = None
    activation: ttnn.UnaryOpType | None = None


@dataclass
class ReshapeConfig(OpConfigBase):
    """Common parameters for a ttnn.reshape op"""

    shape: tuple[int, int, int, int] | None = None


@dataclass
class TopKConfig(OpConfigBase):
    """Common parameters for a ttnn.topk op"""

    k: int
    dim: int
    largest: bool = True
    sorted: bool = True


@dataclass
class ScatterConfig(OpConfigBase):
    """Common parameters for a ttnn.experimental.scatter op"""

    input: ttnn.Tensor
    dim: int
    src: ttnn.Tensor


@dataclass
class AllToAllDispatchConfig(OpConfigBase):
    """Common parameters for a ttnn.all_to_all_dispatch op"""

    cluster_axis: int
    memory_config: ttnn.MemoryConfig
    num_links: int | None = None
    global_semaphore: object | None = None
    init_semaphore: object | None = None
    topology: ttnn.Topology = ttnn.Topology.Linear
    subdevice_id: int | None = None


@dataclass
class AllToAllCombineConfig(OpConfigBase):
    """Common parameters for a ttnn.all_to_all_combine op"""

    axis: int
    memory_config: ttnn.MemoryConfig
    num_links: int | None = None
    global_semaphore: object | None = None
    init_semaphore: object | None = None
    topology: ttnn.Topology = ttnn.Topology.Linear


@dataclass
class RepeatConfig(OpConfigBase):
    """Common parameters for a ttnn.repeat op"""

    repeat_dims: ttnn.Shape


@dataclass
class TopKFallbackConfig(OpConfigBase):
    """Common parameters for a ttnn.topk_fallback op"""

    mesh_device: ttnn.Device
    dtype: ttnn.DataType
    memory_config: ttnn.MemoryConfig
    use_bitonic_sort: bool = False


@dataclass
class LinearFallbackConfig(OpConfigBase):
    """Common parameters for a ttnn.linear_fallback op"""

    mesh_device: ttnn.Device
    dtype: ttnn.DataType


@dataclass
class TypecastConfig(OpConfigBase):
    """Common parameters for a ttnn.typecast op"""

    dtype: ttnn.DataType
    memory_config: ttnn.MemoryConfig | None = None
    sub_core_grids: ttnn.CoreRangeSet | None = None
