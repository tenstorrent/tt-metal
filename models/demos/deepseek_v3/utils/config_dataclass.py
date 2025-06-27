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


@dataclass(eq=True)
class WeightStub:
    filepath: str
    mesh_shape: tuple[int, int]

    @classmethod
    def from_weight(cls, tensor: ttnn.Tensor, filepath: Path) -> "WeightStub":
        """Save a weight tensor and create a stub from it."""
        assert tensor.storage_type() != ttnn.StorageType.HOST, "Weight tensor must be allocated on device"
        ttnn.dump_tensor(filepath, tensor)
        return cls(filepath=str(filepath), mesh_shape=tuple(tensor.device().shape))

    def to_weight(self, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
        """Load the weight tensor from the file and return it as a ttnn.Tensor."""
        assert self.mesh_shape == tuple(mesh_device.shape), "Mesh device shape mismatch"
        return ttnn.load_tensor(self.filepath, device=mesh_device)


@dataclass(frozen=True, eq=True)
class MeshDeviceStub:
    """A stub that gets replaced with a real ttnn.MeshDevice when creating the RunConfig."""

    mesh_shape: tuple[int, int]


@dataclass(frozen=True)
class TensorStub:
    """A stub that gets replaced with a real ttnn.Tensor when creating the RunConfig."""


ConfigDevice = ttnn.MeshDevice | MeshDeviceStub
ConfigTensor = ttnn.Tensor | TensorStub


@dataclass
class OpConfigBase:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return tuple(f.name for f in fields(self))


@dataclass
class LinearConfig(OpConfigBase):
    """Common parameters for a ttnn.linear layer, weights are in input_tensor_b"""

    input_tensor_b: ConfigTensor = TensorStub()
    memory_config: ttnn.MemoryConfig | None = None
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    program_config: ProgramConfig = None


@dataclass
class EmbeddingConfig(OpConfigBase):
    """Common parameters for a ttnn.embedding layer"""

    # mesh_device: ConfigDevice
    weight: ConfigTensor = TensorStub()
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    layout: ttnn.Layout = ttnn.TILE_LAYOUT


@dataclass
class MulConfig(OpConfigBase):
    memory_config: ttnn.MemoryConfig | None = None
    input_tensor_a_activations: list[ttnn.UnaryOpType] | None = None


@dataclass
class AllReduceConfig(OpConfigBase):
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
    memory_config: ttnn.MemoryConfig
    mesh_device: ConfigDevice


ModelConfig = dict[str, "ModelConfig"] | OpConfigBase | str
WeightsConfig = dict[str, "WeightsConfig"] | WeightStub
RunConfig = dict[str, "RunConfig"] | OpConfigBase
