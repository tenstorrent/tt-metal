import itertools
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
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
class TensorStub:
    """A stub in ModelConfig that gets replaced with a real ttnn.Tensor when creating the RunConfig.
    Needs a matching WeightStub in the WeightConfig"""


@dataclass(eq=True)
class WeightStub:
    filepath: str

    @classmethod
    def from_weight(cls, tensor: ttnn.Tensor, filepath: Path) -> "WeightStub":
        """Save a weight tensor and create a stub from it."""
        assert tensor.storage_type() != ttnn.StorageType.HOST, "Weight tensor must be allocated on device"
        ttnn.dump_tensor(filepath, tensor)
        return cls(filepath=str(filepath))

    def to_weight(self, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
        """Load the weight tensor from the file and return it as a ttnn.Tensor."""
        return ttnn.load_tensor(self.filepath, device=mesh_device)


@dataclass(frozen=True, eq=True)
class MeshDeviceStub:
    """A stub that gets replaced with a real ttnn.MeshDevice when creating the RunConfig."""

    mesh_shape: tuple[int, int]

    def __init__(self, mesh_shape: tuple[int, int] | ttnn.MeshDevice):
        object.__setattr__(self, "mesh_shape", tuple(mesh_shape))


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
    """Common parameters for a ttnn.linear op, weights are in input_tensor_b"""

    input_tensor_b: ConfigTensor
    memory_config: ttnn.MemoryConfig | None = None
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    program_config: ProgramConfig = None


@dataclass
class EmbeddingConfig(OpConfigBase):
    """Common parameters for a ttnn.embedding op"""

    weight: ConfigTensor
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
        return cfg_a.__class__(**_merge_configs(op_config_dict, cfg_b, mesh_device))  # type: ignore

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


_PRIMITIVE_COPYABLE_TYPES = bool | int | float | complex | str | bytes | None
ModelConfig = (
    dict[str, "ModelConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase
)  # In general, we require ModelConfig to be deepcopyable
WeightConfig = dict[str, "WeightConfig | WeightStub"]
RunConfig = dict[str, "RunConfig | _PRIMITIVE_COPYABLE_TYPES"] | OpConfigBase
