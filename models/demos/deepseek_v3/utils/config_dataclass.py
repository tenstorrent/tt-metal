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


@dataclass
class OpConfigBase:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return tuple(f.name for f in fields(self))


@dataclass
class LinearConfig(OpConfigBase):
    """Common parameters for a ttnn.linear layer, weights are in input_tensor_b"""

    input_tensor_b: ttnn.Tensor | Path | None = None
    memory_config: ttnn.MemoryConfig | None = None
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    program_config: ProgramConfig = None


@dataclass
class EmbeddingConfig(OpConfigBase):
    """Common parameters for a ttnn.embedding layer"""

    weight: ttnn.Tensor | Path | None = None
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


@dataclass
class ReshardConfig(OpConfigBase):
    """Simple config for operations that only need memory configuration"""

    memory_config: ttnn.MemoryConfig
