# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for throughput-optimized MoE experts with all_to_all operations.

This module implements expert distribution across 32 Galaxy devices (4 experts per device)
using all_to_all_dispatch and all_to_all_combine for dynamic batching based on expert routing.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

import ttnn


@dataclass
class ThroughputExpertConfig:
    """Configuration for throughput-optimized experts with all_to_all operations.

    This config extends the base expert config to support distributing experts
    across multiple devices and using all_to_all operations for dynamic batching.

    Attributes:
        intermediate_size: MLP intermediate dimension
        num_experts: Total number of experts (e.g., 128 for 32 devices × 4 experts)
        hidden_size: Model hidden dimension
        num_experts_per_tok: Number of experts activated per token
        num_devices: Total number of devices in the mesh (e.g., 32 for Galaxy)
        num_experts_per_device: Experts per device (num_experts / num_devices)
        sparsity_block_size: Block size for sparse operations (default: 32)
        swiglu_limit: Clamp limit for SwiGLU activation
        alpha: SwiGLU alpha parameter
    """

    intermediate_size: int
    num_experts: int
    hidden_size: int
    num_experts_per_tok: int
    num_devices: int
    sparsity_block_size: int = 32
    swiglu_limit: float = 7.0
    alpha: float = 1.702
    use_fused_gate_up: bool = True  # If True, fuse w1 and w3 into single matmul

    def __post_init__(self):
        """Validate and compute derived values."""
        if self.num_experts % self.num_devices != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by " f"num_devices ({self.num_devices})"
            )
        self.num_experts_per_device = self.num_experts // self.num_devices

    @classmethod
    def from_hf_config(cls, hf_config, mesh_device) -> "ThroughputExpertConfig":
        """Create config from HuggingFace model config.

        Args:
            hf_config: HuggingFace PretrainedConfig with MoE settings
            mesh_device: TTNN mesh device

        Returns:
            ThroughputExpertConfig instance
        """
        num_devices = mesh_device.get_num_devices()
        return cls(
            intermediate_size=hf_config.intermediate_size,
            num_experts=hf_config.num_local_experts,
            hidden_size=hf_config.hidden_size,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            num_devices=num_devices,
            swiglu_limit=getattr(hf_config, "swiglu_limit", 7.0),
            alpha=getattr(hf_config, "alpha", 1.702),
        )


@dataclass
class AllToAllDispatchConfig:
    """Configuration for ttnn.all_to_all_dispatch operation.

    This operation dispatches tokens to experts across devices based on routing.
    """

    cluster_axis: int = 0
    memory_config: ttnn.MemoryConfig = field(default_factory=lambda: ttnn.L1_MEMORY_CONFIG)
    num_links: int = 4
    topology: ttnn.Topology = field(default_factory=lambda: ttnn.Topology.Ring)
    subdevice_id: Optional[int] = None
    output_concat_dim: Optional[int] = 2  # 2 for tokens on seq_len dim (decode and prefill)

    def as_dict(self):
        """Convert to kwargs dict for ttnn.all_to_all_dispatch."""
        result = {
            "cluster_axis": self.cluster_axis,
            "memory_config": self.memory_config,
            "num_links": self.num_links,
            "topology": self.topology,
            "output_concat_dim": self.output_concat_dim,
        }
        if self.subdevice_id is not None:
            result["subdevice_id"] = self.subdevice_id
        return result


@dataclass
class AllToAllCombineConfig:
    """Configuration for ttnn.all_to_all_combine operation.

    This operation combines expert outputs back to their original token positions.
    """

    cluster_axis: int = 0
    memory_config: ttnn.MemoryConfig = field(default_factory=lambda: ttnn.L1_MEMORY_CONFIG)
    num_links: int = 4
    topology: ttnn.Topology = field(default_factory=lambda: ttnn.Topology.Ring)
    output_shard_dim: int = 2  # 1 for batch dim, 2 for seq_len dim (prefer 2 for decode)

    def as_dict(self):
        """Convert to kwargs dict for ttnn.all_to_all_combine."""
        return {
            "cluster_axis": self.cluster_axis,
            "memory_config": self.memory_config,
            "num_links": self.num_links,
            "topology": self.topology,
            "output_shard_dim": self.output_shard_dim,
        }


@dataclass
class ThroughputProgramConfig:
    """Program configuration for throughput-optimized expert computations.

    Provides matmul program configs for the MLP operations within each expert.
    Supports both legacy batched 1D multicast configs and DRAM sharded configs.

    DRAM sharded matmuls (use_dram_sharded=True):
        Weights are width-sharded across DRAM banks, and each expert is computed
        individually in a loop. This maximizes DRAM read bandwidth utilization
        (from ~30% to ~80%), giving ~2-3x speedup on bandwidth-bound matmuls.
        Based on the pattern used in tt_transformers for decode-mode matmuls.

    Legacy batched 1D multicast (use_dram_sharded=False):
        All experts are batched in a single 4D matmul using
        MatmulMultiCoreReuseMultiCast1DProgramConfig.
    """

    # Whether to use DRAM sharded matmuls (per-expert loop) vs batched 1D multicast
    use_dram_sharded: bool = True

    # =========================================================================
    # Legacy batched 1D multicast parameters (used when use_dram_sharded=False)
    # =========================================================================
    # Core grid sizes for unfused gate/up projections
    gate_up_cores: tuple[int, int] = (5, 6)  # 30 cores - divides N=90 evenly (90/30=3)
    down_cores: tuple[int, int] = (5, 6)  # 30 cores - divides N=90 evenly (90/30=3)

    # Core grid sizes for fused gate/up projection (when use_fused_gate_up=True)
    # If None, defaults to gate_up_cores
    fused_gate_up_cores: tuple[int, int] | None = (5, 6)  # 30 cores - divides N=180 evenly (180/30=6)

    # Matmul parameters for unfused mode
    in0_block_w: int = 15  # K=90 tiles, 90/15 = 6 iterations
    out_subblock_h: int = 1  # M is small (4 tiles)
    out_subblock_w: int = 1  # Conservative for unfused

    # Matmul parameters for fused mode (when use_fused_gate_up=True)
    fused_in0_block_w: int | None = 15  # Same K blocking as unfused
    fused_out_subblock_h: int | None = 1  # M is small
    fused_out_subblock_w: int | None = 2  # Wider output (N=180) benefits from larger subblock

    def __post_init__(self):
        """Validate configuration."""
        self._validate_cores("gate_up_cores", self.gate_up_cores)
        self._validate_cores("down_cores", self.down_cores)
        if self.fused_gate_up_cores is not None:
            self._validate_cores("fused_gate_up_cores", self.fused_gate_up_cores)

    def _validate_cores(self, name: str, cores: tuple[int, int]):
        """Validate core grid dimensions."""
        if not isinstance(cores, tuple) or len(cores) != 2:
            raise ValueError(f"{name} must be a tuple of (x, y), got {cores}")
        core_x, core_y = cores
        if core_x <= 0 or core_y <= 0:
            raise ValueError(f"{name} must have positive dimensions, got {cores}")

    # =========================================================================
    # DRAM sharded matmul helpers (based on tt_transformers model_config.py)
    # =========================================================================

    @staticmethod
    def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
        """Find the largest divisor of n up to max_divisor."""
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1

    @staticmethod
    def _find_grid_k_n(K: int, N: int) -> tuple[int, int]:
        """Find a core grid (rows, cols) where both K and N tile counts divide evenly.

        Prefers larger core counts for maximum parallelism.

        Args:
            K: Number of K tiles
            N: Number of N tiles

        Returns:
            (rows, cols) tuple for the core grid
        """
        max_rows = 8
        max_cols = 8
        max_cores = max_rows * max_cols

        # Find all core counts where both K and N divide evenly, prefer largest
        possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
        possible_cores.sort(reverse=True)

        for cores in possible_cores:
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        raise AssertionError(
            f"Cannot find a grid configuration such that both {K} and {N} tiles "
            f"evenly divide into cores of max size {max_rows}x{max_cols}."
        )

    @staticmethod
    def create_dram_weight_grid(mesh_device) -> tuple[ttnn.CoreRangeSet, int]:
        """Create the DRAM weight grid covering all DRAM cores on the device.

        Args:
            mesh_device: TTNN mesh device

        Returns:
            (dram_weight_grid, dram_cores) tuple
        """
        # Get DRAM grid size from the first device in the mesh
        device = mesh_device.get_devices()[0]
        dram_grid_size = device.dram_grid_size()
        assert dram_grid_size.y == 1, "Current DRAM sharding assumes y dim is 1"

        dram_weight_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                )
            }
        )
        return dram_weight_grid, dram_grid_size.x

    @staticmethod
    def create_dram_sharded_mem_config(
        k: int, n: int, dram_weight_grid: ttnn.CoreRangeSet, dram_cores: int, tile_size: int = 32
    ) -> ttnn.MemoryConfig:
        """Create DRAM width-sharded memory config for weight tensors.

        Weights are sharded along the N dimension across DRAM cores, with each
        core holding the full K dimension and a slice of N.

        Args:
            k: K dimension (input features)
            n: N dimension (output features)
            dram_weight_grid: CoreRangeSet spanning all DRAM cores
            dram_cores: Number of DRAM cores (e.g., 12 for Wormhole)
            tile_size: Tile size (default 32)

        Returns:
            MemoryConfig with WIDTH_SHARDED in DRAM
        """
        padded_n = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
        shard_spec = ttnn.ShardSpec(
            dram_weight_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def get_dram_sharded_config(
        self, m: int, k: int, n: int, num_cores: int | None = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        """Get DRAM sharded matmul program config for a single 2D matmul.

        This config reads weights directly from DRAM shards, maximizing
        DRAM bandwidth utilization for bandwidth-bound matmuls.

        Args:
            m: M dimension (tokens)
            k: K dimension (input features)
            n: N dimension (output features)
            num_cores: Override core count (auto-computed if None)

        Returns:
            MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
        """
        tile_size = 32
        if num_cores is None:
            rows, cols = self._find_grid_k_n(k // tile_size, n // tile_size)
            num_cores = rows * cols
            assert k % (tile_size * num_cores) == 0, (
                f"k must be divisible by tile_size * num_cores: "
                f"{k} % {tile_size * num_cores} != 0"
            )

        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self._find_largest_divisor(k // (tile_size * num_cores)),
            per_core_M=math.ceil(m / tile_size),
            per_core_N=math.ceil(n / (tile_size * num_cores)),
            fused_activation=None,
        )

    # =========================================================================
    # Legacy batched 1D multicast configs (used when use_dram_sharded=False)
    # =========================================================================

    def get_gate_up_config(self, n: int, m: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for gate/up projections (legacy batched mode).

        Args:
            n: Output feature dimension
            m: M dimension (total_tokens) for the matmul

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig
        """
        core_x, core_y = self.gate_up_cores
        n_tiles = math.ceil(n / ttnn.TILE_SIZE)
        m_tiles = math.ceil(m / ttnn.TILE_SIZE)
        per_core_N = n_tiles // (core_x * core_y)

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=self.in0_block_w,
            out_subblock_h=self.out_subblock_h,
            out_subblock_w=self.out_subblock_w,
            out_block_h=max(1, m_tiles),  # Same as per_core_M for 1D mcast
            out_block_w=max(1, per_core_N),  # Same as per_core_N for 1D mcast
            per_core_M=max(1, m_tiles),
            per_core_N=max(1, per_core_N),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_fused_gate_up_config(self, n: int, m: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for FUSED gate/up projection (legacy batched mode).

        This is used when use_fused_gate_up=True, where a single matmul produces
        output of size 2*intermediate_size (twice the size of unfused).

        Args:
            n: Output feature dimension (2*intermediate_size for fused)
            m: M dimension (total_tokens) for the matmul

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig
        """
        # Use fused-specific cores if provided, otherwise default to gate_up_cores
        cores = self.fused_gate_up_cores if self.fused_gate_up_cores is not None else self.gate_up_cores
        core_x, core_y = cores

        # Use fused-specific parameters if provided, otherwise default to unfused parameters
        in0_block_w = self.fused_in0_block_w if self.fused_in0_block_w is not None else self.in0_block_w
        out_subblock_h = self.fused_out_subblock_h if self.fused_out_subblock_h is not None else self.out_subblock_h
        out_subblock_w = self.fused_out_subblock_w if self.fused_out_subblock_w is not None else self.out_subblock_w

        n_tiles = math.ceil(n / ttnn.TILE_SIZE)
        m_tiles = math.ceil(m / ttnn.TILE_SIZE)
        per_core_N = n_tiles // (core_x * core_y)

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=max(1, m_tiles),  # Same as per_core_M for 1D mcast
            out_block_w=max(1, per_core_N),  # Same as per_core_N for 1D mcast
            per_core_M=max(1, m_tiles),
            per_core_N=max(1, per_core_N),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_down_config(self, n: int, m: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for down projection (legacy batched mode).

        Args:
            n: Output feature dimension
            m: M dimension (total_tokens) for the matmul

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig
        """
        core_x, core_y = self.down_cores
        n_tiles = math.ceil(n / ttnn.TILE_SIZE)
        m_tiles = math.ceil(m / ttnn.TILE_SIZE)
        per_core_N = n_tiles // (core_x * core_y)

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=self.in0_block_w,
            out_subblock_h=self.out_subblock_h,
            out_subblock_w=self.out_subblock_w,
            out_block_h=max(1, m_tiles),  # Same as per_core_M for 1D mcast
            out_block_w=max(1, per_core_N),  # Same as per_core_N for 1D mcast
            per_core_M=max(1, m_tiles),
            per_core_N=max(1, per_core_N),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )


def create_expert_mapping_tensors(
    num_devices: int,
    num_experts_per_device: int,
    mesh_device,
    cluster_axis: Optional[int] = None,
    mesh_shape: tuple[int, int] = None,
) -> ttnn.Tensor:
    """Create expert-to-device mapping tensors for all_to_all operations.

    Args:
        num_devices: Total number of devices
        num_experts_per_device: Experts per device
        mesh_device: TTNN mesh device

    Returns:
        Mapping tensor [1, 1, num_experts, num_devices]
    """
    # Create identity matrix showing which device owns which expert
    # Shape: [num_experts, num_devices] where mapping[e, d] = 1 if expert e is on device d
    mapping = (
        torch.eye(num_devices, dtype=torch.int32)
        .repeat_interleave(num_experts_per_device, dim=0)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return ttnn.from_torch(
        mapping,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def create_remap_topk_mask(
    num_dispatch_device_rows: int,
    num_experts: int,
    mesh_device,
) -> ttnn.Tensor:
    """Create mask for remapping top-k expert indices.

    Args:
        num_dispatch_device_rows: Number of device rows for dispatch
        num_experts: Total number of experts
        mesh_device: TTNN mesh device

    Returns:
        Remap mask tensor [1, num_dispatch_device_rows, 1, num_experts]
    """
    mask = torch.ones(
        (1, num_dispatch_device_rows, 1, num_experts),
        dtype=torch.bfloat16,
    )

    return ttnn.from_torch(
        mask,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
