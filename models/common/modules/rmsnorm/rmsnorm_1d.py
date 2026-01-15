# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm1D: RMSNorm for 1D mesh topologies (single-chip, n150, n300, T3K row/column).

Execution paths:
  - Path 1: _decode_local - Local decode (sharded) - always for 1D decode
  - Path 2: _prefill_local - Local prefill (interleaved) - dim <= 4096
  - Path 3: _prefill_1d_distributed - 1D distributed prefill - dim > 4096, Ring topology

Key design:
  - decode_forward always uses local sharded path (no 1D distributed decode)
  - prefill_forward switches between local and distributed based on dim
  - Config provides program_config/memory_config for sharded, None for interleaved
"""

import math
from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.tensor_utils import TILE_SIZE
from models.common.modules.tt_ccl import get_tt_ccl

# =============================================================================
# Constants
# =============================================================================

SHARD_HEIGHT = TILE_SIZE  # Current ttnn.rms_norm requires shard height = single tile


# =============================================================================
# RMSNorm1DConfig
# =============================================================================


@dataclass
class RMSNorm1DConfig:
    """
    Configuration for RMSNorm1D - only fields relevant to 1D mesh topologies.

    Paths:
      - Path 1: Local decode (sharded) - set decode_program_config & decode_memory_config
      - Path 2: Local prefill (interleaved) - no extra config needed
      - Path 3: 1D distributed prefill - requires tt_ccl for Ring all-gather

    Simple usage:
        config = RMSNorm1DConfig(weight)

    Override any field:
        config = RMSNorm1DConfig(weight, eps=1e-6, prefill_distributed=True)
    """

    # Required: weight
    weight: LazyWeight

    # Device and collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: "TT_CCL | None" = None  # type: ignore

    # Normalization settings
    eps: float = 1e-5
    add_unit_offset: bool = False

    # Distributed control (None = auto-detect: True if dim > 4096)
    prefill_distributed: bool | None = None

    # Dimensions
    dim: int | None = None
    max_batch_size: int = 32

    # Local decode configs (Path 1) - sharded
    decode_program_config: ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    decode_memory_config: ttnn.MemoryConfig | None = None

    # Compute kernel config (shared across paths)
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    # Weight settings
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    weight_memory_config: ttnn.MemoryConfig | None = None

    def is_resolved(self) -> bool:
        """Check if all required fields are resolved."""
        required = ["weight", "mesh_device", "dim", "compute_kernel_config"]

        # Multi-device needs CCL for distributed prefill
        num_devices = self.mesh_device.get_num_devices() if self.mesh_device else 1
        if num_devices > 1 and self.prefill_distributed:
            required.append("tt_ccl")

        # Decode always needs sharded config
        required.extend(["decode_program_config", "decode_memory_config"])

        return all(getattr(self, f) is not None for f in required)


# =============================================================================
# RMSNorm1D
# =============================================================================


class RMSNorm1D(LightweightModule):
    """
    RMSNorm for 1D mesh topologies.

    Execution paths (bound at construction):
      - decode_forward -> _decode_local (Path 1) - always local sharded
      - prefill_forward -> _prefill_local (Path 2) or _prefill_1d_distributed (Path 3)

    Simple API:
        norm = RMSNorm1D(weight)

    Power API:
        config = RMSNorm1DConfig(weight, max_batch_size=64)
        norm = RMSNorm1D.from_config(config)
    """

    def __init__(self, weight: LazyWeight, eps: float = 1e-5, add_unit_offset: bool = False):
        """
        Simple API - derives all config from weight.

        Args:
            weight: RMSNorm weight tensor of shape (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)
            eps: Small value for numerical stability (default 1e-5)
            add_unit_offset: Whether to add 1.0 to weights (default False)
        """
        super().__init__()
        self.config = _resolve_1d_config(RMSNorm1DConfig(weight=weight, eps=eps, add_unit_offset=add_unit_offset))
        self._device_weights_loaded = False
        self._bind_forward_methods()

    @classmethod
    def from_config(cls, config: RMSNorm1DConfig):
        """Power API - any level of customization via config."""
        instance = object.__new__(cls)
        super(RMSNorm1D, instance).__init__()
        instance.config = _resolve_1d_config(config)
        instance._device_weights_loaded = False
        instance._bind_forward_methods()
        return instance

    def _bind_forward_methods(self):
        """Bind decode_forward and prefill_forward based on config."""
        cfg = self.config

        # Decode: always local sharded for 1D
        self.decode_forward = self._decode_local

        # Prefill: local or 1D distributed based on dim
        if cfg.prefill_distributed:
            self.prefill_forward = self._prefill_1d_distributed
        else:
            self.prefill_forward = self._prefill_local

    def load_device_weights(self):
        """Load weights to device lazily."""
        if self._device_weights_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device weights!"
        self.weight = self.config.weight.get_device_weight()
        self._device_weights_loaded = True

    # =========================================================================
    # Path 1: Local decode (sharded)
    # =========================================================================

    def _decode_local(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Local decode - sharded RMSNorm.

        Execution: to_sharded -> rms_norm(sharded) with program_config and memory_config
        """
        self.load_device_weights()
        cfg = self.config

        # Convert to sharded memory config
        x = ttnn.to_memory_config(x, memory_config=cfg.decode_memory_config)

        return ttnn.rms_norm(
            x,
            epsilon=cfg.eps,
            weight=self.weight,
            program_config=cfg.decode_program_config,
            memory_config=cfg.decode_memory_config,
            compute_kernel_config=cfg.compute_kernel_config,
        )

    # =========================================================================
    # Path 2: Local prefill (interleaved)
    # =========================================================================

    def _prefill_local(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Local prefill - interleaved RMSNorm for dim <= 4096.

        Execution: rms_norm(interleaved) without program_config
        """
        self.load_device_weights()
        cfg = self.config

        return ttnn.rms_norm(
            x,
            epsilon=cfg.eps,
            weight=self.weight,
            program_config=None,
            memory_config=None,
            compute_kernel_config=cfg.compute_kernel_config,
        )

    # =========================================================================
    # Path 3: 1D distributed prefill (Ring topology, no cluster_axis)
    # =========================================================================

    def _prefill_1d_distributed(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        1D distributed prefill - for dim > 4096.

        Uses Ring topology, no cluster_axis.

        Execution:
          rms_norm_pre_all_gather -> all_gather(Ring) -> rms_norm_post_all_gather
        """
        self.load_device_weights()
        cfg = self.config

        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=cfg.compute_kernel_config, dtype=ttnn.bfloat16)

        # AllGather stats (Ring topology, no cluster_axis)
        tt_stats = ttnn.experimental.all_gather_async(
            tt_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            tt_stats,
            epsilon=cfg.eps,
            weight=self.weight,
            compute_kernel_config=cfg.compute_kernel_config,
        )
        tt_stats.deallocate(True)

        return tt_out

    # =========================================================================
    # Forward dispatcher
    # =========================================================================

    def forward(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """Dispatch to the appropriate forward method based on mode."""
        if mode == "decode":
            return self.decode_forward(x)
        else:
            return self.prefill_forward(x)


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_1d_config(config: RMSNorm1DConfig) -> RMSNorm1DConfig:
    """Resolve config defaults for RMSNorm1D."""

    to_set = {}

    # --- Phase 1: Foundational fields ---

    # Derive dim from weight
    dim = config.dim
    if config.dim is None:
        dim = config.weight.source.shape[-2] * config.weight.source.shape[-1]
        to_set["dim"] = dim

    # Derive mesh_device from weight
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.weight.device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    assert mesh_device is not None, "mesh_device must be available!"

    num_devices = mesh_device.get_num_devices()

    # --- Phase 2: Auto-detect distributed prefill ---

    if config.prefill_distributed is None:
        if num_devices == 1:
            to_set["prefill_distributed"] = False
        elif dim > 4096:
            to_set["prefill_distributed"] = True
        else:
            to_set["prefill_distributed"] = False

    prefill_distributed = to_set.get("prefill_distributed", config.prefill_distributed)

    # Derive tt_ccl (only needed for distributed prefill)
    if config.tt_ccl is None and num_devices > 1 and prefill_distributed:
        to_set["tt_ccl"] = get_tt_ccl(mesh_device)

    # --- Phase 3: Compute kernel config ---

    if config.compute_kernel_config is None:
        to_set["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # --- Phase 4: Local decode configs (Path 1) ---

    tile_size = TILE_SIZE
    tile_padded_batch_rows = tile_size * math.ceil(config.max_batch_size / tile_size)

    if config.decode_program_config is None:
        norm_core_grid = _compute_norm_core_grid(dim)
        to_set["decode_program_config"] = _create_sharded_norm_program_config(
            dim, norm_core_grid, tile_padded_batch_rows, tile_size
        )

    if config.decode_memory_config is None:
        norm_core_grid = _compute_norm_core_grid(dim)
        to_set["decode_memory_config"] = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // norm_core_grid.num_cores),
            norm_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # --- Phase 5: Weight memory config ---

    if config.weight_memory_config is None:
        to_set["weight_memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    # --- Phase 6: Resolve LazyWeight ---

    if num_devices == 1:
        mesh_mapper_config = None
    else:
        mesh_mapper_config = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate()],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        )

    resolved_weight = resolve_lazy_weight(
        config.weight,
        dtype=config.weight_dtype,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=to_set.get("weight_memory_config", config.weight_memory_config),
        mesh_mapper_config=mesh_mapper_config,
    )
    to_set["weight"] = resolved_weight

    return replace(config, **to_set)


# =============================================================================
# Helper functions
# =============================================================================


def _compute_norm_core_grid(dim: int, tile_size: int = TILE_SIZE) -> ttnn.CoreGrid:
    """Compute core grid for RMSNorm that evenly divides dim."""
    n_tiles = dim // tile_size
    rows, cols = _find_grid(n_tiles)
    return ttnn.CoreGrid(x=cols, y=rows)


def _find_grid(n_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    """Find grid dimensions (rows, cols) that evenly divide n_tiles."""
    max_cores = max_rows * max_cols
    target = 32
    possible_cores = [k for k in range(1, max_cores + 1) if n_tiles % k == 0]
    possible_cores.sort(key=lambda x: abs(x - target))

    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    raise AssertionError(f"Cannot find grid for {n_tiles} tiles within {max_rows}x{max_cols}")


def _create_sharded_norm_program_config(
    dim: int, grid: ttnn.CoreGrid, tile_padded_batch_rows: int, tile_size: int = TILE_SIZE
) -> ttnn.LayerNormShardedMultiCoreProgramConfig:
    """Create LayerNormShardedMultiCoreProgramConfig for RMSNorm."""
    block_w = dim // grid.num_cores // tile_size
    subblock_w = 4
    while subblock_w > 0:
        if block_w % subblock_w == 0:
            break
        subblock_w -= 1

    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid.x, grid.y],
        subblock_w=subblock_w,
        block_h=tile_padded_batch_rows // tile_size,
        block_w=block_w,
        inplace=False,
    )
