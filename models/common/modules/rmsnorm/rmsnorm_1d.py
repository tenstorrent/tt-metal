# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm1D: RMSNorm for 1D mesh topologies (single-chip, n150, n300, T3K row/column).

Execution paths:
  - Path 1: _decode_local - Local decode (sharded or interleaved) - always for 1D decode
  - Path 2: _prefill_local - Local prefill (interleaved) - single-device or prefill_distributed=False
  - Path 3: _prefill_1d_distributed - 1D distributed prefill - multi-device, Ring topology

Key design:
  - decode_forward always uses local sharded path (no 1D distributed decode)
  - prefill_forward switches between local and distributed based on prefill_distributed config
  - Config provides program_config/memory_config for sharded, None for interleaved
  - from_model_args() sets prefill_distributed=True only when dim > 4096 for backward compatibility
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.tensor_utils import TILE_SIZE

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

    # Distributed control (None = auto-detect: True for multi-device, False for single-device)
    prefill_distributed: bool | None = None

    # Batch size for decode
    max_batch_size: int = 32

    # Local decode configs (Path 1)
    decode_in_sharded: bool = True  # If True, use sharded decode; if False, use interleaved
    decode_out_sharded: bool = True  # If True, output stays sharded (only valid if in_sharded=True)
    decode_program_config: ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    decode_memory_config: ttnn.MemoryConfig | None = None

    # Input memory configs (optional, for prepare_input_tensor)
    decode_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_input_memcfg: ttnn.MemoryConfig | None = None

    # Compute kernel config (shared across paths)
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    # Internal: distributed weight LazyWeight (sharded across devices for Path 3)
    # Note: Uses LazyWeight with explicit mesh_mapper_config to shard weights correctly.
    # The get_device_weight() call respects the mesh_mapper_config for proper distribution.
    _weight_distributed: LazyWeight | None = None

    def is_resolved(self) -> bool:
        """Check if all required fields are resolved."""
        required = ["weight", "mesh_device", "compute_kernel_config"]

        # Multi-device distributed prefill needs CCL and distributed weight
        num_devices = self.mesh_device.get_num_devices() if self.mesh_device else 1
        if num_devices > 1 and self.prefill_distributed:
            required.extend(["tt_ccl", "_weight_distributed"])

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

    def __init__(self, weight: LazyWeight):
        """
        Simple API - derives all config from weight.

        Args:
            weight: RMSNorm weight tensor of shape (dim,) or (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)

        Note: Use from_config() to customize eps, add_unit_offset, or other settings.
        """
        super().__init__()
        self.config = _resolve_1d_config(RMSNorm1DConfig(weight=weight))
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

        # Decode: sharded or interleaved based on config
        if cfg.decode_in_sharded:
            self.decode_forward = self._decode_local_sharded
        else:
            self.decode_forward = self._decode_local_interleaved

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

        cfg = self.config

        # Always load replicated weight (decode always needs it)
        self.weight = cfg.weight.get_device_weight()

        # Load sharded weight if distributed prefill is enabled
        if cfg.prefill_distributed and cfg._weight_distributed is not None:
            self.weight_distributed = cfg._weight_distributed.get_device_weight()

        self._device_weights_loaded = True

    # =========================================================================
    # Path 1a: Local decode - sharded
    # =========================================================================

    def _decode_local_sharded(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Sharded decode - standard LLM decode path.

        Execution: to_sharded -> rms_norm(sharded) with program_config and memory_config
        Output: sharded if decode_out_sharded=True, interleaved otherwise
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config

        assert cfg.decode_in_sharded, "decode_in_sharded must be True for sharded decode"

        x = ttnn.to_memory_config(x, memory_config=cfg.decode_memory_config)

        x = ttnn.rms_norm(
            x,
            epsilon=cfg.eps,
            weight=self.weight,
            program_config=cfg.decode_program_config,
            memory_config=cfg.decode_memory_config,
            compute_kernel_config=cfg.compute_kernel_config,
        )

        if not cfg.decode_out_sharded:
            x = ttnn.sharded_to_interleaved(x)

        return x

    # =========================================================================
    # Path 1b: Local decode - interleaved
    # =========================================================================

    def _decode_local_interleaved(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Interleaved decode - for vision encoder decode.

        Execution: rms_norm with program_config=None, memory_config=None
        Output: always interleaved (decode_out_sharded must be False)
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config

        assert not cfg.decode_in_sharded, "decode_in_sharded must be False for interleaved decode"
        assert not cfg.decode_out_sharded, "decode_out_sharded must be False when decode_in_sharded=False"

        return ttnn.rms_norm(
            x,
            epsilon=cfg.eps,
            weight=self.weight,
            program_config=None,
            memory_config=None,
            compute_kernel_config=cfg.compute_kernel_config,
        )

    # =========================================================================
    # Path 2: Local prefill (interleaved)
    # =========================================================================

    def _prefill_local(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Local prefill - interleaved RMSNorm when prefill_distributed=False.

        Execution: rms_norm(interleaved) without program_config
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="prefill")
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

    def _prefill_1d_distributed(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        1D distributed prefill - when prefill_distributed=True.

        Uses Ring topology, no cluster_axis.

        Execution:
          rms_norm_pre_all_gather -> all_gather(Ring) -> rms_norm_post_all_gather
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="prefill")
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

        # Run distributed rmsnorm part 2 (uses sharded weight)
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            tt_stats,
            epsilon=cfg.eps,
            weight=self.weight_distributed,
            compute_kernel_config=cfg.compute_kernel_config,
        )
        tt_stats.deallocate(True)

        return tt_out

    # =========================================================================
    # Forward dispatcher
    # =========================================================================

    def forward(self, x: "ttnn.Tensor | LazyWeight", mode: str) -> ttnn.Tensor:
        """
        Dispatch to the appropriate forward method based on mode.

        Args:
            x: Input tensor (ttnn.Tensor or LazyWeight wrapping torch tensor)
            mode: "decode" or "prefill"

        Note: Sharding behavior for decode is controlled via config:
            - decode_in_sharded: True for sharded (LLM), False for interleaved (vision)
            - decode_out_sharded: True to keep output sharded, False to convert to interleaved
        """
        if mode == "decode":
            return self.decode_forward(x)
        else:
            return self.prefill_forward(x)

    # [INFO] this is the entry point for TTTv1 model_config.py and will retire with TTTv1
    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num: int,
        weight_key: str,
        state_dict_prefix: Optional[str] = None,
        sharded_program_config=None,
        sharded_output_config=None,
    ):
        """Factory method for backward compatibility with ModelArgs."""

        if args.is_galaxy:
            raise ValueError("RMSNorm1D cannot be used for Galaxy devices. Use RMSNorm2D instead.")

        # Build weight name from state_dict_prefix and weight_key
        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
        else:
            if layer_num is None:
                weight_name = f"{weight_key}.weight"
            else:
                weight_name = f"layers.{layer_num}.{weight_key}.weight"

        # Transform weight to expected shape: (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)
        dim = args.dim
        torch_weight = (
            state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )

        # Add offset before creating LazyWeight
        if args.rms_norm_add_unit_offset:
            torch_weight = torch_weight + 1.0

        # Create LazyWeight
        lazy_weight = LazyWeight(
            source=torch_weight,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_dir_weight_name=(weight_cache_path, weight_name) if weight_cache_path else None,
            mesh_mapper_config=(
                ttnn.MeshMapperConfig(
                    placements=[ttnn.PlacementReplicate()],
                    mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
                )
                if mesh_device.get_num_devices() > 1
                else None
            ),
        )

        # Get compute kernel config (e.g., fp32_dest_acc_en for Qwen models)
        fp32_dest_acc_en = True
        if hasattr(args, "base_model_name") and args.base_model_name in ("Qwen2.5-7B", "Qwen2.5-VL-7B"):
            fp32_dest_acc_en = False

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=False,
        )

        # Determine prefill_distributed based on dim and num_devices
        # This preserves backward compatibility with TTTv1's is_distributed_norm() logic
        num_devices = mesh_device.get_num_devices()
        prefill_distributed = num_devices > 1 and dim > 4096

        # Build config
        config = RMSNorm1DConfig(
            weight=lazy_weight,
            eps=args.norm_eps,
            add_unit_offset=False,  # Already applied above
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            max_batch_size=args.max_batch_size,
            prefill_distributed=prefill_distributed,
            decode_program_config=sharded_program_config,
            decode_memory_config=sharded_output_config,
            compute_kernel_config=compute_kernel_config,
        )

        return cls.from_config(config)


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_1d_config(config: RMSNorm1DConfig) -> RMSNorm1DConfig:
    """Resolve config defaults for RMSNorm1D."""

    to_set = {}

    # --- Phase 1: Foundational fields ---

    # Derive dim from weight (works for any shape: [dim], [1, dim], [1, 1, dim//32, 32], etc.)
    dim = config.weight.source.numel()

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
    # Weight shape is (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT), shard on dim 2
    # Need at least num_devices tiles to shard across devices
    num_weight_tiles = dim // SHARD_HEIGHT
    can_shard_weights = num_weight_tiles >= num_devices

    if config.prefill_distributed is None:
        if num_devices == 1:
            to_set["prefill_distributed"] = False
        elif not can_shard_weights:
            # Can't shard weight tiles across num_devices, use local prefill
            to_set["prefill_distributed"] = False
        else:
            to_set["prefill_distributed"] = True

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

    # --- Phase 5: Resolve LazyWeight (replicated for decode and local prefill) ---

    if num_devices == 1:
        mesh_mapper_config_replicated = None
        mesh_mapper_config_sharded = None
    else:
        # For 1D mesh topology, use 2D placement: [Replicate, Replicate]
        # This replicates the weight to all devices
        mesh_mapper_config_replicated = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
            mesh_shape_override=ttnn.MeshShape(1, num_devices),
        )
        # For distributed prefill, shard on tensor dim 2 (hidden tiles)
        # Weight shape is (1, 1, dim // 32, 32), so dim 2 is the tiles dimension
        mesh_mapper_config_sharded = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(2)],
            mesh_shape_override=ttnn.MeshShape(1, num_devices),
        )

    # Reshape weight to expected shape (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)
    assert dim % SHARD_HEIGHT == 0, "dim must be divisible by SHARD_HEIGHT"
    expected_shape = (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)
    if config.weight.source.shape != expected_shape:
        transformed_source = config.weight.source.reshape(*expected_shape)
    else:
        transformed_source = config.weight.source

    # Apply unit offset if requested (e.g., for Gemma models)
    if config.add_unit_offset:
        transformed_source = transformed_source + 1.0

    # Create a new LazyWeight with the transformed source
    transformed_weight = replace(config.weight, source=transformed_source)

    resolved_weight = resolve_lazy_weight(
        transformed_weight,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=mesh_mapper_config_replicated,
    )
    to_set["weight"] = resolved_weight

    # --- Phase 7: Create distributed weight LazyWeight (for Path 3) ---
    #
    # Note: We create a LazyWeight with explicit mesh_mapper_config (ShardTensor2dMesh)
    # to shard weights across devices. The get_device_weight() call at line 182 respects
    # this config for proper distributed RMSNorm alignment.

    if prefill_distributed and num_devices > 1:
        # Create LazyWeight with sharded mesh_mapper_config
        # The mesh_mapper_config with ShardTensor2dMesh ensures proper weight distribution
        weight_distributed = replace(
            transformed_weight,
            dtype=ttnn.bfloat16,  # RMSNorm weights are always bfloat16
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=mesh_mapper_config_sharded,
        )
        to_set["_weight_distributed"] = weight_distributed

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
    target = 32  # based on TTTv1 tuned configs
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


# =============================================================================
# Input Tensor Utilities
# =============================================================================


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: RMSNorm1DConfig, mode: str) -> ttnn.Tensor:
    """
    Resolve the input tensor to ttnn tensor if x is a LazyWeight, otherwise return as is.

    For distributed prefill, uses sharded mesh_mapper_config to shard input across devices.
    For decode/non-distributed prefill, replicates input (mesh_mapper_config=None).

    Args:
        x: Input tensor (ttnn.Tensor or LazyWeight wrapping torch tensor)
        config: Resolved RMSNorm1DConfig
        mode: "decode" or "prefill"

    Returns:
        ttnn.Tensor ready for the actual forward computation
    """
    assert mode in ("decode", "prefill"), f"mode must be 'decode' or 'prefill', got {mode}"

    if isinstance(x, LazyWeight):
        # Determine memory config based on mode (default to DRAM if not specified)
        if mode == "decode":
            mem_cfg = config.decode_input_memcfg or ttnn.DRAM_MEMORY_CONFIG
        else:
            mem_cfg = config.prefill_input_memcfg or ttnn.DRAM_MEMORY_CONFIG

        # For distributed prefill, shard input on last dim; otherwise replicate
        is_distributed = mode == "prefill" and config.prefill_distributed
        if is_distributed:
            num_devices = config.mesh_device.get_num_devices()
            mesh_mapper_config = ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
                mesh_shape_override=ttnn.MeshShape(1, num_devices),
            )
        else:
            mesh_mapper_config = None

        resolved_x = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            memory_config=mem_cfg,
            mesh_mapper_config=mesh_mapper_config,
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    # Already a ttnn.Tensor - return as is
    assert isinstance(x, ttnn.Tensor), f"x must be ttnn.Tensor or LazyWeight, got {type(x)}"
    return x
