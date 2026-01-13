# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style RMSNorm module for 1D-topology devices: N150 (1x1), N300 (1x2), T3K (1x8).

Single unified RMSNorm1D class with separate forward methods:
  - decode_forward(): For decode mode (sharded input/output)
  - prefill_forward(): For prefill mode (interleaved input/output)
  - forward(x, mode): Dispatcher that calls the appropriate method

Execution paths:
  Decode:  rms_norm(sharded) with sharded program_config and output_memcfg
  Prefill: rms_norm(interleaved) with no program_config
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.tensor_utils import TILE_SIZE
from models.common.modules.tt_ccl import TT_CCL, get_tt_ccl

# =============================================================================
# Constants
# =============================================================================

SHARD_HEIGHT = TILE_SIZE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


# =============================================================================
# Top-level config dataclass
# =============================================================================

# from TTTv1 uses:
# ┌──────────────────────────────┬────────────────┬───────────────────────────┐
# │           Scenario           │  distributed   │        in_sharded         │
# ├──────────────────────────────┼────────────────┼───────────────────────────┤
# │ Non-TG decode (layer norms)  │ False          │ True (sharded)            │
# ├──────────────────────────────┼────────────────┼───────────────────────────┤
# │ Non-TG prefill (layer norms) │ False or True* │ False (non-sharded)       │
# ├──────────────────────────────┼────────────────┼───────────────────────────┤
# │ Q/K norms (all modes)        │ False          │ False (no sharded config) │
# └──────────────────────────────┴────────────────┴───────────────────────────┘


@dataclass
class RMSNorm1DConfig:
    """
    Central configuration for RMSNorm1D - the single source of truth for all settings.

    Simple usage (all defaults):
        config = RMSNorm1DConfig(weight)

    Override any field:
        config = RMSNorm1DConfig(weight, eps=1e-6, max_batch_size=64)

    Full customization:
        config = RMSNorm1DConfig(
            weight,
            mesh_device=custom_device,
            decode_sharded_program_config=my_program_config,
            ...
        )
    """

    # Required: weight (LazyWeight)
    weight: LazyWeight

    # Normalization settings
    eps: float = 1e-5
    add_unit_offset: bool = False

    # Optional: device and collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: TT_CCL | None = None
    topology: Optional[ttnn.Topology] = None  # None = auto-detect

    # Optional: derived from weight if None
    dim: int | None = None

    # Optional: sensible defaults
    max_batch_size: int = 32

    # Optional: power-user overrides for decode (sharded) mode
    decode_sharded_program_config: ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    decode_sharded_output_memcfg: ttnn.MemoryConfig | None = None

    # Optional: power-user overrides for prefill (interleaved) mode
    prefill_output_memcfg: ttnn.MemoryConfig | None = None

    # Weight settings
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    weight_memory_config: ttnn.MemoryConfig | None = None

    # Compute kernel config
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None
    fp32_dest_acc_en: bool = True

    def is_resolved(self) -> bool:
        """Check if all required fields are resolved."""
        # For single device, topology is optional (no CCL ops needed)
        optional = set()
        if self.mesh_device and self.mesh_device.get_num_devices() == 1:
            optional.add("topology")
            optional.add("tt_ccl")
        # prefill_output_memcfg is optional (defaults to None for interleaved)
        optional.add("prefill_output_memcfg")

        required_fields = [f for f in self.__dataclass_fields__ if f not in optional]
        return all(getattr(self, f) is not None for f in required_fields)


# =============================================================================
# RMSNorm1D - Unified RMSNorm for 1D-topology devices with decode and prefill modes
# =============================================================================


class RMSNorm1D(LightweightModule):
    """
    RMSNorm for non-TG devices supporting both decode and prefill modes.

    Simple API (90% of users):
        norm = RMSNorm1D(weight)

    Power API (10% of users) - any level of customization via config:
        config = RMSNorm1DConfig(weight, max_batch_size=64, eps=1e-6)
        norm = RMSNorm1D.from_config(config)

    Execution paths:
      Decode:  rms_norm(sharded) with program_config and memory_config
      Prefill: rms_norm(interleaved) without program_config
    """

    def __init__(self, weight: LazyWeight, eps: float = 1e-5, add_unit_offset: bool = False):
        """
        Simple API for 90% of users - derives all config from weight.

        Args:
            weight: RMSNorm weight tensor of shape (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)
            eps: Small value for numerical stability (default 1e-5)
            add_unit_offset: Whether to add 1.0 to weights (default False)

        The mesh_device is derived from weight.device(). tt_ccl is created/cached automatically.
        All other settings use sensible defaults.
        """
        super().__init__()
        self.config = _resolve_rmsnorm1d_config(
            RMSNorm1DConfig(weight=weight, eps=eps, add_unit_offset=add_unit_offset)
        )
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: RMSNorm1DConfig):
        """
        Power API for 10% of users - any level of customization via config.

        Override any subset of fields in RMSNorm1DConfig:
            config = RMSNorm1DConfig(weight, max_batch_size=64)
            norm = RMSNorm1D.from_config(config)

        Full customization:
            config = RMSNorm1DConfig(
                weight,
                mesh_device=custom_device,
                decode_sharded_program_config=my_program_config,
                ...
            )
            norm = RMSNorm1D.from_config(config)
        """
        instance = object.__new__(cls)
        super(RMSNorm1D, instance).__init__()
        instance.config = _resolve_rmsnorm1d_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        if self._device_weights_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device weights!"
        self.weight = self.config.weight.get_device_weight()
        self._device_weights_loaded = True

    def decode_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Decode forward - sharded RMSNorm.

        Execution path:
          to_sharded -> rms_norm(sharded) with sharded program_config and output_memcfg
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config

        # Convert to sharded memory config (input and output have same shape for RMSNorm)
        x = ttnn.to_memory_config(x, memory_config=cfg.decode_sharded_output_memcfg)

        return ttnn.rms_norm(
            x,
            epsilon=cfg.eps,
            weight=self.weight,
            program_config=cfg.decode_sharded_program_config,
            memory_config=cfg.decode_sharded_output_memcfg,
            compute_kernel_config=cfg.compute_kernel_config,
        )

    def prefill_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Prefill forward - NO if-else, fully flattened.

        Execution path:
          rms_norm(interleaved) without program_config
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="prefill")
        cfg = self.config

        out = ttnn.rms_norm(
            x,
            epsilon=cfg.eps,
            weight=self.weight,
            program_config=None,
            memory_config=cfg.prefill_output_memcfg,
            compute_kernel_config=cfg.compute_kernel_config,
        )

        return out

    def forward(self, x: ttnn.Tensor | LazyWeight, mode: str) -> ttnn.Tensor:
        """Dispatch to the appropriate forward method based on mode."""
        if mode == "decode":
            return self.decode_forward(x)
        else:
            return self.prefill_forward(x)

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
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementReplicate()],
                mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
            )
            if mesh_device.get_num_devices() > 1
            else None,
        )

        # Get extra kwargs (e.g., fp32_dest_acc_en for Qwen models)
        fp32_dest_acc_en = True
        if hasattr(args, "base_model_name") and args.base_model_name in ("Qwen2.5-7B", "Qwen2.5-VL-7B"):
            fp32_dest_acc_en = False

        # Build config
        config = RMSNorm1DConfig(
            weight=lazy_weight,
            eps=args.norm_eps,
            add_unit_offset=False,  # Already applied above
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=args.ccl_topology() if hasattr(args, "ccl_topology") else None,
            dim=dim,
            max_batch_size=args.max_batch_size,
            decode_sharded_program_config=sharded_program_config,
            decode_sharded_output_memcfg=sharded_output_config,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

        return cls.from_config(config)


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_rmsnorm1d_config(config: RMSNorm1DConfig) -> RMSNorm1DConfig:
    """Materialize the config to known good defaults using replace pattern."""

    to_set = {}

    # --- Phase 1: Foundational fields (order matters due to dependencies) ---

    # Derive dim from weight
    dim = config.dim
    if config.dim is None:
        # Weight shape is (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)
        dim = config.weight.source.shape[-2] * config.weight.source.shape[-1]
        to_set["dim"] = dim

    # Derive mesh_device from weight, fall back to default
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.weight.device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    assert mesh_device is not None, "mesh_device must be available at this point!"

    # Derive tt_ccl (only needed for multi-device)
    tt_ccl = config.tt_ccl
    if config.tt_ccl is None and mesh_device.get_num_devices() > 1:
        tt_ccl = get_tt_ccl(mesh_device)
        to_set["tt_ccl"] = tt_ccl

    # Auto-detect topology (only needed for multi-device)
    topology = config.topology
    if config.topology is None and mesh_device.get_num_devices() > 1:
        topology = _default_topology(mesh_device)
        to_set["topology"] = topology

    # --- Phase 2: Derived fields ---

    tile_size = TILE_SIZE
    tile_padded_batch_rows = tile_size * math.ceil(config.max_batch_size / tile_size)

    # Compute kernel config
    if config.compute_kernel_config is None:
        to_set["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=config.fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    # Weight memory config
    if config.weight_memory_config is None:
        to_set["weight_memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    # --- Phase 3: Decode (sharded) program configs ---

    if config.decode_sharded_program_config is None:
        # Compute core grid for sharded norm
        norm_core_grid = _compute_norm_core_grid(dim)
        to_set["decode_sharded_program_config"] = _create_sharded_norm_program_config(
            dim, norm_core_grid, tile_padded_batch_rows, tile_size
        )

    if config.decode_sharded_output_memcfg is None:
        norm_core_grid = _compute_norm_core_grid(dim)
        to_set["decode_sharded_output_memcfg"] = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // norm_core_grid.num_cores),
            norm_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # --- Phase 4: Resolve LazyWeight ---

    # Build mesh mapper config if multi-device
    mesh_mapper_config = None
    if mesh_device.get_num_devices() > 1:
        mesh_mapper_config = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate()],
            mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
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
    """
    Compute core grid for RMSNorm that evenly divides dim.

    Returns a CoreGrid that can shard the dim across cores.
    Uses the same algorithm as model_config.py: find_grid.
    """
    n_tiles = dim // tile_size
    rows, cols = _find_grid(n_tiles)
    return ttnn.CoreGrid(x=cols, y=rows)


def _find_grid(n_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    """Find grid dimensions (rows, cols) that evenly divide n_tiles.

    Targets ~32 cores for optimal performance, matching model_config.py logic.
    """
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
    """
    Create LayerNormShardedMultiCoreProgramConfig for RMSNorm.

    This is adapted from model_config.py:create_sharded_norm_config.
    """
    block_w = dim // grid.num_cores // tile_size
    # Find largest value <= 4 that evenly divides block_w
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


def _default_topology(mesh_device: ttnn.MeshDevice) -> ttnn.Topology:
    """Auto-detect Ring or Linear topology based on mesh shape."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 1:
        return ttnn.Topology.Ring  # Doesn't matter for single device
    # For 1D mesh (1x2, 1x4, 1x8), use Ring topology
    return ttnn.Topology.Ring


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: RMSNorm1DConfig, mode: str) -> ttnn.Tensor:
    """Load input tensor to device, handling LazyWeight if needed."""
    if isinstance(x, LazyWeight):
        # Resolve LazyWeight to device tensor
        resolved = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return resolved.get_device_weight()
    return x
