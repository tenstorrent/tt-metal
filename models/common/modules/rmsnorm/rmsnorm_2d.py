# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm2D: RMSNorm for 2D mesh topologies (TG, Galaxy).

Execution paths:
  - decode_forward - Sharded, Linear topology, cluster_axis=1
  - prefill_forward - Interleaved, Linear topology, cluster_axis=1

Key design:
  - Both decode and prefill are ALWAYS distributed for 2D mesh
  - Uses Linear topology with cluster_axis=1 (gather across columns)
  - Decode uses sharded configs, Prefill uses interleaved
  - Separate weight for distributed path (sharded across columns, replicated across rows)
"""

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

SHARD_HEIGHT = TILE_SIZE


# =============================================================================
# RMSNorm2DConfig
# =============================================================================


@dataclass
class RMSNorm2DConfig:
    """
    Configuration for RMSNorm2D - only fields relevant to 2D mesh topologies.

    Both paths are always distributed with Linear topology and cluster_axis=1.

    Paths:
      - Path 4: 2D distributed decode (sharded) - requires decode_* configs
      - Path 5: 2D distributed prefill (interleaved) - no program_config needed

    Simple usage:
        config = RMSNorm2DConfig(weight, cluster_shape=(4, 8))

    Override any field:
        config = RMSNorm2DConfig(weight, cluster_shape=(4, 8), eps=1e-6)
    """

    # Required: weight and cluster_shape
    weight: LazyWeight
    cluster_shape: tuple[int, int] | None = None  # (rows, cols), e.g. (4, 8) for TG

    # Normalization settings
    eps: float = 1e-5
    add_unit_offset: bool = False

    # Device and collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: "TT_CCL | None" = None  # type: ignore

    # Batch size for decode
    max_batch_size: int = 32

    # Input memory configs (for _load_input_device_tensor_2d)
    decode_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_input_memcfg: ttnn.MemoryConfig | None = None

    # 2D distributed decode configs - sharded
    decode_progcfg: ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    decode_stats_memcfg: ttnn.MemoryConfig | None = None

    # Compute kernel config (only for prefill - decode uses program_config)
    compute_kernel_config_prefill: ttnn.WormholeComputeKernelConfig | None = None

    def is_resolved(self) -> bool:
        """Check if all required fields are resolved."""
        required = [
            "mesh_device",
            "cluster_shape",
            "tt_ccl",
            # Decode sharded configs
            "decode_input_memcfg",
            "decode_progcfg",
            "decode_stats_memcfg",
            # Compute config
            "compute_kernel_config_prefill",
            # Weight (2D always uses distributed/sharded)
            "weight",
        ]
        return all(getattr(self, f) is not None for f in required)


# =============================================================================
# RMSNorm2D
# =============================================================================


class RMSNorm2D(LightweightModule):
    """
    RMSNorm for 2D mesh topologies (TG, Galaxy).

    Both decode and prefill are always distributed with Linear topology.

    Execution paths (bound at construction):
      - decode_forward -> _decode_2d_distributed (Path 4) - sharded
      - prefill_forward -> _prefill_2d_distributed (Path 5) - interleaved

    Simple API:
        norm = RMSNorm2D(weight)

    Power API:
        config = RMSNorm2DConfig(weight, cluster_shape=(4, 8), max_batch_size=64)
        norm = RMSNorm2D.from_config(config)
    """

    def __init__(self, weight: LazyWeight):
        """
        Simple API - derives all config from weight.

        Args:
            weight: RMSNorm weight tensor of shape (dim,) or (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)

        Note: Use from_config() to customize eps, add_unit_offset, cluster_shape, or other settings.
        """
        super().__init__()
        self.config = _resolve_2d_config(RMSNorm2DConfig(weight=weight))
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: RMSNorm2DConfig):
        """Power API - any level of customization via config."""
        instance = object.__new__(cls)
        super(RMSNorm2D, instance).__init__()
        instance.config = _resolve_2d_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        """Load weights to device lazily."""
        if self._device_weights_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device weights!"

        # Load weight (sharded across columns, replicated across rows)
        self.weight = self.config.weight.get_device_weight()

        self._device_weights_loaded = True

    def decode_forward(self, x: "ttnn.Tensor | LazyWeight") -> ttnn.Tensor:
        """
        2D distributed decode - sharded for TG/Galaxy.

        Uses Linear topology, cluster_axis=1, with program_config.

        Execution:
          to_sharded -> rms_norm_pre_all_gather(sharded) -> all_gather(cluster_axis=1)
          -> rms_norm_post_all_gather(sharded)
        """
        self.load_device_weights()
        x = _load_input_device_tensor_2d(x, self.config, mode="decode")
        cfg = self.config

        # Convert to sharded memory config
        x = ttnn.to_memory_config(x, memory_config=cfg.decode_input_memcfg)

        # Run distributed rmsnorm part 1 (sharded)
        tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=cfg.decode_progcfg)

        # All gather stats along cluster axis 1 (columns)
        cluster_axis = 1
        tt_stats = ttnn.experimental.all_gather_async(
            tt_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=1,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Linear,
            memory_config=cfg.decode_stats_memcfg,
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            epsilon=cfg.eps,
            weight=self.weight,
            program_config=cfg.decode_progcfg,
            stats=tt_stats,
        )
        tt_stats.deallocate(True)

        return tt_out

    def prefill_forward(self, x: "ttnn.Tensor | LazyWeight") -> ttnn.Tensor:
        """
        2D distributed prefill - interleaved for TG/Galaxy.

        Uses Linear topology, cluster_axis=1.

        Execution:
          rms_norm_pre_all_gather -> reshape -> all_gather(cluster_axis=1)
          -> rms_norm_post_all_gather
        """
        self.load_device_weights()
        x = _load_input_device_tensor_2d(x, self.config, mode="prefill")
        cfg = self.config

        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            x, compute_kernel_config=cfg.compute_kernel_config_prefill, dtype=ttnn.bfloat16
        )

        # Reshape stats for all_gather (preserve batch dimension for multi-batch prefill)
        batch = x.shape[0]
        padded_shape = (batch, 1, x.shape[-2], 32)
        tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))

        # All gather stats along cluster axis 1 (columns)
        cluster_axis = 1
        tt_stats_gathered = ttnn.experimental.all_gather_async(
            tt_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=1,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        tt_stats.deallocate(True)

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            tt_stats_gathered,
            epsilon=cfg.eps,
            weight=self.weight,
            compute_kernel_config=cfg.compute_kernel_config_prefill,
        )
        tt_stats_gathered.deallocate(True)

        return tt_out

    # =========================================================================
    # Forward dispatcher
    # =========================================================================

    def forward(self, x: "ttnn.Tensor | LazyWeight", mode: str) -> ttnn.Tensor:
        """Dispatch to decode_forward or prefill_forward based on mode."""
        if mode == "decode":
            return self.decode_forward(x)
        else:
            return self.prefill_forward(x)

    # =========================================================================
    # Factory method for TTTv1 compatibility
    # =========================================================================

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
    ):
        """Factory method for backward compatibility with ModelArgs."""
        # Validate TG topology
        cluster_shape = tuple(mesh_device.shape)
        valid_shapes = [(4, 8), (8, 4)]
        if cluster_shape not in valid_shapes:
            raise ValueError(
                f"RMSNorm2D requires Galaxy topology (4x8 or 8x4), got {cluster_shape}. "
                "Use RMSNorm1D for non-Galaxy devices."
            )

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

        # Create LazyWeight (mesh_mapper_config is handled by _resolve_2d_config)
        lazy_weight = LazyWeight(
            source=torch_weight,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_dir_weight_name=(weight_cache_path, weight_name + "_2d") if weight_cache_path else None,
        )

        # Build config (dim derived from weight in _resolve_2d_config)
        config = RMSNorm2DConfig(
            weight=lazy_weight,
            cluster_shape=cluster_shape,
            eps=args.norm_eps,
            add_unit_offset=False,  # Already applied above
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            max_batch_size=args.max_batch_size,
        )

        return cls.from_config(config)


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_2d_config(config: RMSNorm2DConfig) -> RMSNorm2DConfig:
    """Resolve config defaults for RMSNorm2D."""

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

    # Derive cluster_shape from mesh_device if not provided
    cluster_shape = config.cluster_shape
    if cluster_shape is None:
        cluster_shape = tuple(mesh_device.shape)
        to_set["cluster_shape"] = cluster_shape

    assert len(cluster_shape) == 2 and all(
        d > 1 for d in cluster_shape
    ), f"cluster_shape must be 2D with both dims > 1, got {cluster_shape}"

    num_rows, num_cols = cluster_shape

    # Derive tt_ccl (always needed for 2D)
    if config.tt_ccl is None:
        to_set["tt_ccl"] = get_tt_ccl(mesh_device)

    # --- Phase 2: Compute kernel config (prefill only - decode uses program_config) ---

    if config.compute_kernel_config_prefill is None:
        # 2D uses fp32=False, packer_l1=False (from DistributedNorm.ln_cfg)
        to_set["compute_kernel_config_prefill"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    # --- Phase 3: 2D distributed decode configs (Path 4) ---

    hidden_size_per_device = dim // num_cols
    core_grid_ln = (
        min(4, hidden_size_per_device // TILE_SIZE // 8),
        8,
    )
    num_cores_ln = core_grid_ln[0] * core_grid_ln[1]

    if config.decode_input_memcfg is None:
        to_set["decode_input_memcfg"] = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, hidden_size_per_device),
            core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
        )

    if config.decode_progcfg is None:
        to_set["decode_progcfg"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
            subblock_w=(hidden_size_per_device // num_cores_ln) // TILE_SIZE,
            block_h=1,
            block_w=(hidden_size_per_device // num_cores_ln) // TILE_SIZE,
            inplace=False,
        )

    if config.decode_stats_memcfg is None:
        # Stats memory: 32 x (32 * num_cols) where num_cols is the cluster width
        to_set["decode_stats_memcfg"] = ttnn.create_sharded_memory_config(
            shape=[1, 1, 32, 32 * num_cols],
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.WIDTH,
        )

    # --- Phase 4: Resolve weight (sharded across columns, replicated across rows) ---
    # 2D always uses distributed paths
    # Weight shape must be (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT), shard on dim 2

    # Reshape weight to expected shape if needed (handles [dim], [1, dim], etc.)
    assert dim % SHARD_HEIGHT == 0, f"dim must be divisible by SHARD_HEIGHT={SHARD_HEIGHT}, got {dim}"
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

    # Shard on tensor dim 2 across columns (cluster axis 1), replicate across rows (cluster axis 0)
    mesh_mapper_config = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(2)],
        mesh_shape_override=ttnn.MeshShape(list(cluster_shape)),
    )

    resolved_weight = resolve_lazy_weight(
        transformed_weight,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=mesh_mapper_config,
    )
    to_set["weight"] = resolved_weight

    return replace(config, **to_set)


# =============================================================================
# Input Tensor Utilities
# =============================================================================


def _load_input_device_tensor_2d(x: "ttnn.Tensor | LazyWeight", config: RMSNorm2DConfig, mode: str) -> ttnn.Tensor:
    """
    Resolve the input tensor to ttnn tensor if x is a LazyWeight, otherwise return as is.

    For 2D distributed, shards input on last dim across columns (cluster_axis=1).

    Args:
        x: Input tensor (ttnn.Tensor or LazyWeight wrapping torch tensor)
        config: Resolved RMSNorm2DConfig
        mode: "decode" or "prefill"

    Returns:
        ttnn.Tensor ready for the actual forward computation
    """
    assert mode in ("decode", "prefill"), f"mode must be 'decode' or 'prefill', got {mode}"

    if isinstance(x, LazyWeight):
        num_rows, num_cols = config.cluster_shape

        # Determine memory config based on mode
        if mode == "decode":
            mem_cfg = config.decode_input_memcfg or ttnn.DRAM_MEMORY_CONFIG
        else:
            mem_cfg = config.prefill_input_memcfg or ttnn.DRAM_MEMORY_CONFIG

        # Shard input on last dim across columns, replicate across rows
        mesh_mapper_config = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape(num_rows, num_cols),
        )

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
