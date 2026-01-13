# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style RMSNorm module for 2D-topology devices: TG/Galaxy (4x8 or 8x4 mesh).

Single unified RMSNorm2D class with separate forward methods:
  - decode_forward(): For decode mode (sharded distributed norm)
  - prefill_forward(): For prefill mode (interleaved distributed norm)
  - forward(x, mode): Dispatcher that calls the appropriate method

Execution paths (distributed across mesh columns):
  Decode:  to_sharded -> rms_norm_pre_all_gather -> all_gather(axis=1) -> rms_norm_post_all_gather
  Prefill: rms_norm_pre_all_gather -> reshape -> all_gather(axis=1) -> rms_norm_post_all_gather

This module inlines the distributed norm logic from DistributedNorm and ccl.py.
"""

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
# ┌─────────────────────────────────┬────────────────┬───────────────────────────┐
# │            Scenario             │  distributed   │        in_sharded         │
# ├─────────────────────────────────┼────────────────┼───────────────────────────┤
# │ TG decode (layer norms)         │ True           │ True (sharded)            │
# ├─────────────────────────────────┼────────────────┼───────────────────────────┤
# │ TG prefill (layer norms)        │ True           │ False (interleaved)       │
# └─────────────────────────────────┴────────────────┴───────────────────────────┘


@dataclass
class RMSNorm2DConfig:
    """
    Central configuration for RMSNorm2D - for TG/Galaxy 2D mesh topology.

    Simple usage (all defaults):
        config = RMSNorm2DConfig(weight)

    Override any field:
        config = RMSNorm2DConfig(weight, eps=1e-6, max_batch_size=64)

    Full customization:
        config = RMSNorm2DConfig(
            weight,
            mesh_device=custom_device,
            decode_ln_sharded_progcfg=my_program_config,
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

    # Optional: derived from weight if None
    dim: int | None = None

    # Optional: sensible defaults
    max_batch_size: int = 32

    # TG decode (sharded distributed norm) configs
    decode_ln_sharded_input_memcfg: ttnn.MemoryConfig | None = None
    decode_ln_sharded_progcfg: ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    decode_ln_sharded_stats_memcfg: ttnn.MemoryConfig | None = None

    # TG prefill compute kernel
    prefill_compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    # Weight settings
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    weight_memory_config: ttnn.MemoryConfig | None = None

    # Compute kernel config for decode
    decode_compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None
    fp32_dest_acc_en: bool = False  # TG uses fp32_dest_acc_en=False by default

    def is_resolved(self) -> bool:
        """Check if all required fields are resolved."""
        required_fields = [
            "weight",
            "mesh_device",
            "tt_ccl",
            "dim",
            "decode_ln_sharded_input_memcfg",
            "decode_ln_sharded_progcfg",
            "decode_ln_sharded_stats_memcfg",
            "prefill_compute_kernel_config",
            "decode_compute_kernel_config",
            "weight_memory_config",
        ]
        return all(getattr(self, f) is not None for f in required_fields)


# =============================================================================
# RMSNorm2D - Distributed RMSNorm for 2D-topology (TG/Galaxy) devices
# =============================================================================


class RMSNorm2D(LightweightModule):
    """
    RMSNorm for TG/Galaxy devices supporting both decode and prefill modes.

    Simple API (90% of users):
        norm = RMSNorm2D(weight)

    Power API (10% of users) - any level of customization via config:
        config = RMSNorm2DConfig(weight, max_batch_size=64, eps=1e-6)
        norm = RMSNorm2D.from_config(config)

    Execution paths (distributed across mesh columns):
      Decode:  to_sharded -> rms_norm_pre_all_gather -> all_gather(axis=1) -> rms_norm_post_all_gather
      Prefill: rms_norm_pre_all_gather -> reshape -> all_gather(axis=1) -> rms_norm_post_all_gather
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
        self.config = _resolve_rmsnorm2d_config(
            RMSNorm2DConfig(weight=weight, eps=eps, add_unit_offset=add_unit_offset)
        )
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: RMSNorm2DConfig):
        """
        Power API for 10% of users - any level of customization via config.

        Override any subset of fields in RMSNorm2DConfig:
            config = RMSNorm2DConfig(weight, max_batch_size=64)
            norm = RMSNorm2D.from_config(config)
        """
        instance = object.__new__(cls)
        super(RMSNorm2D, instance).__init__()
        instance.config = _resolve_rmsnorm2d_config(config)
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
        Decode forward - sharded distributed RMSNorm for TG.

        Inlined from tt_sharded_distributed_rmsnorm in ccl.py.

        Execution path:
          to_sharded -> rms_norm_pre_all_gather -> all_gather(axis=1) -> rms_norm_post_all_gather
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config

        # Move to sharded memory config
        x = ttnn.to_memory_config(x, memory_config=cfg.decode_ln_sharded_input_memcfg)

        # Run distributed rmsnorm part 1 (sharded)
        tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=cfg.decode_ln_sharded_progcfg)

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
            memory_config=cfg.decode_ln_sharded_stats_memcfg,
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
            program_config=cfg.decode_ln_sharded_progcfg,
            stats=tt_stats,
        )
        tt_stats.deallocate(True)

        return tt_out

    def prefill_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Prefill forward - interleaved distributed RMSNorm for TG.

        Inlined from tt_distributed_rmsnorm in ccl.py.

        Execution path:
          rms_norm_pre_all_gather -> reshape -> all_gather(axis=1) -> rms_norm_post_all_gather
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="prefill")
        cfg = self.config

        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            x, compute_kernel_config=cfg.prefill_compute_kernel_config, dtype=ttnn.bfloat16
        )

        # Reshape stats for all_gather
        padded_shape = (1, 1, x.shape[-2], 32)
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
            compute_kernel_config=cfg.prefill_compute_kernel_config,
        )
        tt_stats_gathered.deallocate(True)

        return tt_out

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
    ):
        """Factory method for backward compatibility with ModelArgs."""
        # Validate TG topology
        cluster_shape = list(mesh_device.shape)
        valid_shapes = [(4, 8), (8, 4)]
        if tuple(cluster_shape) not in valid_shapes:
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

        # Create LazyWeight with 2D sharding (sharded on dim -1 for columns, replicated for rows)
        # Weight is sharded across the 4 columns of the TG mesh
        num_cols = cluster_shape[1]  # 8 for 4x8, 4 for 8x4
        lazy_weight = LazyWeight(
            source=torch_weight,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_dir_weight_name=(weight_cache_path, weight_name + "_2d") if weight_cache_path else None,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-1), ttnn.PlacementReplicate()],
                mesh_shape_override=ttnn.MeshShape(cluster_shape),
            ),
        )

        # Get extra kwargs
        fp32_dest_acc_en = False  # TG uses fp32_dest_acc_en=False
        if hasattr(args, "base_model_name") and args.base_model_name in ("Qwen2.5-7B", "Qwen2.5-VL-7B"):
            fp32_dest_acc_en = False

        # Build config
        config = RMSNorm2DConfig(
            weight=lazy_weight,
            eps=args.norm_eps,
            add_unit_offset=False,  # Already applied above
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dim=dim,
            max_batch_size=args.max_batch_size,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

        return cls.from_config(config)


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_rmsnorm2d_config(config: RMSNorm2DConfig) -> RMSNorm2DConfig:
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

    # Validate 2D mesh topology
    cluster_shape = list(mesh_device.shape)
    valid_shapes = [(4, 8), (8, 4)]
    if tuple(cluster_shape) not in valid_shapes:
        raise ValueError(
            f"RMSNorm2D requires Galaxy topology (4x8 or 8x4), got {cluster_shape}. "
            "Use RMSNorm1D for non-Galaxy devices."
        )

    # Derive tt_ccl
    tt_ccl = config.tt_ccl
    if config.tt_ccl is None:
        tt_ccl = get_tt_ccl(mesh_device)
        to_set["tt_ccl"] = tt_ccl

    assert tt_ccl is not None, "tt_ccl must be available at this point!"

    # --- Phase 2: TG decode configs (from DistributedNorm.__init__) ---

    # Core grid for layer norm: dividing by 4 (num_rows of TG) and 8 (num_cores_per_row) and 32 (tile size)
    num_cols = cluster_shape[1]  # 8 for 4x8, 4 for 8x4
    hidden_size_per_device = dim // num_cols
    core_grid_ln = (
        min(4, hidden_size_per_device // TILE_SIZE // 8),
        8,
    )
    num_cores_ln = core_grid_ln[0] * core_grid_ln[1]

    if config.decode_ln_sharded_input_memcfg is None:
        to_set["decode_ln_sharded_input_memcfg"] = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, hidden_size_per_device),
            core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
        )

    if config.decode_ln_sharded_progcfg is None:
        to_set["decode_ln_sharded_progcfg"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
            subblock_w=(hidden_size_per_device // num_cores_ln) // TILE_SIZE,
            block_h=1,
            block_w=(hidden_size_per_device // num_cores_ln) // TILE_SIZE,
            inplace=False,
        )

    if config.decode_ln_sharded_stats_memcfg is None:
        # Stats memory: 32 x (32 * num_cols) where num_cols is the cluster width
        to_set["decode_ln_sharded_stats_memcfg"] = ttnn.create_sharded_memory_config(
            shape=[1, 1, 32, 32 * num_cols],
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.WIDTH,
        )

    # --- Phase 3: Compute kernel configs ---

    if config.decode_compute_kernel_config is None:
        to_set["decode_compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=config.fp32_dest_acc_en,
            packer_l1_acc=False,
        )

    if config.prefill_compute_kernel_config is None:
        to_set["prefill_compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=config.fp32_dest_acc_en,
            packer_l1_acc=False,
        )

    # Weight memory config
    if config.weight_memory_config is None:
        to_set["weight_memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    # --- Phase 4: Resolve LazyWeight with 2D sharding ---

    # Weight is sharded across columns (axis 1) and replicated across rows (axis 0)
    mesh_mapper_config = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1), ttnn.PlacementReplicate()],
        mesh_shape_override=ttnn.MeshShape(cluster_shape),
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


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: RMSNorm2DConfig, mode: str) -> ttnn.Tensor:
    """Load input tensor to device, handling LazyWeight if needed."""
    if isinstance(x, LazyWeight):
        resolved = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return resolved.get_device_weight()
    return x
