# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm2D: distributed RMSNorm for the Wormhole Galaxy (8, 4) mesh.

Execution paths:
  - decode_forward - Sharded, Linear topology, cluster_axis=1
  - prefill_forward - Interleaved, Linear topology, cluster_axis=1

Key design:
  - Model-width norms are distributed; 128-wide Q/K norms are head-local
  - Uses Linear topology with cluster_axis=1 (gather across columns)
  - Decode uses sharded configs, Prefill uses interleaved
  - Separate weight for distributed path (sharded across columns, replicated across rows)
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.tensor_utils import TILE_SIZE

# =============================================================================
# Constants
# =============================================================================

SHARD_HEIGHT = TILE_SIZE
WH_GALAXY_MESH_SHAPE = (8, 4)


class RMSNorm2DResidualPolicy(Enum):
    """Static ownership policy for residual accumulation."""

    NONE = "none"
    FUSED_DECODE = "fused_decode"


class RMSNorm2DGeometry(Enum):
    """Immutable normalization geometry selected during host resolution."""

    DISTRIBUTED = "distributed"
    HEAD_LOCAL = "head_local"


# =============================================================================
# RMSNorm2DConfig
# =============================================================================


@dataclass(frozen=True)
class RMSNorm2DConfig:
    """
    Configuration for RMSNorm2D - only fields relevant to 2D mesh topologies.

    Distributed paths use Linear topology and cluster_axis=1. Head-local paths
    use the regular RMSNorm operation without a collective.

    Paths:
      - Path 4: 2D distributed decode (sharded) - requires decode_* configs
      - Path 5: 2D distributed prefill (interleaved) - no program_config needed

    Simple usage:
        config = RMSNorm2DConfig(weight, cluster_shape=(8, 4))

    Override any field:
        config = RMSNorm2DConfig(weight, cluster_shape=(8, 4), eps=1e-6)
    """

    # Required: weight and cluster_shape
    weight: LazyWeight
    cluster_shape: tuple[int, int] | None = None

    # Normalization settings
    eps: float = 1e-5
    add_unit_offset: bool = False
    residual_policy: RMSNorm2DResidualPolicy = RMSNorm2DResidualPolicy.NONE
    geometry: RMSNorm2DGeometry | None = None

    # Device and collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: Any = None
    decode_ccl_context: Any = None
    prefill_ccl_context: Any = None
    decode_prefetch_context: Any = None
    prefill_prefetch_context: Any = None
    decode_all_gather_resources: Any = None
    prefill_all_gather_resources: Any = None
    collective_resource_selector: Callable[[Any, str, int, Any], Any] | None = None
    ccl_chunks_per_sync: int = 10
    ccl_num_workers_per_link: int = 2
    ccl_num_buffers_per_channel: int = 2

    # Batch size for decode
    max_batch_size: int = 32

    # Input memory configs (for _load_input_device_tensor_2d)
    decode_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_input_memcfg: ttnn.MemoryConfig | None = None
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    decode_output_memcfg: ttnn.MemoryConfig | None = None
    prefill_output_memcfg: ttnn.MemoryConfig | None = None

    # 2D distributed decode configs - sharded
    decode_progcfg: ttnn.LayerNormShardedMultiCoreProgramConfig | None = None
    decode_stats_memcfg: ttnn.MemoryConfig | None = None

    #: Which cores a ``HEAD_LOCAL`` decode norm runs its kernel on. Optional,
    #: and only meaningful for that geometry; see ``_decode_head_local``. The
    #: caller names the cores because only the caller knows which ones its
    #: sub-device owns; the shard shape is derived from the tensor, because the
    #: created-head padding is the op's business and not the caller's.
    decode_compute_cores: ttnn.CoreRangeSet | None = None

    # Compute kernel config (only for prefill - decode uses program_config)
    compute_kernel_config_prefill: ttnn.WormholeComputeKernelConfig | None = None

    def is_resolved(self) -> bool:
        """Check if all required fields are resolved."""
        required = [
            "mesh_device",
            "cluster_shape",
            "prefill_input_memcfg",
            "prefill_residual_memcfg",
            "prefill_output_memcfg",
            "compute_kernel_config_prefill",
            "weight",
            "geometry",
        ]
        if self.geometry is RMSNorm2DGeometry.DISTRIBUTED:
            required.extend(
                [
                    "tt_ccl",
                    "decode_ccl_context",
                    "prefill_ccl_context",
                    "decode_all_gather_resources",
                    "prefill_all_gather_resources",
                    "decode_input_memcfg",
                    "decode_progcfg",
                    "decode_stats_memcfg",
                    "decode_residual_memcfg",
                    "decode_output_memcfg",
                ]
            )
            if self.collective_resource_selector is not None:
                required.remove("decode_all_gather_resources")
                required.remove("prefill_all_gather_resources")
        return all(getattr(self, f) is not None for f in required)


# =============================================================================
# RMSNorm2D
# =============================================================================


class RMSNorm2D(LightweightModule):
    """
    RMSNorm for the Wormhole Galaxy (8, 4) mesh.

    Simple API:
        norm = RMSNorm2D(weight, tt_ccl=galaxy_ccl)

    Power API:
        config = RMSNorm2DConfig(weight, cluster_shape=(8, 4), max_batch_size=32)
        norm = RMSNorm2D.from_config(config)
    """

    def __init__(
        self,
        weight: LazyWeight,
        *,
        tt_ccl: Any = None,
        mesh_device: ttnn.MeshDevice | None = None,
        geometry: RMSNorm2DGeometry | None = None,
    ):
        """
        Simple API - derives all config from weight.

        Args:
            weight: RMSNorm weight tensor of shape (dim,) or (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)

        Note: Use from_config() to customize eps, add_unit_offset, cluster_shape, or other settings.
        """
        super().__init__()
        self.config = _resolve_2d_config(
            RMSNorm2DConfig(weight=weight, tt_ccl=tt_ccl, mesh_device=mesh_device, geometry=geometry)
        )
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

    def decode_forward(
        self, x: "ttnn.Tensor | LazyWeight", residual: "ttnn.Tensor | LazyWeight | None" = None
    ) -> "ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]":
        """
        Wormhole Galaxy distributed decode.

        Uses Linear topology, cluster_axis=1, with program_config.

        Execution:
          to_sharded -> rms_norm_pre_all_gather(sharded) -> all_gather(cluster_axis=1)
          -> rms_norm_post_all_gather(sharded)
        """
        self.load_device_weights()
        x = _load_input_device_tensor_2d(x, self.config, mode="decode")
        cfg = self.config

        if residual is not None:
            residual = _load_input_device_tensor_2d(residual, cfg, mode="decode", residual=True)

        if cfg.geometry is RMSNorm2DGeometry.HEAD_LOCAL:
            return self._decode_head_local(x, residual)

        if cfg.residual_policy is RMSNorm2DResidualPolicy.FUSED_DECODE:
            if residual is None:
                raise ValueError("FUSED_DECODE requires a residual tensor")
            return self._decode_fused_residual_norm(x, residual)

        if residual is not None:
            x = ttnn.add(x, residual, memory_config=cfg.decode_residual_memcfg)

        return self._decode_distributed(x, release_input=residual is not None)

    def _decode_head_local(self, x: ttnn.Tensor, residual: ttnn.Tensor | None) -> ttnn.Tensor:
        """Normalize each ``head_dim``-wide head, inside the decode partition.

        A head-local norm is a plain ``ttnn.rms_norm`` - no column reduction, no
        collective - and on an unpartitioned device the obvious spelling is to
        run it straight on whatever placement the caller hands over. On a WH
        Galaxy decode step it is not that simple, and both obvious spellings
        fail, for different reasons:

        * **interleaved input.** ``ttnn.rms_norm`` resolves
          ``LayerNormDefaultProgramConfig``, which splits its tile rows over
          ``device->compute_with_storage_grid_size()`` - the whole compute grid,
          including the prefetch sender columns the loaded decode sub-device
          manager does not own::

              TT_FATAL: Kernel group cores do not match sub device cores for
                        programmable core type TENSIX
              program.cpp:2205: num_intersections == num_cores

          This is Milestone A's D2 seen from the other side: D2's fix made
          interleaved DRAM the *default* for this geometry, which is right for
          prefill - whose mode plan is one sub-device over the full grid - and
          unplaceable for decode. Nothing had run the decode side until
          Milestone B's Qwen3-32B bring-up (D-B26).
        * **the created heads' own placement.** ``nlp_create_qkv_heads_decode``
          writes Q and K height-sharded, one user per core, and the op rejects
          that outright::

              TT_FATAL: Height sharded inputs are not supported.
              layernorm_device_operation.cpp:166

          (a standing TODO in that file, not a property of the maths.)

        So the kernel runs in a third placement, block-sharded over
        ``decode_compute_cores``: the sharded factory accepts block sharding,
        takes its core ranges from the tensor's own shard spec - so the program
        lands exactly on those cores - and demands a rectangular grid. One core
        wide keeps the whole ``head_dim``-wide row on a single core, so the
        reduction needs no multicast.

        **The shard shape is derived from the tensor, not configured.** The
        decode Q and K carry ``users_per_column`` users of one *tile* of padded
        heads - 8 x 32 = 256 rows on this mesh, whether the model has 8 local Q
        heads or 1 local K head - and that padding is the create-heads op's
        business. A configured shape would encode it, and encoding it wrongly is
        how this landed: a first version sized the rectangle for the full
        physical batch and got::

            TT_FATAL: Shard layout requires 2x1 = 2 shards but shard grid has
                      8 cores
            shard_spec_validation.cpp:81

        The result goes back into **the input's own placement**, for the same
        reason: the rotary that follows expects Q and K exactly as
        ``nlp_create_qkv_heads_decode`` left them, so the norm has to be
        invisible to it.

        Getting in and out uses ``sharded_to_interleaved`` (runs on its
        *input's* shard grid) and ``interleaved_to_sharded`` (runs on its
        *output's* cores) rather than ``to_memory_config``, which between two
        shard specs resolves to ``reshard`` and would abort exactly like the
        interleaved case. That is four extra ops per norm and a real
        decode-latency cost; it belongs on the performance follow-up list, not
        in a correctness argument.

        With ``decode_compute_cores`` unset the behaviour is exactly what it was
        before: one ``ttnn.rms_norm`` on the input as given.
        """

        cfg = self.config
        cores = cfg.decode_compute_cores
        if cores is None:
            return ttnn.rms_norm(
                x,
                epsilon=cfg.eps,
                weight=self.weight,
                residual_input_tensor=residual,
                memory_config=cfg.decode_output_memcfg,
            )
        if residual is not None:
            # The op requires the residual to carry the input's shard spec
            # exactly, so it would have to be relocated in step with `x`. No
            # caller needs that - a per-head Q/K norm has no residual - and
            # guessing is worse than saying so.
            raise ValueError("decode_compute_cores does not support a residual input")

        source_memcfg = x.memory_config()
        compute_memcfg = _head_local_compute_memory_config(x, cores)
        placed = _place_without_leaving_subdevice(x, compute_memcfg)
        try:
            normalized = ttnn.rms_norm(
                placed,
                epsilon=cfg.eps,
                weight=self.weight,
                memory_config=compute_memcfg,
            )
        finally:
            if placed is not x:
                placed.deallocate(True)
        output = _place_without_leaving_subdevice(normalized, source_memcfg)
        if output is not normalized:
            normalized.deallocate(True)
        return output

    def _decode_fused_residual_norm(self, x: ttnn.Tensor, residual: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run WH fused residual accumulation and distributed normalization."""
        cfg = self.config
        stats_shape = (1, 1, int(x.shape[-2]), TILE_SIZE)
        resources = _select_all_gather_resources(cfg, mode="decode", tensor=stats_shape)
        _require_fused_stats_placement(cfg, resources.persistent_output_buffers[0])
        output = ttnn.fused_rms_minimal(
            x,
            cfg.decode_progcfg,
            1,
            cfg.mesh_device,
            _next_single_semaphore(cfg.decode_ccl_context, resources),
            topology=resources.topology,
            subdevice_id=cfg.decode_ccl_context.worker_sub_device_id,
            residual_input_tensor=residual,
            num_links=resources.num_links,
            epsilon=cfg.eps,
            weight=self.weight,
            stats=resources.persistent_output_buffers[0],
            memory_config=cfg.decode_output_memcfg,
        )
        return output, residual

    def _decode_distributed(self, x: ttnn.Tensor, *, release_input: bool = False) -> ttnn.Tensor:
        """Run the module-owned distributed decode recipe on a resolved tensor."""
        cfg = self.config
        # Convert to sharded memory config, but only if it is not already there.
        # `ttnn.to_memory_config` returns *the same* tt_metal tensor when the
        # requested config already matches, and nanobind hands that back as a
        # fresh Python wrapper - so `is not` below cannot tell "no copy was made"
        # from "a copy was made". Guarding on the config, the way
        # `attention_2d.py::_place_qk` and the Galaxy models' `_relocate` already
        # do, keeps the identity test meaningful.
        distributed_input = (
            ttnn.to_memory_config(x, memory_config=cfg.decode_input_memcfg)
            if x.memory_config() != cfg.decode_input_memcfg
            else x
        )

        # Run distributed rmsnorm part 1 (sharded)
        tt_stats = ttnn.rms_norm_pre_all_gather(distributed_input, program_config=cfg.decode_progcfg)

        # All gather stats along cluster axis 1 (columns)
        cluster_axis = 1
        resources = _select_all_gather_resources(cfg, mode="decode", tensor=tt_stats)
        gathered_stats = ttnn.experimental.all_gather_async(
            tt_stats,
            3,
            cluster_axis,
            cfg.mesh_device,
            resources.topology,
            _next_semaphore(cfg.decode_ccl_context, resources),
            persistent_output_tensor=resources.persistent_output_buffers[0],
            num_links=resources.num_links,
            memory_config=cfg.decode_stats_memcfg,
            barrier_semaphore=None,
            subdevice_id=cfg.decode_ccl_context.worker_sub_device_id,
            num_workers_per_link=cfg.ccl_num_workers_per_link,
            num_buffers_per_channel=cfg.ccl_num_buffers_per_channel,
        )
        tt_stats.deallocate(True)

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            distributed_input,
            epsilon=cfg.eps,
            weight=self.weight,
            program_config=cfg.decode_progcfg,
            stats=gathered_stats,
        )
        if distributed_input is not x:
            distributed_input.deallocate(True)
        if release_input:
            x.deallocate(True)

        if cfg.decode_output_memcfg is not None and tt_out.memory_config() != cfg.decode_output_memcfg:
            unplaced_output = tt_out
            tt_out = ttnn.to_memory_config(unplaced_output, cfg.decode_output_memcfg)
            if tt_out is not unplaced_output:
                unplaced_output.deallocate(True)

        return tt_out

    def prefill_forward(
        self, x: "ttnn.Tensor | LazyWeight", residual: "ttnn.Tensor | LazyWeight | None" = None
    ) -> ttnn.Tensor:
        """
        Wormhole Galaxy distributed prefill.

        Uses Linear topology, cluster_axis=1.

        Execution:
          rms_norm_pre_all_gather -> reshape -> all_gather(cluster_axis=1)
          -> rms_norm_post_all_gather
        """
        self.load_device_weights()
        x = _load_input_device_tensor_2d(x, self.config, mode="prefill")
        cfg = self.config

        if residual is not None:
            residual = _load_input_device_tensor_2d(residual, cfg, mode="prefill", residual=True)
        if cfg.geometry is RMSNorm2DGeometry.HEAD_LOCAL:
            return ttnn.rms_norm(
                x,
                epsilon=cfg.eps,
                weight=self.weight,
                residual_input_tensor=residual,
                compute_kernel_config=cfg.compute_kernel_config_prefill,
                memory_config=cfg.prefill_output_memcfg,
            )

        if residual is not None:
            x = ttnn.add(x, residual, memory_config=cfg.prefill_residual_memcfg)

        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            x, compute_kernel_config=cfg.compute_kernel_config_prefill, dtype=ttnn.bfloat16
        )

        # Reshape stats for all_gather (preserve batch dimension for multi-batch prefill)
        padded_shape = _prefill_stats_shape(x.shape)
        tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))

        # All gather stats along cluster axis 1 (columns)
        cluster_axis = 1
        resources = _select_all_gather_resources(cfg, mode="prefill", tensor=tt_stats)
        tt_stats_gathered = ttnn.all_gather(
            tt_stats,
            3,
            cluster_axis=cluster_axis,
            topology=resources.topology,
            num_links=resources.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=cfg.prefill_ccl_context.worker_sub_device_id,
        )

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            tt_stats_gathered,
            epsilon=cfg.eps,
            weight=self.weight,
            compute_kernel_config=cfg.compute_kernel_config_prefill,
            memory_config=cfg.prefill_output_memcfg,
        )

        return tt_out

    # =========================================================================
    # Forward dispatcher
    # =========================================================================

    def forward(
        self,
        x: "ttnn.Tensor | LazyWeight",
        mode: str,
        residual: "ttnn.Tensor | LazyWeight | None" = None,
    ) -> "ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]":
        """Dispatch to decode_forward or prefill_forward based on mode."""
        if mode == "decode":
            return self.decode_forward(x, residual=residual)
        if mode == "prefill":
            return self.prefill_forward(x, residual=residual)
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode}")


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_2d_config(config: RMSNorm2DConfig) -> RMSNorm2DConfig:
    """Resolve config defaults for RMSNorm2D."""

    to_set = {}

    # --- Phase 1: Foundational fields ---

    # Derive dim from weight (works for any shape: [dim], [1, dim], [1, 1, dim//32, 32], etc.)
    dim = config.weight.source.numel()
    geometry = config.geometry or (RMSNorm2DGeometry.HEAD_LOCAL if dim == 128 else RMSNorm2DGeometry.DISTRIBUTED)
    to_set["geometry"] = geometry

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

    assert (
        tuple(cluster_shape) == WH_GALAXY_MESH_SHAPE
    ), f"RMSNorm2D requires WH Galaxy mesh {WH_GALAXY_MESH_SHAPE}, got {tuple(cluster_shape)}"
    assert tuple(mesh_device.shape) == tuple(cluster_shape), "cluster_shape must match the configured mesh"
    assert mesh_device.get_num_devices() == 32, "RMSNorm2D requires exactly 32 devices"
    assert mesh_device.arch() == ttnn.device.Arch.WORMHOLE_B0, "RMSNorm2D requires Wormhole"

    num_rows, num_cols = cluster_shape
    if geometry is RMSNorm2DGeometry.DISTRIBUTED:
        assert dim % num_cols == 0, f"dim={dim} must be divisible by Galaxy columns={num_cols}"

    weight_device = getattr(config.weight, "device", None)
    assert weight_device is None or weight_device is mesh_device, "weight must belong to the configured mesh"
    for mode, context in (
        ("decode", config.decode_prefetch_context),
        ("prefill", config.prefill_prefetch_context),
    ):
        context_mesh = getattr(context, "mesh_device", mesh_device)
        assert context_mesh is mesh_device, "prefetch context must belong to the configured mesh"
        context_mode = getattr(context, "mode", mode)
        assert context_mode == mode, f"{mode} prefetch context has mode={context_mode}"

    if geometry is RMSNorm2DGeometry.DISTRIBUTED:
        tt_ccl = config.tt_ccl
        if tt_ccl is None:
            raise ValueError("distributed RMSNorm2D requires an injected Galaxy CCL collaborator")
        ccl_mesh = getattr(tt_ccl, "mesh_device", mesh_device)
        assert ccl_mesh is mesh_device, "CCL collaborator must belong to the configured mesh"
        for mode in ("decode", "prefill"):
            context = _resolve_ccl_context(
                getattr(config, f"{mode}_ccl_context"), tt_ccl=tt_ccl, mode=mode, mesh_device=mesh_device
            )
            to_set[f"{mode}_ccl_context"] = context
            if config.collective_resource_selector is None:
                to_set[f"{mode}_all_gather_resources"] = _resolve_all_gather_resources(context, mode=mode)
    elif config.residual_policy is RMSNorm2DResidualPolicy.FUSED_DECODE:
        raise ValueError("FUSED_DECODE is only valid for distributed norm geometry")

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

    hidden_size_per_device = dim // num_cols if geometry is RMSNorm2DGeometry.DISTRIBUTED else dim
    hidden_tiles = hidden_size_per_device // TILE_SIZE
    if geometry is RMSNorm2DGeometry.HEAD_LOCAL:
        # A head-local norm is a plain `rms_norm` over one 128-wide head: no
        # collective, and the program config is derived from the input placement.
        # Decode feeds it `batch * local heads` rows, not the fixed 32 rows a
        # width-sharded L1 recipe would pin, so keep decode interleaved like
        # prefill and let callers opt into an explicit sharded placement.
        if config.decode_input_memcfg is None:
            to_set["decode_input_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    else:
        # The canonical column-dispatch Galaxy layout reserves x=0..1. Keep four
        # width tiles per core and use the proven two-column norm grid.
        grid_width = 2
        if hidden_tiles % (grid_width * 4) != 0:
            raise ValueError(f"distributed decode width requires four tiles per norm core, got {hidden_tiles} tiles")
        grid_height = hidden_tiles // (grid_width * 4)
        if not 1 <= grid_height <= 10:
            raise ValueError(f"distributed decode norm grid height must fit Wormhole, got {grid_height}")
        num_cores_ln = grid_height * grid_width
        norm_origin = ttnn.CoreCoord(2, 0)
        decode_core_range = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    norm_origin, ttnn.CoreCoord(norm_origin.x + grid_width - 1, norm_origin.y + grid_height - 1)
                )
            }
        )

        if config.decode_input_memcfg is None:
            to_set["decode_input_memcfg"] = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device // num_cores_ln),
                core_grid=decode_core_range,
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )

        if config.decode_progcfg is None:
            to_set["decode_progcfg"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(grid_width, grid_height),
                subblock_w=(hidden_size_per_device // num_cores_ln) // TILE_SIZE,
                block_h=1,
                block_w=(hidden_size_per_device // num_cores_ln) // TILE_SIZE,
                inplace=False,
            )

        if config.decode_stats_memcfg is None:
            # Stats memory: 32 x (32 * num_cols) where num_cols is the cluster width.
            # The stats shard must sit on the *first* core of the norm input shard
            # grid: the sharded RMS kernels build their stats circular buffer on
            # that core and bind it to this tensor's L1 address, so a shard placed
            # anywhere else makes the kernel read unrelated L1 (see
            # _require_fused_stats_placement).
            to_set["decode_stats_memcfg"] = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * num_cols],
                core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(norm_origin, norm_origin)}),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )

    if config.prefill_input_memcfg is None:
        to_set["prefill_input_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    for field_name, default in (
        ("decode_residual_memcfg", config.decode_input_memcfg or to_set["decode_input_memcfg"]),
        ("prefill_residual_memcfg", config.prefill_input_memcfg or to_set["prefill_input_memcfg"]),
        ("decode_output_memcfg", config.decode_input_memcfg or to_set["decode_input_memcfg"]),
        ("prefill_output_memcfg", ttnn.DRAM_MEMORY_CONFIG),
    ):
        if getattr(config, field_name) is None:
            to_set[field_name] = default

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

    placements = (
        [ttnn.PlacementReplicate(), ttnn.PlacementShard(2)]
        if geometry is RMSNorm2DGeometry.DISTRIBUTED
        else [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
    )
    mesh_mapper_config = ttnn.MeshMapperConfig(
        placements=placements,
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

    resolved = replace(config, **to_set)
    assert resolved.is_resolved(), "RMSNorm2D config must be resolved"
    return resolved


# =============================================================================
# Input Tensor Utilities
# =============================================================================


def _head_local_compute_memory_config(x: ttnn.Tensor, cores: ttnn.CoreRangeSet) -> ttnn.MemoryConfig:
    """Block-shard `x`'s rows evenly over `cores`, one core wide.

    Rectangular and one core wide are both requirements of the sharded
    layernorm: it rejects a non-rectangular grid outright, and a wider grid would
    split the normalized dimension across cores and need a multicast reduction
    for no benefit at these widths.
    """

    box = cores.bounding_box()
    width_in_cores = box.end.x - box.start.x + 1
    if width_in_cores != 1:
        raise ValueError(f"a head-local compute grid must be one core wide, got {width_in_cores}")
    num_cores = cores.num_cores()
    if num_cores != box.end.y - box.start.y + 1:
        raise ValueError("a head-local compute grid must be a rectangle")
    # `ttnn.Shape` indexes by position but does not slice, so this walks it.
    shape = x.padded_shape
    height = 1
    for axis in range(len(shape) - 1):
        height *= int(shape[axis])
    width = int(shape[len(shape) - 1])
    if height % num_cores or (height // num_cores) % TILE_SIZE:
        raise ValueError(f"{height} rows do not divide over {num_cores} cores in whole tiles")
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (height // num_cores, width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _place_without_leaving_subdevice(tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Move `tensor` into `memory_config` using only sub-device-aware ops.

    ``to_memory_config`` is not usable here: between two shard specs it resolves
    to ``reshard``, and between two interleaved configs to ``ttnn::prim::copy``,
    both of which build their programs over the device's whole compute grid. A
    partitioned decode manager rejects that with

        TT_FATAL ... Kernel group cores do not match sub device cores

    ``sharded_to_interleaved`` runs on its input's ``shard_spec.grid`` and
    ``interleaved_to_sharded`` on its output shard's cores, so as long as both
    ends are inside the partition, so is every program this builds. Returns the
    input itself when it is already placed, so callers can test identity to know
    whether they own the result.
    """

    if tensor.memory_config() == memory_config:
        return tensor
    if not memory_config.is_sharded():
        if not tensor.memory_config().is_sharded():
            raise ValueError("an interleaved-to-interleaved move is ttnn::prim::copy, which is not sub-device aware")
        # One hop, straight into the requested interleaved config:
        # `sharded_to_interleaved` runs on its input's grid whatever the
        # destination buffer type is, so staging through DRAM first would only
        # add an interleaved-to-interleaved copy on the full grid.
        return ttnn.sharded_to_interleaved(tensor, memory_config)
    staged = tensor
    if staged.memory_config().is_sharded():
        staged = ttnn.sharded_to_interleaved(staged, ttnn.DRAM_MEMORY_CONFIG)
    placed = ttnn.interleaved_to_sharded(staged, memory_config)
    if staged is not tensor:
        staged.deallocate(True)
    return placed


def _load_input_device_tensor_2d(
    x: "ttnn.Tensor | LazyWeight", config: RMSNorm2DConfig, mode: str, residual: bool = False
) -> ttnn.Tensor:
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
            mem_cfg = config.decode_residual_memcfg if residual else config.decode_input_memcfg
        else:
            mem_cfg = config.prefill_residual_memcfg if residual else config.prefill_input_memcfg

        # Shard input on last dim across columns, replicate across rows
        placements = (
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)]
            if config.geometry is RMSNorm2DGeometry.DISTRIBUTED
            else [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
        )
        mesh_mapper_config = ttnn.MeshMapperConfig(
            placements=placements,
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


def _shard_origin(memory_config: Any) -> Any:
    """First core of a sharded memory config, or None when interleaved."""
    shard_spec = getattr(memory_config, "shard_spec", None)
    if shard_spec is None:
        return None
    return shard_spec.grid.bounding_box().start


def _require_fused_stats_placement(config: RMSNorm2DConfig, stats: Any) -> None:
    """
    Reject a fused-decode stats buffer that is not sharded on the norm sender core.

    `fused_rms_minimal` creates its stats circular buffer on the first core of the
    norm input shard grid and binds it to the stats tensor's L1 address; it never
    reads that tensor over the NoC. A stats shard on any other core therefore makes
    the kernel reduce whatever else the allocator left at that address on the sender
    core, which silently corrupts the normalization scale - on hardware this showed up
    as an output scaled by ~1e37 (rsqrt of zero) or, when the aliased L1 happened to
    hold other activations, as a plausible-looking but unearned PCC.
    """
    expected = _shard_origin(config.decode_input_memcfg)
    actual = _shard_origin(stats.memory_config())
    if expected is None:
        raise ValueError("fused decode requires a width-sharded L1 decode input placement")
    if actual is None or (actual.x, actual.y) != (expected.x, expected.y):
        raise ValueError(
            "fused decode stats buffer must be L1-sharded on the first core of the norm input shard grid "
            f"({expected}); got {actual}. The fused RMS stats circular buffer is created on that core and "
            "bound to this buffer's address, so any other placement reads unrelated L1."
        )


def _prefill_stats_shape(input_shape: Any) -> tuple[int, int, int, int]:
    """Preserve every token-bearing axis in the distributed statistics tensor."""
    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 4:
        raise ValueError(f"RMSNorm input must have rank 4, got {shape}")
    return (*shape[:3], TILE_SIZE)


def _resolve_ccl_context(context: Any, *, tt_ccl: Any, mode: str, mesh_device: Any) -> Any:
    if context is None:
        factory = getattr(tt_ccl, "context", None)
        if not callable(factory):
            raise TypeError("Galaxy CCL collaborator must provide context(mode)")
        context = factory(mode)
    if getattr(context, "mesh_device", None) is not mesh_device:
        raise ValueError(f"{mode} CCL context must belong to the configured mesh")
    if getattr(context, "mode", None) != mode:
        raise ValueError(f"{mode} CCL context has mode={getattr(context, 'mode', None)}")
    for method in ("resources", "next_semaphore_handles", "next_barrier_semaphore_handle"):
        if not callable(getattr(context, method, None)):
            raise TypeError(f"{mode} CCL context must provide {method}()")
    if getattr(context, "worker_sub_device_id", None) is None:
        raise ValueError(f"{mode} CCL context requires worker_sub_device_id")
    return context


def _resolve_all_gather_resources(context: Any, *, mode: str) -> Any:
    resources = context.resources("all_gather", 1)
    if resources is None or resources.cluster_axis != 1:
        raise ValueError(f"{mode} all_gather resources must target cluster_axis=1")
    if resources.topology is None or resources.num_links < 1:
        raise ValueError(f"{mode} all_gather topology and num_links must be resolved")
    if not resources.persistent_output_buffers:
        raise ValueError(f"{mode} all_gather requires a persistent output buffer")
    if getattr(resources, "key", None) is None:
        raise ValueError(f"{mode} all_gather resources require an exact resource key")
    return resources


def _select_all_gather_resources(config: RMSNorm2DConfig, *, mode: str, tensor: Any) -> Any:
    selector = config.collective_resource_selector
    if selector is None:
        return getattr(config, f"{mode}_all_gather_resources")
    context = config.decode_ccl_context if mode == "decode" else config.prefill_ccl_context
    resources = selector(context, "all_gather", 1, tensor)
    if resources is None or resources.cluster_axis != 1:
        raise ValueError(f"{mode} all_gather resources must target cluster_axis=1")
    if resources.topology is None or resources.num_links < 1:
        raise ValueError(f"{mode} all_gather topology and num_links must be resolved")
    if not resources.persistent_output_buffers:
        raise ValueError(f"{mode} all_gather requires a persistent output buffer")
    if getattr(resources, "key", None) is None:
        raise ValueError(f"{mode} all_gather resources require an exact resource key")
    return resources


def _next_semaphore(context: Any, resources: Any) -> Any:
    key = resources.key
    return context.next_semaphore_handles(key.operation, key.cluster_axis, key.geometry, key.sequence_key)


def _next_single_semaphore(context: Any, resources: Any) -> Any:
    handles = _next_semaphore(context, resources)
    if isinstance(handles, (tuple, list)):
        if len(handles) != 1:
            raise ValueError("fused RMSNorm requires exactly one semaphore handle")
        return handles[0]
    return handles
