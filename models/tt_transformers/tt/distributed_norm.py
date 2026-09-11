# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm
from models.tt_transformers.tt.common import Mode


def galaxy_distributed_norm_core_grid(dim: int) -> tuple[int, int]:
    """Choose the Galaxy decode RMSNorm grid as (y, x).

    Returns the legacy ``(min(4, dim // 4 // 32 // 8), 8)`` grid for every dim it
    already handled. Only dims that grid cannot tile-align get an override, and a
    dim with no tile-aligned override keeps the legacy grid with a warning rather
    than raising -- ``gather_in_mem_cfg`` / ``ln_prg_cfg`` are consumed only on the
    sharded decode path, so raising here would break prefill-only Galaxy runs that
    never read them (e.g. every Gemma variant).
    """
    hidden_size_per_device = dim // 4
    legacy_core_grid = (min(4, hidden_size_per_device // 32 // 8), 8)

    if hidden_size_per_device == 1280:
        # Qwen's silicon-validated Galaxy layout: 10 cores with four tiles per shard.
        # The legacy grid would be (4, 8) = 32 cores, which cannot tile-align 1280.
        core_grid = (5, 2)
    else:
        core_grid = legacy_core_grid

    num_cores = core_grid[0] * core_grid[1]
    if num_cores == 0 or hidden_size_per_device % (num_cores * 32) != 0:
        logger.warning(
            f"Galaxy distributed norm hidden size {hidden_size_per_device} is not tile-shardable "
            f"across grid {core_grid}; keeping the legacy grid {legacy_core_grid}. The sharded "
            "decode norm config will be misaligned if it is used."
        )
        return legacy_core_grid
    return core_grid


class DistributedNorm(LightweightModule):
    """Wraps a norm with the TP gather it implies.

    ``norm=None`` turns this into a gather-only identity: the input is gathered
    (or left fractured + gathered after, in distributed-norm mode) exactly as it
    would be around a real norm, but no normalization is applied. Post-norm
    decoders (EXAONE-4.x) use this where pre-norm models have input_layernorm /
    pre_feedforward_layernorm, since those slots double as the fractured->
    replicated gather points. Note an RMSNorm with all-ones gamma is NOT an
    identity (it still divides by RMS), hence this explicit mode.
    """

    def __init__(self, norm, args, tt_ccl, prefetcher=None, TG=False, ag_config_key=None, enable_all_gather=True):
        if norm is None and TG:
            raise NotImplementedError("Gather-only DistributedNorm (norm=None) is not supported on Galaxy (TG)")
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.ag_config_key = ag_config_key

        # Flag to control whether all_gather is performed after distributed norm (can be disabled when output should remain sharded)
        self.enable_all_gather = enable_all_gather

        if TG:
            core_grid_ln = galaxy_distributed_norm_core_grid(args.dim)
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            hidden_size_per_device_distributed_ln = args.dim // 4
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
                core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        self.TG = TG
        # Lazily-built block-shard configs for the interleaved prefill norm, keyed by input shape.
        self._prefill_shard_cfg_cache = {}

    def _prefill_block_shard_cfg(self, x):
        """Block-shard configs for a prefill ``[1, 1, S, D]`` norm input, or ``(None, None)``.

        An interleaved ``rms_norm`` only parallelises over tile ROWS, so a 128-token
        prefill chunk runs its reduction on ``S/32 = 4`` cores of a 130-core grid --
        that is the ``grid=tiny`` tag on this op. Block-sharding the activation into
        L1 gives every core an ``(S/gy, D/gx)`` slice instead. The grid comes from the
        shape's tile factors (never hard-coded); if that grid is no wider than the
        ``S/32`` cores the interleaved kernel already gets, the reshard is not worth
        paying for and we stay interleaved.
        """
        key = tuple(x.shape)
        cached = self._prefill_shard_cfg_cache.get(key)
        if cached is not None:
            return cached

        cfg = (None, None)
        seq, dim = int(x.shape[-2]), int(x.shape[-1])
        if seq % 32 == 0 and dim % 32 == 0:
            grid = self.args.mesh_device.compute_with_storage_grid_size()
            m_tiles, n_tiles = seq // 32, dim // 32
            gy = max((y for y in range(1, min(grid.y, m_tiles) + 1) if m_tiles % y == 0), default=1)
            gx = max((c for c in range(1, min(grid.x, n_tiles) + 1) if n_tiles % c == 0), default=1)
            if gy * gx > m_tiles:
                block_h, block_w = m_tiles // gy, n_tiles // gx
                # fp32 dest accumulation (mandatory for norms) halves the dest register
                # budget, so cap the math subblock at 4 tiles.
                subblock_w = max(
                    (s for s in range(1, block_w + 1) if block_w % s == 0 and s * block_h <= 4),
                    default=1,
                )
                cfg = (
                    ttnn.create_sharded_memory_config(
                        shape=(seq // gy, dim // gx),
                        core_grid=ttnn.CoreGrid(y=gy, x=gx),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    ),
                    ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        subblock_w=subblock_w,
                        block_h=block_h,
                        block_w=block_w,
                        inplace=False,
                    ),
                )

        self._prefill_shard_cfg_cache[key] = cfg
        return cfg

    def update(self, *, weight: ttnn.Tensor) -> None:
        """Pass-through to the wrapped ``RMSNorm.update`` (``DistributedNorm``
        owns no weights of its own). Same HF-format contract: ``(1, 1, 1, dim)``,
        TILE, bf16, DRAM-interleaved, replicated.
        """
        if self.norm is None:
            raise ValueError("Gather-only DistributedNorm (norm=None) owns no weights to update")
        self.norm.update(weight=weight)

    def forward(self, x, mode: Mode, norm_config=None):
        """Apply a norm, possibly gathering inputs if required."""

        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None

        if self.TG:
            if mode == Mode.DECODE:
                return tt_sharded_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                    ln_sharded_progcfg=self.ln_prg_cfg,
                    ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                )
            else:
                return tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    compute_kernel_config=self.ln_cfg,
                )

        input_mem_cfg = sharded_output_config if mode == Mode.DECODE else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            # NOTE: the `mode == "decode"` tests below are comparing a Mode enum member against
            # a str, so they are always False and the per-model *_LN_AG_CONFIG has never been
            # read here. Left as-is: the only configs it would select are the Galaxy-tuned ones
            # (num_links=4), which cannot be validated on this board. The fallbacks are what
            # actually run, so they are what is tuned -- see ModelArgs.ccl_sync_params.
            ag_chunks_per_sync, ag_num_workers = self.args.ccl_sync_params(mode)
            # Halve what the gather moves. This collective did not respond to more
            # workers per link or a coarser sync, so it is pinned by the inter-chip link
            # rather than by cores -- which leaves bytes as the only lever. bf8_b is the
            # documented floor for a normalization input (never below), and the residual
            # stream itself is untouched: only the copy handed to the gather is narrowed.
            #
            # PREFILL only. The cast is a whole extra program, and it only pays when the
            # payload is big enough: prefill gathers ~1 MB in 22 us, decode ~0.2 MB in
            # 12 us. Measured with the cast in BOTH modes, prefill improved but the
            # per-token time went 8.1227 -> 8.1289 ms -- in decode the 64 extra typecast
            # launches cost more than the narrower gather saves.
            if mode == Mode.PREFILL and x.dtype == ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat8_b)
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.args.model_config[self.ag_config_key]["num_links"]
                if self.ag_config_key and mode == "decode"
                else self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=self.args.model_config[self.ag_config_key]["chunks_per_sync"]
                if self.ag_config_key and mode == "decode"
                else ag_chunks_per_sync,
                num_workers_per_link=self.args.model_config[self.ag_config_key]["num_workers_per_link"]
                if self.ag_config_key and mode == "decode"
                else ag_num_workers,
                num_buffers_per_channel=2,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        if self.norm is not None:
            shard_mem_cfg, shard_prg_cfg = (
                self._prefill_block_shard_cfg(x)
                if (mode == Mode.PREFILL and not self.args.is_distributed_norm(mode))
                else (None, None)
            )
            if shard_mem_cfg is not None:
                # Occupy the grid: run the prefill norm block-sharded in L1, then hand
                # DRAM-interleaved back to the projection that consumes it.
                x_sharded = ttnn.to_memory_config(x, shard_mem_cfg)
                y = self.norm(
                    x_sharded,
                    mode=mode,
                    in_sharded=True,
                    out_sharded=True,
                    norm_config={
                        "sharded_program_config": shard_prg_cfg,
                        "sharded_output_config": shard_mem_cfg,
                    },
                )
                ttnn.deallocate(x_sharded)
                # L1, not DRAM: this feeds the QKV / ff1-ff3 projection in the same layer
                # and nothing else, so a DRAM round-trip here is ~2 MB of pure waste per
                # norm. Interleaved, so the projection's in0 contract is unchanged.
                x = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(y)
            else:
                x = self.norm(
                    x,
                    mode=mode,
                    in_sharded=(mode == Mode.DECODE),
                    out_sharded=(mode == Mode.DECODE),
                    norm_config=norm_config,
                )

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode) and self.enable_all_gather:
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        return x
