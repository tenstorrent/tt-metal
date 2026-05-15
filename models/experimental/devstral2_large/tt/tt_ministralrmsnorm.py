# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Tenstorrent RMSNorm for Devstral-2 large (Ministral3 text stack, ~12k hidden).

Builds on :class:`models.common.rmsnorm.RMSNorm` with the same per-layer weight keys as the small
Ministral3 stack (``attention_norm`` / ``ffn_norm``), plus Blackhole-specific handling for **wide**
hidden (~12k):

- **Distributed** path: ``rms_norm_pre_all_gather`` / post all-gather can overflow L1 without a
  multicore program config; mitigated with ``LayerNormShardedMultiCoreProgramConfig`` + seq chunks.
- **Local** ``ttnn.rms_norm`` path (e.g. tests that set ``is_distributed_norm`` to ``False`` for PCC):
  the **default** fused RMSNorm still allocates huge **kernel** circular buffers on BH at 12288
  width × long prefill — **not** fixed by ``l1_small_size`` (that only sizes the small L1 allocator at
  device open). Same multicore program config + sequence chunking applies here.

``l1_small_size`` (:data:`DEVSTRAL2_LARGE_L1_SMALL_SIZE`) is only for ``open_mesh_device`` /
pytest ``device_params``; it does not change per-op CB allocation.
"""

from __future__ import annotations

import ttnn

from models.common.rmsnorm import RMSNorm
from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.common import Mode

TILE = 32

# ``l1_small_size`` for ``ttnn.open_mesh_device`` / pytest ``device_params`` (L1 small allocator
# region on device init). Not read by :class:`TtDevstral2LargeRMSNorm` — the mesh is already open.
# Same order of magnitude as multi-device fabric demos (e.g. Gemma3 vision tests use 24576).
DEVSTRAL2_LARGE_L1_SMALL_SIZE = 24576

# Max sequence tokens per RMSNorm invocation on BH + ~12k hidden (L1 CB budget).
# Width-sharded ``interleaved_to_sharded`` requires shard shapes tile-aligned (32).
_BH_WIDE_SEQ_CHUNK = 32


def _resolve_norm_core_grid(mesh_device, hidden_dim: int) -> ttnn.CoreGrid:
    """Pick a core rectangle within the worker grid that evenly divides hidden width in tiles."""
    gs = mesh_device.compute_with_storage_grid_size()
    width_tiles = hidden_dim // TILE
    max_cores = gs.x * gs.y
    for ncores in range(max_cores, 0, -1):
        if width_tiles % ncores != 0:
            continue
        for gy in range(min(gs.y, ncores), 0, -1):
            if ncores % gy != 0:
                continue
            gx = ncores // gy
            if gx <= gs.x and gy <= gs.y:
                return ttnn.CoreGrid(x=gx, y=gy)
    raise RuntimeError(
        f"Cannot build LayerNorm sharded program config: hidden_dim={hidden_dim} "
        f"(width_tiles={width_tiles}) vs worker grid {gs.x}x{gs.y}"
    )


class TtDevstral2LargeRMSNorm(RMSNorm):
    """
    Ministral3 RMSNorm for very wide models on Blackhole + fabric mesh.

    See module docstring for chunking and sharded program config behavior.

    ``model_final_norm=True`` loads ``norm.weight`` (HF final ``Ministral3Model`` RMSNorm) via
    :class:`models.common.rmsnorm.RMSNorm` instead of a layer ``attention_norm`` / ``ffn_norm`` tensor.
    """

    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        tt_ccl,
        *,
        post_attention: bool = False,
        model_final_norm: bool = False,
    ):
        self._hidden_dim = args.dim
        if model_final_norm:
            super().__init__(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                weight_key="norm",
                layer_num=None,
                state_dict_prefix=None,
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=args.is_distributed_norm,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            )
        else:
            weight_key = "ffn_norm" if post_attention else "attention_norm"
            super().__init__(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                weight_key=weight_key,
                layer_num=None,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            )

    def _bh_wide_norm_program_config(self, x: ttnn.Tensor) -> ttnn.LayerNormShardedMultiCoreProgramConfig:
        grid = _resolve_norm_core_grid(self.device, self._hidden_dim)
        seq_len = int(x.shape[2])
        tile_padded_seq = ((seq_len + TILE - 1) // TILE) * TILE
        block_h = tile_padded_seq // TILE
        num_cores = grid.x * grid.y
        block_w = self._hidden_dim // num_cores // TILE
        subblock_w = 4
        while subblock_w > 0:
            if block_w % subblock_w == 0:
                break
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )

    def _use_bh_wide_distributed_chunked_norm(self, mode: Mode) -> bool:
        return (
            is_blackhole()
            and self.device.get_num_devices() > 1
            and self._hidden_dim >= 8192
            and bool(self.is_distributed and self.is_distributed(mode))
        )

    def _distributed_rmsnorm_sharded_program(
        self,
        inp: ttnn.Tensor,
        *,
        program_config: ttnn.LayerNormShardedMultiCoreProgramConfig,
        epsilon,
        weight,
        compute_kernel_config,
    ) -> ttnn.Tensor:
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            program_config=program_config,
        )
        tt_stats = ttnn.experimental.all_gather_async(
            tt_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=self.ccl_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=epsilon,
            weight=weight,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
        )
        tt_stats.deallocate(True)
        return tt_out

    def _forward_bh_wide_distributed(self, x: ttnn.Tensor, weight) -> ttnn.Tensor:
        """Distributed RMSNorm with sharded program config + optional sequence chunking."""
        seq_len = int(x.shape[2])
        if seq_len <= _BH_WIDE_SEQ_CHUNK:
            pc = self._bh_wide_norm_program_config(x)
            return self._distributed_rmsnorm_sharded_program(
                x,
                program_config=pc,
                epsilon=self.eps,
                weight=weight,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )

        b, _, _, h = x.shape
        parts: list = []
        for start in range(0, seq_len, _BH_WIDE_SEQ_CHUNK):
            end = min(start + _BH_WIDE_SEQ_CHUNK, seq_len)
            chunk = ttnn.slice(
                x,
                [0, 0, start, 0],
                [b, 1, end, h],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pc = self._bh_wide_norm_program_config(chunk)
            parts.append(
                self._distributed_rmsnorm_sharded_program(
                    chunk,
                    program_config=pc,
                    epsilon=self.eps,
                    weight=weight,
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                )
            )
            chunk.deallocate(True)
        out = ttnn.concat(parts, dim=2)
        for p in parts:
            p.deallocate(True)
        return out

    def _need_bh_wide_local_rmsnorm_workaround(self, in_sharded: bool, distributed: bool) -> bool:
        """Wide hidden + BH + non-distributed ``ttnn.rms_norm`` hits default LayerNormDeviceOperation L1 CB limits."""
        return not in_sharded and is_blackhole() and self._hidden_dim >= 8192 and not distributed

    def _forward_bh_wide_local_rmsnorm(self, x: ttnn.Tensor, *, weight) -> ttnn.Tensor:
        """Width-shard activations, run ``ttnn.rms_norm`` (multicore), return interleaved.

        Plain ``ttnn.rms_norm`` only accepts ``LayerNormShardedMultiCoreProgramConfig`` with **sharded**
        activations; interleaved + sharded PC hits optional-access errors in the kernel.
        """
        seq_len = int(x.shape[2])

        def _run_chunk(t: ttnn.Tensor) -> ttnn.Tensor:
            bsz = int(t.shape[0])
            sl = int(t.shape[2])
            sl_pad = ((sl + TILE - 1) // TILE) * TILE
            if sl_pad > sl:
                t_work = ttnn.pad(t, padding=[(0, 0), (0, 0), (0, sl_pad - sl), (0, 0)], value=0.0)
            else:
                t_work = t
            grid = _resolve_norm_core_grid(self.device, self._hidden_dim)
            num_cores = grid.x * grid.y
            shard_w_el = self._hidden_dim // num_cores
            input_shard_cfg = ttnn.create_sharded_memory_config(
                shape=[bsz * sl_pad, shard_w_el],
                core_grid=grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            xs = ttnn.interleaved_to_sharded(t_work, input_shard_cfg)
            if sl_pad > sl:
                t_work.deallocate(True)
            activation_grid = xs.memory_config().shard_spec.grid.bounding_box().grid_size()
            shard_height, shard_width = xs.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=activation_grid,
                subblock_w=1,
                block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
                block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
                inplace=False,
            )
            out_mem = ttnn.get_memory_config(xs)
            out = ttnn.rms_norm(
                xs,
                epsilon=self.eps,
                weight=weight,
                program_config=program_config,
                memory_config=out_mem,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )
            xs.deallocate(True)
            y = ttnn.sharded_to_interleaved(out)
            out.deallocate(True)
            if sl_pad > sl:
                y = ttnn.slice(y, [0, 0, 0, 0], [bsz, 1, sl, self._hidden_dim])
            return y

        if seq_len <= _BH_WIDE_SEQ_CHUNK:
            return _run_chunk(x)

        b, _, _, h = x.shape
        parts: list = []
        for start in range(0, seq_len, _BH_WIDE_SEQ_CHUNK):
            end = min(start + _BH_WIDE_SEQ_CHUNK, seq_len)
            chunk = ttnn.slice(
                x,
                [0, 0, start, 0],
                [b, 1, end, h],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            parts.append(_run_chunk(chunk))
            chunk.deallocate(True)
        out = ttnn.concat(parts, dim=2)
        for p in parts:
            p.deallocate(True)
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        mode: Mode | str,
        in_sharded=False,
        out_sharded=False,
        norm_config=None,
    ) -> ttnn.Tensor:
        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError:
                raise ValueError(f"Invalid mode: {mode}")
        elif not isinstance(mode, Mode):
            raise ValueError(f"Invalid mode: {mode}")

        sharded_program_config = norm_config.get("sharded_program_config") if norm_config else None
        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        output_mem_config = norm_config.get("output_mem_config") if norm_config else None

        program_config = sharded_program_config if in_sharded else None
        memory_config = sharded_output_config if out_sharded else None
        distributed = self.is_distributed and self.is_distributed(mode)
        weight = self.weight_distributed if distributed else self.weight

        if in_sharded:
            assert not distributed, "Distributed RMSNorm does not support sharded inputs"
        else:
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"

        if distributed and self._use_bh_wide_distributed_chunked_norm(mode):
            x = self._forward_bh_wide_distributed(x, weight)
        elif self._need_bh_wide_local_rmsnorm_workaround(in_sharded, distributed):
            x = self._forward_bh_wide_local_rmsnorm(x, weight=weight)
        else:
            norm = self._distributed_rmsnorm if distributed else ttnn.rms_norm
            x = norm(
                x,
                epsilon=self.eps,
                weight=weight,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )

        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(x)
        if output_mem_config is not None:
            x = ttnn.to_memory_config(x, output_mem_config)
        return x


__all__ = [
    "DEVSTRAL2_LARGE_L1_SMALL_SIZE",
    "TtDevstral2LargeRMSNorm",
]
