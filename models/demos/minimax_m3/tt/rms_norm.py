# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 RMSNorm — single-pass (replicated residual) or distributed (emb/tp-sharded residual).

Which one runs is decided ONCE at construction from ``tt/residual.py::use_sharded_residual`` (override
per instance with ``is_distributed=``), because the two need differently-sharded gains:

  * single-pass: ``ttnn.rms_norm`` over the full emb, gain REPLICATED — every TP column normalizes the
    same full vector, so the result is already replicated.
  * distributed: ``rms_norm_pre_all_gather`` -> all-gather the per-column sum(x^2) -> \
    ``rms_norm_post_all_gather``, gain sharded on dim -2 across TP. Three ops instead of one, but each
    touches only ``emb/tp``.

Supports both DRAM-interleaved and L1-sharded activations on the distributed path (the underlying ops
do: ``program_config=None`` selects the interleaved kernels). M3's residual is DRAM-interleaved, so the
interleaved path is the one that runs in the model; the sharded branch is kept for L1-sharded callers.
"""

from loguru import logger
from torch import nn

import ttnn
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.utils.general_utils import get_cache_file_name, get_default_num_links

from .residual import use_distributed_norm


class RMSNorm(nn.Module):
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        tensor_cache_path=None,
        mesh_config=None,
        ccl_manager=None,
        is_distributed=None,
    ):
        """is_distributed: None => derive from the residual scheme (sharded residual + tp > 1). Pass
        False to pin a norm to the single-pass form regardless (the final norm does: its input is
        gathered to full emb for the column-parallel LM head anyway)."""
        super().__init__()
        # MiniMax-M3 uses Gemma-style RMSNorm: out = x_normed * (1 + weight), whereas
        # a plain RMSNorm is out = x_normed * weight. ttnn.rms_norm only does the
        # `* weight` form, so when use_gemma_norm is set we fold the +1 into the weight
        # at load time (in fp32, before the bf16 cast) — equivalent and free at runtime.
        self.use_gemma_norm = bool(getattr(hf_config, "use_gemma_norm", False))
        if state_dict:
            weight = state_dict["weight"]
            if self.use_gemma_norm:
                weight = weight.float() + 1.0
            torch_weight = weight.reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.ccl_manager = ccl_manager
        if is_distributed is None:
            is_distributed = use_distributed_norm() and self.mesh_config.tp > 1
        self.is_distributed = is_distributed

        # The REPLICATED gain is always built: it is the cached artefact (weight_cache.py checks for
        # exactly this file), and on the distributed path it is also the only source of the gain in
        # cache-only mode. Its cache key is unchanged, so switching residual schemes never invalidates
        # or rebuilds the tilized weight cache.
        replicated = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=None,
        )
        if self.is_distributed:
            # rms_norm_post_all_gather requires the gain to match the input width (emb/tp), so re-shard
            # it on dim -2 across the TP cols. Derived here rather than cached: in the production
            # cache-only path there is no torch source, so the gain is recovered from the replicated
            # device tensor with one tiny init-time D2H (12 KiB per norm) — no new cache entry, hence no
            # dependence on the (often other-user-owned, read-only) weight cache being writable.
            self.tt_weight = ttnn.from_torch(
                torch_weight if torch_weight is not None else ttnn.to_torch(ttnn.get_device_tensors(replicated)[0]),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2)),
            )
            replicated.deallocate(True)
        else:
            self.tt_weight = replicated

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        self._stats_gather_logged = False

    def _gather_stats(self, tt_stats, memory_config=None):
        """All-gather the per-column sum(x^2) across the TP axis.

        Prefers the MANAGED all_gather_async (mesh_config.allgather + the CCL manager's ping-pong /
        barrier semaphores and its topology) — the raw ``ttnn.all_gather`` here previously hardcoded
        ``Topology.Ring``, which hangs on an unwrapped axis (this galaxy's TP axis under FABRIC_1D).
        Falls back to the raw op only when no CCL manager was supplied (standalone unit tests), and
        then with the mesh's actual topology rather than an assumed ring.
        """
        if self.ccl_manager is not None:
            if memory_config is not None:
                return self.mesh_config.allgather(
                    tt_stats, self.ccl_manager, memory_config=memory_config, axis=self.mesh_config.tp_axis, dim=3
                )
            return self.mesh_config.allgather(tt_stats, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3)
        if not self._stats_gather_logged:
            self._stats_gather_logged = True
            logger.warning(
                "[RMSNorm] distributed norm without a CCL manager: gathering stats with the raw "
                "ttnn.all_gather (Linear topology, unmanaged semaphores). Pass ccl_manager= in the model."
            )
        return ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=get_default_num_links(self.mesh_device),
            cluster_axis=self.mesh_config.tp_axis,
            mesh_device=self.mesh_device,
            memory_config=memory_config,
            topology=ttnn.Topology.Linear,
        )

    def forward(self, x):
        if not self.is_distributed:
            return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps)

        # Distributed: program_config/stats memory_config are None on the interleaved path (M3's
        # residual is DRAM-interleaved) and derived from the shard spec on the L1-sharded path.
        program_config = None
        stats_memory_config = None
        if x.memory_config().shard_spec is not None:
            shard_height, shard_width = x.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=x.memory_config().shard_spec.grid.bounding_box().grid_size(),
                subblock_w=1,
                block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
                block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
                inplace=False,
            )
            stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * self.mesh_device.shape[self.mesh_config.tp_axis]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )

        tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, dtype=ttnn.bfloat16)
        tt_gathered_stats = self._gather_stats(tt_stats, stats_memory_config)
        ttnn.deallocate(tt_stats)
        tt_output = ttnn.rms_norm_post_all_gather(
            x,
            tt_gathered_stats,
            program_config=program_config,
            epsilon=self.eps,
            weight=self.tt_weight,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(tt_gathered_stats)
        return tt_output
