# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B RMSNorm — single-pass (replicated residual) or distributed (emb/tp-sharded residual).

Structure borrowed from `minimax_m3/tt/rms_norm.py` (hidden 6144, sp8×tp4, this mesh). Which form
runs is decided ONCE at construction from `tt/residual.py::use_sharded_residual`, because the two
need differently-sharded gains:

  * single-pass: `ttnn.rms_norm` over the full emb, gain REPLICATED — every TP column normalizes the
    same full vector, so the result is already replicated.
  * distributed: `rms_norm_pre_all_gather` -> all-gather the per-column sum(x^2) ->
    `rms_norm_post_all_gather`, gain sharded on dim -2 across TP. Three ops instead of one, but each
    touches only `emb/tp`.

## Two things NOT carried over from the donor

**No Gemma fold.** The donor folds `+1` into the gain at load time for MiniMax-M3's
`out = x_normed * (1 + weight)` form. Llama is plain `out = x_normed * weight` — `LlamaRMSNorm`
multiplies by the raw gain — so there is no fold here, and enabling one would be a silent accuracy
bug rather than an error.

**`ttnn.rms_norm` is safe at this width.** The interleaved kernel's circular buffers exceed the L1
budget above roughly hidden 12288 (a sibling dense-GQA bring-up had to compose the norm out of
`multiply`/`mean`/`rsqrt` for that reason). Llama's hidden is 4096 — a third of that — so the
single ttnn op is used directly and the decomposition is not needed. `hidden/tp = 1024` on the
distributed path is smaller still.
"""

from loguru import logger
from torch import nn

import ttnn
from models.demos.llama_3_1_8b_d_p.config import MeshConfig
from models.demos.llama_3_1_8b_d_p.utils.general_utils import get_cache_file_name, get_default_num_links

from .residual import use_distributed_norm

RMS_NORM_L1_HIDDEN_CEILING = 12288
"""Above roughly this hidden width `ttnn.rms_norm`'s interleaved kernel overflows L1 and throws.
Asserted at construction so a future width change fails loudly here rather than deep in a kernel."""


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
        """Build the norm's device gain and pin its form.

        Args:
            mesh_device: the open mesh.
            hf_config: anything exposing `.hidden_size` and `.rms_norm_eps`.
            state_dict: this norm's substate — `{"weight": [hidden]}` — or `{}` for cache-only.
            tensor_cache_path: prefix for the tilized weight cache, or None to skip caching.
            mesh_config: :class:`MeshConfig`; defaults to TP on the cols.
            ccl_manager: required on the distributed path (the stats all-gather is managed).
            is_distributed: None => derive from the residual scheme. Pass False to pin a norm to the
                single-pass form regardless — the model's FINAL norm does, because its input is
                gathered to full emb for the column-parallel LM head anyway.
        """
        super().__init__()
        self.hidden_size = hf_config.hidden_size
        assert self.hidden_size <= RMS_NORM_L1_HIDDEN_CEILING, (
            f"hidden {self.hidden_size} is above the ~{RMS_NORM_L1_HIDDEN_CEILING} ceiling where "
            "ttnn.rms_norm's interleaved kernel overflows L1; compose it from multiply/mean/rsqrt "
            "instead (and buy fp32 accumulation with dtype=float32 on the squaring multiply plus a "
            "compute_kernel_config on ttnn.mean - only mean accepts one)"
        )
        # Llama is PLAIN RMSNorm. Asserted rather than assumed: the donor this was borrowed from
        # reads `use_gemma_norm` off the config and folds +1 into the gain when it is set.
        assert not getattr(hf_config, "use_gemma_norm", False), "Llama has no Gemma (1 + weight) norm fold"

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.ccl_manager = ccl_manager
        if is_distributed is None:
            is_distributed = use_distributed_norm() and self.mesh_config.tp > 1
        self.is_distributed = is_distributed
        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        self.tensor_cache_path = tensor_cache_path
        self.state_dict = state_dict
        self._stats_gather_logged = False
        self.tt_weight = None
        self._build_weight()

    def _build_weight(self):
        """Tilize the gain onto the mesh.

        The REPLICATED gain is always built: it is the cached artefact (its cache key is what a
        weight-cache check looks for), and on the distributed path it is also the only source of the
        gain in cache-only mode. Its key does not depend on the residual scheme, so switching
        schemes never invalidates the tilized cache.
        """
        weight = self.state_dict.get("weight") if self.state_dict else None
        torch_weight = None if weight is None else weight.reshape((1, 1, -1, ttnn.TILE_SIZE))

        replicated = ttnn.as_tensor(
            torch_weight,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(self.tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=None,
        )
        if not self.is_distributed:
            self.tt_weight = replicated
            return

        # rms_norm_post_all_gather needs the gain to match the input width (emb/tp), so re-shard it
        # on dim -2 across the TP cols. Derived here rather than cached: in cache-only mode there is
        # no torch source, so the gain is recovered from the replicated device tensor with one tiny
        # init-time D2H. That adds no cache entry, hence no dependence on the (often other-user-owned,
        # read-only) shared weight cache being writable.
        source = torch_weight if torch_weight is not None else ttnn.to_torch(ttnn.get_device_tensors(replicated)[0])
        self.tt_weight = ttnn.from_torch(
            source,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.shard_mapper(self.mesh_device, mesh_dims=(None, -2)),
        )
        replicated.deallocate(True)

    def _gather_stats(self, tt_stats, memory_config=None):
        """All-gather the per-column sum(x^2) across the TP axis.

        Must go through the MANAGED all_gather (`mesh_config.allgather` + the CCL manager's
        ping-pong / barrier semaphores and its topology). The raw `ttnn.all_gather` with a hardcoded
        `Topology.Ring` HANGS on an unwrapped axis, which is what this galaxy's TP axis is under
        FABRIC_1D — and it hangs silently, with no error.
        """
        if self.ccl_manager is not None:
            kwargs = {} if memory_config is None else {"memory_config": memory_config}
            return self.mesh_config.allgather(
                tt_stats, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3, **kwargs
            )
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
            topology=ttnn.Topology.Linear,  # never Ring: this galaxy's TP axis is unwrapped
        )

    def forward(self, x):
        """x -> normed x, same sharding as the input."""
        if not self.is_distributed:
            return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps)

        # program_config / stats memory_config are None on the interleaved path (this model's
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
        tt_gathered = self._gather_stats(tt_stats, stats_memory_config)
        ttnn.deallocate(tt_stats)
        out = ttnn.rms_norm_post_all_gather(
            x,
            tt_gathered,
            program_config=program_config,
            epsilon=self.eps,
            weight=self.tt_weight,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(tt_gathered)
        return out
