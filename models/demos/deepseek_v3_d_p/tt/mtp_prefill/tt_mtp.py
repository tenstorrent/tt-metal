# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN modules for GLM-5.3 multi-token prediction during prefill.

Each level projects the shifted embedding and the previous level's hidden state through one decoder
layer, then normalizes the output to seed the next level. CPU reference in reference/glm_5_3/mtp.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import mesh_shard_dims
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import COMPUTE_KERNEL_CONFIG_HIFI2
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.mtp_config import MTPConfig
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import eh_proj_to_tt_layout
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TopologyArg, TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.weight_cache_dump import weight_cache_dump_mode


class TtFusedMTP(LightweightModule):
    """The MTP input projection: two distributed RMSNorms, a concat, and a TP matmul.

    Concatenating two TP-sharded activations leaves each chip a non-contiguous slice of eh_proj's
    input, so the weight rows are permuted chip-major on the host before sharding. Under TP the
    matmul and its reduce-scatter run as one fused op.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        mtp_config: MTPConfig,
        state_dict: Optional[dict] = None,
        *,
        tp_axis: int = 1,
        num_links: int = 1,
        topology: TopologyArg = ttnn.Topology.Linear,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        matmul_weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: str = "mtp_0",
        use_fused_rmsnorm: bool = False,
    ):
        """Build both norms and upload eh_proj.

        ``state_dict`` holds enorm/hnorm/eh_proj in HF layout (eh_proj is ``[H, 2H]``, un-transposed);
        it may be None only when the weights are loaded from ``weight_cache_path``. ``weights_dtype`` is
        the cached dtype; ``matmul_weights_dtype`` is what the matmul reads (cast on device after load).
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.mtp_config = mtp_config
        self.hidden_size = mtp_config.hidden_size
        self.tp_axis = tp_axis
        self.tp = mesh_device.shape[tp_axis]
        self.num_links = num_links
        self.topology = topology[tp_axis] if isinstance(topology, tuple) else topology
        self.compute_kernel_config = compute_kernel_config
        self._is_blackhole = is_blackhole()

        if state_dict is None and weight_cache_path is None:
            raise ValueError(
                "TtFusedMTP needs either state_dict or weight_cache_path. There is deliberately no "
                "random-weight fallback: a random eh_proj the CPU reference never saw turns a PCC "
                "failure into a hunt. Tests generate the torch weights and pass them in."
            )

        norm_kwargs = dict(
            mesh_device=mesh_device,
            emb_dim=self.hidden_size,
            epsilon=mtp_config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=self.topology,
            weight_cache_path=weight_cache_path,
            use_fused=use_fused_rmsnorm,
        )
        sd = state_dict or {}
        self.enorm = TtDistributedRmsNorm(
            torch_weight=sd.get("enorm"), cache_name_prefix=f"{cache_name_prefix}.enorm", **norm_kwargs
        )
        self.hnorm = TtDistributedRmsNorm(
            torch_weight=sd.get("hnorm"), cache_name_prefix=f"{cache_name_prefix}.hnorm", **norm_kwargs
        )

        self.eh_proj = self._convert_and_cache_eh_proj(
            sd.get("eh_proj"),
            hidden_size=self.hidden_size,
            tp=self.tp,
            mesh_device=mesh_device,
            tp_axis=tp_axis,
            weights_dtype=weights_dtype,
            cache_path=weight_cache_path,
            cache_name_prefix=cache_name_prefix,
            device=mesh_device,
        )
        if self.eh_proj.dtype != matmul_weights_dtype:
            # bfp8 halves the weight read: 0.26 vs 0.34 ms per 5k chunk on Galaxy 8x4, PCC 0.99992 vs bf16.
            cast = ttnn.typecast(self.eh_proj, matmul_weights_dtype)
            ttnn.deallocate(self.eh_proj)
            self.eh_proj = cast

    @staticmethod
    def _convert_and_cache_eh_proj(
        eh_proj_weight: Optional[torch.Tensor],
        *,
        hidden_size: int,
        tp: int,
        mesh_device: ttnn.MeshDevice,
        tp_axis: int,
        weights_dtype: ttnn.DataType,
        cache_path: Optional[Path],
        cache_name_prefix: Optional[str],
        device: Optional[ttnn.MeshDevice],
    ) -> ttnn.Tensor:
        """Transpose, permute, shard and optionally cache ``eh_proj``.

        With ``device=None`` this only writes cache files.
        """
        h = hidden_size
        dims = mesh_shard_dims(-2, tp_axis)

        def _to_ttnn(tensor: torch.Tensor, name: str) -> ttnn.Tensor:
            mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=dims)
            cache_file_name = (
                str(cache_path / f"{cache_name_prefix}.{name}") if cache_path and cache_name_prefix else None
            )
            return ttnn.as_tensor(
                tensor,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
                cache_file_name=cache_file_name,
                cache_dump_mode=weight_cache_dump_mode(),
            )

        if eh_proj_weight is not None:
            w = eh_proj_to_tt_layout(eh_proj_weight, tp)
        else:
            w = torch.empty(2 * h, h)
        return _to_ttnn(w, "eh_proj")

    @staticmethod
    def check_cache_complete(
        cache_path: Path, cache_name_prefix: str = "mtp_0", weights_dtype: ttnn.DataType = ttnn.bfloat16
    ) -> bool:
        """Whether the fused-MTP weight cache (both norms + eh_proj at ``weights_dtype``, the cached dtype) is present."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{cache_name_prefix}.enorm"):
            return False
        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{cache_name_prefix}.hnorm"):
            return False
        # Pinned to the cached dtype; the matmul's bfp8 copy is cast after load.
        eh_proj_file = f"{cache_name_prefix}.eh_proj_dtype_{weights_dtype.name}_layout_TILE.tensorbin"
        if not pattern_exists(eh_proj_file, "FusedMTP"):
            logger.debug(f"TTNN cache missing: {cache_name_prefix}.eh_proj")
            return False
        return True

    @staticmethod
    def build_ttnn_cache(
        state_dict: dict,
        mtp_config: MTPConfig,
        mesh_device: ttnn.MeshDevice,
        cache_path: Path,
        cache_name_prefix: str = "mtp_0",
        *,
        tp_axis: int = 1,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        """Build the fused-MTP weight cache without copying anything to the device."""
        for name in ("enorm", "hnorm"):
            TtDistributedRmsNorm.build_ttnn_cache(
                state_dict[name],
                mtp_config.hidden_size,
                mesh_device,
                cache_path,
                f"{cache_name_prefix}.{name}",
                cluster_axis=tp_axis,
            )
        TtFusedMTP._convert_and_cache_eh_proj(
            state_dict["eh_proj"],
            hidden_size=mtp_config.hidden_size,
            tp=mesh_device.shape[tp_axis],
            mesh_device=mesh_device,
            tp_axis=tp_axis,
            weights_dtype=weights_dtype,
            cache_path=cache_path,
            cache_name_prefix=cache_name_prefix,
            device=None,
        )

    # Live L1 bytes per chip allowed for the two norm outputs plus their concat (bf16): a packed pair of 5k chunks
    # on Galaxy 8x4 (1280 rows x 1536) is 15 MiB.
    L1_ACT_MAX_BYTES = 16 * 2**20

    def act_memory_config(self, rows_local: int, width_local: int) -> ttnn.MemoryConfig:
        """Where the norm outputs and concat live: L1 on Blackhole when they fit L1_ACT_MAX_BYTES (0.30 vs 0.34 ms per
        5k chunk on Galaxy 8x4), else DRAM."""
        rows = -(-rows_local // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        live_bytes = 8 * rows * width_local  # e and h (2 B x r x w each) live with their concat (2 B x r x 2w), bf16
        if self._is_blackhole and live_bytes <= self.L1_ACT_MAX_BYTES:
            return ttnn.L1_MEMORY_CONFIG
        return ttnn.DRAM_MEMORY_CONFIG

    def forward(self, embed: ttnn.Tensor, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Project the shifted embedding and the previous level's hidden into this level's input.

        Both inputs are ``[1, 1, seq_local, H/tp]`` and TP-sharded.
        """
        rows = embed.shape[-2]
        mem = self.act_memory_config(rows, embed.shape[-1])
        self.enorm.output_memcfg = mem
        self.hnorm.output_memcfg = mem
        e = self.enorm(embed)
        h = self.hnorm(hidden)

        x = ttnn.concat([e, h], dim=-1, memory_config=mem)
        ttnn.deallocate(e)
        ttnn.deallocate(h)
        if self.tp > 1 and rows % ttnn.TILE_SIZE != 0:
            # The fused op needs whole tiles of rows; keep the separate matmul and reduce-scatter for the rest.
            out_full = ttnn.matmul(
                x, self.eh_proj, compute_kernel_config=self.compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = ttnn.reduce_scatter(
                out_full, dim=-1, cluster_axis=self.tp_axis, num_links=self.num_links, topology=self.topology
            )
            ttnn.deallocate(out_full)
        elif self.tp > 1:
            out = ttnn.experimental.matmul_reduce_scatter_prefill(
                x,
                self.eh_proj,
                cluster_axis=self.tp_axis,
                scatter_dim=-1,
                topology=self.topology,
                num_links=min(self.num_links, 2),  # the fused op supports 1 or 2 links
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.compute_kernel_config.math_fidelity
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                # Ports in a column leave a 10x10 matmul grid (10 tiles per core) instead of ~84 cores at 12:
                # 0.172 vs 0.197 ms per 5k chunk on Galaxy 8x4, bit-identical.
                use_column_major_ccl=True,
            )
        else:
            out = ttnn.matmul(
                x, self.eh_proj, compute_kernel_config=self.compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        ttnn.deallocate(x)
        return out


class TtMTPModule(LightweightModule):
    """One MTP level: :class:`TtFusedMTP`, one ``TtPrefillBlock``, and ``shared_head.norm``.

    :class:`TtMTPPredictor` replays a single instance of this across every level.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config,
        model_cfg,
        state_dict: dict,
        mtp_config: Optional[MTPConfig] = None,
        *,
        seq_len: int,
        layer_idx: Optional[int] = None,
        tp_axis: int = 1,
        num_links: int = 1,
        topology: TopologyArg = ttnn.Topology.Linear,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: str = "mtp_0",
        kv_only: bool = False,
        **block_kwargs,
    ):
        """Build the fused projection, the decoder layer and ``shared_head.norm``.

        ``state_dict`` is ``{"mtp": {enorm, hnorm, eh_proj, shared_head_norm}, "layer": {...}}``, where
        ``"layer"`` is an ordinary ``TtPrefillBlock`` state dict. ``block_kwargs`` passes straight through.
        ``kv_only`` builds a block that only fills its caches: no query path, FFN/MoE or ``shared_head.norm``.
        """
        super().__init__()
        self.mtp_config = mtp_config or MTPConfig.from_hf_config(config)
        self.layer_idx = self.mtp_config.mtp_layer_idx if layer_idx is None else layer_idx
        self.mesh_device = mesh_device
        # Same fused-norm rule as the trunk's TtPrefillBlock.
        use_fused_rmsnorm = block_kwargs.get("use_fused_rmsnorm")
        if use_fused_rmsnorm is None:
            use_fused_rmsnorm = (
                getattr(model_cfg, "USE_FUSED_PREFILL_RMSNORM", False)
                and is_blackhole()
                and block_kwargs.get("is_chunked", False)
            )

        mtp_weights = state_dict["mtp"]
        self.fused = TtFusedMTP(
            mesh_device,
            self.mtp_config,
            mtp_weights,
            tp_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            weights_dtype=weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=cache_name_prefix,
            use_fused_rmsnorm=use_fused_rmsnorm,
        )

        self.layer = TtPrefillBlock(
            mesh_device=mesh_device,
            config=config,
            model_cfg=model_cfg,
            state_dict=state_dict["layer"],
            layer_idx=self.layer_idx,
            seq_len=seq_len,
            num_links=num_links,
            topology=topology,
            tp_axis=tp_axis,
            weight_cache_path=weight_cache_path,
            kv_only=kv_only,
            indexer_key_only=kv_only,  # its indexer only writes keys: no wq_b / weights_proj
            # (the MTP layer's MoE: the model's default (tt/moe_impl.py; GLM-5.3 the MoE block - its gate / shared expert
            # from the checkpoint's layer mtp_layer_idx, its routed experts from this cache's TtMoe tensorbins) unless the
            # caller passes moe_impl)
            **block_kwargs,
        )

        self.shared_head_norm = (
            None
            if kv_only
            else TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=self.mtp_config.hidden_size,
                epsilon=self.mtp_config.rms_norm_eps,
                torch_weight=mtp_weights.get("shared_head_norm"),
                cluster_axis=tp_axis,
                num_links=num_links,
                topology=self.fused.topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"{cache_name_prefix}.shared_head_norm",
                use_fused=use_fused_rmsnorm,
            )
        )

    @staticmethod
    def check_cache_complete(
        cache_path: Path,
        layer_idx: int,
        *,
        cache_name_prefix: str = "mtp_0",
        experts_per_chip: int = 8,
        model_cfg: type | None = None,
        kv_only: bool = False,
        owns_indexer: bool = True,
    ) -> bool:
        """Whether the module's cache is present: fused MTP + block + shared_head.norm, or with ``kv_only`` what a
        KV-only build loads (fused MTP, attn_norm, the MLA KV branch). When the layer ``owns_indexer``, also its
        indexer: every file, or the key side only for ``kv_only``."""
        if not TtFusedMTP.check_cache_complete(cache_path, cache_name_prefix):
            return False
        if kv_only:
            block_ok = TtPrefillBlock.check_cache_complete(
                cache_path, layer_idx, is_dense=False, model_cfg=model_cfg, kv_only=True
            )
        else:
            block_ok = TtDistributedRmsNorm.check_cache_complete(
                cache_path, f"{cache_name_prefix}.shared_head_norm"
            ) and TtPrefillBlock.check_cache_complete(
                cache_path, layer_idx, is_dense=False, experts_per_chip=experts_per_chip, model_cfg=model_cfg
            )
        if not block_ok:
            return False
        return not owns_indexer or TtIndexer.check_cache_complete(
            cache_path, f"layer_{layer_idx}.mla", key_only=kv_only
        )

    def forward(self, embed: ttnn.Tensor, hidden: ttnn.Tensor, rope_tensors: dict, kvpe_cache, **fwd_kwargs):
        """Run one MTP level.

        Returns ``(x, out, out_head_normed, *block_extras)``; ``out_head_normed`` is what the next level
        consumes. Both are None when the level runs kv_only.

        A ``chunks`` keyword (None in the traced packed form) runs the block packed (TtPrefillBlock.forward_packed).
        """
        x = self.fused(embed, hidden)
        packed = "chunks" in fwd_kwargs
        chunks = fwd_kwargs.pop("chunks", None)
        if packed:
            out, *extras = self.layer.forward_packed(x, rope_tensors, kvpe_cache, chunks, **fwd_kwargs)
        else:
            out, *extras = self.layer(x, rope_tensors, kvpe_cache, **fwd_kwargs)
        out_head_normed = self.shared_head_norm(out) if out is not None else None
        return (x, out, out_head_normed, *extras)


# Forward kwargs the predictor owns; passing one of these through would break the level loop.
_RESERVED_FWD_KWARGS = (
    "cache_layer_idx",
    "indexer_indices",
    "return_indexer_indices",
    "return_kv_cache",
    "return_kv_intermediates",
    "ack_layer_idx",
    "ack_alias_layer_ids",
    "force_kv_only",
)


@dataclass
class MTPPredictorOutput:
    """Per-level results from :meth:`TtMTPPredictor.forward`, ordered by level."""

    x: list  # fused-projection output, the decoder layer's input
    out: list  # decoder-layer output, before shared_head.norm
    out_head_normed: list  # what the next level consumes
    kv_cache: object = None
    indexer_indices: list | None = None


class TtMTPPredictor(LightweightModule):
    """K MTP levels over one shared weight module, each writing its own KV cache slot.

    ``TtPrefillBlock`` holds no per-call state, so a single built block serves every level. With index
    sharing on, level 1's top-k is injected into the rest and only one index-cache slot is written.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config,
        model_cfg,
        state_dict: dict,
        mtp_config: Optional[MTPConfig] = None,
        *,
        seq_len: int,
        num_levels: Optional[int] = None,
        decode_levels: Optional[int] = None,
        layer_idx: Optional[int] = None,
        first_cache_slot: int = 0,
        index_share: Optional[bool] = None,
        kv_only_last_level: bool = False,
        **module_kwargs,
    ):
        """Build the one shared module and fix the replay policy.

        ``first_cache_slot`` is the KV slot level 1 writes and level k writes the k-th slot after it, so
        the caller's KVPE cache must be allocated with at least that many slots. ``decode_levels`` (default
        ``num_levels``) past ``num_levels`` reuse level 1's KV on decode, so level 1 acks them too.
        """
        super().__init__()
        self.mtp_config = mtp_config or MTPConfig.from_hf_config(config)
        self.num_levels = int(self.mtp_config.num_levels if num_levels is None else num_levels)
        assert self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        self.decode_levels = self.num_levels if decode_levels is None else int(decode_levels)
        assert self.decode_levels == self.num_levels or (
            self.num_levels == 1 and self.decode_levels > 1
        ), f"decode_levels={self.decode_levels} can only extend num_levels=1, got num_levels={self.num_levels}"
        self.first_cache_slot = int(first_cache_slot)
        self.index_share = self.mtp_config.index_share_for_mtp_iteration if index_share is None else bool(index_share)
        self.kv_only_last_level = bool(kv_only_last_level)
        self.mesh_device = mesh_device
        # One KV-only level is the whole replay, so the shared block needs only its cache-filling weights.
        self.kv_only_module = self.num_levels == 1 and self.kv_only_last_level

        self.module = TtMTPModule(
            mesh_device,
            config,
            model_cfg,
            state_dict,
            self.mtp_config,
            seq_len=seq_len,
            layer_idx=layer_idx,
            kv_only=self.kv_only_module,
            **module_kwargs,
        )
        self.layer_idx = self.module.layer_idx

    @staticmethod
    def check_cache_complete(cache_path: Path, layer_idx: int, **kwargs) -> bool:
        """Whether the shared module's cache is present. K levels reuse one weight set."""
        return TtMTPModule.check_cache_complete(cache_path, layer_idx, **kwargs)

    def forward(
        self,
        get_embed,
        hidden: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache,
        *,
        index_share: Optional[bool] = None,
        return_kv_cache: bool = False,
        return_indexer_indices: bool = False,
        layer_ack_base: Optional[int] = None,
        **fwd_kwargs,
    ) -> MTPPredictorOutput:
        """Run every level, chaining each level's normed output into the next.

        ``get_embed(k, hidden)`` supplies level k's embedding lazily. ``layer_ack_base`` numbers level
        k's migration ack ``base + k``; level 1 also acks ``base + num_levels .. base + decode_levels - 1``.
        """
        for name in _RESERVED_FWD_KWARGS:
            if name in fwd_kwargs:
                raise TypeError(f"{name} is owned by TtMTPPredictor and must not be passed through fwd_kwargs")
        assert not (
            return_indexer_indices and self.kv_only_module
        ), "a KV-only MTP level computes no top-k; build with kv_only_last_level=False to read it"
        share = self.index_share if index_share is None else bool(index_share)

        xs, outs, normeds, per_level_indices = [], [], [], []
        shared_indices = None
        kv_host = None
        h = hidden

        for k in range(self.num_levels):
            embed = get_embed(k, h)
            is_last = k == self.num_levels - 1
            # Level 1's top-k only matters if a later level reuses it; with one level nothing does.
            feeds_next_level = share and k == 0 and not is_last
            want_indices = return_indexer_indices or feeds_next_level
            kwargs = dict(fwd_kwargs)
            kwargs["cache_layer_idx"] = self.first_cache_slot + k
            if layer_ack_base is not None:
                kwargs["ack_layer_idx"] = layer_ack_base + k
                if k == 0:
                    kwargs["ack_alias_layer_ids"] = tuple(
                        range(layer_ack_base + self.num_levels, layer_ack_base + self.decode_levels)
                    )
            if share and k > 0:
                kwargs["indexer_indices"] = shared_indices
            if want_indices:
                kwargs["return_indexer_indices"] = True
            if is_last and return_kv_cache:
                kwargs["return_kv_cache"] = True
            # The last level only has to fill its KV and index-key slots: prefill discards its output.
            # With one level this skips the indexer scoring, SDPA, output projection and MoE.
            if is_last and self.kv_only_last_level:
                kwargs["force_kv_only"] = True

            x, out, out_head_normed, *extras = self.module.forward(embed, h, rope_tensors, kvpe_cache, **kwargs)
            ttnn.deallocate(embed)
            if want_indices:
                kv, indices = extras
            else:
                (kv,) = extras
                indices = None
            if feeds_next_level:
                assert indices is not None, "index_share is on but level 1's MLA returned no top-k indices"
                shared_indices = indices

            xs.append(x)
            outs.append(out)
            normeds.append(out_head_normed)
            per_level_indices.append(indices)
            if is_last:
                kv_host = kv
            h = out_head_normed

        if shared_indices is not None and not return_indexer_indices:
            # A packed forward's indices are one tensor per chunk (None for an empty one).
            for t in shared_indices if isinstance(shared_indices, list) else [shared_indices]:
                if t is not None:
                    ttnn.deallocate(t)

        return MTPPredictorOutput(
            x=xs,
            out=outs,
            out_head_normed=normeds,
            kv_cache=kv_host,
            indexer_indices=per_level_indices if return_indexer_indices else None,
        )
