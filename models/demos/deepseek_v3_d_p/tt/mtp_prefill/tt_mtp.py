# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN modules for GLM-5.2 multi-token prediction during prefill.

Each level projects the shifted embedding and the previous level's hidden state through one decoder
layer, then normalizes the output to seed the next level. CPU reference in reference/glm_5_2/mtp.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import COMPUTE_KERNEL_CONFIG_HIFI2
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.mtp_config import MTPConfig
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import eh_proj_to_tt_layout
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TopologyArg, TtPrefillBlock


class TtFusedMTP(LightweightModule):
    """The MTP input projection: two distributed RMSNorms, a concat, and a TP matmul.

    Concatenating two TP-sharded activations leaves each chip a non-contiguous slice of eh_proj's
    input, so the weight rows are permuted chip-major on the host before sharding.
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
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: str = "mtp_0",
    ):
        """Build both norms and upload eh_proj.

        ``state_dict`` holds enorm/hnorm/eh_proj in HF layout (eh_proj is ``[H, 2H]``, un-transposed);
        it may be None only when the weights are loaded from ``weight_cache_path``.
        """
        super().__init__()
        assert tp_axis == 1, (
            f"tp_axis={tp_axis} unsupported: TtDistributedRmsNorm shards its weight with a hard-coded "
            "dims=(None, 2) mapper (tt_distributed_rms_norm.py:_convert_and_cache_weight), so enorm/hnorm "
            "shard over mesh columns whatever cluster_axis says. tp_axis=0 needs that fixed first."
        )
        self.mesh_device = mesh_device
        self.mtp_config = mtp_config
        self.hidden_size = mtp_config.hidden_size
        self.tp_axis = tp_axis
        self.tp = mesh_device.shape[tp_axis]
        self.num_links = num_links
        # Every collective here runs on the TP axis, so a per-axis tuple resolves to that entry.
        self.topology = topology[tp_axis] if isinstance(topology, tuple) else topology
        self.compute_kernel_config = compute_kernel_config

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

        With ``device=None`` this only writes cache files, so the bytes are produced by the same code
        that later consumes them.
        """
        h = hidden_size
        # Shard the contracted dim over the TP axis and replicate over the other.
        dims = (None, -2) if tp_axis == 1 else (-2, None)

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
            )

        if eh_proj_weight is not None:
            w = eh_proj_to_tt_layout(eh_proj_weight, tp)
        else:
            w = torch.empty(2 * h, h)
        return _to_ttnn(w, "eh_proj")

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str = "mtp_0") -> bool:
        """Whether the fused-MTP weight cache (both norms + eh_proj) is present."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{cache_name_prefix}.enorm"):
            return False
        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{cache_name_prefix}.hnorm"):
            return False
        if not pattern_exists(f"{cache_name_prefix}.eh_proj*.tensorbin", "FusedMTP"):
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
                state_dict[name], mtp_config.hidden_size, mesh_device, cache_path, f"{cache_name_prefix}.{name}"
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

    def forward(self, embed: ttnn.Tensor, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Project the shifted embedding and the previous level's hidden into this level's input.

        Both inputs are ``[1, 1, seq_local, H/tp]`` and TP-sharded.
        """
        e = self.enorm(embed)
        h = self.hnorm(hidden)

        x = ttnn.concat([e, h], dim=-1)
        # Freed as the consumer is enqueued: the predictor replays this per level, so anything left
        # behind is multiplied by K. `embed` and `hidden` belong to the caller.
        ttnn.deallocate(e)
        ttnn.deallocate(h)
        out_full = ttnn.matmul(x, self.eh_proj, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(x)

        # Contracted dim was sharded, so every chip holds a full-width partial sum.
        if self.mesh_device.shape[self.tp_axis] > 1:
            out = ttnn.reduce_scatter(
                out_full, dim=-1, cluster_axis=self.tp_axis, num_links=self.num_links, topology=self.topology
            )
            ttnn.deallocate(out_full)
            return out
        return out_full


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
        **block_kwargs,
    ):
        """Build the fused projection, the decoder layer and ``shared_head.norm``.

        ``state_dict`` is ``{"mtp": {enorm, hnorm, eh_proj, shared_head_norm}, "layer": {...}}``, where
        ``"layer"`` is an ordinary ``TtPrefillBlock`` state dict. ``block_kwargs`` passes straight through.
        """
        super().__init__()
        self.mtp_config = mtp_config or MTPConfig.from_hf_config(config)
        self.layer_idx = self.mtp_config.mtp_layer_idx if layer_idx is None else layer_idx
        self.mesh_device = mesh_device

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
            **block_kwargs,
        )

        self.shared_head_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=self.mtp_config.hidden_size,
            epsilon=self.mtp_config.rms_norm_eps,
            torch_weight=mtp_weights.get("shared_head_norm"),
            cluster_axis=tp_axis,
            num_links=num_links,
            # TtPrefillBlock resolves a per-axis topology itself; a bare norm needs it resolved.
            topology=self.fused.topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"{cache_name_prefix}.shared_head_norm",
        )

    @staticmethod
    def check_cache_complete(
        cache_path: Path,
        layer_idx: int,
        *,
        cache_name_prefix: str = "mtp_0",
        experts_per_chip: int = 8,
        model_cfg: type | None = None,
    ) -> bool:
        """Whether the whole module's cache (fused MTP + block + shared_head.norm) is present."""
        if not TtFusedMTP.check_cache_complete(cache_path, cache_name_prefix):
            return False
        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{cache_name_prefix}.shared_head_norm"):
            return False
        # The MTP layer is always MoE, never dense.
        return TtPrefillBlock.check_cache_complete(
            cache_path, layer_idx, is_dense=False, experts_per_chip=experts_per_chip, model_cfg=model_cfg
        )

    def forward(self, embed: ttnn.Tensor, hidden: ttnn.Tensor, rope_tensors: dict, kvpe_cache, **fwd_kwargs):
        """Run one MTP level.

        Returns ``(x, out, out_head_normed, *block_extras)``. ``out_head_normed`` is what the next
        level consumes; ``out`` is returned beside it as its own comparison point.
        """
        x = self.fused(embed, hidden)
        out, *extras = self.layer(x, rope_tensors, kvpe_cache, **fwd_kwargs)
        out_head_normed = self.shared_head_norm(out) if out is not None else None
        return (x, out, out_head_normed, *extras)


# Forward kwargs the predictor owns; passing one of these through would break the level loop.
_RESERVED_FWD_KWARGS = (
    "cache_layer_idx",  # the per-level KV slot
    "indexer_indices",  # level 1's top-k, injected into the later levels
    "return_indexer_indices",  # promoted to a named argument
    "return_kv_cache",  # promoted to a named argument
    "return_kv_intermediates",  # would change TtPrefillBlock's return arity
    "ack_layer_idx",  # renumbered per level off layer_ack_base
)


@dataclass
class MTPPredictorOutput:
    """Per-level results from :meth:`TtMTPPredictor.forward`, ordered by level."""

    x: list  # the fused-projection output, i.e. the decoder layer's input
    out: list  # decoder-layer output, before shared_head.norm
    out_head_normed: list  # the level's output, what the next level consumes
    kv_cache: object = None  # host KVPE for every slot, or None
    indexer_indices: list | None = None  # per-level top-k, or None


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
        layer_idx: Optional[int] = None,
        first_cache_slot: int = 0,
        index_share: Optional[bool] = None,
        **module_kwargs,
    ):
        """Build the one shared module and fix the replay policy.

        ``first_cache_slot`` is the KV slot level 1 writes and level k writes the k-th slot after it, so
        the caller's KVPE cache must be allocated with at least that many slots.
        """
        super().__init__()
        self.mtp_config = mtp_config or MTPConfig.from_hf_config(config)
        self.num_levels = int(self.mtp_config.num_levels if num_levels is None else num_levels)
        assert self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        self.first_cache_slot = int(first_cache_slot)
        self.index_share = self.mtp_config.index_share_for_mtp_iteration if index_share is None else bool(index_share)
        self.mesh_device = mesh_device

        # One module, replayed: rebuilding per level would re-upload the MTP layer's experts each time.
        self.module = TtMTPModule(
            mesh_device,
            config,
            model_cfg,
            state_dict,
            self.mtp_config,
            seq_len=seq_len,
            layer_idx=layer_idx,
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

        ``get_embed(k, hidden)`` supplies level k's embedding lazily. Each embedding is deallocated
        once its level has run. ``layer_ack_base`` numbers level k's migration ack ``base + k``: one
        replayed module would otherwise ack every level under the same layer.
        """
        for name in _RESERVED_FWD_KWARGS:
            if name in fwd_kwargs:
                raise TypeError(f"{name} is owned by TtMTPPredictor and must not be passed through fwd_kwargs")
        share = self.index_share if index_share is None else bool(index_share)

        xs, outs, normeds, per_level_indices = [], [], [], []
        shared_indices = None
        kv_host = None
        h = hidden

        for k in range(self.num_levels):
            # h is H^k here: the trunk output at k=0, the previous level's chained output after.
            embed = get_embed(k, h)
            is_last = k == self.num_levels - 1
            # Level 1 always computes its own top-k; the others share it.
            want_indices = return_indexer_indices or (share and k == 0)
            kwargs = dict(fwd_kwargs)
            kwargs["cache_layer_idx"] = self.first_cache_slot + k
            if layer_ack_base is not None:
                kwargs["ack_layer_idx"] = layer_ack_base + k
            if share and k > 0:
                kwargs["indexer_indices"] = shared_indices
            if want_indices:
                kwargs["return_indexer_indices"] = True
            if is_last and return_kv_cache:
                kwargs["return_kv_cache"] = True

            x, out, out_head_normed, *extras = self.module.forward(embed, h, rope_tensors, kvpe_cache, **kwargs)
            # The fused projection reads `embed` once, so it is dead after the module call.
            ttnn.deallocate(embed)
            if want_indices:
                kv, indices = extras
            else:
                (kv,) = extras
                indices = None
            if share and k == 0:
                # Without this the later levels would silently compute their own top-k instead.
                assert indices is not None, "index_share is on but level 1's MLA returned no top-k indices"
                shared_indices = indices

            xs.append(x)
            outs.append(out)
            normeds.append(out_head_normed)
            per_level_indices.append(indices)
            if is_last:
                kv_host = kv
            # The next level's hnorm consumes the normed output, not the raw block output.
            h = out_head_normed

        if shared_indices is not None and not return_indexer_indices:
            # The holder frees the shared indices once the last consumer has run.
            ttnn.deallocate(shared_indices)

        # With sharing on, every entry after the first is level 1's tensor: free unique objects only.
        return MTPPredictorOutput(
            x=xs,
            out=outs,
            out_head_normed=normeds,
            kv_cache=kv_host,
            indexer_indices=per_level_indices if return_indexer_indices else None,
        )
