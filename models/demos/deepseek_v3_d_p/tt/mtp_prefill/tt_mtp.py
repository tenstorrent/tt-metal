# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN modules for one GLM-5.2 MTP (Multi-Token-Prediction) level during PREFILL.

    x^k = eh_proj( cat[ enorm(embed(t_{p+k})) , hnorm(h^{k-1}[p]) ] )      TtFusedMTP
    h^k = GLM_decoder_layer(x^k)                                           TtPrefillBlock
    out = shared_head.norm(h^k)                                            TtMTPModule

``TtFusedMTP`` is the only new math on the device; the decoder layer is an ordinary
``TtPrefillBlock`` at ``layer_idx = 78`` and is not re-implemented here. CPU truth for both lives in
``reference/glm_5_2/mtp.py``. See issue #53533 / tt-blaze#1674.
"""

from __future__ import annotations

import os
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

# Debug lever. `cat([e,h]) @ Wt` is algebraically `e @ W[:, :H].T + h @ W[:, H:].T` at identical
# FLOPs, and the split form needs no row permutation at all. If PCC passes with this set and fails
# without it, `eh_proj_to_tt_layout` is wrong and nothing else is.
SPLIT_MM_ENV = "PREFILL_MTP_SPLIT_MM"


def _split_mm_enabled() -> bool:
    return os.environ.get(SPLIT_MM_ENV, "0") == "1"


class TtFusedMTP(LightweightModule):
    """The MTP input projection: two distributed RMSNorms, a concat, and a TP matmul.

    Sharding. Activations arrive TP-sharded on the last dim, so chip ``c`` holds global hidden
    columns ``[c*H/tp, (c+1)*H/tp)``. Both norms preserve that. Concatenating therefore gives chip
    ``c`` the *non-contiguous* global input columns ``{c*H/tp..} u {H + c*H/tp..}`` of ``eh_proj`` —
    which is why the weight rows are permuted chip-major on the host before sharding (see
    :func:`~models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils.eh_proj_to_tt_layout`). The contracted
    dim is the sharded one, so the matmul yields a full-width partial sum per chip and a
    ``reduce_scatter`` closes it — the same structure as ``TtFfn``'s ``down_proj`` step.
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
        """
        Args:
            mesh_device: the mesh.
            mtp_config: geometry (``hidden_size``, ``rms_norm_eps``).
            state_dict: ``{"enorm", "hnorm", "eh_proj"}`` torch tensors in **HF layout**
                (``eh_proj`` is ``[H, 2H]``, un-transposed). None only when loading from cache.
            tp_axis: mesh axis the hidden dim is sharded over. Must be 1 (see the assert).
            topology: either a single ``ttnn.Topology`` or the ``(sp, tp)`` tuple
                ``per_axis_topology()`` returns; only the TP entry is used, since every collective
                here runs on the TP axis. Resolved the same way as ``TtPrefillBlock``.
            weights_dtype: ``eh_proj``'s device dtype. Defaults to bfloat16 to match the
                checkpoint exactly — ``eh_proj`` is the one weight on layer 78 that ships BF16 with
                no ``weight_scale_inv``, and this is the module's only matmul, so there is nothing
                to gain by degrading it.
            weight_cache_path / cache_name_prefix: ttnn weight cache location.
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
        # per_axis_topology() returns (sp_topology, tp_topology); every collective in this module
        # (both norms' stat gather, and the reduce_scatter) runs on the TP axis. Mirrors
        # tt_prefill_block.py:280 so a tuple never reaches ttnn.
        self.topology = topology[tp_axis] if isinstance(topology, tuple) else topology
        self.compute_kernel_config = compute_kernel_config
        self.split_mm = _split_mm_enabled()
        if self.split_mm:
            logger.warning(f"{SPLIT_MM_ENV}=1: running the two-matmul form of eh_proj (debug lever, not serving)")

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

        weights = self._convert_and_cache_eh_proj(
            sd.get("eh_proj"),
            hidden_size=self.hidden_size,
            tp=self.tp,
            mesh_device=mesh_device,
            tp_axis=tp_axis,
            weights_dtype=weights_dtype,
            cache_path=weight_cache_path,
            cache_name_prefix=cache_name_prefix,
            split_mm=self.split_mm,
            device=mesh_device,
        )
        self.eh_proj = weights.get("eh_proj")
        self.eh_proj_e = weights.get("eh_proj_e")
        self.eh_proj_h = weights.get("eh_proj_h")

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
        split_mm: bool,
        device: Optional[ttnn.MeshDevice],
    ) -> dict:
        """Transpose, permute, shard and (optionally) cache ``eh_proj``.

        ``device=None`` builds cache files only. Shared by ``__init__`` and :meth:`build_ttnn_cache`
        so the cached bytes are produced by exactly the code that consumes them.
        """
        h = hidden_size
        # dims is (rows_axis, cols_axis) of the mesh: shard the contracted dim over the TP axis,
        # replicate over the other. Matches TtSharedExpert's down_proj entry.
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

        if split_mm:
            # e @ W[:, :H].T and h @ W[:, H:].T: each half is already contiguous in the sharded dim,
            # so no permutation is involved. That is the whole point of the lever.
            if eh_proj_weight is not None:
                e_w = eh_proj_weight[:, :h].t().contiguous()
                h_w = eh_proj_weight[:, h:].t().contiguous()
            else:
                e_w = torch.empty(h, h)
                h_w = torch.empty(h, h)
            return {"eh_proj_e": _to_ttnn(e_w, "eh_proj_e"), "eh_proj_h": _to_ttnn(h_w, "eh_proj_h")}

        if eh_proj_weight is not None:
            w = eh_proj_to_tt_layout(eh_proj_weight, tp)
        else:
            w = torch.empty(2 * h, h)
        return {"eh_proj": _to_ttnn(w, "eh_proj")}

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str = "mtp_0") -> bool:
        """Whether the fused-MTP weight cache (both norms + eh_proj) is present."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{cache_name_prefix}.enorm"):
            return False
        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{cache_name_prefix}.hnorm"):
            return False
        # The concat form only — the split form is a debug lever and is never the serving cache.
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
            split_mm=False,
            device=None,
        )

    def forward(self, embed: ttnn.Tensor, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Project the shifted embedding and the previous level's hidden into this level's input.

        Args:
            embed: ``[1, 1, seq_local, H/tp]`` TILE_LAYOUT, TP-sharded — the embedding of the
                SHIFTED token ids. **Rows at absolute position 0 must already be zeroed by the
                caller**, matching vLLM (``torch.where(positions.unsqueeze(-1) == 0, 0, embeds)``)
                and ``fused_mtp_reference``. It is not done here because under SP the row index is
                not the absolute position, so only the caller knows which rows qualify.
            hidden: ``[1, 1, seq_local, H/tp]`` — ``h^{k-1}``. For level 1 this is the trunk output
                taken AFTER ``model.norm``.

        Returns:
            ``x^k``, ``[1, 1, seq_local, H/tp]``, TP-sharded like the inputs.
        """
        e = self.enorm(embed)
        h = self.hnorm(hidden)

        if self.split_mm:
            out_full = ttnn.matmul(e, self.eh_proj_e, compute_kernel_config=self.compute_kernel_config)
            out_full = ttnn.add(
                out_full, ttnn.matmul(h, self.eh_proj_h, compute_kernel_config=self.compute_kernel_config)
            )
        else:
            x = ttnn.concat([e, h], dim=-1)
            out_full = ttnn.matmul(x, self.eh_proj, compute_kernel_config=self.compute_kernel_config)

        # Contracted dim was sharded, so every chip holds a full-width partial sum.
        if self.mesh_device.shape[self.tp_axis] > 1:
            return ttnn.reduce_scatter(
                out_full, dim=-1, cluster_axis=self.tp_axis, num_links=self.num_links, topology=self.topology
            )
        return out_full


class TtMTPModule(LightweightModule):
    """One complete MTP module: :class:`TtFusedMTP` + one ``TtPrefillBlock`` + ``shared_head.norm``.

    The name is the DeepSeek-V3 paper's own term for {norms, concat, projection} plus one
    Transformer Block. It is one *level*; a future ``TtMTPPredictor`` holds K of them (K caches,
    one shared weight set) for MTP4.
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
        """
        Args:
            config: the GLM HF-attribute config (``glm_5_2_hf_config()``).
            model_cfg: ``GLM52Config``.
            state_dict: ``{"mtp": {enorm, hnorm, eh_proj, shared_head_norm}, "layer": {...}}`` where
                ``"layer"`` is an ordinary ``TtPrefillBlock`` state dict (``attn_norm_weight``,
                ``ffn_norm_weight``, ``mla_weights``, ``gate_weights``, ``routed_expert_weights``,
                ``shared_expert_weights``).
            mtp_config: defaults to ``MTPConfig.from_hf_config(config)``.
            seq_len: sequence length the block is built for.
            layer_idx: the layer the MTP weights live on. Defaults to ``mtp_config.mtp_layer_idx``
                (78). ``TtPrefillBlock`` derives ``is_moe = layer_idx >= NUM_DENSE_LAYERS``, so 78
                selects the MoE path with no config change.
            block_kwargs: passed through to ``TtPrefillBlock`` unchanged (gate mode, chunking,
                trace, dispatch sizing, ...), so its knobs stay available without re-declaring them.
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
            # TtPrefillBlock takes the raw arg (it resolves per-axis itself); a bare norm does not.
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
        # The MTP layer is MoE (78 >= first_k_dense_replace), never dense.
        return TtPrefillBlock.check_cache_complete(
            cache_path, layer_idx, is_dense=False, experts_per_chip=experts_per_chip, model_cfg=model_cfg
        )

    def forward(self, embed: ttnn.Tensor, hidden: ttnn.Tensor, rope_tensors: dict, kvpe_cache, **fwd_kwargs):
        """Run one MTP level.

        Args:
            embed / hidden: see :meth:`TtFusedMTP.forward`.
            rope_tensors / kvpe_cache / fwd_kwargs: passed to ``TtPrefillBlock.forward`` unchanged.

        Returns:
            ``(x, out, out_head_normed, *block_extras)`` — the same leading triple as
            ``glm_mtp_module_reference``, followed by whatever else the block returned (the KV cache,
            plus indexer indices when ``return_indexer_indices=True``).

            Both ``out`` (pre-``shared_head.norm``) and ``out_head_normed`` are returned on purpose:
            which one feeds level k+1's ``hnorm`` is a live question at MTP2, and returning both
            makes it a PCC comparison rather than a guess.
        """
        x = self.fused(embed, hidden)
        out, *extras = self.layer(x, rope_tensors, kvpe_cache, **fwd_kwargs)
        out_head_normed = self.shared_head_norm(out) if out is not None else None
        return (x, out, out_head_normed, *extras)
