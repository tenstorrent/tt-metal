# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN modules for GLM-5.2 MTP (Multi-Token-Prediction) during PREFILL.

    x^k = eh_proj( cat[ enorm(embed(t_{p+k})) , hnorm(H^{k-1}[p]) ] )      TtFusedMTP
    h^k = GLM_decoder_layer(x^k)                                           TtPrefillBlock
    H^k = shared_head.norm(h^k)                                            TtMTPModule
    for k in 1..K, seeded by H^0 = the trunk output after model.norm       TtMTPPredictor

``TtFusedMTP`` is the only new math on the device; the decoder layer is an ordinary
``TtPrefillBlock`` at ``layer_idx = 78`` and is not re-implemented here. ``TtMTPPredictor`` is the
K-level container — one weight module replayed K times over K KV caches, with the indexer run once
and shared. CPU truth for all three lives in ``reference/glm_5_2/mtp.py``. See issue #53533 /
tt-blaze#1674.
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

        x = ttnn.concat([e, h], dim=-1)
        # Every intermediate here is local to this call and consumed exactly once, so each is freed as
        # soon as its consumer is enqueued. TtMTPPredictor replays this per level, so anything left
        # behind is multiplied by K. `embed` and `hidden` are the caller's and are never freed here:
        # `hidden` is the previous level's output, which the predictor still holds.
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
    """One complete MTP module: :class:`TtFusedMTP` + one ``TtPrefillBlock`` + ``shared_head.norm``.

    The name is the DeepSeek-V3 paper's own term for {norms, concat, projection} plus one
    Transformer Block. It is one *level*; :class:`TtMTPPredictor` replays a single instance of it
    across K levels (K caches, one shared weight set) for MTP4.
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
            makes it a PCC comparison rather than a guess. :class:`TtMTPPredictor` exposes the choice
            as ``chain_from``.
        """
        x = self.fused(embed, hidden)
        out, *extras = self.layer(x, rope_tensors, kvpe_cache, **fwd_kwargs)
        out_head_normed = self.shared_head_norm(out) if out is not None else None
        return (x, out, out_head_normed, *extras)


# ``h^{k-1}`` — which of level k-1's two output forms feeds level k's ``hnorm``. Mirrors
# reference/glm_5_2/mtp.py's constants of the same name; kept a runtime choice on both sides so
# settling it is a flag flip and a PCC comparison rather than an edit.
CHAIN_FROM_NORM = "norm"  # out_head_normed, i.e. shared_head.norm(h^{k-1})  [default]
CHAIN_FROM_RAW = "raw"  # out, i.e. h^{k-1} straight off the decoder layer
CHAIN_FROM_CHOICES = (CHAIN_FROM_NORM, CHAIN_FROM_RAW)

# fwd_kwargs the predictor owns and a caller must not set: they are what makes it a *predictor*
# rather than K independent module calls.
_RESERVED_FWD_KWARGS = (
    "cache_layer_idx",  # the per-level KV slot; the whole point of K caches
    "indexer_indices",  # level 1's top-k, injected into 2..K by index_share
    "return_indexer_indices",  # promoted to a named argument below
    "return_kv_cache",  # promoted to a named argument below
    "return_kv_intermediates",  # changes TtPrefillBlock's return arity out from under the loop
)


def _embed_getter(embeds, num_levels: int):
    """Normalize ``TtMTPPredictor.forward``'s ``embeds`` to ``(k, H^k) -> ttnn.Tensor``.

    A pre-built sequence is the simple case and stays supported: tests feed K unrelated random
    activations, and an interior chunk's K windows are all pure prompt slices that could equally
    well be built up front.

    A callable is what the LAST chunk of a request needs. There, level ``k``'s *input* depends on
    level ``k-1``'s *output* -- the window's last ``k`` rows want ``t_P .. t_{P+k-1}``, and
    ``t_{P+k-1}`` is ``argmax lm_head(H^{k-1})`` at the last real row -- so no list can be
    materialized before the loop runs. See
    :class:`~models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows.MTPDeviceEmbedSource`.

    The callable is handed ``H^k`` (the trunk output after ``model.norm`` at ``k=0``, and the
    previous level's chained output after that), which is exactly the tensor its lm_head needs.
    """
    if callable(embeds):
        return embeds
    materialized = list(embeds)
    assert len(materialized) == num_levels, f"expected {num_levels} embeddings (one per level), got {len(materialized)}"
    return lambda k, _prev: materialized[k]


@dataclass
class MTPPredictorOutput:
    """Per-level results from :meth:`TtMTPPredictor.forward`. Lists are level-ordered, k = 1..K."""

    x: list  # x^k, the fused-projection output = level k's decoder-layer input
    out: list  # h^k, the decoder-layer output BEFORE shared_head.norm
    out_head_normed: list  # H^k = shared_head.norm(h^k)
    kv_cache: object = None  # host KVPE for ALL slots, [K, 1, seq, kv+pe], or None
    indexer_indices: list | None = None  # per-level top-k, or None


class TtMTPPredictor(LightweightModule):
    """K MTP levels over ONE shared weight module — the GLM-5.2 / DeepSeek-V3 MTP scheme.

    K levels are predicted at ONE position and write K separate KV caches, all from a single set of
    MTP weights replayed K times. It is NOT EAGLE-style autoregressive drafting, and
    ``num_nextn_predict_layers`` in the checkpoint counts weight *modules* (1), not levels::

        H^0 = hidden                                   # trunk output, taken AFTER model.norm
        for k in 1..K:
            x^k = eh_proj( cat[ enorm(embed(t_{p+k})) , hnorm(H^{k-1}) ] )
            h^k = GLM_decoder_layer_78(x^k)            # writes KV slot first_cache_slot + k - 1
            H^k = shared_head.norm(h^k)

    Replay is safe because ``TtPrefillBlock`` carries no per-call state: the KV slot arrives as the
    forward argument ``cache_layer_idx``, so one built block serves every level and the K levels cost
    one block's worth of device memory, not K.

    **Index sharing.** GLM-5.2 sets ``index_share_for_mtp_iteration``, so the indexer runs once, on
    level 1, and levels 2..K attend at its top-k. On device this needs no special block: ttMLA's
    indexer call is ``indices = indexer_indices if indexer_indices is not None else
    self._indexer.forward(...)``, which does not consult ``self._indexer_reuse``. Injecting level 1's
    indices into a block built "full" therefore skips both the top-k computation *and* its index-K
    cache write, which is exactly the sharing semantics — and is why the whole MTP stack costs one
    index-cache slot, matching GLM-5.2's 22 full-indexer layers.

    With ``index_share=False`` every level runs the indexer, and because the index-cache slot is
    derived from the block's static ``layer_idx`` (``TtIndexer._cache_slot``) rather than from
    ``cache_layer_idx``, all K levels write the SAME index-K slot, last-writer-wins. That is
    self-consistent inside one single-shot prefill — each level writes the full row range and reads
    it back in the same call — but it leaves the index cache holding only level K's keys, so it is
    not a valid handoff to decode. Non-shared MTP would need one index slot per level.
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
        chain_from: str = CHAIN_FROM_NORM,
        **module_kwargs,
    ):
        """
        Args:
            config / model_cfg / state_dict / mtp_config / seq_len / layer_idx / module_kwargs:
                passed to :class:`TtMTPModule` unchanged — one module is built and replayed.
            num_levels: K. Defaults to ``mtp_config.num_levels``.
            first_cache_slot: KV slot level 1 writes; level k writes ``first_cache_slot + k - 1``.
                The caller's KVPE cache must therefore have at least ``first_cache_slot + K`` slots
                (``init_mla_kv_cache(num_kvpe_cache_layers=...)``).
            index_share: default for :meth:`forward`'s override. Defaults to
                ``mtp_config.index_share_for_mtp_iteration`` (True for GLM-5.2).
            chain_from: default for :meth:`forward`'s override; see :data:`CHAIN_FROM_CHOICES`.
        """
        super().__init__()
        self.mtp_config = mtp_config or MTPConfig.from_hf_config(config)
        self.num_levels = int(self.mtp_config.num_levels if num_levels is None else num_levels)
        assert self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        self.first_cache_slot = int(first_cache_slot)
        self.index_share = self.mtp_config.index_share_for_mtp_iteration if index_share is None else bool(index_share)
        assert chain_from in CHAIN_FROM_CHOICES, f"chain_from must be one of {CHAIN_FROM_CHOICES}, got {chain_from!r}"
        self.chain_from = chain_from
        self.mesh_device = mesh_device

        # ONE module, K activations, K caches. Rebuilding it per level would upload layer 78's
        # 256 experts K times for identical weights.
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
        embeds,
        hidden: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache,
        *,
        index_share: Optional[bool] = None,
        chain_from: Optional[str] = None,
        return_kv_cache: bool = False,
        return_indexer_indices: bool = False,
        **fwd_kwargs,
    ) -> MTPPredictorOutput:
        """Run K MTP levels.

        Args:
            embeds: either K tensors, ``embeds[k-1]`` = the embedding of ``t_{p+k}``, each
                ``[1, 1, seq_local, H/tp]``; or a callable ``(k, H^k) -> ttnn.Tensor`` returning
                level ``k+1``'s embedding, called once per level in order. The callable form is
                required on the last chunk of a request, where level k's window contains tokens
                only level k-1's lm_head can produce, and it keeps peak embedding memory at one
                level's worth instead of K. **A tensor from a callable is deallocated by the
                predictor once its level has run; tensors from a sequence stay the caller's.**
                **Rows at absolute position 0 must already be zeroed by the caller, on every
                level** — vLLM zeroes at position 0 for all k, not just k=1. See
                :meth:`TtFusedMTP.forward` for why the module cannot do it itself.
            hidden: ``H^0``, the trunk output taken AFTER ``model.norm``.
            rope_tensors / kvpe_cache / fwd_kwargs: forwarded to ``TtPrefillBlock.forward``. The
                predictor owns ``cache_layer_idx``, ``indexer_indices``, ``return_indexer_indices``,
                ``return_kv_cache`` and ``return_kv_intermediates``; passing any of them raises.
            index_share / chain_from: per-call overrides of the construction-time defaults. Both are
                pure runtime policy — nothing about the built block depends on either — so an A/B is
                two forwards over one set of device weights.
            return_kv_cache: read the KVPE cache back to host. Requested on the LAST level only: the
                cache is persistent and cumulative, so one readback already holds every level's slot.
            return_indexer_indices: also return the per-level top-k. With ``index_share`` on, every
                entry after the first IS level 1's tensor (ttMLA returns the injected object
                unchanged), so the list holds one unique tensor — deallocate unique objects only.
                With it off, all K are distinct and all K are the caller's to free.

        Returns:
            :class:`MTPPredictorOutput`. ``kv_cache`` is ``[K, 1, seq, kv_lora_rank +
            qk_rope_head_dim]`` when requested, directly comparable to
            ``glm_mtp_predictor_reference``'s fourth return value.
        """
        for name in _RESERVED_FWD_KWARGS:
            if name in fwd_kwargs:
                raise TypeError(f"{name} is owned by TtMTPPredictor and must not be passed through fwd_kwargs")
        share = self.index_share if index_share is None else bool(index_share)
        chain = self.chain_from if chain_from is None else chain_from
        assert chain in CHAIN_FROM_CHOICES, f"chain_from must be one of {CHAIN_FROM_CHOICES}, got {chain!r}"
        # A callable builds each level's embedding on demand, so the predictor owns and frees it;
        # a sequence was built by the caller, who keeps it.
        owns_embeds = callable(embeds)
        get_embed = _embed_getter(embeds, self.num_levels)

        xs, outs, normeds, per_level_indices = [], [], [], []
        shared_indices = None
        kv_host = None
        h = hidden

        for k in range(self.num_levels):
            # h is H^k here: the trunk output at k=0, the previous level's chained output after.
            embed = get_embed(k, h)
            is_last = k == self.num_levels - 1
            # Level 1 always computes its own top-k: it is the level the others share FROM.
            want_indices = return_indexer_indices or (share and k == 0)
            kwargs = dict(fwd_kwargs)
            kwargs["cache_layer_idx"] = self.first_cache_slot + k
            if share and k > 0:
                kwargs["indexer_indices"] = shared_indices
            if want_indices:
                kwargs["return_indexer_indices"] = True
            if is_last and return_kv_cache:
                kwargs["return_kv_cache"] = True

            x, out, out_head_normed, *extras = self.module.forward(embed, h, rope_tensors, kvpe_cache, **kwargs)
            if owns_embeds:
                # TtFusedMTP reads `embed` once (enorm) and does not consume it, so it is dead here.
                ttnn.deallocate(embed)
            if want_indices:
                kv, indices = extras
            else:
                (kv,) = extras
                indices = None
            if share and k == 0:
                # A block whose MLA returned no indices (kv_only) would leave this None, and levels
                # 2..K would silently fall back to computing their own top-k — sharing off, no error.
                assert indices is not None, "index_share is on but level 1's MLA returned no top-k indices"
                shared_indices = indices

            xs.append(x)
            outs.append(out)
            normeds.append(out_head_normed)
            per_level_indices.append(indices)
            if is_last:
                kv_host = kv
            h = out_head_normed if chain == CHAIN_FROM_NORM else out

        if shared_indices is not None and not return_indexer_indices:
            # Same lifetime rule as the transformer's reuse loop (tt_prefill_transformer.py): the
            # holder frees the held indices once the last consumer has run.
            ttnn.deallocate(shared_indices)

        return MTPPredictorOutput(
            x=xs,
            out=outs,
            out_head_normed=normeds,
            kv_cache=kv_host,
            indexer_indices=per_level_indices if return_indexer_indices else None,
        )
