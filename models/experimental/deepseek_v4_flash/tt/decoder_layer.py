"""The decoder block: hyper-connection -> attention -> mix -> hyper-connection -> MoE -> mix.

``B`` = users decoded per step, ``S`` = query length (1 on the decode path), ``hc`` =
``hc_mult`` (the hyper-connection residual-stream count) and ``D`` = ``hidden_size``. The
residual is a stack of ``hc`` parallel streams ``[B, S, hc, D]`` carried through the whole
block, so both sublayers read and write that one tensor.
"""

from typing import Optional

import torch
import ttnn

from .attention import (
    DeepSeekV4Attention,
    _StaticLayerCache,
)
from .common import DeepSeekV4Module, _HIFI4, _profile, _region
from .hyperconnection import DeepSeekV4HyperConnection
from .layers import DeepSeekV4RMSNorm
from .moe import DeepSeekV4SparseMoeBlock
from .paged_cache import PagedLayerView
from .weight_cache import WeightCache, _as_cache


def _strip_prefix(weights: dict, prefix: str) -> dict:
    """Sub-dict of ``weights`` whose keys start with ``prefix.`` (prefix stripped).

    The values are passed through untouched, so a weight keeps its own shape (``[K, N]`` for a
    projection) and a lazy thunk stays lazy.
    """
    p = f"{prefix}."
    return {k[len(p) :]: v for k, v in weights.items() if k.startswith(p)}


class DeepSeekV4DecoderLayer(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4DecoderLayer`` (decode).

    The residual is a stack of ``hc`` parallel streams kept in ``[B, S, hc, D]`` throughout the
    block, mixed in/out by two :class:`DeepSeekV4HyperConnection` modules. For each sublayer
    (attention, then MoE) the matching HC collapses the streams into the sublayer input, and the
    sublayer output is folded back into the streams via the learned ``post`` placement weights
    plus the Sinkhorn ``comb`` stream-mixing matrix::

        post, comb, collapsed = hc(streams)
        out = sublayer(norm(collapsed))
        streams = post * out + (comb.T @ streams)

    ``comb`` is consumed *transposed* (mix over the first hc axis), matching the
    reference ``torch.matmul(comb.transpose(-1, -2), streams)``.

    ``weights`` keys mirror the HF decoder-layer param names: ``self_attn.*``,
    ``mlp.*``, ``attn_hc.{fn,base,scale}``, ``ffn_hc.{fn,base,scale}``,
    ``input_layernorm.weight``, ``post_attention_layernorm.weight``. RoPE tables
    and the additive mask are inputs (built by the surrounding model / test, per
    :func:`make_rope_table`), exactly as for :class:`DeepSeekV4Attention`.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        experts=None,
        gate=None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        use_prefetcher: bool = False,
        prefetch_buffers: Optional[dict] = None,
        tp_size: int = 1,
    ):
        """Build layer ``layer_idx``: attention, MoE, the two hyper-connections and the two
        ``[D]`` RMSNorms.

        ``weights`` is this layer's HF-named parameter dict (see the class docstring),
        ``experts`` the injected routed-expert compute module and ``gate`` an optional router
        override (e.g. :class:`~.moe.DeepSeekV4HashRouter` for the hash layers), ``cache`` the
        tile-cache namespace for this layer. ``prefetch_buffers`` is the per-device mapping from
        :func:`~.decode_prefetch.make_decode_prefetch_buffers`; ``use_prefetcher`` streams
        weights through it, and ``tp_size`` is the stage's tensor-parallel width.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.self_attn = DeepSeekV4Attention(
            config,
            layer_idx,
            _strip_prefix(weights, "self_attn"),
            device,
            cache=cache.sub("self_attn"),
            weight_dtype=weight_dtype,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            tp_size=tp_size,
        )
        self.mlp = DeepSeekV4SparseMoeBlock(
            config,
            _strip_prefix(weights, "mlp"),
            device,
            experts=experts,
            gate=gate,
            cache=cache.sub("mlp"),
            weight_dtype=weight_dtype,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            tp_size=tp_size,
        )
        self.input_layernorm = DeepSeekV4RMSNorm(
            weights["input_layernorm.weight"], eps, device, cache.file("input_layernorm"), sharded=True
        )
        self.post_attention_layernorm = DeepSeekV4RMSNorm(
            weights["post_attention_layernorm.weight"],
            eps,
            device,
            cache.file("post_attention_layernorm"),
            sharded=True,
        )
        self.attn_hc = DeepSeekV4HyperConnection(
            config,
            _strip_prefix(weights, "attn_hc"),
            device,
            cache=cache.sub("attn_hc"),
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            weight_dtype=weight_dtype,
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            config,
            _strip_prefix(weights, "ffn_hc"),
            device,
            cache=cache.sub("ffn_hc"),
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            weight_dtype=weight_dtype,
        )
        _profile(self.device)

    def prefetch_weights(self, *, index_sparse: bool = False):
        """Stage this layer's prefetched weights ahead of the :meth:`decode_static` that uses them.

        Every weight staged here answers that next call on this layer's ``[B,S,hc,D]`` streams.
        Hyper-connection ``fn`` first (private ring, consumed before attention), then
        attention (its own four projections and its compressor's pair), then the FFN
        hyper-connection, then the MoE (router gate on its 8-receiver ring, then the shared
        expert's down on the shared ring; at TP4 the shared gate/up are per-step DRAM -> L1
        copies and queue on no ring). The requests queued here must be consumed by this layer's
        own decode before any later layer queues its own. ``index_sparse`` stages the CSA
        indexer's score projections; the short-sequence trace leaves them unqueued.
        """
        self.attn_hc.prefetch_weights()
        self.self_attn.prefetch_weights(index_sparse=index_sparse)
        self.ffn_hc.prefetch_weights()
        self.mlp.prefetch_weights()

    def _mix(
        self, post: ttnn.Tensor, comb: ttnn.Tensor, sublayer_out: ttnn.Tensor, streams: ttnn.Tensor
    ) -> ttnn.Tensor:
        """``post[..,None] * out[..,None,:] + comb.T @ streams`` -> new streams.

        ``post`` ``[B,S,hc,1]``, ``comb`` ``[B,S,hc,hc]``, ``sublayer_out`` ``[B,S,1,D]``,
        ``streams`` ``[B,S,hc,D]``; returns ``[B,S,hc,D]``.

        Fused into a single composite device op (``ttnn.experimental.deepseek.mix_streams``)
        that folds the broadcast-multiply, the ``comb`` transpose (via ``transpose_a=True``)
        and the add into one op call, matching the eager math at HiFi4 / fp32 dest acc.
        """
        _profile(self.device)
        return ttnn.experimental.deepseek.mix_streams(post, comb, sublayer_out, streams, compute_kernel_config=_HIFI4)

    def decode(
        self,
        hidden_streams: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        mask: ttnn.Tensor | None,
        scache: "_StaticLayerCache",
        sliding_pos: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
        paged: PagedLayerView | None,
        input_ids: Optional[torch.Tensor] = None,
        pool_compressor: bool = True,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        win_slot: ttnn.Tensor | None = None,
        win_row: ttnn.Tensor | None = None,
        hash_token: ttnn.Tensor | None = None,
        index_sparse: bool = False,
    ) -> ttnn.Tensor:
        """Single-token decode: same graph as :meth:`decode_static` (the capture).

        ``hidden_streams`` ``[B,S,hc,D]`` -> ``[B,S,hc,D]``; the remaining tensors are as in
        :meth:`decode_static`. Host ``input_ids`` ``[B]`` are uploaded only for hash-routed
        layers, as the static path's on-device ``hash_token`` ``[1,B]``.
        """
        if hash_token is None and input_ids is not None and self.mlp.is_hash:
            t = hidden_streams.shape[0]
            hash_token = ttnn.from_torch(
                torch.as_tensor(input_ids).reshape(1, t).long().to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=(ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None),
            )
        return self.decode_static(
            hidden_streams,
            cos,
            sin,
            neg_sin,
            cos_win,
            sin_win,
            mask,
            scache,
            sliding_pos,
            compress_pos,
            paged,
            hash_token=hash_token,
            pool_compressor=pool_compressor,
            sdpa_cur_pos=sdpa_cur_pos,
            win_slot=win_slot,
            win_row=win_row,
            index_sparse=index_sparse,
        )

    def decode_static(
        self,
        hidden_streams: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        mask: ttnn.Tensor | None,
        scache: "_StaticLayerCache",
        sliding_pos: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
        paged: PagedLayerView | None,
        hash_token: ttnn.Tensor | None = None,
        pool_compressor: bool = True,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        win_slot: ttnn.Tensor | None = None,
        win_row: ttnn.Tensor | None = None,
        index_sparse: bool = False,
    ) -> ttnn.Tensor:
        """Trace-safe single-token decode (see :meth:`decode`). Uses the fixed-size
        in-place attention cache + the host-sync-free MoE so the whole block can be
        captured into a reusable ``ttnn`` trace.

        ``hidden_streams`` ``[B,S,hc,D]`` -> ``[B,S,hc,D]``; ``cos``/``sin``/``neg_sin`` are the
        bf16 RoPE rows ``[1,1,1,Rd]`` (``cos_win``/``sin_win`` the compressor family's, ``None``
        for sliding layers); ``mask`` is the additive bf16 TILE mask ``[1,1,1,kv_len]`` or
        ``None`` when the step is causal-only; ``sliding_pos`` and ``compress_pos`` are INT32
        ``[B]`` absolute positions; ``sdpa_cur_pos``/``win_slot``/``win_row`` INT32 ``[B]``
        per-user indices; ``hash_token`` a uint32 ROW_MAJOR ``[1,B]`` for a hash-routed MoE;
        ``scache`` the layer's dense KV and compressor window buffers (``_StaticLayerCache``)
        and ``paged`` the shared-block-pool view of an HCA layer's KV (``None`` otherwise).

        ``pool_compressor`` is fixed at capture time: the traced path captures one
        variant per window phase (see :meth:`DeepSeekV4Model._capture_traces`)."""
        with _region("ATTN_HC"):
            post, comb, collapsed = self.attn_hc(hidden_streams)
        with _region("INPUT_NORM"):
            normed = self.input_layernorm(collapsed)
        with _region("ATTENTION"):
            attn_out = self.self_attn.decode_static(
                normed,
                cos,
                sin,
                neg_sin,
                cos_win,
                sin_win,
                mask,
                scache,
                sliding_pos,
                compress_pos,
                paged,
                pool_compressor=pool_compressor,
                sdpa_cur_pos=sdpa_cur_pos,
                win_slot=win_slot,
                win_row=win_row,
                index_sparse=index_sparse,
            )
        with _region("ATTN_MIX"):
            hidden_streams = self._mix(post, comb, attn_out, hidden_streams)
        # Both are spent, and at a wide batch the L1 they hold is the difference
        # between the MoE's circular buffers fitting and not (a user's row is padded to
        # a whole tile, so these grow with the batch). Nothing below reads them.
        ttnn.deallocate(normed)
        ttnn.deallocate(attn_out)
        with _region("FFN_HC"):
            post, comb, collapsed = self.ffn_hc(hidden_streams)
        with _region("POST_NORM"):
            collapsed = self.post_attention_layernorm(collapsed)
        with _region("MOE"):
            mlp_out = self.mlp.decode_static(collapsed, hash_token=hash_token)
        with _region("FFN_MIX"):
            return self._mix(post, comb, mlp_out, hidden_streams)
