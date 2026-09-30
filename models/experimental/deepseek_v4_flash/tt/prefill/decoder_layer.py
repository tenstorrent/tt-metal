# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill decoder block: hyper-connection -> attention -> mix -> hyper-connection -> MoE -> mix.

:class:`DeepSeekV4PrefillDecoderLayer` is the multi-token counterpart of the decode
:class:`~..decode.decoder_layer.DeepSeekV4DecoderLayer` and composes the three prefill blocks:
:class:`~.attention.DeepSeekV4PrefillAttention`, :class:`~.moe.DeepSeekV4PrefillMoE` and two
:class:`~.hyperconnection.DeepSeekV4PrefillHyperConnection`.

``T`` = tokens in the chunk, ``hc`` = ``hc_mult`` (the hyper-connection residual-stream count) and
``D`` = ``hidden_size``. The residual is a stack of ``hc`` parallel streams ``[1, T, hc, D]`` carried
through the whole block, so both sublayers read and write that one tensor (the decode layout with
``B = 1`` and ``S = T``). For each sublayer the matching hyper-connection collapses the streams into
the sublayer input, and the sublayer output is folded back into the streams through the learned
``post`` placement weights and the Sinkhorn ``comb`` stream-mixing matrix::

    post, comb, collapsed = hc(streams)
    out = sublayer(norm(collapsed))
    streams = post * out + (comb.T @ streams)

Chunking works as in the attention block: a prompt is fed as consecutive chunks through one
:class:`~.attention.PrefillAttentionState`. The hyper-connections and the MoE are per-token, so the
attention state is the only thing that crosses a chunk boundary.

Supported shapes (v1) are the union of the blocks': ``T`` and the chunk's start position multiples of
``ALIGNMENT`` (128), CSA layers up to ``index_topk * 4`` tokens (more with ``dense_csa``), ``tp_size``
1 or the width of a 1xTP mesh (the streams and the sublayer inputs/outputs are replicated across the
ranks, and only the attention and the MoE communicate), and the routed op's ``D == 4096`` / per-chip
``I == 512`` geometry (see :mod:`.attention` and :mod:`.moe`).
"""

from typing import Optional

import ttnn

from ..common import _HIFI4, DeepSeekV4Module
from ..decode.decoder_layer import _strip_prefix
from ..decode.moe import DeepSeekV4HashRouter, DeepSeekV4PreloadedExperts
from ..weight_cache import WeightCache, _as_cache, _load_weight, _materialize
from .attention import (
    ALIGNMENT,
    DeepSeekV4PrefillAttention,
    PrefillAttentionState,
    PrefillStaticBuffers,
    PrefillStaticStep,
)
from .hyperconnection import DeepSeekV4PrefillHyperConnection, wide_rms_norm
from .moe import DeepSeekV4PrefillMoE


def load_norm_gamma(weight, device: ttnn.MeshDevice, cache_file_name: Optional[str] = None) -> ttnn.Tensor:
    """A ``[D]`` RMSNorm weight (tensor or thunk) as a bf16 TILE ``[1, 1, 1, D]`` device tensor."""
    w = _materialize(weight, cache_file_name, ttnn.bfloat16)
    return _load_weight(
        w.detach().float().reshape(1, 1, 1, -1) if w is not None else None,
        device,
        cache_file_name=cache_file_name,
    )


def rows_from_tokens(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[1, T, 1, D]`` TILE (one token per "batch" row, the hyper-connection's layout) -> ``[1, 1, T, D]`` TILE.

    The token axis moves from dim 1 to the row axis. In TILE layout that would repack every token's
    one-row tile, so it goes through ROW_MAJOR where it is a page-preserving reshape.
    """
    _, t, _, d = x.shape
    rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    rows = ttnn.reshape(rows, [1, 1, t, d])
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)


def tokens_from_rows(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[1, 1, T, D]`` TILE -> ``[1, T, 1, D]`` ROW_MAJOR, the sublayer-output form ``mix_streams`` reads."""
    _, _, t, d = x.shape
    rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(rows, [1, t, 1, d])


def mix_streams(post: ttnn.Tensor, comb: ttnn.Tensor, sublayer_out: ttnn.Tensor, streams: ttnn.Tensor) -> ttnn.Tensor:
    """``post[..., None] * out[..., None, :] + comb.T @ streams`` -> the new streams.

    ``post`` ``[1, T, hc, 1]``, ``comb`` ``[1, T, hc, hc]``, ``streams`` ``[1, T, hc, D]`` (all TILE) and
    ``sublayer_out`` ``[1, 1, T, D]`` TILE (the sublayer's own layout); returns ``[1, T, hc, D]``.

    It is the same fused ``ttnn.experimental.deepseek.mix_streams`` device op decode uses, which folds
    the broadcast-multiply, the ``comb`` transpose and the add into one call at HiFi4 / fp32 accumulation
    and spreads the ``T * D / 32`` output tiles over the core grid.
    """
    return ttnn.experimental.deepseek.mix_streams(
        post, comb, tokens_from_rows(sublayer_out), streams, compute_kernel_config=_HIFI4
    )


class DeepSeekV4PrefillDecoderLayer(DeepSeekV4Module):
    """ttnn prefill port of ``DeepseekV4DecoderLayer`` (see the module docstring).

    ``weights`` keys mirror the HF decoder-layer param names, as for the decode layer:
    ``self_attn.*``, ``mlp.*``, ``attn_hc.{fn,base,scale}``, ``ffn_hc.{fn,base,scale}``,
    ``input_layernorm.weight`` and ``post_attention_layernorm.weight``, each value a torch tensor or a
    zero-arg thunk returning one (the blocks document their own keys).

    ``rope`` is the host rotary bundle the attention block reads (``rope["main"]`` / ``rope["compress"]``,
    each ``(cos_half, sin_half)`` ``[L, qk_rope_head_dim / 2]`` with ``L`` at least the longest prompt).
    ``experts`` is the layer's :class:`~..decode.moe.DeepSeekV4PreloadedExperts`: decode already owns
    it and prefill reads the same weights in place. ``gate`` overrides the router; by default a hash
    layer (``config.mlp_layer_types[layer_idx] == "hash_moe"``) gets a
    :class:`~..decode.moe.DeepSeekV4HashRouter` built from ``weights`` and every other layer the learned
    top-k router.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        rope: dict,
        experts: DeepSeekV4PreloadedExperts,
        gate=None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        moe_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        tp_size: int = 1,
        dense_csa: bool = False,
        lightning_indexer: bool = False,
    ):
        """Build layer ``layer_idx``: attention, MoE, the two hyper-connections and the two ``[D]`` RMSNorms.

        ``dense_csa`` is forwarded to the attention (see :class:`~.attention.DeepSeekV4PrefillAttention`):
        CSA layers attend to every visible compressed entry instead of the indexer's top-k, which lifts the
        2048-token limit at the price of not being the model's exact attention.

        ``weight_dtype`` is the dtype of the attention projections and ``moe_weight_dtype`` that of the
        router gate and the shared expert; the routed experts keep whatever dtype ``experts`` was built
        with. ``hc``'s ``fn`` projection and the norms are bf16. ``cache`` is the tile-cache namespace
        for this layer.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.hidden = config.hidden_size
        self.eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.self_attn = DeepSeekV4PrefillAttention(
            config,
            layer_idx,
            _strip_prefix(weights, "self_attn"),
            device,
            rope,
            cache=cache.sub("self_attn"),
            weight_dtype=weight_dtype,
            tp_size=tp_size,
            dense_csa=dense_csa,
            lightning_indexer=lightning_indexer,
        )

        mlp_weights = _strip_prefix(weights, "mlp")
        mlp_cache = cache.sub("mlp")
        is_hash = (getattr(config, "mlp_layer_types", None) or [None] * (layer_idx + 1))[layer_idx] == "hash_moe"
        if gate is None and is_hash:
            gate = DeepSeekV4HashRouter(config, mlp_weights, device, cache=mlp_cache, weight_dtype=moe_weight_dtype)
        self.mlp = DeepSeekV4PrefillMoE(
            config,
            mlp_weights,
            device,
            experts=experts,
            gate=gate,
            cache=mlp_cache,
            weight_dtype=moe_weight_dtype,
            tp_size=tp_size,
        )

        self.attn_hc = DeepSeekV4PrefillHyperConnection(
            config, _strip_prefix(weights, "attn_hc"), device, cache=cache.sub("attn_hc")
        )
        self.ffn_hc = DeepSeekV4PrefillHyperConnection(
            config, _strip_prefix(weights, "ffn_hc"), device, cache=cache.sub("ffn_hc")
        )
        self.input_layernorm_weight = load_norm_gamma(
            weights["input_layernorm.weight"], device, cache.file("input_layernorm.prefill")
        )
        self.post_attention_layernorm_weight = load_norm_gamma(
            weights["post_attention_layernorm.weight"], device, cache.file("post_attention_layernorm.prefill")
        )

    @property
    def is_hash(self) -> bool:
        """Whether this layer's MoE routes by the frozen token-id table (and so needs ``token_ids``)."""
        return self.mlp.is_hash

    def new_state(self) -> PrefillAttentionState:
        """An empty attention state: the start of a prompt."""
        return self.self_attn.new_state()

    def _norm(self, collapsed: ttnn.Tensor, gamma: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, T, 1, D]`` collapsed streams -> the RMSNormed sublayer input ``[1, 1, T, D]``."""
        rows = rows_from_tokens(collapsed)
        normed = wide_rms_norm(rows, self.eps)
        out = ttnn.multiply(normed, gamma)
        ttnn.deallocate(normed)
        return out

    def _check_chunk(self, hidden_streams: ttnn.Tensor, state: PrefillAttentionState, token_ids) -> None:
        """Reject a chunk this version cannot process, before any device work."""
        shape = tuple(hidden_streams.shape)
        hc = self.attn_hc.hc
        if len(shape) != 4 or shape[0] != 1 or shape[2] != hc or shape[3] != self.hidden:
            raise ValueError(f"expected streams [1, T, {hc}, {self.hidden}], got {shape}")
        if shape[1] == 0 or shape[1] % ALIGNMENT:
            raise ValueError(f"chunk length {shape[1]} must be a positive multiple of {ALIGNMENT}")
        if state.seq_len % ALIGNMENT:
            raise ValueError(f"chunk start {state.seq_len} must be a multiple of {ALIGNMENT}")
        if self.is_hash and token_ids is None:
            raise ValueError(f"layer {self.layer_idx} is hash-routed and needs the chunk's token ids")

    def forward(
        self, hidden_streams: ttnn.Tensor, state: Optional[PrefillAttentionState] = None, token_ids=None
    ) -> ttnn.Tensor:
        """The block over one prompt chunk.

        ``hidden_streams`` is ``[1, T, hc, D]`` bf16 TILE on the device; ``state`` is this layer's
        :class:`~.attention.PrefillAttentionState` (``None`` runs a whole prompt from scratch and
        discards the state), advanced in place; ``token_ids`` (``[T]`` / ``[1, T]`` torch ints, or a
        ``[1, T]`` uint32 ROW_MAJOR device tensor) is required for hash-routed layers and ignored
        otherwise. Returns the new streams ``[1, T, hc, D]`` TILE; ``hidden_streams`` is left intact.
        """
        if state is None:
            state = self.new_state()
        self._check_chunk(hidden_streams, state, token_ids)
        return self._block(hidden_streams, lambda normed: self.self_attn(normed, state), token_ids)

    def forward_static(
        self,
        hidden_streams: ttnn.Tensor,
        bufs: PrefillStaticBuffers,
        step: PrefillStaticStep,
        token_ids: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Trace-safe :meth:`forward` for traced prefill (see :class:`~..model.TracedPrefill`).

        The attention state is the persistent ``bufs`` (updated in place) and its per-chunk inputs come
        from ``step``; ``token_ids`` must already be a ``[1, T]`` uint32 ROW_MAJOR device tensor (hash layers).
        Shapes are static, so nothing is validated against a running position. ``hidden_streams`` is left intact.
        """
        if self.is_hash and token_ids is None:
            raise ValueError(f"layer {self.layer_idx} is hash-routed and needs the chunk's token ids")
        return self._block(hidden_streams, lambda normed: self.self_attn.forward_static(normed, bufs, step), token_ids)

    def _block(self, hidden_streams: ttnn.Tensor, attend, token_ids) -> ttnn.Tensor:
        """The shared body: ``attend(normed [1, 1, T, D]) -> [1, 1, T, D]`` is the only part that differs."""
        post, comb, collapsed = self.attn_hc(hidden_streams)
        normed = self._norm(collapsed, self.input_layernorm_weight)
        ttnn.deallocate(collapsed)
        attn_out = attend(normed)
        ttnn.deallocate(normed)
        streams = mix_streams(post, comb, attn_out, hidden_streams)
        for tensor in (post, comb, attn_out):
            ttnn.deallocate(tensor)

        post, comb, collapsed = self.ffn_hc(streams)
        normed = self._norm(collapsed, self.post_attention_layernorm_weight)
        ttnn.deallocate(collapsed)
        mlp_out = self.mlp(normed, token_ids)
        ttnn.deallocate(normed)
        out = mix_streams(post, comb, mlp_out, streams)
        for tensor in (post, comb, mlp_out, streams):
            ttnn.deallocate(tensor)
        return out
