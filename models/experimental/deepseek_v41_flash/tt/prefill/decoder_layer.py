# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill decoder layer, on top of the V4-Flash prefill block.

The multi-token counterpart of :class:`~..decode.decoder_layer.DeepSeekV41DecoderLayer`: V4's block --
hyper-connection -> attention -> mix -> hyper-connection -> MoE -> mix over ``hc`` residual streams
``[1, T, hc, D]`` -- with V4.1's differences:

* A sublayer collapses the streams with the ``pre`` its *predecessor's* hyper-connection computed: attention with
  the previous layer's FFN ``pre``, the FFN with this layer's attention ``pre``. :meth:`forward` therefore takes
  ``(streams, pre_mix)`` and returns the same pair for the next layer; the first layer starts from the one-hot
  ``pre_mix`` ``[1, 0, ..., 0]``, ``[1, T, 1, hc]`` bf16 TILE.
* Engram layers (1 and 14) first write their n-gram lookup (host rows, see :mod:`.engram`) into the streams.
* Attention and MoE are the V4.1 blocks. Compressed layers share state through a
  :class:`~.attention.PrefillShared`, so the layers of a chunk run in order.

Chunking is V4's: a prompt is fed as consecutive chunks, each layer carrying its own
:class:`~models.experimental.deepseek_v4_flash.tt.prefill.attention.PrefillAttentionState`.
"""

from typing import Optional

import ttnn

from models.experimental.deepseek_v41_flash.tt.prefill.attention import (
    DeepSeekV41PrefillAttention,
    PrefillShared,
    PrefillStaticShared,
)
from models.experimental.deepseek_v41_flash.tt.prefill.engram import DeepSeekV41PrefillEngram
from models.experimental.deepseek_v41_flash.tt.prefill.moe import DeepSeekV41PrefillMoE
from models.experimental.deepseek_v4_flash.tt.decode.decoder_layer import _strip_prefix
from models.experimental.deepseek_v4_flash.tt.prefill.attention import (
    PrefillAttentionState,
    PrefillStaticBuffers,
    PrefillStaticStep,
)
from models.experimental.deepseek_v4_flash.tt.prefill.decoder_layer import (
    DeepSeekV4PrefillDecoderLayer,
    load_norm_gamma,
    mix_streams,
)
from models.experimental.deepseek_v4_flash.tt.prefill.hyperconnection import DeepSeekV4PrefillHyperConnection
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache, _as_cache


class DeepSeekV41PrefillDecoderLayer(DeepSeekV4PrefillDecoderLayer):
    """V4.1 prefill decoder layer ``layer_idx``.

    ``weights`` comes from :func:`~..config.decoder_layer_weights`, ``rope`` is V4's bundle (see
    :class:`~.attention.DeepSeekV41PrefillAttention`) and ``experts`` the layer's
    :class:`~..decode.moe.DeepSeekV41PreloadedExperts`: decode owns it and prefill reads the same weights in
    place. ``weight_dtype`` is for the attention's fp8 projections and the Engram's ``wkv``; ``moe_weight_dtype``
    for the router and the shared expert. The hyper-connections and norms stay bf16.

    ``cache`` is the layer's namespace. Its sublayers sit under the names the decode layer uses (``attn``,
    ``ffn``, ``engram``, ``hc_attn``, ``hc_ffn``) with a ``.prefill`` suffix on every weight, so decode and
    prefill share a cache directory without colliding.

    ``tp_size > 1`` needs ``device`` to be a 1xTP mesh and ``experts`` built at the same ``tp_size``; the
    streams, ``pre_mix``, hyper-connections, norms and Engram are replicated over it.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        rope: dict,
        experts,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        moe_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        tp_size: int = 1,
        maskless: bool = False,
    ):
        cache = _as_cache(cache)
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.hidden = config.hidden_size
        self.eps = config.rms_norm_eps

        self.engram = (
            DeepSeekV41PrefillEngram(
                config,
                layer_idx,
                _strip_prefix(weights, "engram"),
                device,
                cache=cache.sub("engram"),
                weight_dtype=weight_dtype,
            )
            if layer_idx in config.engram_layer_ids
            else None
        )
        self.self_attn = DeepSeekV41PrefillAttention(
            config,
            layer_idx,
            _strip_prefix(weights, "self_attn"),
            device,
            rope,
            cache=cache.sub("attn"),
            weight_dtype=weight_dtype,
            tp_size=tp_size,
            maskless=maskless,
        )
        self.mlp = DeepSeekV41PrefillMoE(
            config,
            _strip_prefix(weights, "mlp"),
            device,
            experts=experts,
            cache=cache.sub("ffn"),
            weight_dtype=moe_weight_dtype,
            tp_size=tp_size,
        )
        self.attn_hc = DeepSeekV4PrefillHyperConnection(
            config, _strip_prefix(weights, "attn_hc"), device, cache=cache.sub("hc_attn")
        )
        self.ffn_hc = DeepSeekV4PrefillHyperConnection(
            config, _strip_prefix(weights, "ffn_hc"), device, cache=cache.sub("hc_ffn")
        )
        self.input_layernorm_weight = load_norm_gamma(
            weights["input_layernorm.weight"], device, cache.file("input_layernorm.prefill")
        )
        self.post_attention_layernorm_weight = load_norm_gamma(
            weights["post_attention_layernorm.weight"], device, cache.file("post_attention_layernorm.prefill")
        )

    def forward(
        self,
        hidden_streams: ttnn.Tensor,
        pre_mix: ttnn.Tensor,
        state: Optional[PrefillAttentionState] = None,
        shared: Optional[PrefillShared] = None,
        engram_rows: Optional[ttnn.Tensor] = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The block over one prompt chunk: ``(streams [1, T, hc, D], pre_mix [1, T, 1, hc])`` -> the same pair
        for the next layer, both bf16 TILE. ``hidden_streams`` and ``pre_mix`` are left intact.

        ``state`` is this layer's attention state and ``shared`` the prompt's :class:`~.attention.PrefillShared`
        (see :meth:`~.attention.DeepSeekV41PrefillAttention.forward`). An Engram layer takes the chunk's host-looked-up
        ``engram_rows`` ``[1, 1, T, kin]`` bf16 TILE; any other layer takes none.
        """
        if state is None:
            state = self.new_state()
        self._check_chunk(hidden_streams, state, None)
        return self._v41_block(
            hidden_streams, pre_mix, lambda normed: self.self_attn(normed, state, shared), engram_rows
        )

    def forward_static(
        self,
        hidden_streams: ttnn.Tensor,
        pre_mix: ttnn.Tensor,
        bufs: PrefillStaticBuffers,
        step: PrefillStaticStep,
        shared: PrefillStaticShared,
        engram_rows: Optional[ttnn.Tensor] = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Trace-safe :meth:`forward` (see :class:`~.model.DeepSeekV41TracedPrefill`): the attention state is the
        persistent ``bufs``, the chunk's inputs come from ``step`` and the stage's group state is ``shared``."""
        return self._v41_block(
            hidden_streams,
            pre_mix,
            lambda normed: self.self_attn.forward_static(normed, bufs, step, shared),
            engram_rows,
        )

    def _v41_block(self, hidden_streams, pre_mix, attend, engram_rows) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The shared body: ``attend(normed [1, 1, T, D]) -> [1, 1, T, D]`` is the only part that differs."""
        if (self.engram is None) != (engram_rows is None):
            raise ValueError(f"layer {self.layer_idx}: engram_rows go to an Engram layer, and only to one")

        streams = hidden_streams if self.engram is None else self.engram(hidden_streams, engram_rows)
        post, comb, collapsed, attn_pre = self.attn_hc(streams, pre_mix=pre_mix)
        normed = self._norm(collapsed, self.input_layernorm_weight)
        ttnn.deallocate(collapsed)
        attn_out = attend(normed)
        ttnn.deallocate(normed)
        mixed = mix_streams(post, comb, attn_out, streams)
        for tensor in (post, comb, attn_out):
            ttnn.deallocate(tensor)
        if streams is not hidden_streams:
            ttnn.deallocate(streams)

        post, comb, collapsed, ffn_pre = self.ffn_hc(mixed, pre_mix=attn_pre)
        ttnn.deallocate(attn_pre)
        normed = self._norm(collapsed, self.post_attention_layernorm_weight)
        ttnn.deallocate(collapsed)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        out = mix_streams(post, comb, mlp_out, mixed)
        for tensor in (post, comb, mlp_out, mixed):
            ttnn.deallocate(tensor)
        return out, ffn_pre
