# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash decode decoder layer, on top of the V4-Flash block.

V4.1 keeps V4's block -- hyper-connection -> attention -> mix -> hyper-connection -> MoE -> mix
over ``hc`` residual streams ``[B, S, hc, D]`` -- with these differences:

* A sublayer collapses the streams with the ``pre`` its *predecessor's* hyper-connection
  computed: attention with the previous layer's FFN ``pre``, the FFN with this layer's attention
  ``pre``. The state passed between layers is therefore ``(streams, pre_mix)``; the first layer
  starts from the one-hot ``pre_mix`` ``[1, 0, ..., 0]``.
* Engram layers (1 and 14) first write their n-gram lookup into the streams.
* Attention and MoE are the V4.1 blocks. Compressed layers share caches, so within a step the
  layers run in order.

Tensor parallelism is V4's: on a 1xTP mesh the attention heads and the MoE's ``I`` are split
over the ranks, each sublayer all-reduces its output, and everything else -- the streams,
``pre_mix``, hyper-connections, norms, Engram and caches -- is replicated.
"""

import ttnn

from models.experimental.deepseek_v41_flash.tt.decode.attention import DeepSeekV41Attention
from models.experimental.deepseek_v41_flash.tt.decode.engram import DeepSeekV41Engram
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41SparseMoeBlock
from models.experimental.deepseek_v4_flash.tt.common import _region
from models.experimental.deepseek_v4_flash.tt.decode.attention import _StaticLayerCache
from models.experimental.deepseek_v4_flash.tt.decode.decoder_layer import DeepSeekV4DecoderLayer, _strip_prefix
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import DeepSeekV4HyperConnection
from models.experimental.deepseek_v4_flash.tt.layers import DeepSeekV4RMSNorm
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache


class DeepSeekV41DecoderLayer(DeepSeekV4DecoderLayer):
    """V4.1 decode decoder layer ``layer_idx``.

    ``weights`` comes from :func:`~..config.decoder_layer_weights` and ``experts`` is the layer's
    :class:`~.moe.DeepSeekV41PreloadedExperts`. ``weight_dtype`` is for the attention's fp8
    projections; the shared expert and the Engram keep their own default (bfloat8_b), and the
    hyper-connections and norms (fp32 / bf16 in the checkpoint) stay bf16.

    ``cache`` is the layer's namespace. Its sublayers sit under the checkpoint's names (``attn``,
    ``ffn``, ``engram``), the ones the module tests use, so they share converted weights. An
    Engram layer's H2D sockets allocate L1, so build the layer before any trace capture.

    ``tp_size > 1`` needs ``device`` to be a 1xTP mesh with fabric, and ``experts`` built at the
    same ``tp_size``. The inputs and outputs of :meth:`decode` are replicated over it.
    ``use_prefetcher`` streams the Engram, attention and MoE projections through ``prefetch_buffers``,
    the per-device decode rings V4 uses; the hyper-connections stay off them.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device,
        experts,
        cache=None,
        weight_dtype=ttnn.bfloat16,
        tp_size: int = 1,
        use_prefetcher: bool = False,
        prefetch_buffers=None,
    ):
        cache = _as_cache(cache)
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.tp_size = tp_size
        eps = config.rms_norm_eps
        self.engram = (
            DeepSeekV41Engram(
                config,
                layer_idx,
                _strip_prefix(weights, "engram"),
                device,
                cache=cache.sub("engram"),
                use_prefetcher=use_prefetcher,
                prefetch_buffers=prefetch_buffers,
            )
            if layer_idx in config.engram_layer_ids
            else None
        )
        self.self_attn = DeepSeekV41Attention(
            config,
            layer_idx,
            _strip_prefix(weights, "self_attn"),
            device,
            cache=cache.sub("attn"),
            weight_dtype=weight_dtype,
            tp_size=tp_size,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
        )
        self.mlp = DeepSeekV41SparseMoeBlock(
            config,
            _strip_prefix(weights, "mlp"),
            device,
            experts,
            cache=cache.sub("ffn"),
            tp_size=tp_size,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
        )
        self.input_layernorm = DeepSeekV4RMSNorm(
            weights["input_layernorm.weight"], eps, device, cache.file("attn_norm"), sharded=True
        )
        self.post_attention_layernorm = DeepSeekV4RMSNorm(
            weights["post_attention_layernorm.weight"], eps, device, cache.file("ffn_norm"), sharded=True
        )
        self.attn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "attn_hc"), device, cache=cache.sub("hc_attn")
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "ffn_hc"), device, cache=cache.sub("hc_ffn")
        )

    def prefetch_weights(self, *, index_sparse: bool = False):
        """Queue this layer's ring requests: Engram's wkv, attention's q_a, q_b, kv, o_a, o_b, then the
        shared expert's down -- the order :meth:`decode` pops them.

        Only weights on a ring (see :func:`~.decode_prefetch.prefetch_rings`) are queued. The
        router gate and the shared gate/up keep their per-call DRAM -> L1 copy, which V4's
        ``prefetch_weights`` would instead run here and hold in L1 until this layer decodes.
        """
        if self.engram is not None and self.engram.use_prefetcher:
            self.engram.wkv.fetch_weights()
        self.self_attn.prefetch_weights()
        down = self.mlp.shared_experts.down_proj
        if down.use_prefetcher:
            down.fetch_weights()

    def decode(
        self, hidden_streams: ttnn.Tensor, pre_mix: ttnn.Tensor, scache: _StaticLayerCache, **attn_inputs
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One decode step: ``(streams [B, 1, hc, D], pre_mix [B, 1, 1, hc])`` -> the same pair for the next layer.

        ``hidden_streams`` is TILE and ``pre_mix`` bf16 TILE. ``scache`` is this layer's
        :func:`~.attention.build_decode_caches` entry and ``attn_inputs`` what
        :meth:`~.attention.DeepSeekV41Attention.decode_inputs` builds for the position. On an
        Engram layer the step's rows go to ``self.engram.write_rows`` before this call: under TP
        its all-reduces can sync the host, which would otherwise wait on rows never sent.
        """
        if self.engram is not None:
            with _region("ENGRAM"):
                hidden_streams = self.engram(hidden_streams)
        with _region("ATTN_HC"):
            post, comb, collapsed, attn_pre = self.attn_hc(hidden_streams, pre_mix=pre_mix)
        with _region("INPUT_NORM"):
            normed = self.input_layernorm(collapsed)
        with _region("ATTENTION"):
            attn_out = self.self_attn.decode(normed, scache=scache, **attn_inputs)
        with _region("ATTN_MIX"):
            hidden_streams = self._mix(post, comb, attn_out, hidden_streams)
        ttnn.deallocate(normed)
        ttnn.deallocate(attn_out)
        with _region("FFN_HC"):
            post, comb, collapsed, ffn_pre = self.ffn_hc(hidden_streams, pre_mix=attn_pre)
        with _region("POST_NORM"):
            collapsed = self.post_attention_layernorm(collapsed)
        with _region("MOE"):
            mlp_out = self.mlp.decode_static(collapsed)
        with _region("FFN_MIX"):
            return self._mix(post, comb, mlp_out, hidden_streams), ffn_pre
