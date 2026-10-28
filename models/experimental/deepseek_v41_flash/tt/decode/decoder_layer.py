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

from typing import Optional

import ttnn

from models.experimental.deepseek_v41_flash.tt.decode.attention import DeepSeekV41Attention
from models.experimental.deepseek_v41_flash.tt.decode.engram import DeepSeekV41Engram
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41SparseMoeBlock
from models.experimental.deepseek_v4_flash.tt.common import _region, width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.attention import _StaticLayerCache
from models.experimental.deepseek_v4_flash.tt.decode.decoder_layer import DeepSeekV4DecoderLayer, _strip_prefix
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import DeepSeekV4HyperConnection
from models.experimental.deepseek_v4_flash.tt.layers import DeepSeekV4RMSNorm
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache


def _verify_rows(normed: ttnn.Tensor, device) -> ttnn.Tensor:
    """A verify block's normed rows ``[1, 1, S, D]`` (TILE, sharded) -> ``[1, S, 1, D]`` ROW_MAJOR, width-sharded
    in L1 with all ``S`` rows on every core: the layout attention's and the MoE's ``S``-row paths take.

    A sharded tensor cannot move its rows onto separate tile-rows in place, so the rows go through DRAM.
    """
    s, d = normed.shape[-2], normed.shape[-1]
    rows = ttnn.to_layout(ttnn.to_memory_config(normed, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(normed)
    out = ttnn.to_memory_config(ttnn.reshape(rows, [1, s, 1, d]), width_sharded_l1_config(s, d, device, tile_height=1))
    ttnn.deallocate(rows)
    return out


class DeepSeekV41DecoderLayer(DeepSeekV4DecoderLayer):
    """V4.1 decode decoder layer ``layer_idx``.

    ``weights`` comes from :func:`~..config.decoder_layer_weights` and ``experts`` is the layer's
    :class:`~.moe.DeepSeekV41PreloadedExperts`. ``weight_dtype`` is for the attention's fp8
    projections; the shared expert and the Engram keep their own default (bfloat8_b), and the
    hyper-connections and norms (fp32 / bf16 in the checkpoint) stay bf16.

    ``cache`` is the layer's namespace. Its sublayers sit under the checkpoint's names (``attn``,
    ``ffn``, ``engram``), the ones the module tests use, so they share converted weights. An
    Engram layer's H2D sockets allocate L1, so build the layer before any trace capture;
    ``engram_sockets`` passes ones opened earlier (see :func:`~.engram.open_rows_sockets`).

    ``tp_size > 1`` needs ``device`` to be a 1xTP mesh with fabric, and ``experts`` built at the
    same ``tp_size``. The inputs and outputs of :meth:`decode` are replicated over it.
    ``use_prefetcher`` streams the Engram, attention and MoE projections through ``prefetch_buffers``,
    the per-device decode rings V4 uses; the hyper-connections stay off them.

    ``verify_rows > 1`` also lets :meth:`decode` step one session's ``S == verify_rows`` consecutive
    positions at once (a DSpark verify block): it builds the MoE's separate shared expert and the
    Engram's second rows buffer. A block's rows outnumber the Engram socket's pages, so they are
    written from another thread while the step is dispatched (see :meth:`~.engram.DeepSeekV41Engram.write_rows`).
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
        engram_sockets=None,
        verify_rows: int = 1,
    ):
        cache = _as_cache(cache)
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.tp_size = tp_size
        self.verify_rows = verify_rows
        eps = config.rms_norm_eps
        self.engram = (
            DeepSeekV41Engram(
                config,
                layer_idx,
                _strip_prefix(weights, "engram"),
                device,
                prefetch_buffers,
                cache=cache.sub("engram"),
                sockets=engram_sockets,
                tp_size=tp_size,
                verify_rows=verify_rows,
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
            verify_rows=verify_rows,
        )
        self.input_layernorm = DeepSeekV4RMSNorm(
            weights["input_layernorm.weight"], eps, device, cache.file("attn_norm"), sharded=True
        )
        self.post_attention_layernorm = DeepSeekV4RMSNorm(
            weights["post_attention_layernorm.weight"], eps, device, cache.file("ffn_norm"), sharded=True
        )
        self.attn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "attn_hc"), device, cache=cache.sub("hc_attn"), row_major_streams=True
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "ffn_hc"), device, cache=cache.sub("hc_ffn"), row_major_streams=True
        )

    def prefetch_weights(self, *, index_sparse: bool = False):
        """Queue this layer's ring requests: Engram's wkv, attention's q_a, q_b, kv, o_a, o_b, then the
        shared expert's down -- the order :meth:`decode` pops them.

        Only weights on a ring (see :func:`~.decode_prefetch.prefetch_rings`) are queued. The
        router gate and the shared gate/up are not: their matmuls read the DRAM ND-sharded
        weight themselves, so nothing is copied into L1 here.
        """
        if self.engram is not None:
            self.engram.wkv.fetch_weights()
        self.self_attn.prefetch_weights()
        # None when ``fused_experts`` runs the shared expert from DRAM itself.
        if self.mlp.shared_experts is None:
            return
        down = self.mlp.shared_experts.down_proj
        if down.use_prefetcher:
            down.fetch_weights()

    def decode(
        self,
        hidden_streams: ttnn.Tensor,
        pre_mix: ttnn.Tensor,
        scache: _StaticLayerCache,
        image_mask: Optional[ttnn.Tensor] = None,
        **attn_inputs,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One decode step: ``(streams [B, 1, hc, D], pre_mix [B, 1, 1, hc])`` -> the same pair for the next layer.

        A verify block (``B == 1``, ``S == verify_rows``) passes ``[1, S, hc, D]`` / ``[1, S, 1, hc]`` instead,
        with ``attn_inputs`` from ``decode_inputs(..., rows=S)``; row ``j`` is position ``p + j``, and the
        outputs and caches equal ``S`` single-row steps (up to bf16). Its streams are width-sharded as
        ``[S * hc, D / 8]`` over the 8 collapse cores (the Engram's ``verify_streams_memory_config``).

        ``hidden_streams`` is ROW_MAJOR, width-sharded in L1, and is consumed: each streams tensor is
        freed once its successor exists, so a stale one cannot sit in L1 through the rest of the
        step (the caller's reference would otherwise keep the input alive). ``pre_mix`` is bf16 TILE.
        ``scache`` is this layer's :func:`~.attention.build_decode_caches` entry and ``attn_inputs``
        what :meth:`~.attention.DeepSeekV41Attention.decode_inputs` builds for the position. On an
        Engram layer the step's rows go to ``self.engram.write_rows`` before this call: under TP
        its all-reduces can sync the host, which would otherwise wait on rows never sent.

        ``image_mask`` (``[1, B]`` uint32, 1 for a token of an image span) makes the router rank those tokens with
        ``bias_vl``.
        """
        b, s = hidden_streams.shape[0], hidden_streams.shape[1]
        if s > 1:
            assert b == 1 and s == self.verify_rows, f"a [{b}, {s}] block needs B == 1 and verify_rows == {s}"

        def _l1(tag):  # TEMP probe
            if s > 1:
                from loguru import logger
                import os

                if os.environ.get("DSV_SYNC"):
                    ttnn.synchronize_device(self.device)

                v = ttnn.get_memory_view(self.device, ttnn.BufferType.L1)
                lo = sorted((int(x["address"]), int(x["size"])) for x in v.block_table if x["allocated"] == "yes")[:3]
                logger.error(f"L1 layer {self.layer_idx} {tag}: {lo}")

        _l1("enter")
        if self.engram is not None:
            with _region("ENGRAM"):
                streams_in = hidden_streams
                hidden_streams = self.engram(streams_in)
                ttnn.deallocate(streams_in)
        _l1("engram")
        with _region("ATTN_HC"):
            post, comb, collapsed, attn_pre = self.attn_hc(hidden_streams, pre_mix=pre_mix)
            _l1("attn_hc")
        with _region("INPUT_NORM"):
            if s == 1:
                normed = self.input_layernorm(collapsed)
            else:
                # The hyper-connection leaves the rows in DRAM, one tile-row each; the norm packs one tile-row.
                packed = ttnn.reshape(collapsed, [1, 1, s, self.config.hidden_size])
                normed = _verify_rows(self.input_layernorm(packed), self.device)
                _l1("norm")
        with _region("ATTENTION"):
            attn_out = self.self_attn.decode(normed, scache=scache, **attn_inputs)
            _l1("attn")
        with _region("ATTN_MIX"):
            streams_in = hidden_streams
            hidden_streams = self._mix(post, comb, attn_out, streams_in)
        ttnn.deallocate(streams_in)
        ttnn.deallocate(normed)
        ttnn.deallocate(attn_out)
        _l1("attn_mix")
        with _region("FFN_HC"):
            post, comb, collapsed, ffn_pre = self.ffn_hc(hidden_streams, pre_mix=attn_pre)
            _l1("ffn_hc")
        with _region("POST_NORM"):
            if s == 1:
                collapsed = self.post_attention_layernorm(collapsed)
            else:
                packed = ttnn.reshape(collapsed, [1, 1, s, self.config.hidden_size])
                collapsed = _verify_rows(self.post_attention_layernorm(packed), self.device)
                _l1("post_norm")
        if s > 1:
            # The MoE's S-row fused_experts CBs reach past where the streams sit in L1, so they wait in DRAM.
            streams_config = hidden_streams.memory_config()
            streams_l1 = hidden_streams
            hidden_streams = ttnn.to_memory_config(streams_l1, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(streams_l1)
        with _region("MOE"):
            mlp_out = self.mlp.decode_static(collapsed, image_mask=image_mask)
            _l1("moe")
        if s > 1:
            streams_dram = hidden_streams
            hidden_streams = ttnn.to_memory_config(streams_dram, streams_config)
            ttnn.deallocate(streams_dram)
        with _region("FFN_MIX"):
            streams_out = self._mix(post, comb, mlp_out, hidden_streams)
        ttnn.deallocate(hidden_streams)
        _l1("exit")
        return streams_out, ffn_pre
