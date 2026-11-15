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

import os
from typing import NamedTuple, Optional

import torch
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

# The input RMSNorm's gain is folded into attention's projections (see :class:`~.attention.DeepSeekV41Attention`'s
# ``input_gain``): a one-row step on Blackhole then hands attention the collapsed row without a norm, except where
# a ratio-2 compressor needs the ``1 / rms`` (a unit-gain norm). ``DEEPSEEK_V41_FOLD_INPUT_NORM=0``: always normalize.
FOLD_INPUT_NORM = os.environ.get("DEEPSEEK_V41_FOLD_INPUT_NORM", "1") not in ("0", "", "false", "False")
# Each one-row hyper-connection folds its ``fn`` dot into its single-user program (``DeepSeekV4HyperConnection``'s
# ``fused_fn``), spread over 80 cores; ``DEEPSEEK_V41_FUSED_HC_FN=0`` runs ``fn`` as its own 8-core op instead.
# ``DEEPSEEK_V41_HC_FN_IN_L1=1`` also keeps the weight resident in L1 (~16 KB of each dot core's L1 per module)
# instead of reading it from DRAM per call.
FUSED_HC_FN = os.environ.get("DEEPSEEK_V41_FUSED_HC_FN", "1") not in ("0", "", "false", "False")
HC_FN_IN_L1 = os.environ.get("DEEPSEEK_V41_HC_FN_IN_L1", "0") not in ("0", "", "false", "False")
# A one-row step folds attention's mix (ATTN_MIX, ``mix_streams``) into the FFN hyper-connection's program
# (``DeepSeekV4HyperConnection.forward``'s ``mix``): its collapse cores mix their shard of the streams first, which
# saves a launch and the streams' round trip through L1. Needs ``DEEPSEEK_V41_FUSED_HC_FN``.
# ``DEEPSEEK_V41_FUSED_HC_MIX=0`` runs ``mix_streams`` on its own.
FUSED_HC_MIX = os.environ.get("DEEPSEEK_V41_FUSED_HC_MIX", "1") not in ("0", "", "false", "False")
# Likewise the FFN's mix (FFN_MIX) into the next layer's attention hyper-connection: :meth:`DeepSeekV41DecoderLayer.decode`
# with ``defer_mix`` hands the step on as a :class:`PendingMix`. ``DEEPSEEK_V41_DEFER_FFN_MIX=0`` always mixes in place.
# A one-row step with the persistent MoE fuses the post-attention RMSNorm onto the FFN hyper-connection's collapse
# (``DeepSeekV4HyperConnection.forward``'s ``norm``), which hands the MoE request the normed DRAM row: one op less per
# layer. Needs ``DEEPSEEK_V41_FUSED_HC_FN``. ``DEEPSEEK_V41_FOLD_POST_NORM=0`` runs ``sharded_rmsnorm`` on its own.
FOLD_POST_NORM = os.environ.get("DEEPSEEK_V41_FOLD_POST_NORM", "1") not in ("0", "", "false", "False")
# Likewise the input RMSNorm the ratio-2 compressor layers still need (see FOLD_INPUT_NORM) onto the attention
# hyper-connection's collapse, kept on its collapse cores where the norm's output was.
# ``DEEPSEEK_V41_FUSE_INPUT_NORM=0`` runs ``sharded_rmsnorm`` on its own.
FUSE_INPUT_NORM = os.environ.get("DEEPSEEK_V41_FUSE_INPUT_NORM", "1") not in ("0", "", "false", "False")
# A one-row step has the attention hyper-connection's collapse cores multicast the (normed) collapsed row straight
# into the token replica attention's QKV reads, instead of a separate all_gather_for_matmul.
# ``DEEPSEEK_V41_HC_TOKEN_REPLICA=0`` gathers it in attention.
HC_TOKEN_REPLICA = os.environ.get("DEEPSEEK_V41_HC_TOKEN_REPLICA", "1") not in ("0", "", "false", "False")
DEFER_FFN_MIX = FUSED_HC_MIX and os.environ.get("DEEPSEEK_V41_DEFER_FFN_MIX", "1") not in ("0", "", "false", "False")


class PendingMix(NamedTuple):
    """A layer's FFN mix step not yet run: the streams it leaves are ``mix_streams(post, comb, sublayer_out,
    streams)``. The next layer's :meth:`~DeepSeekV41DecoderLayer.decode` takes it in place of the streams and folds it
    into its attention hyper-connection (or runs it first where it cannot); it consumes ``streams``, and
    ``sublayer_out`` too when ``free_sublayer_out``."""

    post: ttnn.Tensor
    comb: ttnn.Tensor
    sublayer_out: ttnn.Tensor
    streams: ttnn.Tensor
    free_sublayer_out: bool


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
        self._build_non_moe(
            config,
            layer_idx,
            weights,
            device,
            cache,
            weight_dtype=weight_dtype,
            tp_size=tp_size,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            engram_sockets=engram_sockets,
            verify_rows=verify_rows,
            build_mlp=lambda: DeepSeekV41SparseMoeBlock(
                config,
                _strip_prefix(weights, "mlp"),
                device,
                experts,
                cache=cache.sub("ffn"),
                tp_size=tp_size,
                use_prefetcher=use_prefetcher,
                prefetch_buffers=prefetch_buffers,
                verify_rows=verify_rows,
            ),
        )

    def _build_non_moe(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device,
        cache,
        *,
        weight_dtype,
        tp_size: int,
        use_prefetcher: bool,
        prefetch_buffers,
        engram_sockets,
        verify_rows: int,
        build_mlp=None,
        hc_bias_in_l1: bool = True,
        hc_fn_in_l1: bool = HC_FN_IN_L1,
    ):
        """The Engram, attention, both hyper-connections and both norms; ``self.mlp`` is ``build_mlp()`` (built
        after attention, where the layer always built it, so the allocations keep their order), or ``None``."""
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.tp_size = tp_size
        self.verify_rows = verify_rows
        eps = config.rms_norm_eps
        self.fold_input_norm = FOLD_INPUT_NORM
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
            input_gain=weights["input_layernorm.weight"] if self.fold_input_norm else None,
        )
        self.mlp = build_mlp() if build_mlp is not None else None
        # With the gain folded into attention, the steps that still normalize divide by rms(x) only.
        self.input_layernorm = (
            DeepSeekV4RMSNorm(torch.ones(config.hidden_size), eps, device, cache.file("attn_norm.unit"), sharded=True)
            if self.fold_input_norm
            else DeepSeekV4RMSNorm(
                weights["input_layernorm.weight"], eps, device, cache.file("attn_norm"), sharded=True
            )
        )
        self.post_attention_layernorm = DeepSeekV4RMSNorm(
            weights["post_attention_layernorm.weight"], eps, device, cache.file("ffn_norm"), sharded=True
        )
        # The post-attention norm's gain as the ROW_MAJOR DRAM row the FFN hyper-connection's fused norm reads.
        self.post_norm_row = None
        if FOLD_POST_NORM and FUSED_HC_FN and device.arch() == ttnn.Arch.BLACKHOLE:
            self.post_norm_row = ttnn.to_layout(self.post_attention_layernorm.weight, ttnn.ROW_MAJOR_LAYOUT)
        self.input_norm_row = None
        if (
            FUSE_INPUT_NORM
            and FUSED_HC_FN
            and device.arch() == ttnn.Arch.BLACKHOLE
            and (not self.fold_input_norm or self.self_attn.needs_normed_input)
        ):
            self.input_norm_row = ttnn.to_layout(self.input_layernorm.weight, ttnn.ROW_MAJOR_LAYOUT)
        self.attn_hc = DeepSeekV4HyperConnection(
            config,
            _strip_prefix(weights, "attn_hc"),
            device,
            cache=cache.sub("hc_attn"),
            row_major_streams=True,
            bias_in_l1=hc_bias_in_l1,
            fused_fn=FUSED_HC_FN,
            fn_in_l1=FUSED_HC_FN and hc_fn_in_l1,
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            config,
            _strip_prefix(weights, "ffn_hc"),
            device,
            cache=cache.sub("hc_ffn"),
            row_major_streams=True,
            bias_in_l1=hc_bias_in_l1,
            fused_fn=FUSED_HC_FN,
            fn_in_l1=FUSED_HC_FN and hc_fn_in_l1,
        )

    def prefetch_weights(self, *, index_sparse: bool = False, one_row: bool = True, fused_qkv: Optional[bool] = None):
        """Queue this layer's ring requests: Engram's wkv, attention's q_a, q_b, kv, o_a, o_b, then the
        shared expert's down -- the order :meth:`decode` pops them. ``one_row``: the step decodes one row
        (see :meth:`~.attention.DeepSeekV41Attention.prefetch_weights`); a verify block's does not. ``fused_qkv``
        selects whether a one-row step queues attention_qkv_decode's slabs.

        Only weights on a ring (see :func:`~.decode_prefetch.prefetch_rings`) are queued. The
        router gate and the shared gate/up are not: their matmuls read the DRAM ND-sharded
        weight themselves, so nothing is copied into L1 here.
        """
        if self.engram is not None:
            self.engram.wkv.fetch_weights()
        self.self_attn.prefetch_weights(one_row=one_row, fused_qkv=fused_qkv)
        # None when ``fused_experts`` runs the shared expert from DRAM itself.
        if self.mlp.shared_experts is None:
            return
        down = self.mlp.shared_experts.down_proj
        if down.use_prefetcher:
            down.fetch_weights()

    def _run_pending_mix(self, pending: PendingMix) -> ttnn.Tensor:
        """The previous layer's deferred FFN mix, run on its own: the streams it leaves."""
        with _region("FFN_MIX"):
            streams = self._mix(pending.post, pending.comb, pending.sublayer_out, pending.streams)
        ttnn.deallocate(pending.streams)
        if pending.free_sublayer_out:
            ttnn.deallocate(pending.sublayer_out)
        return streams

    def _attn_hc(self, hidden_streams, pre_mix, collapsed_layout, one_row: bool, norm=None, replica_grid=None):
        """ATTN_HC: ``(post, comb, collapsed, attn_pre, streams)``. ``hidden_streams`` is the streams, which are
        returned as they are, or a :class:`PendingMix`, folded into the hyper-connection where it can be (one row,
        see :data:`FUSED_HC_MIX`) and else run first; ``streams`` are then the mixed ones. With ``replica_grid``
        ``collapsed`` is the pair ``(collapsed, its replica on replica_grid)`` (see :data:`HC_TOKEN_REPLICA`)."""
        if isinstance(hidden_streams, PendingMix):
            pending = hidden_streams
            if one_row and self.attn_hc.can_fold_mix(pending.post, pending.comb, pending.sublayer_out):
                with _region("ATTN_HC"):
                    post, comb, collapsed, attn_pre, streams, *replica = self.attn_hc(
                        pending.streams,
                        pre_mix=pre_mix,
                        collapsed_layout=collapsed_layout,
                        mix=(pending.post, pending.comb, pending.sublayer_out),
                        norm=norm,
                        replica_grid=replica_grid,
                    )
                ttnn.deallocate(pending.streams)
                if pending.free_sublayer_out:
                    ttnn.deallocate(pending.sublayer_out)
                if replica:
                    collapsed = (collapsed, replica[0])
                return post, comb, collapsed, attn_pre, streams
            hidden_streams = self._run_pending_mix(pending)
        with _region("ATTN_HC"):
            post, comb, collapsed, attn_pre, *replica = self.attn_hc(
                hidden_streams, pre_mix=pre_mix, collapsed_layout=collapsed_layout, norm=norm, replica_grid=replica_grid
            )
            if replica:
                collapsed = (collapsed, replica[0])
        return post, comb, collapsed, attn_pre, hidden_streams

    def _fuses_post_norm(self, one_row: bool) -> bool:
        """Whether a one-row step's FFN hyper-connection returns ``collapsed`` already RMS-normed (see
        :data:`FOLD_POST_NORM`)."""
        return one_row and self.post_norm_row is not None and self.ffn_hc.can_fuse_norm()

    def _attn_mix_and_ffn_hc(self, post, comb, attn_out, hidden_streams, attn_pre, one_row: bool, norm=None):
        """ATTN_MIX then FFN_HC: ``(post, comb, collapsed, ffn_pre, streams)`` for the FFN, ``streams`` the mixed
        ones. A one-row step folds the mix into the FFN hyper-connection's program when it can (see
        :data:`FUSED_HC_MIX`); otherwise ``mix_streams`` runs first. ``hidden_streams`` is consumed. ``norm``
        ``(weight, eps)``: ``collapsed`` comes back RMS-normed (``DeepSeekV4HyperConnection.forward``'s ``norm``)."""
        if FUSED_HC_MIX and one_row and self.ffn_hc.can_fold_mix(post, comb, attn_out):
            with _region("FFN_HC"):
                post, comb, collapsed, ffn_pre, streams = self.ffn_hc(
                    hidden_streams, pre_mix=attn_pre, mix=(post, comb, attn_out), norm=norm
                )
            ttnn.deallocate(hidden_streams)
            return post, comb, collapsed, ffn_pre, streams
        with _region("ATTN_MIX"):
            streams = self._mix(post, comb, attn_out, hidden_streams)
        ttnn.deallocate(hidden_streams)
        with _region("FFN_HC"):
            post, comb, collapsed, ffn_pre = self.ffn_hc(streams, pre_mix=attn_pre, norm=norm)
        return post, comb, collapsed, ffn_pre, streams

    def decode(
        self,
        hidden_streams: ttnn.Tensor,
        pre_mix: ttnn.Tensor,
        scache: _StaticLayerCache,
        image_mask: Optional[ttnn.Tensor] = None,
        defer_mix: bool = False,
        **attn_inputs,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One decode step: ``(streams [B, 1, hc, D], pre_mix [B, 1, 1, hc])`` -> the same pair for the next layer.

        ``hidden_streams`` may also be the previous layer's :class:`PendingMix`. ``defer_mix`` lets a one-row step
        return its own FFN mix as one in place of the streams (see :data:`DEFER_FFN_MIX`); only pass it when the
        result goes straight to the next layer's :meth:`decode`.

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
        shape = (hidden_streams.streams if isinstance(hidden_streams, PendingMix) else hidden_streams).shape
        b, s = shape[0], shape[1]
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
        one_row = b == 1 and s == 1 and self.device.arch() == ttnn.Arch.BLACKHOLE
        if self.engram is not None:
            if isinstance(hidden_streams, PendingMix):
                hidden_streams = self._run_pending_mix(hidden_streams)
            with _region("ENGRAM"):
                streams_in = hidden_streams
                hidden_streams = self.engram(streams_in)
                ttnn.deallocate(streams_in)
        _l1("engram")
        skip_input_norm = (
            self.fold_input_norm
            and b == 1
            and s == 1
            and self.device.arch() == ttnn.Arch.BLACKHOLE
            and not self.self_attn.needs_normed_input
        )
        # ROW_MAJOR [1, D/8] rows on the collapse cores: the layout sharded_rmsnorm would have written.
        post, comb, collapsed, attn_pre, hidden_streams = self._attn_hc(
            hidden_streams, pre_mix, ttnn.ROW_MAJOR_LAYOUT if skip_input_norm else None, one_row
        )
        _l1("attn_hc")
        with _region("INPUT_NORM"):
            if skip_input_norm:
                normed = collapsed
            elif b == 1 and s == 1 and self.device.arch() == ttnn.Arch.BLACKHOLE:
                normed = ttnn.experimental.deepseek.sharded_rmsnorm(
                    collapsed,
                    self.input_layernorm.weight,
                    epsilon=self.input_layernorm.eps,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                ttnn.deallocate(collapsed)
            elif s == 1:
                normed = self.input_layernorm(collapsed)
            else:
                # The hyper-connection leaves the rows in DRAM, one tile-row each; the norm packs one tile-row.
                packed = ttnn.reshape(collapsed, [1, 1, s, self.config.hidden_size])
                normed = _verify_rows(self.input_layernorm(packed), self.device)
                _l1("norm")
        with _region("ATTENTION"):
            attn_out = self.self_attn.decode(normed, scache=scache, **attn_inputs)
            _l1("attn")
        post, comb, collapsed, ffn_pre, hidden_streams = self._attn_mix_and_ffn_hc(
            post, comb, attn_out, hidden_streams, attn_pre, one_row
        )
        ttnn.deallocate(normed)
        ttnn.deallocate(attn_out)
        _l1("ffn_hc")
        with _region("POST_NORM"):
            if b == 1 and s == 1 and self.device.arch() == ttnn.Arch.BLACKHOLE:
                # When fused_experts also runs the router and shared expert, decode_static skips
                # all_gather_for_matmul for ROW_MAJOR input and passes it directly to fused_experts,
                # whose broadcast reader expects DRAM-interleaved (or replicated HEIGHT_SHARDED).
                post_mem = (
                    ttnn.DRAM_MEMORY_CONFIG
                    if self.mlp is not None and self.mlp.fused_router and self.mlp.fused_shared is not None
                    else None
                )
                norm_in = collapsed
                collapsed = ttnn.experimental.deepseek.sharded_rmsnorm(
                    norm_in,
                    self.post_attention_layernorm.weight,
                    epsilon=self.post_attention_layernorm.eps,
                    memory_config=post_mem,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                ttnn.deallocate(norm_in)
            elif s == 1:
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
        if defer_mix and DEFER_FFN_MIX and one_row:
            # The MoE's output is not freed here, so the pending mix leaves it be too.
            return PendingMix(post, comb, mlp_out, hidden_streams, False), ffn_pre
        with _region("FFN_MIX"):
            streams_out = self._mix(post, comb, mlp_out, hidden_streams)
        ttnn.deallocate(hidden_streams)
        _l1("exit")
        return streams_out, ffn_pre


class DeepSeekV41PersistentMoeDecoderLayer(DeepSeekV41DecoderLayer):
    """V4.1 decode decoder layer ``layer_idx`` whose MoE runs on a persistent TP12 MoE server.

    Everything but the MoE -- Engram, attention, both hyper-connections and both norms -- is built as in
    :class:`DeepSeekV41DecoderLayer` on ``device``, a 1x4 TP4 mesh; one such mesh can hold all 40 layers. The MoE
    (router, routed experts and shared expert) is ``moe_server``'s (a :class:`~.persistent_moe.PersistentMoE`
    over the layer's TP12 MoE block on its 3x4 mesh), and ``device`` is its requester ``requester``. So this
    layer holds no MoE weights: ``weights`` needs no ``mlp.*`` entries and :attr:`mlp` is ``None``.

    The FFN step sends the normed row to the server (``persistent_moe_request``) and mixes in the reduced row it
    gets back, replicated over the TP4 chips, in place of the MoE block's all-reduced output.

    Decode only, one row: ``B == 1`` and ``S == 1``. The server takes one ``[1, D]`` row per request, and its
    layer table holds the router's text bias only, so image tokens (``image_mask``) are not supported.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device,
        moe_server,
        requester: int = 0,
        cache=None,
        weight_dtype=ttnn.bfloat16,
        tp_size: int = 4,
        use_prefetcher: bool = False,
        prefetch_buffers=None,
        engram_sockets=None,
    ):
        if layer_idx not in moe_server.table_rows:
            raise ValueError(f"layer {layer_idx} is not on the persistent MoE (it serves {moe_server.layer_ids})")
        if moe_server.request_meshes[requester] is not device:
            raise ValueError(f"requester {requester} of the persistent MoE is not this layer's device")
        self._build_non_moe(
            config,
            layer_idx,
            weights,
            device,
            _as_cache(cache),
            weight_dtype=weight_dtype,
            tp_size=tp_size,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            engram_sockets=engram_sockets,
            verify_rows=1,
            # One device holds every layer here, and each HC module's L1 bias tiles take 6 KB of every core's L1:
            # ~480 KB over 40 layers, past where the step's circular buffers have to end. The single-row program
            # reads them from DRAM instead, ~0.2 us per call.
            hc_bias_in_l1=False,
            # Resident fn weights would take ~16 KB of each dot core's L1 per module, 1.3 MB over 80 modules.
            hc_fn_in_l1=False,
        )
        self.moe_server = moe_server
        self.requester = requester

    def prefetch_weights(self, *, index_sparse: bool = False, one_row: bool = True, fused_qkv: Optional[bool] = None):
        """Queue this layer's ring requests: Engram's wkv, then attention's -- the order :meth:`decode` pops them.
        The MoE's weights live on the server."""
        if self.engram is not None:
            self.engram.wkv.fetch_weights()
        self.self_attn.prefetch_weights(one_row=one_row, fused_qkv=fused_qkv)

    def moe(self, normed: ttnn.Tensor, post=None, comb=None, streams=None) -> ttnn.Tensor:
        """The server's MoE of ``normed`` ``[1, 1, 1, D]``: the reduced row as ``[1, 1, 1, D]`` bf16 ROW_MAJOR in
        DRAM, replicated over the mesh. ``normed`` is not consumed.

        With ``post``, ``comb`` and ``streams`` the request also runs the FFN hyper-connection's post step on the
        row (``mix_streams``, on the stream-collapse cores as the partials arrive) and returns the new streams, in
        ``streams``' spec, which must be the server's :meth:`~.persistent_moe.PersistentMoE.streams_memory_config`."""
        d = self.config.hidden_size
        row = normed
        if row.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            row = ttnn.to_memory_config(row, ttnn.DRAM_MEMORY_CONFIG)
        if row.layout != ttnn.ROW_MAJOR_LAYOUT:
            prev = row
            row = ttnn.to_layout(prev, ttnn.ROW_MAJOR_LAYOUT)
            if prev is not normed and prev.buffer_address() != row.buffer_address():
                ttnn.deallocate(prev)
        if row.dtype != ttnn.bfloat16:
            prev = row
            row = ttnn.typecast(prev, ttnn.bfloat16)
            if prev is not normed and prev.buffer_address() != row.buffer_address():
                ttnn.deallocate(prev)
        if list(row.shape) != [1, 1, 1, d]:
            prev = row
            row = ttnn.reshape(prev, [1, 1, 1, d])
            if prev is not normed and prev.buffer_address() != row.buffer_address():
                ttnn.deallocate(prev)
        out = self.moe_server.request(self.requester, self.layer_idx, row, streams=streams, post=post, comb=comb)
        if row is not normed and row.buffer_address() != normed.buffer_address():
            ttnn.deallocate(row)
        return out

    def _fuses_post(self, post: ttnn.Tensor, comb: ttnn.Tensor, streams: ttnn.Tensor) -> bool:
        """Whether the MoE request can run the FFN post step itself (see :meth:`moe`)."""
        return (
            self.moe_server.fuse_hc
            and post.dtype == ttnn.bfloat16
            and comb.dtype == ttnn.bfloat16
            and post.layout == ttnn.TILE_LAYOUT
            and comb.layout == ttnn.TILE_LAYOUT
            and streams.dtype == ttnn.bfloat16
            and streams.layout == ttnn.ROW_MAJOR_LAYOUT
            and streams.memory_config()
            == self.moe_server.streams_memory_config(self.device, num_streams=streams.shape[2])
        )

    def decode(
        self,
        hidden_streams: ttnn.Tensor,
        pre_mix: ttnn.Tensor,
        scache: _StaticLayerCache,
        image_mask: Optional[ttnn.Tensor] = None,
        defer_mix: bool = False,
        **attn_inputs,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One decode step: ``(streams [1, 1, hc, D], pre_mix [1, 1, 1, hc])`` -> the same pair for the next layer.

        As :meth:`DeepSeekV41DecoderLayer.decode` for one row, the MoE on the server; ``hidden_streams`` may be a
        :class:`PendingMix` and ``defer_mix`` returns one, unless the request runs the FFN mix itself. On an Engram
        layer the step's rows go to ``self.engram.write_rows`` before this call."""
        shape = (hidden_streams.streams if isinstance(hidden_streams, PendingMix) else hidden_streams).shape
        b, s = shape[0], shape[1]
        assert b == 1 and s == 1, f"the persistent MoE decodes one row per step, got [{b}, {s}]"
        assert image_mask is None, "the persistent MoE's layer table holds the text router bias only"

        one_row = self.device.arch() == ttnn.Arch.BLACKHOLE
        if self.engram is not None:
            if isinstance(hidden_streams, PendingMix):
                hidden_streams = self._run_pending_mix(hidden_streams)
            with _region("ENGRAM"):
                streams_in = hidden_streams
                hidden_streams = self.engram(streams_in)
                ttnn.deallocate(streams_in)
        skip_input_norm = (
            self.fold_input_norm and self.device.arch() == ttnn.Arch.BLACKHOLE and not self.self_attn.needs_normed_input
        )
        fuse_input_norm = (
            not skip_input_norm and one_row and self.input_norm_row is not None and self.attn_hc.can_fuse_norm()
        )
        replica_grid = None
        if HC_TOKEN_REPLICA and one_row and (skip_input_norm or fuse_input_norm) and self.attn_hc.can_fuse_norm():
            replica_grid = self.self_attn.token_replica_grid(
                select=bool(attn_inputs.get("select", False)), dense=bool(attn_inputs.get("dense", False))
            )
        post, comb, collapsed, attn_pre, hidden_streams = self._attn_hc(
            hidden_streams,
            pre_mix,
            ttnn.ROW_MAJOR_LAYOUT if skip_input_norm else None,
            one_row,
            norm=(self.input_norm_row, self.input_layernorm.eps, True) if fuse_input_norm else None,
            replica_grid=replica_grid,
        )
        tokens = None
        if isinstance(collapsed, tuple):
            collapsed, tokens = collapsed
            attn_inputs["tokens"] = tokens
        with _region("INPUT_NORM"):
            if skip_input_norm or fuse_input_norm:
                # Either unnormed (the gain folded into attention), or ATTN_HC's collapse cores normed it.
                normed = collapsed
            elif self.device.arch() == ttnn.Arch.BLACKHOLE:
                normed = ttnn.experimental.deepseek.sharded_rmsnorm(
                    collapsed,
                    self.input_layernorm.weight,
                    epsilon=self.input_layernorm.eps,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                ttnn.deallocate(collapsed)
            else:
                normed = self.input_layernorm(collapsed)
        with _region("ATTENTION"):
            attn_out = self.self_attn.decode(normed, scache=scache, **attn_inputs)
        fold_post_norm = self._fuses_post_norm(one_row)
        post, comb, collapsed, ffn_pre, hidden_streams = self._attn_mix_and_ffn_hc(
            post,
            comb,
            attn_out,
            hidden_streams,
            attn_pre,
            self.device.arch() == ttnn.Arch.BLACKHOLE,
            norm=(self.post_norm_row, self.post_attention_layernorm.eps) if fold_post_norm else None,
        )
        ttnn.deallocate(normed)
        ttnn.deallocate(attn_out)
        with _region("POST_NORM"):
            if fold_post_norm:
                # FFN_HC's collapse cores normed it (the MoE request's DRAM row).
                normed = collapsed
            elif self.device.arch() == ttnn.Arch.BLACKHOLE:
                normed = ttnn.experimental.deepseek.sharded_rmsnorm(
                    collapsed,
                    self.post_attention_layernorm.weight,
                    epsilon=self.post_attention_layernorm.eps,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                ttnn.deallocate(collapsed)
            else:
                normed = self.post_attention_layernorm(collapsed)
        if self._fuses_post(post, comb, hidden_streams):
            # FFN_MIX runs inside the request, as the partials arrive.
            with _region("MOE"):
                streams_out = self.moe(normed, post=post, comb=comb, streams=hidden_streams)
            ttnn.deallocate(normed)
            ttnn.deallocate(hidden_streams)
            return streams_out, ffn_pre
        with _region("MOE"):
            mlp_out = self.moe(normed)
        ttnn.deallocate(normed)
        if defer_mix and DEFER_FFN_MIX and one_row:
            return PendingMix(post, comb, mlp_out, hidden_streams, True), ffn_pre
        with _region("FFN_MIX"):
            streams_out = self._mix(post, comb, mlp_out, hidden_streams)
        ttnn.deallocate(hidden_streams)
        ttnn.deallocate(mlp_out)
        return streams_out, ffn_pre
