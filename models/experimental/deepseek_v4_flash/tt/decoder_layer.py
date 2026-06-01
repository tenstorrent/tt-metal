from typing import Optional

import torch
import ttnn

from .attention import DeepSeekV4Attention, _LayerKVCache, _StaticLayerCache
from .common import DeepSeekV4Module, _HIFI4, _profile, _region
from .hyperconnection import DeepSeekV4HyperConnection
from .layers import DeepSeekV4RMSNorm
from .moe import DeepSeekV4SparseMoeBlock
from .weight_cache import WeightCache, _as_cache


def _strip_prefix(weights: dict, prefix: str) -> dict:
    """Sub-dict of ``weights`` whose keys start with ``prefix.`` (prefix stripped)."""
    p = f"{prefix}."
    return {k[len(p) :]: v for k, v in weights.items() if k.startswith(p)}


class DeepSeekV4DecoderLayer(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4DecoderLayer`` (decode).

    The residual is a stack of ``hc_mult`` parallel streams kept in
    ``[B, S, H, D]`` (``H`` = ``hc_mult``, ``D`` = ``hidden_size``) throughout the
    block, mixed in/out by two :class:`DeepSeekV4HyperConnection` modules. For
    each sublayer (attention, then MoE) the matching HC collapses the streams
    into the sublayer input, and the sublayer output is folded back into the
    streams via the learned ``post`` placement weights plus the Sinkhorn
    ``comb`` stream-mixing matrix::

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
    ):
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
        )
        self.mlp = DeepSeekV4SparseMoeBlock(
            config,
            _strip_prefix(weights, "mlp"),
            device,
            experts=experts,
            gate=gate,
            cache=cache.sub("mlp"),
            weight_dtype=weight_dtype,
        )
        self.input_layernorm = DeepSeekV4RMSNorm(
            weights["input_layernorm.weight"], eps, device, cache.file("input_layernorm")
        )
        self.post_attention_layernorm = DeepSeekV4RMSNorm(
            weights["post_attention_layernorm.weight"], eps, device, cache.file("post_attention_layernorm")
        )
        self.attn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "attn_hc"), device, cache=cache.sub("attn_hc")
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "ffn_hc"), device, cache=cache.sub("ffn_hc")
        )
        _profile(self.device)

    def _mix(
        self, post: ttnn.Tensor, comb: ttnn.Tensor, sublayer_out: ttnn.Tensor, streams: ttnn.Tensor
    ) -> ttnn.Tensor:
        """``post[..,None] * out[..,None,:] + comb.T @ streams`` -> new streams.

        ``post`` ``[B,S,H,1]``, ``comb`` ``[B,S,H,H]``, ``sublayer_out`` ``[B,S,1,D]``,
        ``streams`` ``[B,S,H,D]``; returns ``[B,S,H,D]``.
        """
        b, s, hc, d = streams.shape
        t = b * s
        _profile(self.device)

        # placement = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) -> [1,T,H,D].
        out = ttnn.reshape(sublayer_out, [1, t, 1, d])
        out = ttnn.repeat(out, ttnn.Shape([1, 1, hc, 1]))  # broadcast over the stream axis
        placement = ttnn.multiply(out, ttnn.reshape(post, [1, t, hc, 1]))

        # mix = matmul(comb.transpose(-1, -2), streams): sum over the FIRST hc axis.
        comb_t = ttnn.transpose(ttnn.reshape(comb, [1, t, hc, hc]), -2, -1)
        mixed = ttnn.matmul(comb_t, ttnn.reshape(streams, [1, t, hc, d]), compute_kernel_config=_HIFI4)

        return ttnn.reshape(ttnn.add(placement, mixed), [b, s, hc, d])

    def decode(
        self,
        hidden_streams: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        kv_cache: "_LayerKVCache",
        input_ids: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Single-token decode: ``hidden_streams`` ``[B, 1, hc_mult, D]`` -> same.

        Everything outside attention (hyper-connections, norms, MoE / MLP) is
        per-token; attention runs against the running ``kv_cache``.
        """
        with _region("ATTN_HC"):
            post, comb, collapsed = self.attn_hc(hidden_streams)
        with _region("INPUT_NORM"):
            normed = self.input_layernorm(collapsed)
        with _region("ATTENTION"):
            attn_out = self.self_attn.decode(normed, cos, sin, neg_sin, cos_win, sin_win, kv_cache)
        with _region("ATTN_MIX"):
            hidden_streams = self._mix(post, comb, attn_out, hidden_streams)
        _profile(self.device)
        with _region("FFN_HC"):
            post, comb, collapsed = self.ffn_hc(hidden_streams)
        with _region("POST_NORM"):
            normed = self.post_attention_layernorm(collapsed)
        with _region("MOE"):
            mlp_out = self.mlp(normed, input_ids=input_ids)
        with _region("FFN_MIX"):
            return self._mix(post, comb, mlp_out, hidden_streams)

    def decode_static(
        self,
        hidden_streams: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        mask: ttnn.Tensor,
        scache: "_StaticLayerCache",
        sliding_pos: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
        hash_token: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Trace-safe single-token decode (see :meth:`decode`). Uses the fixed-size
        in-place attention cache + the host-sync-free MoE so the whole block can be
        captured into a reusable ``ttnn`` trace."""
        # return ttnn.assign(hidden_streams, memory_config=hidden_streams.memory_config())
        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_out = self.self_attn.decode_static(
            self.input_layernorm(collapsed),
            cos,
            sin,
            neg_sin,
            cos_win,
            sin_win,
            mask,
            scache,
            sliding_pos,
            compress_pos,
        )
        hidden_streams = self._mix(post, comb, attn_out, hidden_streams)
        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_out = self.mlp.decode_static(self.post_attention_layernorm(collapsed), hash_token=hash_token)
        return self._mix(post, comb, mlp_out, hidden_streams)
