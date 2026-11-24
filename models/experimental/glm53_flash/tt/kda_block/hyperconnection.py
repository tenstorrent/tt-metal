# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC hyper-connection of a GLM-5.3-Flash decoder block (``Glm5NextTextHyperConnection``), for decode and prefill.

Same contract as the DeepSeek-V4 hyper-connections: the residual is ``[B, S, hc, D]`` and

    post, comb, collapsed = hc(streams)          # [B,S,hc,1], [B,S,hc,hc], [B,S,1,D]
    streams' = mix_streams(post, comb, out, streams) = post * out + comb^T @ streams

The projection runs in fp32 like the reference (``F.linear(rmsnorm(flat).float(), fn.float())``), with the
unweighted RMSNorm folded after it: ``fn(x * rstd) = fn(x) * rstd``. The pre / post / comb split, the affine
maps and the Sinkhorn projection are ``ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn``; the collapse and the
mix are batched matmuls over the ``B * S`` tokens.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.experimental.glm53_flash.tt.ds_v4.common import DeepSeekV4Module, _HIFI4
from models.experimental.glm53_flash.tt.ds_v4.layers import Linear
from models.experimental.glm53_flash.tt.ds_v4.streams import flatten_streams
from models.experimental.glm53_flash.tt.ds_v4.weight_cache import (
    WeightCache,
    _as_cache,
    _load_weight,
    _materialize,
    _memo,
)
from models.experimental.glm53_flash.tt.hyperconnection import build_sinkhorn_consts


def _fp32(x: ttnn.Tensor) -> ttnn.Tensor:
    return x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)


def mix_streams(post: ttnn.Tensor, comb: ttnn.Tensor, sublayer_out: ttnn.Tensor, streams: ttnn.Tensor) -> ttnn.Tensor:
    """``post[..., None] * out[..., None, :] + comb^T @ streams`` -> the new streams, in ``streams``' dtype.

    ``post`` ``[B, S, hc, 1]``, ``comb`` ``[B, S, hc, hc]`` (fp32), ``sublayer_out`` ``[B, S, 1, D]`` and
    ``streams`` ``[B, S, hc, D]``, all TILE; accumulates in fp32. No input is consumed.
    """
    dtype = streams.dtype
    mixed = ttnn.matmul(comb, _fp32(streams), transpose_a=True, compute_kernel_config=_HIFI4, dtype=ttnn.float32)
    placed = ttnn.multiply(post, _fp32(sublayer_out))
    out = ttnn.add(mixed, placed)
    for tensor in (mixed, placed):
        ttnn.deallocate(tensor)
    return ttnn.typecast(out, dtype) if dtype != ttnn.float32 else out


class GLMHyperConnection(DeepSeekV4Module):
    """ttnn port of ``Glm5NextTextHyperConnection`` for any token count (one decode row per user, or a prefill chunk).

    ``weights`` holds the HF module's ``fn`` ``[(2+hc)*hc, hc*D]``, ``base`` ``[(2+hc)*hc]`` and ``scale`` ``[3]``
    (torch tensors or zero-arg thunks). ``base`` / ``scale`` are folded into the eight constant tiles of
    ``mhc_split_sinkhorn``.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        fn_dtype: ttnn.DataType = ttnn.float32,
    ):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        hc = self.hc
        mixes = (2 + hc) * hc
        fn = _memo(weights["fn"])
        base = _memo(weights["base"])
        scale = _memo(weights["scale"])
        self.fn = Linear(lambda: fn()[:mixes].detach().float(), device, cache.file("fn.glm"), dtype=fn_dtype)
        consts_file = cache.file("sinkhorn_consts")
        consts = _materialize(
            lambda: build_sinkhorn_consts(hc, scale().detach().flatten(), base().detach().flatten()),
            consts_file,
            ttnn.float32,
        )
        self.consts = _load_weight(consts, device, cache_file_name=consts_file, dtype=ttnn.float32)

    def forward(self, hidden_streams: ttnn.Tensor):
        """``hidden_streams`` ``[B, S, hc, D]`` -> ``(post [B,S,hc,1], comb [B,S,hc,hc], collapsed [B,S,1,D])``.

        ``post`` and ``comb`` are fp32; ``collapsed`` comes back in ``hidden_streams``' dtype, as in the reference.
        """
        shape = tuple(hidden_streams.shape)
        if len(shape) != 4 or shape[2] != self.hc or shape[3] != self.hidden:
            raise ValueError(f"expected streams [B, S, {self.hc}, {self.hidden}], got {shape}")
        b, s, hc, d = shape
        t = b * s

        flat = flatten_streams(hidden_streams)  # [1, 1, T, hc*D]
        flat32 = ttnn.typecast(flat, ttnn.float32)
        ttnn.deallocate(flat)
        raw_w = self.fn(flat32)  # [1, 1, T, (2+hc)*hc]
        sq = ttnn.square(flat32)
        ttnn.deallocate(flat32)
        mean = ttnn.mean(sq, dim=-1, keepdim=True)
        ttnn.deallocate(sq)
        rstd = ttnn.rsqrt(ttnn.add(mean, self.norm_eps))
        ttnn.deallocate(mean)
        fused_w = ttnn.multiply(raw_w, rstd)
        for tensor in (raw_w, rstd):
            ttnn.deallocate(tensor)

        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            fused_w, self.consts, hc, self.iters, self.eps
        )
        ttnn.deallocate(fused_w)
        pre = ttnn.reshape(pre, (b, s, 1, hc))
        post = ttnn.reshape(post, (b, s, hc, 1))
        comb = ttnn.reshape(comb, (b, s, hc, hc))

        collapsed = ttnn.matmul(
            pre, _fp32(hidden_streams), compute_kernel_config=_HIFI4, dtype=ttnn.float32
        )  # [B, S, 1, D]
        ttnn.deallocate(pre)
        if hidden_streams.dtype != ttnn.float32:
            collapsed = ttnn.typecast(collapsed, hidden_streams.dtype)
        return post, comb, collapsed
