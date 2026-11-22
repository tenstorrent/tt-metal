# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill hyper-connection over stream-major residual streams ``[1, hc, T, D]``.

The math is V4's :class:`~models.experimental.deepseek_v4_flash.tt.prefill.hyperconnection.DeepSeekV4PrefillHyperConnection`
with V4.1's collapse (each sublayer collapses with the ``pre`` its predecessor computed); only the layout
differs. Token-major ``[1, T, hc, D]`` TILE pads every token's ``hc = 4`` rows up to a 32-row tile (8x the bytes),
and its ``fn`` projection needs the streams flattened to ``[T, hc * D]`` through ROW_MAJOR. Stream-major, each
stream is a dense ``[T, D]`` slab, so:

* ``fn`` is a batched matmul: stream ``j`` against its ``[D, (2+hc)*hc]`` block of the packed weight, summed over
  the streams -- the same ``x @ fn^T`` over the flattened ``hc * D`` channels;
* the unweighted RMSNorm's mean of squares reduces over ``D`` and then over the streams;
* ``ttnn.experimental.deepseek.fused_hyperconnection_weights`` turns the projection into ``post`` / ``comb`` /
  ``pre`` per token (no collapse), and ``ttnn.experimental.deepseek.stream_mix`` collapses the streams with
  ``pre_mix`` straight into the sublayer's ``[1, 1, T, D]`` rows.
"""

from typing import Optional

import ttnn

from models.experimental.deepseek_v4_flash.tt.common import _HIFI4, DeepSeekV4Module
from models.experimental.deepseek_v4_flash.tt.weight_cache import (
    WeightCache,
    _as_cache,
    _load_weight,
    _materialize,
    _memo,
)


class DeepSeekV41PrefillHyperConnection(DeepSeekV4Module):
    """One stream-major prefill hyper-connection (see the module docstring).

    ``weights`` is the dict the V4 block takes (``fn`` ``[(2+hc)*hc, hc*D]``, ``base`` ``[(2+hc)*hc]``, ``scale``,
    each a tensor or a zero-arg thunk). ``fn`` is stored per stream, ``[1, hc, D, (2+hc)*hc]``, under its own
    ``.prefill_sm`` cache name; the bias rows share V4's ``.prefill`` entries.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        hc, d = self.hc, self.hidden
        mixes = (2 + hc) * hc
        fn = _memo(weights["fn"])  # [(2+hc)*hc, hc*D]
        base = _memo(weights["base"])  # [(2+hc)*hc]
        scale_src = weights["scale"]
        scale = (scale_src() if callable(scale_src) else scale_src).flatten().tolist()  # 3 learned scalars

        fn_file = cache.file("fn.prefill_sm")
        fn_streams = _materialize(
            lambda: fn()[:mixes].detach().reshape(mixes, hc, d).permute(1, 2, 0).reshape(1, hc, d, mixes).contiguous(),
            fn_file,
            weight_dtype,
        )
        self.fn = _load_weight(fn_streams, device, cache_file_name=fn_file, dtype=weight_dtype)

        def bias(name: str, lo: int, hi: int) -> ttnn.Tensor:
            file = cache.file(f"{name}.prefill")
            row = _materialize(lambda: base()[lo:hi].detach().reshape(1, 1, 1, hi - lo), file, ttnn.bfloat16)
            return _load_weight(row, device, cache_file_name=file)

        self.pre_b = bias("pre_b", 0, hc)
        self.post_b = bias("post_b", hc, 2 * hc)
        self.comb_b = bias("comb_b", 2 * hc, 2 * hc + hc * hc)
        self.pre_scale, self.post_scale, self.comb_scale = (float(scale[0]), float(scale[1]), float(scale[2]))

    def forward(self, streams: ttnn.Tensor, pre_mix: ttnn.Tensor):
        """``streams`` ``[1, hc, T, D]`` and ``pre_mix`` ``[1, T, 1, hc]`` (bf16 TILE) ->
        ``(post [1, T, hc, 1], comb [1, T, hc, hc], collapsed [1, 1, T, D], pre [1, T, 1, hc])``, all bf16 TILE.

        ``collapsed`` is the streams weighted by ``pre_mix``; ``pre`` is the one this block computes, for the next
        sublayer to collapse with. Both inputs are left intact.
        """
        shape = tuple(streams.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != self.hc or shape[3] != self.hidden:
            raise ValueError(f"expected streams [1, {self.hc}, T, {self.hidden}], got {shape}")

        # The norm is unweighted and fn is linear, so fn(x * rstd) is taken as fn(x) * rstd. Per-stream partial
        # projections are summed in fp32, like the single hc*D-deep matmul they replace.
        partial = ttnn.matmul(streams, self.fn, dtype=ttnn.float32, compute_kernel_config=_HIFI4)  # [1, hc, T, mixes]
        raw_w = ttnn.sum(partial, dim=1, keepdim=True)  # [1, 1, T, mixes]
        ttnn.deallocate(partial)
        sq = ttnn.square(streams)
        sq_sum = ttnn.sum(sq, dim=-1, keepdim=True)  # [1, hc, T, 1]
        ttnn.deallocate(sq)
        total = ttnn.sum(ttnn.typecast(sq_sum, ttnn.float32), dim=1, keepdim=True)  # [1, 1, T, 1]
        ttnn.deallocate(sq_sum)
        rstd = ttnn.rsqrt(ttnn.add(ttnn.multiply(total, 1.0 / (self.hc * self.hidden)), self.norm_eps))
        ttnn.deallocate(total)
        fused_w = ttnn.typecast(ttnn.multiply(raw_w, rstd), ttnn.bfloat16)
        for tensor in (raw_w, rstd):
            ttnn.deallocate(tensor)

        post, comb, pre = ttnn.experimental.deepseek.fused_hyperconnection_weights(
            fused_w=fused_w,
            pre_bias=self.pre_b,
            post_bias=self.post_b,
            comb_bias=self.comb_b,
            num_streams=self.hc,
            sinkhorn_iters=self.iters,
            pre_scale=self.pre_scale,
            post_scale=self.post_scale,
            comb_scale=self.comb_scale,
            eps=self.eps,
        )
        ttnn.deallocate(fused_w)
        collapsed = ttnn.experimental.deepseek.stream_mix(streams, pre_mix)  # [1, 1, T, D]
        return post, comb, collapsed, pre
