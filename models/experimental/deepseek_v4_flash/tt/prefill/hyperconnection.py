# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill hyper-connection (mHC) for DeepSeek-V4-Flash: one block, many tokens per call.

:class:`DeepSeekV4PrefillHyperConnection` is the multi-token counterpart of the decode
:class:`~..decode.hyperconnection.DeepSeekV4HyperConnection`. The math is the same reference
``DeepseekV4HyperConnection``: the ``hc`` residual streams of every token are unweighted-RMSNormed
over their flattened ``hc * D`` channels, projected by the learned ``fn`` to the ``(2 + hc) * hc``
pre / post / comb mixes, and turned into

* ``collapsed`` -- the ``pre``-weighted sum of the streams (the sublayer input),
* ``post``      -- the sublayer-output placement weights (``2 * sigmoid(.)``),
* ``comb``      -- the stream-mixing matrix, Sinkhorn-projected onto the doubly-stochastic manifold.

What differs from decode is only where the work runs. The decode block keeps the flattened streams
width-sharded in L1 and runs ``fn`` through ``matmul_decode``, both built for a few token rows; a
prefill chunk is hundreds to thousands of rows, so here the norm and the ``fn`` projection are plain
DRAM-interleaved ``ttnn`` ops. The pre / post / comb / Sinkhorn stage is the same multi-token
``ttnn.experimental.deepseek.fused_hyperconnection`` device op decode uses for ``T > 1``, which
spreads the tokens over the core grid.

Streams are ``[B, S, hc, D]`` TILE bf16 (the prefill decoder layer carries ``[1, T, hc, D]``); the
outputs are ``post [B, S, hc, 1]``, ``comb [B, S, hc, hc]`` and ``collapsed [B, S, 1, D]``, all TILE.
"""

from typing import Optional

import ttnn

from ..common import DeepSeekV4Module
from ..layers import Linear
from ..weight_cache import WeightCache, _as_cache, _load_weight, _materialize, _memo


def flatten_streams(streams: ttnn.Tensor) -> ttnn.Tensor:
    """``[B, S, hc, D]`` TILE -> ``[1, 1, B*S, hc*D]`` TILE: each token's streams laid side by side.

    ``hc`` is far below a tile row, so this is not a view: it goes through ROW_MAJOR, where the
    merge of ``hc`` rows into one wide row is a plain page reshape.
    """
    b, s, hc, d = streams.shape
    rows = ttnn.to_layout(streams, ttnn.ROW_MAJOR_LAYOUT)
    rows = ttnn.reshape(rows, [1, 1, b * s, hc * d])
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)


def wide_rms_norm(x: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """Unweighted RMSNorm over the last dim of a TILE ``[1, 1, T, W]`` tensor, for very wide rows (``hc*D``).

    ``ttnn.rms_norm``'s default program sizes its circular buffers by the row width, which at ``W = hc*D``
    does not fit next to the decode model's persistent L1 buffers on the shared chips. Eltwise ops and a
    row reduction only ever hold a few tiles per core. The mean of squares is taken in fp32.
    """
    x32 = ttnn.typecast(x, ttnn.float32)
    sq = ttnn.multiply(x32, x32)
    ttnn.deallocate(x32)
    mean = ttnn.mean(sq, dim=-1, keepdim=True)  # [1, 1, T, 1]
    ttnn.deallocate(sq)
    rstd = ttnn.typecast(ttnn.rsqrt(ttnn.add(mean, eps)), x.dtype)
    ttnn.deallocate(mean)
    out = ttnn.multiply(x, rstd)
    ttnn.deallocate(rstd)
    return out


class DeepSeekV4PrefillHyperConnection(DeepSeekV4Module):
    """ttnn prefill port of ``DeepseekV4HyperConnection`` (see the module docstring).

    ``weights`` is the same dict the decode block takes, each value a torch tensor or a zero-arg
    thunk: the packed ``fn`` ``[(2+hc)*hc, hc*D]``, ``base`` ``[(2+hc)*hc]`` and ``scale``, the three
    learned scalars (pre / post / comb). ``base`` is split into its pre / post / comb rows and
    ``scale`` stays a host scalar triple, exactly as in decode.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        """Upload ``fn`` as a ``[hc*D, (2+hc)*hc]`` projection and the ``base`` rows.

        ``cache`` is an optional :class:`~..weight_cache.WeightCache` namespace; this block's entries
        carry a ``.prefill`` suffix so they never collide with the decode layouts of the same weight.
        """
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        hc = self.hc
        mixes = (2 + hc) * hc
        fn = _memo(weights["fn"])  # [(2+hc)*hc, hc*D]
        base = _memo(weights["base"])  # [(2+hc)*hc]
        scale_src = weights["scale"]
        scale = (scale_src() if callable(scale_src) else scale_src).flatten().tolist()  # 3 learned scalars

        # The (2+hc)*hc output columns need no padding here: the op reads the logical width.
        self.fn = Linear(lambda: fn()[:mixes].detach(), device, cache.file("fn.prefill"), dtype=weight_dtype)

        def bias(name: str, lo: int, hi: int) -> ttnn.Tensor:
            file = cache.file(f"{name}.prefill")
            row = _materialize(lambda: base()[lo:hi].detach().reshape(1, 1, 1, hi - lo), file, ttnn.bfloat16)
            return _load_weight(row, device, cache_file_name=file)

        self.pre_b = bias("pre_b", 0, hc)
        self.post_b = bias("post_b", hc, 2 * hc)
        self.comb_b = bias("comb_b", 2 * hc, 2 * hc + hc * hc)
        self.pre_scale, self.post_scale, self.comb_scale = (float(scale[0]), float(scale[1]), float(scale[2]))

    def forward(self, hidden_streams: ttnn.Tensor):
        """``hidden_streams`` ``[B, S, hc, D]`` -> ``(post [B,S,hc,1], comb [B,S,hc,hc], collapsed [B,S,1,D])``."""
        shape = tuple(hidden_streams.shape)
        if len(shape) != 4 or shape[2] != self.hc or shape[3] != self.hidden:
            raise ValueError(f"expected streams [B, S, {self.hc}, {self.hidden}], got {shape}")
        if shape[0] * shape[1] < 2:
            # One token takes the dedicated single-user device program, whose operands are the decode
            # layouts (sharded); this block feeds the multi-token program interleaved operands.
            raise ValueError("the prefill hyper-connection needs at least 2 tokens; single tokens are decode's")

        flat = flatten_streams(hidden_streams)  # [1, 1, T, hc*D]
        normed = wide_rms_norm(flat, self.norm_eps)
        ttnn.deallocate(flat)
        fused_w = self.fn(normed)  # [1, 1, T, (2+hc)*hc]
        ttnn.deallocate(normed)

        outputs = ttnn.experimental.deepseek.fused_hyperconnection(
            hidden_streams,
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
        return outputs
