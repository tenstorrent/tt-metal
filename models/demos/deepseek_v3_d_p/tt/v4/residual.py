# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4's residual: manifold-constrained hyper-connections.

A V4 block carries ``hc_mult`` parallel residual streams instead of one hidden state. At each of a
block's two sublayer sites the streams are collapsed to one hidden state, the sublayer runs on that,
and the result is expanded back over the streams while the incoming streams are re-mixed:

    collapsed = sum_i pre[i] * X[i]
    X'[j]     = post[j] * sublayer(collapsed) + sum_i comb[i, j] * X[i]

where ``X[i]`` is stream i of the input, ``X'[j]`` stream j of the output, ``pre[i]`` and ``post[j]``
scalars per token, and ``comb`` an [hc_mult, hc_mult] doubly-stochastic mixing matrix per token. None
of the three are parameters: TtMHCWrap recomputes all of them from the streams themselves each
forward pass, from one projection of the normed streams.

This module is only the driver. It pairs a TtMHCWrap per site with the dtype boundary between the
fp32 streams and the bf16 sublayer, and exposes ``(x, sublayer)`` so a site reads like the plain
residual it stands in for.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap


class MhcResidual:
    """One hyper-connection site: ``__call__(x, sublayer)`` in, re-mixed streams out.

    The casts live here rather than inside TtMHCWrap because which dtype a sublayer wants is the
    block's business. The streams are fp32 -- the fused parametrization op is fp32-only -- while
    attention and the MoE run bf16.
    """

    def __init__(self, wrap: TtMHCWrap, sublayer_dtype):
        self.wrap = wrap
        self.sublayer_dtype = sublayer_dtype

    @classmethod
    def pair(cls, mesh_device, config, weights, *, tp_axis, num_links, topology, sublayer_dtype):
        """The two sites a block needs -- attention and FFN -- with independent parameters.

        ``weights`` is ``{"attn": (fn, base, scale), "ffn": (...)}``: the projection and the per-group
        scale and bias, one set per site.
        """
        assert weights is not None and set(weights) >= {
            "attn",
            "ffn",
        }, f"both sites need an (fn, base, scale) triple; got keys {sorted(weights or ())}"
        cfg = MHCConfig(
            dim=config.hidden_size,
            n=config.hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
            eps=config.hc_eps,
            norm_eps=config.rms_norm_eps,
        )
        return tuple(
            cls(
                TtMHCWrap(
                    mesh_device,
                    cfg,
                    *weights[site],
                    tp_axis=tp_axis,
                    num_links=num_links,
                    topology=topology,
                ),
                sublayer_dtype,
            )
            for site in ("attn", "ffn")
        )

    def __call__(self, x, sublayer):
        """``x`` is [1, 1, S, hc_mult * hidden] fp32 in and out; ``sublayer`` sees one hidden state."""

        def _run(collapsed):
            out = sublayer(ttnn.typecast(collapsed, self.sublayer_dtype))
            cast = ttnn.typecast(out, ttnn.float32)
            ttnn.deallocate(out)
            return cast

        return self.wrap.forward(x, _run)
