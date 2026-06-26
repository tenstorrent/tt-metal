from typing import Optional

import ttnn

from .common import DeepSeekV4Module, _profile
from .layers import Linear, _rms_norm_unweighted
from .weight_cache import WeightCache, _as_cache, _load_weight, _materialize, _memo


class DeepSeekV4HyperConnection(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HyperConnection`` (Manifold-Constrained Hyper-
    Connections / mHC).

    Given the residual stream stack ``hidden_streams [B, S, H, D]`` (``H`` =
    ``hc_mult`` parallel streams, ``D`` = ``hidden_size``) it returns the triple
    ``(post, comb, collapsed)``:
      * ``collapsed [B, S, 1, D]`` -- the ``pre``-weighted collapse of the streams
        into a single sequence (the sublayer input),
      * ``post [B, S, H, 1]`` -- the sublayer-output placement weights
        (``2 * sigmoid(.)``),
      * ``comb [B, S, H, H]`` -- the stream-mixing matrix projected onto the
        doubly-stochastic manifold by ``hc_sinkhorn_iters`` Sinkhorn-Knopp steps.

    The learned ``fn`` / ``base`` / ``scale`` parameters are split host-side into
    their ``pre`` / ``post`` / ``comb`` parts (and ``fn`` run as three separate
    linears) so we never sub-tile-slice the packed ``(2+H)*H``-wide projection.
    See ``modular_deepseek_v4.py`` for the reference math.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        hc = self.hc
        # ``fn`` / ``base`` are one packed checkpoint tensor each, sliced into
        # pre [H] / post [H] / comb [H*H] parts; memoize so a cache miss reads
        # each source once across its slices. ``scale`` is 3 host scalars (no
        # tile cache), so it is always materialised.
        fn = _memo(weights["fn"])  # [(2+H)*H, H*D]
        base = _memo(weights["base"])  # [(2+H)*H]
        scale_src = weights["scale"]
        scale = (scale_src() if callable(scale_src) else scale_src).flatten().tolist()  # 3 learned scalars

        self.fn_pre = Linear(lambda: fn()[:hc], device, cache.file("fn_pre"))
        self.fn_post = Linear(lambda: fn()[hc : 2 * hc], device, cache.file("fn_post"))
        self.fn_comb = Linear(lambda: fn()[2 * hc : 2 * hc + hc * hc], device, cache.file("fn_comb"))
        self.pre_b = _load_weight(
            _materialize(lambda: base()[:hc].reshape(1, 1, 1, hc), cache.file("pre_b"), ttnn.bfloat16),
            device,
            cache_file_name=cache.file("pre_b"),
        )
        self.post_b = _load_weight(
            _materialize(lambda: base()[hc : 2 * hc].reshape(1, 1, 1, hc), cache.file("post_b"), ttnn.bfloat16),
            device,
            cache_file_name=cache.file("post_b"),
        )
        self.comb_b = _load_weight(
            _materialize(
                lambda: base()[2 * hc : 2 * hc + hc * hc].reshape(1, 1, 1, hc * hc),
                cache.file("comb_b"),
                ttnn.bfloat16,
            ),
            device,
            cache_file_name=cache.file("comb_b"),
        )
        self.pre_scale, self.post_scale, self.comb_scale = (float(scale[0]), float(scale[1]), float(scale[2]))

    def forward(self, hidden_streams: ttnn.Tensor):
        """``hidden_streams`` ``[B, S, H, D]`` -> ``(post [B,S,H,1], comb [B,S,H,H], collapsed [B,S,1,D])``."""
        b, s, hc, d = hidden_streams.shape
        t = b * s

        # Flatten streams to [1,1,T,H*D] and unweighted-RMSNorm over H*D.
        flat = ttnn.reshape(hidden_streams, [1, 1, t, hc * d])
        flat = _rms_norm_unweighted(flat, self.norm_eps)

        pre_w = self.fn_pre(flat)  # [1,1,T,H]
        post_w = self.fn_post(flat)  # [1,1,T,H]
        comb_w = self.fn_comb(flat)  # [1,1,T,H*H]
        _profile(self.device)

        return ttnn.experimental.deepseek.fused_hyperconnection(
            hidden_streams,
            pre_w=pre_w,
            post_w=post_w,
            comb_w=comb_w,
            pre_bias=self.pre_b,
            post_bias=self.post_b,
            comb_bias=self.comb_b,
            num_streams=hc,
            sinkhorn_iters=self.iters,
            pre_scale=self.pre_scale,
            post_scale=self.post_scale,
            comb_scale=self.comb_scale,
            eps=self.eps,
        )


class DeepSeekV4HyperHead(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HyperHead`` (final HC-stream collapse).

    Collapses the ``hc_mult`` residual streams ``[B, S, H, D]`` into a single
    ``[B, S, 1, D]`` sequence before the model's shared RMSNorm + ``lm_head``::

        flat  = unweighted_rmsnorm(streams.flatten(2))
        pre   = sigmoid(hc_fn @ flat * hc_scale + hc_base) + eps
        out   = (pre[..,None] * streams).sum(dim=2)

    ``weights`` keys: ``hc_fn`` ``[H, H*D]``, ``hc_base`` ``[H]``, ``hc_scale``
    (scalar). Unlike :class:`DeepSeekV4HyperConnection` there is no ``post`` /
    ``comb`` placement: the head only produces the collapsed sequence.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.fn = Linear(weights["hc_fn"], device, cache.file("hc_fn"))  # [H, H*D]
        base_src = weights["hc_base"]
        self.base = _load_weight(
            _materialize(
                lambda: (base_src() if callable(base_src) else base_src).reshape(1, 1, 1, self.hc),
                cache.file("hc_base"),
                ttnn.bfloat16,
            ),
            device,
            cache_file_name=cache.file("hc_base"),
        )
        # hc_scale is a host scalar (no tile cache) -- always materialise.
        scale_src = weights["hc_scale"]
        self.scale = float((scale_src() if callable(scale_src) else scale_src).flatten().tolist()[0])

    def forward(self, hidden_streams: ttnn.Tensor) -> ttnn.Tensor:
        """``hidden_streams`` ``[B, S, H, D]`` -> ``[B, S, 1, D]``."""
        b, s, hc, d = hidden_streams.shape
        t = b * s

        flat = ttnn.reshape(hidden_streams, [1, 1, t, hc * d])
        flat = _rms_norm_unweighted(flat, self.norm_eps)

        mixes = self.fn(flat)  # [1,1,T,H]
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mixes, self.scale), self.base)), self.eps)

        hs = ttnn.reshape(hidden_streams, [1, t, hc, d])
        pre_col = ttnn.reshape(pre, [1, t, hc, 1])
        out = ttnn.sum(ttnn.multiply(hs, pre_col), dim=-2, keepdim=True)  # [1,T,1,D]
        return ttnn.reshape(out, [b, s, 1, d])
