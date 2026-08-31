from typing import Optional

import torch
import ttnn

from .common import FULL_TILE, SINGLE_USER_TILE, DeepSeekV4Module, _profile, width_sharded_l1_config, with_tile_height
from .decode_prefetch import (
    DECODE_LAYOUTS,
    HC_FN_GCB,
    HC_FN_GCB_PAGES,
    check_decode_layout,
    ensure_named_gcb,
    hc_fn_page_bytes,
    hc_fn_ring_specs,
)
from .l1_weights import packed_weight_spec
from .layers import Linear, LinearDecode, _rms_norm_unweighted
from .weight_cache import WeightCache, _as_cache, _load_weight, _materialize, _memo

# Partial-K cut for the fused ``fn`` matmul, read off the layout registry rather than
# repeated here: the GCB is sized from that entry before any layer exists, so a second
# copy of the number is a buffer sized for a cut the layer does not build. At the model's
# K = H*D = 16384 it is a [256, 32] slab on 64 B cores, the receiver count every other
# decode weight uses.
_HC_FN_K_BLOCKS = DECODE_LAYOUTS[HC_FN_GCB]["k_blocks"]


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

    The learned ``base`` / ``scale`` parameters are split host-side into their
    ``pre`` / ``post`` / ``comb`` parts; ``fn`` runs as a single fused matmul whose
    output is split into the three parts inside ``fused_hyperconnection``.
    See ``modular_deepseek_v4.py`` for the reference math.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        packed_weights=None,
        packed_name: str | None = None,
        use_prefetcher: bool = False,
        prefetch_buffers: Optional[dict] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
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

        # The pre [H] / post [H] / comb [H*H] projections are contiguous slices of
        # the packed ``fn`` weight, so fuse them into one matmul ([(2+H)*H, H*D])
        # and split the output back into the three parts inside the fused op.
        #
        # Large-K / small-N (K = H*D, N = (2+H)*H padded to a tile): partial-K
        # ``LinearDecode`` over ``_HC_FN_K_BLOCKS`` B cores, ``n_blocks=1`` reducing the
        # K-partials onto one output core.
        #
        # Under the prefetcher this weight streams through ``HC_FN_GCB`` -- a second ring,
        # because its 8-tile slab has no page in common with the shared group's 32 -- and
        # every hyper-connection on the device shares that one ring. Attaching it to the
        # caller's ``prefetch_buffers`` is what makes them share: a GCB costs ~176 B of the
        # DRISC senders' 1 KB state zone, so one per hyper-connection overflows it at the
        # third layer.
        if packed_weights is None:
            k = hc * self.hidden
            n = ((2 * hc + hc * hc + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

            def fn_weight():
                w = fn()[: 2 * hc + hc * hc]
                if w.shape[0] < n:
                    w = torch.nn.functional.pad(w, (0, 0, 0, n - w.shape[0]))
                return w

            prefetch = {}
            if use_prefetcher:
                layout = check_decode_layout(HC_FN_GCB, k, n)
                if prefetch_buffers is None:
                    prefetch_buffers = {}
                prefetch = {
                    "use_prefetcher": True,
                    "global_cb": ensure_named_gcb(
                        prefetch_buffers,
                        HC_FN_GCB,
                        device,
                        hc_fn_ring_specs(),
                        weight_dtype,
                        num_pages=HC_FN_GCB_PAGES,
                    ),
                    "global_cb_page_bytes": hc_fn_page_bytes(weight_dtype),
                }
            self.fn = LinearDecode(
                fn_weight,
                device,
                cache.file("fn.decode"),
                dtype=weight_dtype,
                K=k,
                N=n,
                partial_width_sharded=True,
                k_blocks=_HC_FN_K_BLOCKS,
                n_blocks=1,
                tile_height=1,
                **prefetch,
            )
        else:
            tensor, layout, slot = packed_weights
            spec = packed_weight_spec(layout, slot, packed_name)
            self.fn = LinearDecode(
                lambda: fn()[: 2 * hc + hc * hc],
                device,
                cache.file("fn"),
                dtype=ttnn.bfloat4_b,
                K=spec.K,
                N=spec.N,
                partial_width_sharded=True,
                k_blocks=spec.k_blocks,
                n_blocks=spec.n_blocks,
                packed_weight_tensor=tensor,
                packed_weight_spec=spec,
            )
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

    def prefetch_weights(self):
        """Queue the ``fn`` prefetch when that weight is streamed through a GCB."""
        if isinstance(self.fn, LinearDecode):
            self.fn.fetch_weights()

    def forward(self, hidden_streams: ttnn.Tensor):
        """``hidden_streams`` ``[B, S, H, D]`` -> ``(post [B,S,H,1], comb [B,S,H,H], collapsed [B,S,1,D])``."""
        b, s, hc, d = hidden_streams.shape
        t = b * s

        # Flatten streams to [1,1,T,H*D] and unweighted-RMSNorm over H*D.
        tile_height = hidden_streams.get_tile().tile_shape[0]
        if isinstance(self.fn, LinearDecode):
            flat_mem_config = self.fn.get_input_memory_config(t, hc * d, tile_height)
        else:
            flat_mem_config = width_sharded_l1_config(t, hc * d, self.device)
        flat = ttnn.reshape(hidden_streams, [1, 1, t, hc * d], memory_config=flat_mem_config)
        flat = _rms_norm_unweighted(flat, self.norm_eps)
        flat = ttnn.tilize(
            flat,
            tile=SINGLE_USER_TILE,
            memory_config=with_tile_height(flat.memory_config(), t, tile_height=1),
        )
        fused_w = self.fn(flat)  # [1,1,T,(2+H)*H]
        fused_w = ttnn.tilize(
            fused_w,
            tile=FULL_TILE,
            memory_config=with_tile_height(fused_w.memory_config(), t, tile_height=ttnn.TILE_SIZE),
        )

        fused_w = ttnn.reshape(
            fused_w, [1, 1, t, (2 + hc) * hc], fused_w.padded_shape, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        fused_w = ttnn.to_memory_config(fused_w, ttnn.DRAM_MEMORY_CONFIG)
        _profile(self.device)

        # The pre_w / post_w / comb_w slices are split out of `fused_w` inside the op
        # (fused_hyperconnection_pre_post kernel); pre_w / post_w are consumed in-place
        # and comb_w is returned already laid out as the [1,1,H,H] comb matrix.
        return ttnn.experimental.deepseek.fused_hyperconnection(
            hidden_streams,
            fused_w=fused_w,
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
        flat_mem_config = width_sharded_l1_config(1, hc * d, self.device)
        flat = ttnn.to_memory_config(flat, flat_mem_config)
        flat = _rms_norm_unweighted(flat, self.norm_eps)

        mixes = self.fn(flat)  # [1,1,T,H]
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mixes, self.scale), self.base)), self.eps)
        _profile(self.device)
        hs = ttnn.reshape(hidden_streams, [1, t, hc, d])
        pre_col = ttnn.reshape(pre, [1, t, hc, 1])
        out = ttnn.sum(ttnn.multiply(hs, pre_col), dim=-2, keepdim=True)  # [1,T,1,D]
        return ttnn.reshape(out, [b, s, 1, d])
