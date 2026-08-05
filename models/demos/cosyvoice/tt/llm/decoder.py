# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The autoregressive decoder: 14 pre-norm Transformer blocks with a KV cache.

Architecturally this is the flow encoder's block with two names changed
(`norm1`/`norm2` instead of `norm_mha`/`norm_ff`) and no macaron or conv module,
so `TtRelPosAttention` and the layer body are shared rather than duplicated.
The differences that matter are all at the edges:

**The input layer has a ReLU.** `input_layer: 'linear_legacy'` selects
`LegacyLinearNoSubsampling`, which is `Linear -> LayerNorm(eps=1e-5) -> Dropout ->
ReLU`. The plain `LinearNoSubsampling` the flow encoder uses stops at Dropout.
Nothing downstream would fail if the ReLU were missed -- the model would simply be
wrong. `embed_has_relu` in the exported meta records which one this checkpoint has.

**Two different LayerNorm epsilons.** `subsampling.py` pins `1e-5` on the
embedding norm; `encoder_layer.py` pins `1e-12` on the block norms. Both are
carried in the meta rather than assumed.

**The positional window follows the cache, not the chunk.** `forward_chunk`
recomputes `pos_emb = position_encoding(offset - cache_t1, size=cache_t1 + chunk)`,
discarding what the embedding produced. With `required_cache_size = -1` -- what
CosyVoice always passes -- the cache holds everything, so `offset == cache_t1` and
the offset term vanishes: the window is always the symmetric `2*key_size - 1` one.
That is what makes a single `espnet_rel_positional_encoding(key_size, d_model)`
correct at every step, and it is worth stating because it is only true for this
configuration.

**Prefill is causal; decode is not.** The prefill mask is `tril`, so the prompt
attends to itself autoregressively. A one-token decode step passes a `[1, 1, 1]`
all-true mask, and `forward_attention` slices it to the score width -- so it masks
nothing, which is right: every cached position is real history.
"""
from __future__ import annotations

import torch

import ttnn

from ..flow.encoder import TtRelPosAttention, _layernorm_weights, _linear, espnet_rel_positional_encoding
from ..hifigan.conv import accurate_compute_config

NEG_INF = -1e9  # bfloat16-safe stand-in for -inf under softmax


def causal_bias(size: int, dtype=torch.float32) -> torch.Tensor:
    """Additive `[1, 1, T, T]` mask: 0 on and below the diagonal, very negative above."""
    keep = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return torch.where(
        keep, torch.zeros(size, size, dtype=dtype), torch.full((size, size), NEG_INF, dtype=dtype)
    ).reshape(1, 1, size, size)


class TtTransformerLayer:
    """`x = x + attn(norm1(x))`, `x = x + ff(norm2(x))`, with an optional KV cache."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16, cc=None):
        self.device, self.dtype = device, dtype
        self.cc = accurate_compute_config(device) if cc is None else cc
        self.eps = meta["layer_norm_eps"]
        self.ffn_act = ttnn.relu if meta.get("ffn_activation", "relu") == "relu" else ttnn.silu
        self.attn = TtRelPosAttention(device, bag.sub("self_attn"), meta["n_head"], meta["d_k"], dtype, self.cc)
        self.w1, self.b1 = _linear(device, bag, "feed_forward.w_1", dtype)
        self.w2, self.b2 = _linear(device, bag, "feed_forward.w_2", dtype)
        self.g1, self.bt1 = _layernorm_weights(device, bag, "norm1", dtype)
        self.g2, self.bt2 = _layernorm_weights(device, bag, "norm2", dtype)

    def __call__(self, x, pos_emb, mask=None, cache=None, return_cache=True):
        h = ttnn.layer_norm(x, weight=self.g1, bias=self.bt1, epsilon=self.eps)
        a, new_cache = self.attn.forward_cached(h, pos_emb, mask=mask, cache=cache, return_cache=return_cache)
        ttnn.deallocate(h)
        x1 = ttnn.add(x, a)
        ttnn.deallocate(a)
        ttnn.deallocate(x)

        h = ttnn.layer_norm(x1, weight=self.g2, bias=self.bt2, epsilon=self.eps)
        f = ttnn.linear(h, self.w1, bias=self.b1, compute_kernel_config=self.cc)
        ttnn.deallocate(h)
        f = self.ffn_act(f)  # "relu" for this stack -- the text encoder uses SiLU
        f2 = ttnn.linear(f, self.w2, bias=self.b2, compute_kernel_config=self.cc)
        ttnn.deallocate(f)
        out = ttnn.add(x1, f2)
        ttnn.deallocate(f2)
        ttnn.deallocate(x1)
        return out, new_cache


class TtARDecoder:
    """`TransformerEncoder.forward_chunk` -- prefill and one-token decode alike."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16):
        self.device, self.dtype, self.meta = device, dtype, meta
        self.cc = accurate_compute_config(device)
        self.d_model = meta["d_model"]
        self.xscale = float(self.d_model) ** 0.5
        self.has_relu = bool(meta.get("embed_has_relu", False))
        self.embed_eps = meta.get("embed_norm_eps", 1e-5)

        self.w_in, self.b_in = _linear(device, bag, "embed.out.0", dtype)
        self.g_in, self.bt_in = _layernorm_weights(device, bag, "embed.out.1", dtype)
        self.layers = [
            TtTransformerLayer(device, bag.sub(f"encoders.{i}"), meta, dtype, self.cc) for i in range(meta["n_layers"])
        ]
        self.g_after, self.bt_after = _layernorm_weights(device, bag, "after_norm", dtype)
        self._pos_cache: dict[int, object] = {}

    def positional(self, key_size: int):
        """The `[1, 2*key_size - 1, d_model]` window, cached per size.

        Decoding grows the key size by one per token, so each step needs a window
        one element wider than the last. Generating it on the host and uploading is
        cheap next to 14 blocks of attention, but caching keeps the repeated
        prefill/verification passes from rebuilding it.
        """
        if key_size not in self._pos_cache:
            pos = espnet_rel_positional_encoding(key_size, self.d_model)
            self._pos_cache[key_size] = ttnn.from_torch(
                pos, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._pos_cache[key_size]

    def embed(self, xs):
        """Linear -> LayerNorm -> [ReLU] -> scale by sqrt(d_model)."""
        h = ttnn.linear(xs, self.w_in, bias=self.b_in, compute_kernel_config=self.cc)
        n = ttnn.layer_norm(h, weight=self.g_in, bias=self.bt_in, epsilon=self.embed_eps)
        ttnn.deallocate(h)
        if self.has_relu:
            r = ttnn.relu(n)
            ttnn.deallocate(n)
            n = r
        out = ttnn.multiply(n, self.xscale)
        ttnn.deallocate(n)
        return out

    def forward_chunk(self, xs, caches=None, mask=None):
        """xs `[1, chunk, input_size]`, `caches` a list of `(k, v)` per layer.

        Returns `(ys, new_caches)`. The old caches are consumed -- their k/v are
        concatenated into the new ones and then freed, so the caller must not hold
        a reference past this call.
        """
        chunk = xs.shape[1]
        cache_t1 = 0 if not caches else caches[0][0].shape[2]
        pos = self.positional(cache_t1 + chunk)

        h = self.embed(xs)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches else None
            h, new_cache = layer(h, pos, mask=mask, cache=cache, return_cache=True)
            if cache is not None:
                for t in cache:
                    ttnn.deallocate(t)
            new_caches.append(new_cache)
        out = ttnn.layer_norm(h, weight=self.g_after, bias=self.bt_after, epsilon=self.meta["layer_norm_eps"])
        ttnn.deallocate(h)
        return out, new_caches

    @staticmethod
    def free_caches(caches):
        for cache in caches or []:
            for t in cache:
                ttnn.deallocate(t)
