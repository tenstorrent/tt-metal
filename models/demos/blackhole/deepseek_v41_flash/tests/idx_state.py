# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU helpers to study the accuracy of the decode indexer selection (used by test_paged_long_context.py and idx_noise_floor.py).

``make_state``: builds the reference attention state of ONE user at decode position S-1 of an index layer, either synthetic (random caches,
flat scores) or REAL (the layer's real attention inputs ``attn_in`` from a prefill dump of ``reference/ref_prefill_dump.py``: prefill the first S-1
tokens through the layer's reference Attention, decode token S-1).
``capture``: hook around the model's ``sparse_attn`` that records, for the decode call, the dense softmax probabilities of every head over
[window | all compressed entries] (sink included in the denominator) so the attention-mass covered by ANY selected set can be computed later.
"""

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R


class Capture:
    def __init__(self, mod):
        self.mod, self.p_comp, self.ids = mod, None, None
        self._orig = mod.sparse_attn

        def hook(q, kv, attn_sink, topk_idxs, softmax_scale):
            if q.shape[1] == 1 and topk_idxs.shape[-1] > 128:  # the decode call of a compressed layer
                with torch.no_grad():
                    s = torch.einsum("bmhd,bnd->bmhn", q.float(), kv.float()) * softmax_scale  # dense [b,1,h,n]
                    s = torch.cat([s, attn_sink.float().view(1, 1, -1, 1).expand(s.shape[0], 1, -1, 1)], dim=-1)
                    p = torch.softmax(s, dim=-1)[
                        0, 0, :, :-1
                    ]  # [h, n] (sink excluded from the columns, included in the normaliser)
                    self.p_comp = p[
                        :, 128:
                    ].clone()  # compressed entries only (the first 128 slots are the window ring)
                    self.p_win = p[:, :128].sum(-1)
                    self.ids = (topk_idxs[0, 0, 128:] - 128).clone()
            return self._orig(q, kv, attn_sink, topk_idxs, softmax_scale)

        mod.sparse_attn = hook

    def release(self):
        self.mod.sparse_attn = self._orig

    def coverage(self, ids):
        """Fraction of the dense attention mass of the compressed entries that the id set ``ids`` covers (mean over heads weighted by mass)."""
        ids = ids[(ids >= 0) & (ids < self.p_comp.shape[1])].long().unique()
        return float(self.p_comp[:, ids].sum() / self.p_comp.sum())

    def total_coverage(self, ids):
        """Same including the window: mass of (window + ids) / mass of everything (what the output actually loses)."""
        ids = ids[(ids >= 0) & (ids < self.p_comp.shape[1])].long().unique()
        tot = self.p_comp.sum() + self.p_win.sum()
        return float((self.p_comp[:, ids].sum() + self.p_win.sum()) / tot)


def make_state(blk, S, real_dir=None, layer_id=None, seed=1):
    """Fill blk.attn for ONE user (batch slot 0) and return the decode input x_dec [1,1,5120] at position S-1 (the state holds S-1 prefilled tokens).
    synthetic: random caches (N-1 entries) and a random decode input; real: prefill of real attn_in."""
    torch.manual_seed(seed)
    at = blk.attn
    ratio = at.compress_ratio
    if real_dir is None:
        N = S // ratio if ratio else 0
        at.window_kv_cache.copy_(torch.randn_like(at.window_kv_cache))
        if ratio == 0:
            h, pm = R.embed_tokens(torch.randint(1000, 100000, (blk.attn.window_kv_cache.shape[0], 1)))
            return blk.attn_norm(blk.hc_pre(h, pm))[:1]
        at.compress_kv_cache[:, : N - 1] = torch.randn(at.compress_kv_cache.shape[0], N - 1, 512).to(
            at.compress_kv_cache.dtype
        )
        at.indexer.k_cache[:, : N - 1] = (torch.randn(at.indexer.k_cache.shape[0], N - 1, 128) * 0.7).to(
            at.indexer.k_cache.dtype
        )
        if ratio > 1:
            at.compressor.kv_state.copy_(torch.randn_like(at.compressor.kv_state))
            at.compressor.score_state.copy_(torch.randn_like(at.compressor.score_state))
        h, pm = R.embed_tokens(torch.randint(1000, 100000, (blk.attn.window_kv_cache.shape[0], 1)))
        return blk.attn_norm(blk.hc_pre(h, pm))[:1]
    d = torch.load(f"{real_dir}/layer_{layer_id}.pt")
    a_in = d["prefill"]["attn_in"][:1].to(torch.bfloat16)  # [1, S0, 5120] REAL attention inputs
    assert a_in.shape[1] >= S, (a_in.shape, S)
    with torch.no_grad():
        at(a_in[:, : S - 1], 0)  # reference prefill of the first S-1 real tokens
    return a_in[:, S - 1 : S]


def replicate_user0(at):
    """copy batch slot 0 of every reference cache to all slots (identical users)."""
    for t in (
        at.window_kv_cache,
        getattr(at, "compress_kv_cache", None),
        at.indexer.k_cache if at.indexer is not None else None,
    ):
        if t is not None:
            t[1:] = t[:1]
    c = at.compressor
    if c is not None and hasattr(c, "kv_state"):
        c.kv_state[1:] = c.kv_state[:1]
        c.score_state[1:] = c.score_state[:1]
