# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.xtts.config import NEG_INF  # noqa: F401 — re-exported for callers


class TtSampler:
    def __init__(self, device, vocab_size, temperature, top_k=0, repetition_penalty=1.0, top_p=1.0):
        """Initialize sampler tables and repetition-penalty state."""
        self.device = device
        self.v = vocab_size
        self.temperature = float(temperature)
        self.top_k = int(top_k) if top_k and top_k < vocab_size else 0
        self.rep = float(repetition_penalty)
        self.top_p = float(top_p)
        self._nucleus = 0.0 < self.top_p < 1.0 and self.top_k > 0
        self._neg = ttnn.from_torch(
            torch.full((1, vocab_size), NEG_INF), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._one = ttnn.from_torch(torch.ones((1, 1)), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self._arange_v = ttnn.from_torch(
            torch.arange(vocab_size, dtype=torch.float32).reshape(1, vocab_size),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
        self.reset()

    def reset(self):
        # Clear seen in place so traced pick_dev keeps binding this buffer.
        """Clear or allocate the seen-token buffer for a new sequence."""
        if getattr(self, "seen", None) is not None:
            ttnn.multiply(self.seen, 0.0, output_tensor=self.seen)
            return
        # bf16 required: fp32 seen breaks ttnn.where dtype with bf16 logits.
        self.seen = ttnn.from_torch(
            torch.zeros((1, self.v)), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def release(self):
        """Free device tables after the decode trace is released (not while bound)."""
        for name in ("seen", "_neg", "_one", "_arange_v"):
            t = getattr(self, name, None)
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
            setattr(self, name, None)

    def _mark(self, token):
        """Mark a sampled token as seen for repetition penalty."""
        idx = ttnn.from_torch(
            torch.tensor([[token]], dtype=torch.int32), device=self.device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT
        )
        self.seen = ttnn.scatter(self.seen, 1, idx, self._one)

    def pick(self, logits):
        """Sample one token from logits with penalty, temp, and top-k/p."""
        L = ttnn.typecast(ttnn.reshape(logits, [1, self.v]), ttnn.bfloat16)

        if self.rep != 1.0:
            pos = ttnn.gt(L, 0.0)
            penalized = ttnn.where(pos, ttnn.multiply(L, 1.0 / self.rep), ttnn.multiply(L, self.rep))
            L = ttnn.where(ttnn.gt(self.seen, 0.5), penalized, L)

        if self.temperature != 1.0:
            L = ttnn.multiply(L, 1.0 / self.temperature)

        if self.top_k:
            vals = ttnn.topk(L, self.top_k, dim=-1, largest=True, sorted=True)[0]
            kth = ttnn.slice(vals, [0, self.top_k - 1], [1, self.top_k])
            thr = kth
            if self._nucleus:
                probs = ttnn.softmax(vals, dim=-1)
                excl = ttnn.subtract(ttnn.cumsum(probs, dim=-1), probs)
                keep = ttnn.lt(excl, self.top_p)
                pos_inf = ttnn.add(ttnn.multiply(vals, 0.0), -NEG_INF)
                nuc = ttnn.min(ttnn.where(keep, vals, pos_inf), dim=-1, keepdim=True)
                thr = ttnn.maximum(nuc, kth)
            L = ttnn.where(ttnn.ge(L, thr), L, self._neg)

        # Gumbel in fp32 — bf16 U is too coarse near 0/1.
        u = ttnn.clamp(ttnn.rand([1, self.v], device=self.device, dtype=ttnn.float32), 1e-4, 1.0 - 1e-3)
        g = ttnn.multiply(ttnn.log(ttnn.multiply(ttnn.log(u), -1.0)), -1.0)
        noisy = ttnn.add(ttnn.typecast(L, ttnn.float32), g)
        tok = ttnn.argmax(noisy, dim=-1)
        token = int(ttnn.to_torch(tok).flatten()[0].item())

        if self.rep != 1.0:
            self._mark(token)
        return token

    def _apply_penalty_temp_topk(self, logits):
        """Apply repetition penalty, temperature, and top-k/p to logits."""
        L = ttnn.typecast(ttnn.reshape(logits, [1, self.v]), ttnn.bfloat16)
        if self.rep != 1.0:
            pos = ttnn.gt(L, 0.0)
            penalized = ttnn.where(pos, ttnn.multiply(L, 1.0 / self.rep), ttnn.multiply(L, self.rep))
            L = ttnn.where(ttnn.gt(self.seen, 0.5), penalized, L)
        if self.temperature > 0.0 and self.temperature != 1.0:
            L = ttnn.multiply(L, 1.0 / self.temperature)
        if self.top_k:
            vals = ttnn.topk(L, self.top_k, dim=-1, largest=True, sorted=True)[0]
            kth = ttnn.slice(vals, [0, self.top_k - 1], [1, self.top_k])
            thr = kth
            if self._nucleus:
                probs = ttnn.softmax(vals, dim=-1)
                excl = ttnn.subtract(ttnn.cumsum(probs, dim=-1), probs)
                keep = ttnn.lt(excl, self.top_p)
                pos_inf = ttnn.add(ttnn.multiply(vals, 0.0), -NEG_INF)
                nuc = ttnn.min(ttnn.where(keep, vals, pos_inf), dim=-1, keepdim=True)
                thr = ttnn.maximum(nuc, kth)
            L = ttnn.where(ttnn.ge(L, thr), L, self._neg)
        return L

    def pick_dev(self, logits, gumbel=None, bias=None):
        """Argmax-sample on device with optional Gumbel noise and stop bias."""
        L = self._apply_penalty_temp_topk(logits)
        Lf = ttnn.typecast(L, ttnn.float32)
        if bias is not None:
            Lf = ttnn.add(Lf, bias)
        if self.temperature > 0.0 and gumbel is not None:
            Lf = ttnn.add(Lf, gumbel)
        tok = ttnn.argmax(Lf, dim=-1)
        tok = ttnn.reshape(ttnn.typecast(tok, ttnn.uint32), [1, 1])
        if self.rep != 1.0:
            # Mark seen in place so penalty accumulates across traced replays.
            oh = ttnn.typecast(ttnn.eq(self._arange_v, ttnn.typecast(tok, ttnn.float32)), ttnn.bfloat16)
            ttnn.maximum(self.seen, oh, output_tensor=self.seen)
        return tok
