# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""On-device token sampler for the XTTS-v2 GPT decode loop.

Greedy argmax is deterministic but collapses on long generations (repeats a code
to silence). XTTS's real inference samples with repetition penalty + temperature +
top-k, which is what gives natural, self-terminating audio. This does that entirely
on device (no host fallback in the tensor path):

    repetition penalty  (seen-token mask, `ttnn.scatter`)
    -> temperature       (`ttnn.mul`)
    -> top-k truncation  (`ttnn.topk`, mask below the k-th value to -inf)
    -> top-p / nucleus    (softmax -> exclusive-prefix cumsum < p keeps the smallest
                          set whose mass >= p; the min kept logit is the threshold,
                          mask everything below it to -inf) — coqui XTTS uses top_p 0.85
    -> categorical draw   via the Gumbel-max trick: argmax(logits + Gumbel noise),
                          since ttnn has no multinomial. Gumbel = -log(-log(U)),
                          U ~ uniform(0,1) from `ttnn.rand` (drawn in fp32 — bf16 noise
                          is too coarse and biases the draw toward the top logit).

Only the final sampled id crosses to host (loop control + next embedding index) —
the same one-int-per-step read greedy already needs.
"""

import torch
import ttnn

NEG_INF = -1e30


class TtSampler:
    """Repetition-penalty / temperature / top-k / Gumbel-max sampler. Holds the
    per-generation repetition ``seen`` mask; call :meth:`reset` between generations."""

    def __init__(self, device, vocab_size, temperature, top_k=0, repetition_penalty=1.0, top_p=1.0):
        self.device = device
        self.v = vocab_size
        self.temperature = float(temperature)
        self.top_k = int(top_k) if top_k and top_k < vocab_size else 0
        self.top_p = float(top_p) if top_p and 0.0 < top_p < 1.0 else 0.0
        self.rep = float(repetition_penalty)
        # bf16 throughout — ttnn.topk requires bf16, and the GPT logits are bf16 anyway.
        self._neg = ttnn.from_torch(
            torch.full((1, vocab_size), NEG_INF), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._one = ttnn.from_torch(torch.ones((1, 1)), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.reset()

    def reset(self):
        self.seen = ttnn.from_torch(
            torch.zeros((1, self.v)), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def _mark(self, token):
        idx = ttnn.from_torch(
            torch.tensor([[token]], dtype=torch.int32), device=self.device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT
        )
        self.seen = ttnn.scatter(self.seen, 1, idx, self._one)

    def pick(self, logits):
        """``logits`` is ``[1, 1, V]`` (or ``[1, V]``); returns the sampled id (int)."""
        L = ttnn.typecast(ttnn.reshape(logits, [1, self.v]), ttnn.bfloat16)

        if self.rep != 1.0:
            pos = ttnn.gt(L, 0.0)
            penalized = ttnn.where(pos, ttnn.multiply(L, 1.0 / self.rep), ttnn.multiply(L, self.rep))
            L = ttnn.where(ttnn.gt(self.seen, 0.5), penalized, L)

        if self.temperature != 1.0:
            L = ttnn.multiply(L, 1.0 / self.temperature)

        if self.top_k or self.top_p:
            # Sort a window of the highest logits (descending). With an explicit
            # top_k, that k is the window; otherwise use a cap wide enough to hold
            # the nucleus mass for the audio vocab.
            k = self.top_k or min(self.v, 100)
            vals = ttnn.topk(L, k, dim=-1, largest=True, sorted=True)[0]  # [1, k] desc
            thr = ttnn.slice(vals, [0, k - 1], [1, k])  # k-th value = top-k threshold

            if self.top_p:
                # Nucleus: keep the smallest prefix whose cumulative prob >= top_p.
                # exclusive-prefix cumsum < p keeps token j (and always keeps j=0),
                # so the smallest kept logit is the nucleus threshold.
                probs = ttnn.softmax(vals, dim=-1)  # over the sorted window
                excl = ttnn.sub(ttnn.cumsum(probs, dim=-1), probs)  # exclusive prefix (excl[...,0]=0)
                keep = ttnn.typecast(ttnn.lt(excl, self.top_p), ttnn.bfloat16)  # 1.0 kept / 0.0 dropped
                drop = ttnn.add(ttnn.multiply(keep, -1.0), 1.0)  # 1 - keep
                # push dropped logits to +inf so the row-min ignores them.
                kept = ttnn.add(vals, ttnn.multiply(drop, 1e30))
                p_thr = ttnn.min(kept, dim=-1, keepdim=True)  # smallest kept logit
                thr = ttnn.maximum(thr, p_thr)  # apply the stricter of top_k / top_p

            L = ttnn.where(ttnn.ge(L, thr), L, self._neg)

        # Gumbel-max: argmax(L + g), g = -log(-log(U)), U ~ uniform(0,1). Draw the
        # noise in fp32 — bf16 uniforms are too coarse and bias the draw toward the
        # single top logit (collapsing sampling back to near-greedy).
        u = ttnn.clamp(ttnn.rand([1, self.v], device=self.device, dtype=ttnn.float32), 1e-4, 1.0 - 1e-3)
        g = ttnn.multiply(ttnn.log(ttnn.multiply(ttnn.log(u), -1.0)), -1.0)
        Lf = ttnn.add(ttnn.typecast(L, ttnn.float32), g)
        tok = ttnn.argmax(ttnn.to_layout(Lf, ttnn.ROW_MAJOR_LAYOUT), dim=-1)
        token = int(ttnn.to_torch(tok).flatten()[0].item())

        if self.rep != 1.0:
            self._mark(token)
        return token
