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
    -> top-p / nucleus    (softmax over the sorted top-k window -> exclusive cumsum
                          `ttnn.cumsum` < p -> smallest kept logit = nucleus threshold,
                          combined with the top-k threshold via `ttnn.maximum`)
    -> categorical draw   via the Gumbel-max trick: argmax(logits + Gumbel noise),
                          since ttnn has no multinomial. Gumbel = -log(-log(U)),
                          U ~ uniform(0,1) from `ttnn.rand` (drawn in fp32 — bf16 is
                          too coarse near 0/1 and biases the draw toward greedy).

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
        self.rep = float(repetition_penalty)
        self.top_p = float(top_p)
        # nucleus needs a top-k window to sort/cumsum over; without one it is a no-op.
        self._nucleus = 0.0 < self.top_p < 1.0 and self.top_k > 0
        # bf16 throughout — ttnn.topk requires bf16, and the GPT logits are bf16 anyway.
        self._neg = ttnn.from_torch(
            torch.full((1, vocab_size), NEG_INF), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._one = ttnn.from_torch(torch.ones((1, 1)), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        # Counter-PRNG constants for the trace-compatible Gumbel draw (pick_dev): a GLSL-style hash
        # frac(sin(step*A + idx*B) * C) driven by a per-step counter, in place of ttnn.rand (which
        # replays identical noise inside a trace). ``_arange_b`` = idx * B, precomputed.
        self._arange_b = ttnn.from_torch(
            torch.arange(vocab_size, dtype=torch.float32).reshape(1, vocab_size) * 78.233,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
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

        if self.top_k:
            vals = ttnn.topk(L, self.top_k, dim=-1, largest=True, sorted=True)[0]  # [1, k] desc
            kth = ttnn.slice(vals, [0, self.top_k - 1], [1, self.top_k])  # [1, 1] top-k threshold
            thr = kth
            if self._nucleus:
                # Nucleus over the sorted top-k window: keep the shortest prefix whose
                # cumulative prob first exceeds top_p (the token that crosses is kept).
                probs = ttnn.softmax(vals, dim=-1)  # [1, k] descending probabilities
                excl = ttnn.subtract(ttnn.cumsum(probs, dim=-1), probs)  # exclusive prefix sum
                keep = ttnn.lt(excl, self.top_p)  # [1, k] bool: positions inside the nucleus
                # smallest logit still kept = nucleus threshold (mask dropped to +inf, row-min).
                pos_inf = ttnn.add(ttnn.multiply(vals, 0.0), -NEG_INF)
                nuc = ttnn.min(ttnn.where(keep, vals, pos_inf), dim=-1, keepdim=True)  # [1, 1]
                thr = ttnn.maximum(nuc, kth)
            L = ttnn.where(ttnn.ge(L, thr), L, self._neg)

        # Gumbel-max: argmax(L + g), g = -log(-log(U)), U ~ uniform(0,1). Draw/compute the
        # noise in fp32 (bf16 U is too coarse near 0/1 and biases the draw toward greedy).
        u = ttnn.clamp(ttnn.rand([1, self.v], device=self.device, dtype=ttnn.float32), 1e-4, 1.0 - 1e-3)
        g = ttnn.multiply(ttnn.log(ttnn.multiply(ttnn.log(u), -1.0)), -1.0)
        noisy = ttnn.add(ttnn.typecast(L, ttnn.float32), g)
        tok = ttnn.argmax(ttnn.to_layout(noisy, ttnn.ROW_MAJOR_LAYOUT), dim=-1)
        token = int(ttnn.to_torch(tok).flatten()[0].item())

        if self.rep != 1.0:
            self._mark(token)
        return token

    def _apply_penalty_temp_topk(self, logits):
        """Shared rep-penalty -> temperature -> top-k/top-p logit shaping (returns ``[1, V]``)."""
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
        return L

    def pick_dev(self, logits, step_t):
        """Fully-on-device sample for the TRACED decode loop: same rep/temp/top-k/top-p shaping,
        but the Gumbel noise comes from a counter-based PRNG (``step_t``: a ``[1, 1]`` fp32 tensor
        holding the decode step) instead of ``ttnn.rand`` — so it is trace-capturable and varies
        every step. Returns the sampled id as a DEVICE ``[1, 1]`` uint32 tensor (no host readback)
        and updates the ``seen`` mask on device. Greedy when ``temperature <= 0``."""
        L = self._apply_penalty_temp_topk(logits)
        Lf = ttnn.typecast(L, ttnn.float32)
        if self.temperature > 0.0:
            # U = frac(sin(step*12.9898 + idx*78.233) * 43758.5453) in (0,1); g = -log(-log(U)).
            h = ttnn.add(ttnn.multiply(step_t, 12.9898), self._arange_b)  # [1,1] broadcasts over [1,V]
            u = ttnn.clamp(ttnn.frac(ttnn.multiply(ttnn.sin(h), 43758.5453)), 1e-4, 1.0 - 1e-3)
            g = ttnn.multiply(ttnn.log(ttnn.multiply(ttnn.log(u), -1.0)), -1.0)
            Lf = ttnn.add(Lf, g)
        tok = ttnn.argmax(ttnn.to_layout(Lf, ttnn.ROW_MAJOR_LAYOUT), dim=-1)  # device
        tok = ttnn.reshape(ttnn.typecast(tok, ttnn.uint32), [1, 1])
        if self.rep != 1.0:
            self.seen = ttnn.scatter(self.seen, 1, tok, self._one)  # mark on device (no host int)
        return tok
