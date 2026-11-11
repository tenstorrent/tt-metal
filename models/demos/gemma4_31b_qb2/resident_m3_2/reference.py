# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference of the resident prototype's stubbed decode step, from the same quantized weights.

Weights are the cached device tensors dequantized (BF4 / BF8 values are exact in bf16), so quantization matches the
device; the math is fp32 with bf16 rounding where the device stores activations:
  h = bf16(rmsnorm(x));  per chip c: y_c = bf16(h @ Wqkv_c), o_c = y_c[:, :Ot] (attention stub);
  y = bf16(sum_c bf16(o_c @ Wo_c));  x = bf16(x + w_post_attn * rmsnorm(y));  h = bf16(rmsnorm(x))
  per chip: a_c = bf16(gelu_tanh(bf16(h @ G_c)) * bf16(h @ U_c));  y = bf16(sum_c bf16(a_c @ D_c))
  x = bf16((x + w_post_ff * rmsnorm(y)) * layer_scalar);  h = bf16(rmsnorm(x))
then logits = bf16(h @ Head) (final norm folded in), argmax with the smallest index on ties.
"""
import time

import torch

from models.demos.gemma4_31b_qb2.resident_m3_2.common import (
    E_DOWN,
    E_GU,
    E_HEAD,
    E_O_F,
    E_O_S,
    E_QKV_F,
    E_QKV_S,
    EPS,
    HIDDEN,
    IC,
    N_CHIPS,
    Q_F,
    Q_S,
    is_full,
)
from models.demos.gemma4_31b_qb2.resident_m3_2.weights import dequantized

bf16 = lambda t: t.bfloat16().float()


def rms(v):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)


def gelu_tanh(v):
    return torch.nn.functional.gelu(v, approximate="tanh")


class Reference:
    def __init__(self, cache, layers=60, matmul_dtype=torch.float32):
        self.layers, self.mm_dtype = layers, matmul_dtype
        t0 = time.time()
        self.w = []
        for l in range(layers):
            full = is_full(l)
            ents = (E_QKV_F, E_O_F) if full else (E_QKV_S, E_O_S)
            self.w.append({e: [w.to(matmul_dtype) for w in dequantized(cache, l, e)] for e in ents + (E_GU, E_DOWN)})
            if l % 10 == 9:
                print(f"reference weights: {l + 1} layers, {time.time() - t0:.0f} s", flush=True)
        self.head = [w.to(matmul_dtype) for w in dequantized(cache, 0, E_HEAD)]
        self.vectors = torch.load(f"{cache}/vectors.pt")
        self.embed = torch.load(f"{cache}/embed_scaled.pt", mmap=True)
        self.folded = self.vectors.get("folded", True)

    def pre(self, l, half):
        """the pre-norm weight ahead of layer l's mixer (half 0) / MLP (half 1); l = layers: the final norm (1.0 when
        the cache folds them into the weights)"""
        if self.folded:
            return 1.0
        if l >= self.layers:
            return self.vectors["final_norm"].float()
        return self.vectors["pre"][l][half].float()

    def mm(self, h, w):
        """bf16(h @ w), the product of a bf16 vector and a bf16 matrix accumulated in fp32"""
        return bf16(h.to(self.mm_dtype) @ w.to(self.mm_dtype))

    def step(self, token, rounds_out=None):
        """(logits [VOCAB], h_final [HIDDEN]) of one step from `token`; rounds_out (a list) gets h of every round"""
        x = self.embed[int(token)].float()
        h = bf16(rms(x) * self.pre(0, 0))
        if rounds_out is not None:
            rounds_out.append(h)
        for l in range(self.layers):
            full = is_full(l)
            w = self.w[l]
            ot = Q_F if full else Q_S
            qkv, o = (w[E_QKV_F], w[E_O_F]) if full else (w[E_QKV_S], w[E_O_S])
            parts = [self.mm(self.mm(h, qkv[c])[:ot], o[c]) for c in range(N_CHIPS)]
            y = bf16(sum(parts))
            pa, pf = (p.float() for p in self.vectors["post"][l])
            x = bf16(x + pa * rms(y))
            h = bf16(rms(x) * self.pre(l, 1))
            if rounds_out is not None:
                rounds_out.append(h)
            parts = []
            for c in range(N_CHIPS):
                gu = self.mm(h, w[E_GU][c])
                a = bf16(gelu_tanh(gu[:IC]) * gu[IC:])
                parts.append(self.mm(a, w[E_DOWN][c]))
            y = bf16(sum(parts))
            x = bf16((x + pf * rms(y)) * self.vectors["scalar"][l])
            h = bf16(rms(x) * self.pre(l + 1, 0))
            if rounds_out is not None:
                rounds_out.append(h)
        logits = torch.cat([self.mm(h, self.head[c]) for c in range(N_CHIPS)])
        return logits, h

    @staticmethod
    def argmax(logits):
        m = logits.max()
        return int((logits == m).nonzero()[0])

    def chain(self, token, steps):
        out = []
        for _ in range(steps):
            logits, _ = self.step(token)
            token = self.argmax(logits)
            out.append(token)
        return out
