# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked gated delta rule composed from ttnn primitives, fp32 throughout.

Why not the fused ``ttnn.transformer.chunk_gated_delta_rule``: fed identical bf16 q/k/v from the real
checkpoint (layer 28, 2048 tokens), the fused op is 2.6% off on the output and 6.8% on the final state,
where a CPU fp32 core is 0.14% / 0.13% (scripts/diag_gdn_core.py); none of its knobs move it. The same
chunk algorithm emulated with tf32 matmuls stays at 0.14%, so the loss is in the op's own algorithm,
not the hardware. This composition (upstream ``torch_chunk_gated_delta_rule``, math only) measures
~0.04% / 0.07% with the fp32 conv output as input. It is the default; ``QWEN38_GDN_CORE=fused`` restores
the fused op.

Algorithm (per TP column: H value heads, chunk C, NC = T / C chunks, state S [H, Dk, Dv]):
  vectorised over all chunks
    q, k   L2-normalised (q also * Dk^-0.5); kb = k * beta, vb = v * beta
    gc     within-chunk cumsum of g (ttnn.cumsum, fp32 — a matmul cumsum would round g to tf32 and
           put up to ~0.1 absolute error into exp())
    decay  exp(gc_i - gc_j) on the lower triangle (upper set to -1e4 before exp: no overflow). Every
           exp() argument is clamped at -80: real Qwen3.8 gates reach cumulative decays of -1e4 and
           ttnn.exp returns inf (not 0) that far down.
    M      -(kb @ k^T) * decay, strictly lower  (|M| <= 1, nilpotent)
    T      (I - M)^-1 = (I+M)(I+M^2)(I+M^4)...  — log2(C) products; exact for nilpotent M. Stable
           at C = 32 in tf32-matmul emulation on every layer checked; C = 64 degrades on layer 1's gates
           (0.19% vs 0.016%) and overflows on device, C = 128 is NaN. So C = 32.
    vp = T @ vb,  kc = T @ (kb * exp(gc)),  att = (q @ k^T) * decay,
    qg = q * exp(gc),  kd = k * exp(gc_last - gc),  el = exp(gc_last)
  sequential over chunks
    v_new = vp_c - kc_c @ S;  o_c = qg_c @ S + att_c @ v_new;  S = S * el_c + kd_c^T @ v_new
"""

from __future__ import annotations

import math
import os

import torch

import ttnn
from models.demos.qwen_3_8_27b.tt.common import hifi4_fp32

CHUNK = 32


def gdn_core_mode() -> str:
    m = os.environ.get("QWEN38_GDN_CORE", "composed")
    assert m in ("composed", "fused"), m
    return m


EXP_FLOOR = -80.0  # exp() arguments are clamped here: ttnn.exp returns inf for very negative inputs


def _exp(x):
    c = ttnn.clamp(x, min=EXP_FLOOR)
    y = ttnn.exp(c)
    ttnn.deallocate(c)
    return y


class ChunkDeltaRule:
    """Owns the chunk-size constants (device-resident, built once)."""

    def __init__(self, mesh_device, C: int = CHUNK):
        self.C = C
        self.ckc = hifi4_fp32()
        rp = ttnn.ReplicateTensorToMesh(mesh_device)
        up = lambda t: ttnn.from_torch(
            t[None, None], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rp
        )  # noqa: E731
        tril = torch.tril(torch.ones(C, C))
        self.tril = up(tril)
        self.stril = up(tril - torch.eye(C))
        self.upper_neg = up((1 - tril) * -1e4)
        self.eye = up(torch.eye(C))
        self.two_eye = up(2 * torch.eye(C))
        self.n_refine = int(os.environ.get("QWEN38_GDN_NEWTON", "3"))
        self.ones = up(torch.ones(C, C))
        self.n_sq = int(math.log2(C)) - 1  # (I+M) then (I+M^2) ... (I+M^(C/2))
        self.debug = None  # diagnostics: callable(name, tensor)

    def _mm(self, a, b, **kw):
        return ttnn.matmul(
            a, b, compute_kernel_config=self.ckc, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw
        )

    def __call__(self, q, k, v, g, beta, S0=None):
        """q, k: [H, NC, C, Dk] fp32 (already GVA-expanded, NOT normalised); v [H, NC, C, Dv] fp32;
        g, beta [H, NC, C, 1] fp32; S0 [H, 1, Dk, Dv] fp32 or None.
        Returns o [H, NC, C, Dv] fp32 and S [H, 1, Dk, Dv] fp32."""
        H, NC, C, Dk = q.shape
        Dv = v.shape[-1]
        mm = self._mm
        dealloc = lambda *ts: [ttnn.deallocate(t) for t in ts]  # noqa: E731

        def l2n(x, scale=1.0):
            ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
            sse = ttnn.add(ss, 1e-6)
            inv = ttnn.rsqrt(sse)
            dealloc(ss, sse)
            if scale != 1.0:
                inv2 = ttnn.multiply(inv, scale)
                ttnn.deallocate(inv)
                inv = inv2
            y = ttnn.multiply(x, inv)
            ttnn.deallocate(inv)
            return y

        qn = l2n(q, Dk**-0.5)
        kn = l2n(k)
        kb = ttnn.multiply(kn, beta)
        vb = ttnn.multiply(v, beta)

        gc = ttnn.cumsum(g, dim=2)  # [H, NC, C, 1]
        g_row = ttnn.multiply(self.ones, gc)  # [.., i, j] = gc_i
        g_col = ttnn.transpose(g_row, -1, -2)  # [.., i, j] = gc_j
        diff = ttnn.subtract(g_row, g_col)
        dealloc(g_row, g_col)
        diff_m = ttnn.add(ttnn.multiply(diff, self.tril), self.upper_neg)
        decay = ttnn.multiply(_exp(diff_m), self.tril)
        dealloc(diff, diff_m)

        kkT = mm(kb, kn, transpose_b=True)
        M = ttnn.multiply(ttnn.multiply(kkT, decay), self.stril)
        M = ttnn.neg(M)
        dealloc(kkT)
        P = ttnn.add(M, self.eye)
        Mp = M
        for _ in range(self.n_sq):
            Mp2 = mm(Mp, Mp)
            if Mp is not M:
                ttnn.deallocate(Mp)
            Mp = Mp2
            f = ttnn.add(Mp, self.eye)
            P2 = mm(P, f)
            dealloc(P, f)
            P = P2
        if Mp is not M:
            ttnn.deallocate(Mp)
        # Newton-Schulz refinement P <- P (2I - A P), A = I - M: the product form above is exact in exact
        # arithmetic but loses ~9% on device (chained fp32 matmuls, real-checkpoint gates); each step
        # squares the residual.
        A = ttnn.subtract(self.eye, M)
        for _ in range(self.n_refine):
            AP = mm(A, P)
            R = ttnn.subtract(self.two_eye, AP)
            P2 = mm(P, R)
            dealloc(AP, R, P)
            P = P2
        dealloc(A, M)
        if self.debug:
            for n, t in (("gc", gc), ("decay", decay), ("P", P)):
                self.debug(n, t)

        egc = _exp(gc)
        vp = mm(P, vb)
        kbe = ttnn.multiply(kb, egc)
        kc = mm(P, kbe)
        dealloc(P, vb, kbe, kb)
        qkT = mm(qn, kn, transpose_b=True)
        att = ttnn.multiply(qkT, decay)
        dealloc(qkT, decay)
        qg = ttnn.multiply(qn, egc)
        glast = ttnn.slice(gc, [0, 0, C - 1, 0], [H, NC, C, 1])  # [H, NC, 1, 1]
        kd = ttnn.multiply(kn, _exp(ttnn.subtract(glast, gc)))
        el = _exp(glast)
        if self.debug:
            for n, t in (("vp", vp), ("kc", kc), ("att", att), ("qg", qg), ("kd", kd), ("el", el)):
                self.debug(n, t)
        dealloc(qn, kn, gc, egc, glast)

        S = S0
        if S is None:
            S = ttnn.from_torch(
                torch.zeros(H, 1, Dk, Dv),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=q.device(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(q.device()),
            )
        outs = []
        sl = lambda t, c, w: ttnn.slice(t, [0, c, 0, 0], [H, c + 1, t.shape[2], w])  # noqa: E731
        for c in range(NC):
            vp_c, kc_c, qg_c, att_c, kd_c = sl(vp, c, Dv), sl(kc, c, Dk), sl(qg, c, Dk), sl(att, c, C), sl(kd, c, Dk)
            el_c = ttnn.slice(el, [0, c, 0, 0], [H, c + 1, 1, 1])
            t1 = mm(kc_c, S)
            v_new = ttnn.subtract(vp_c, t1)
            a = mm(qg_c, S)
            b = mm(att_c, v_new)
            outs.append(ttnn.add(a, b))
            upd = mm(kd_c, v_new, transpose_a=True)
            decayed = ttnn.multiply(S, el_c)
            S_new = ttnn.add(decayed, upd)
            dealloc(vp_c, kc_c, qg_c, att_c, kd_c, el_c, t1, v_new, a, b, upd, decayed)
            if S is not S0:
                ttnn.deallocate(S)
            S = S_new
        dealloc(vp, kc, qg, att, kd, el)
        groups = [ttnn.concat(outs[i : i + 32], dim=1) for i in range(0, len(outs), 32)]
        dealloc(*outs)
        o = groups[0] if len(groups) == 1 else ttnn.concat(groups, dim=1)
        if len(groups) > 1:
            dealloc(*groups)
        return o, S
