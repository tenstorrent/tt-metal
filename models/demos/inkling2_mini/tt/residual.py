# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini residual-path ops for batch-1 decode on one card: block attention residual, cross-pass
messages, message gate and the static k=4 short convolution. All on device, no host reads, static shapes:
trace-safe.

Row convention: a per-token vector is a [1, 1, 32, C] TILE tensor whose row 0 is the token. Other rows are
ignored (they must be finite). Row moves are matmuls with 0/1 selector matrices: exact for bf16-valued data,
tf32-rounded otherwise (the FPU reads fp32 operands as tf32).

Module ctors take a weights dict keyed by design.md §4 names (torch or host ttnn tensors, §4 shapes):
    attn_res_q / mlp_res_q / final_res_q  [1,1,32,C] fp32, rows = queries
    msg_q                                 [1,12,32,C] fp32, rows 0-3 of loop layer l = q, k, v, mlp
    msg_gate.{q,k,v,mlp}.down             [1,1,2C,64] bf16; .forget / .gain [1,1,64,C] fp32
    post_attn_conv / post_mlp_conv        [1,1,4,C] fp32, row r = tap at lag r
"""

import math

import torch

import ttnn

TILE = 32
MAX_DEPTH = 5  # 4 sealed block states + 1 partial
NUM_LOOP_LAYERS = 12
NUM_CACHE_SETS = 48  # 24 layers + 12 loop layers x 2 extra passes (HF cache index)


def compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def to_device(t, device, dtype=ttnn.float32):
    """Host torch tensor -> 4D TILE device tensor in DRAM."""
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _torch(w):
    return ttnn.to_torch(w) if isinstance(w, ttnn.Tensor) else w


def _pad_rows(t, rows=TILE):
    """[..., r, C] -> [..., rows, C] zero padded."""
    return torch.nn.functional.pad(t, (0, 0, 0, rows - t.shape[-2]))


def _f32(x):
    return x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)


def cache_index(layer, loop_pass=1):
    """HF cache index of (layer, pass): L on pass 1, 24 + (p-2)*12 + (L-6) on passes 2-3 of loop layers."""
    return layer if loop_pass == 1 else 24 + (loop_pass - 2) * NUM_LOOP_LAYERS + (layer - 6)


class RowOps:
    """Constant selector / mask matrices shared by every op on a device (a few hundred KiB)."""

    def __init__(self, device):
        self.device = device
        self.ckc = compute_config(device)
        self._sel = {}
        self.place = [self.selector(TILE, TILE, [(k, 0)]) for k in range(TILE)]  # row 0 -> row k
        self.depth_mask = {}  # additive softmax mask over depth columns: keep cols < depth
        for d in range(1, MAX_DEPTH + 1):
            m = torch.full((TILE, TILE), -1e9)
            m[:, :d] = 0
            self.depth_mask[d] = to_device(m, device)
        self.shift = self.selector(TILE, TILE, [(i + 1, i) for i in range(3)])  # conv window rows 0..2 -> 1..3
        self.sum4 = self.selector(TILE, TILE, [(0, i) for i in range(4)])  # row 0 = sum of rows 0..3

    def selector(self, rows_out, rows_in, pairs):
        """0/1 matrix [rows_out, rows_in] with 1 at each (out_row, in_row); out = sel @ x."""
        key = (rows_out, rows_in, tuple(pairs))
        if key not in self._sel:
            s = torch.zeros(rows_out, rows_in)
            for o, i in pairs:
                s[o, i] = 1
            self._sel[key] = to_device(s, self.device)
        return self._sel[key]

    def mm(self, a, b, **kw):
        return ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=self.ckc, **kw)

    def row(self, x, j):
        """Row j of x ([1,1,R,C]) moved to row 0 of a [1,1,32,C] tensor (other rows zero)."""
        return self.mm(self.selector(TILE, x.shape[-2], [(0, j)]), x)

    def seal(self, stack, h, k):
        """Block-state stack with token vector h written into row k (row k of stack must be zero).

        stack=None starts an empty stack. Returns a new tensor; the old one stays valid (passes 2-3 of the
        loop restart from the pre-loop stack)."""
        placed = self.mm(self.place[k], _f32(h))
        return placed if stack is None else ttnn.add(stack, placed, dtype=ttnn.float32)

    def attend(self, q, s, mask):
        """softmax(q . rms(s)^T + mask) . rms(s); q [R,C] pre-scaled, s [32,C]. Returns [1,1,R,C] fp32."""
        n = ttnn.rms_norm(s, epsilon=1e-6, compute_kernel_config=self.ckc)  # zero rows stay zero
        logits = ttnn.add(self.mm(q, n, transpose_b=True), mask, dtype=ttnn.float32)
        w = ttnn.softmax(logits, dim=-1, numeric_stable=True, compute_kernel_config=self.ckc)
        return self.mm(w, n)


class AttnRes:
    """Inkling2LoopBlockAttentionResidual: softmax_depth(q . rms(S)^T / sqrt(C)) . rms(S) over
    S = [block states..., partial]. name = attn_res_q (1 query, or 3 = q,k,v in global layers),
    mlp_res_q or final_res_q."""

    def __init__(self, rows: RowOps, weights, name, num_queries=1):
        q = _torch(weights[name]).float().reshape(-1, _torch(weights[name]).shape[-1])[:TILE].clone()
        q[num_queries:] = 0
        self.rows = rows
        self.nq = num_queries
        self.q = to_device(_pad_rows(q) * q.shape[-1] ** -0.5, rows.device)  # 1/sqrt(C) folded into q

    def __call__(self, stack, partial, depth: int, split=True):
        """stack: [1,1,32,C] fp32, rows 0..depth-2 = sealed states, rows >= depth-1 zero (None at depth 1).
        partial: token vector. Returns a list of num_queries fp32 token vectors (split=False: one [1,1,32,C] tensor,
        row j = query j, rows >= num_queries finite garbage)."""
        r = self.rows
        out = r.attend(self.q, r.seal(stack, partial, depth - 1), r.depth_mask[depth])
        if not split:
            return out
        return [out] if self.nq == 1 else [r.row(out, j) for j in range(self.nq)]


class MessageAttnRes:
    """Inkling2LoopMessageAttentionResidual for one whole pass. Keys = [input of L12 (block state), output of
    L17] of the previous pass, shared by all 12 loop layers: one [48, C] query matmul per pass."""

    def __init__(self, rows: RowOps, weights, name="msg_q"):
        q = _torch(weights[name]).float().reshape(NUM_LOOP_LAYERS, TILE, -1)[:, :4]  # [12, 4, C]
        c = q.shape[-1]
        self.rows = rows
        self.q = to_device(_pad_rows(q.reshape(-1, c), 64) * c**-0.5, rows.device)  # row 4l+j
        m = torch.full((64, TILE), -1e9)
        m[:, :2] = 0
        self.mask = to_device(m, rows.device)

    def messages_for_pass(self, h12_in, h17_out):
        """Token vectors of the previous pass -> [12][q, k, v, mlp] fp32 message token vectors."""
        r = self.rows
        s = r.seal(r.seal(None, h12_in, 0), h17_out, 1)
        out = r.attend(self.q, s, self.mask)  # [1,1,64,C], row 4l+j
        return [[r.row(out, 4 * l + j) for j in range(4)] for l in range(NUM_LOOP_LAYERS)]


class MessageGate:
    """Inkling2LoopMessageGate: (1 - tanh(F z)) * h + tanh(G z) * m with z = down(cat(rms h, rms m)).
    which = q, k, v or mlp."""

    def __init__(self, rows: RowOps, weights, which, eps=1e-6):
        d = rows.device
        p = f"msg_gate.{which}."
        w = {k: _torch(weights[p + k]) for k in ("down", "forget", "gain")}
        self.rows = rows
        self.eps = eps
        self.down = to_device(w["down"].reshape(w["down"].shape[-2:]).to(torch.bfloat16), d, ttnn.bfloat16)
        self.forget = to_device(w["forget"].reshape(w["forget"].shape[-2:]).float(), d)
        self.gain = to_device(w["gain"].reshape(w["gain"].shape[-2:]).float(), d)

    def __call__(self, h, m):
        """h, m: token vectors (fp32 or bf16). Row-wise, so rows other than 0 are computed too.
        Returns fp32."""
        r = self.rows
        h, m = _f32(h), _f32(m)
        nh = ttnn.rms_norm(h, epsilon=self.eps, compute_kernel_config=r.ckc)
        nm = ttnn.rms_norm(m, epsilon=self.eps, compute_kernel_config=r.ckc)
        x = ttnn.typecast(ttnn.concat([nh, nm], dim=-1), ttnn.bfloat16)  # HF casts to down_proj dtype
        z = ttnn.typecast(ttnn.matmul(x, self.down, dtype=ttnn.bfloat16, compute_kernel_config=r.ckc), ttnn.float32)
        f = ttnn.tanh(r.mm(z, self.forget))
        g = ttnn.tanh(r.mm(z, self.gain))
        keep = ttnn.subtract(h, ttnn.multiply(f, h, dtype=ttnn.float32), dtype=ttnn.float32)
        return ttnn.add(keep, ttnn.multiply(g, m, dtype=ttnn.float32), dtype=ttnn.float32)


class ShortConv:
    """Inkling2LoopShortConvolution (static): (conv_k4(x) + x) / sqrt(5) in fp32, one decode step.
    name = post_attn_conv or post_mlp_conv. The dynamic k/v conv (/sqrt(9)) lives in attention (T3)."""

    def __init__(self, rows: RowOps, weights, name):
        t = _torch(weights[name]).float().reshape(4, -1).clone()
        t[0] += 1  # fold the residual into the lag-0 tap
        t /= math.sqrt(5)
        self.rows = rows
        self.c = t.shape[-1]
        self.taps = to_device(_pad_rows(t), rows.device)

    def alloc_state(self, c):
        """Zero fp32 state [1,1,32,C] for HF cache index c (see cache_index). After a step, row r holds
        the input at lag r (r < 4)."""
        assert 0 <= c < NUM_CACHE_SETS, c
        return to_device(torch.zeros(TILE, self.c), self.rows.device)

    def __call__(self, x, state):
        """x: token vector. Updates state in place; returns the output token vector in x's dtype."""
        r = self.rows
        window = ttnn.add(r.mm(r.shift, state), r.mm(r.place[0], _f32(x)), dtype=ttnn.float32)
        y = r.mm(r.sum4, ttnn.multiply(window, self.taps, dtype=ttnn.float32))
        ttnn.copy(window, state)
        return y if x.dtype == ttnn.float32 else ttnn.typecast(y, x.dtype)
