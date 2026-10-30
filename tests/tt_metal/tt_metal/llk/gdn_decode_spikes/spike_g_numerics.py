#!/usr/bin/env python3
"""Spike G: host emulation of the device numerics of the GDN decode recurrence, for the forms v2 chooses between.

Device model (requirements S-6 item 5, and the prefill hi/lo probe): FPU operands are truncated to tf32-class
precision, asymmetric: in0 / SrcB keeps 10 explicit mantissa bits, in1 / SrcA keeps 9; products of such operands are
exact in fp32; accumulation is fp32 in k order (the real MVMUL's 16-term face sum holds ~12-13 bits, see the caveat in
the notes). SFPU arithmetic is fp32 with one rounding per op. copy_tile into DEST is exact for fp32 tiles
(unpack-to-dest). FPU eltwise binary ops (sub_tiles, mul_tiles_bcast) go through SrcA/SrcB and truncate like matmuls.

Forms, per token (per head), K = V = 128:
  direct    today's kernel: FPU decay pass (S truncated), k.S', FPU residual, SFPU beta, rank-1 on a copy_tile seed, q.S''
  two_row   tier B: [q;k].S_old in one pass, SFPU residual, SFPU decay in the update acquire + rank-1 accumulate,
            o = e.(q.S_old) + (q.k).u with the scalar on the FPU (two_row) or the SFPU (two_row_sfpu_dot)
  lazy      two_row with a running scale c on the stored state (S_true = c.S~), renormalised when c nears underflow;
            with the scale persisted across calls (lazy_persist) or folded into the state at every call (lazy_fold)
Outputs: PCC and max rel error vs an fp64 recurrence over 64 tokens; torch.equal between one T = 8 call and 8 chained
T = 1 calls for every form.
"""
import math
import sys

import torch

torch.manual_seed(0)
K = V = 128
HV = 12
SRCB, SRCA = 10, 9


def trunc(x, bits):
    """Keep `bits` explicit mantissa bits of an fp32 tensor by truncation (the FPU's operand read)."""
    xi = x.to(torch.float32).contiguous().view(torch.int32)
    mask = torch.tensor(-(1 << (23 - bits)), dtype=torch.int32)  # all ones above the kept bits
    return (xi & mask).view(torch.float32)


def mm_dev(a, b, a_bits=SRCB, b_bits=SRCA, seed=None):
    """a [H, m, k] x b [H, k, n] on the FPU: truncated operands, exact products, fp32 accumulation in k order,
    optionally onto a DEST seed (the copy_tile'd S)."""
    a_t, b_t = trunc(a, a_bits), trunc(b, b_bits)
    acc = torch.zeros(a.shape[0], a.shape[1], b.shape[2], dtype=torch.float32) if seed is None else seed.clone()
    for kk in range(a.shape[2]):
        acc = acc + a_t[:, :, kk : kk + 1] * b_t[:, kk : kk + 1, :]
    return acc


def f32(x):
    return x.to(torch.float32)


# ---- one token of each form -------------------------------------------------------------------------------------------
def step_ref64(S, q, k, v, g, beta):
    S, q, k, v = (t.to(torch.float64) for t in (S, q, k, v))
    e = torch.exp(g.to(torch.float64))[:, None, None]
    S = e * S
    vread = (k[:, None, :] @ S)[:, 0]
    u = beta.to(torch.float64)[:, None] * (v - vread)
    S = S + k[:, :, None] * u[:, None, :]
    o = (q[:, None, :] @ S)[:, 0]
    return S, o


def step_direct(S, q, k, v, g, beta):
    e = f32(torch.exp(g))[:, None, None]  # exp on the host today, fp32
    S1 = f32(trunc(S, SRCB) * trunc(e, SRCA))  # mul_tiles_bcast_scalar: S in0, decay in1
    vread = mm_dev(k[:, None, :], S1)[:, 0]
    diff = f32(trunc(v, SRCB) - trunc(vread, SRCA))  # sub_tiles on the FPU
    u = f32(diff * beta[:, None])  # sfpu_mul_bcast_col, fp32
    kcol = trunc(k, SRCA)  # transpose_tile moves k through SrcA
    S2 = mm_dev(kcol[:, :, None], u[:, None, :], seed=S1)  # copy_tile seed + inner-dim-1 matmul
    o = mm_dev(q[:, None, :], S2)[:, 0]
    return S2, o


def step_two_row(S, q, k, v, g, beta, sfpu_dot=False, s_bits=SRCA, u_bits=SRCA):
    e = f32(torch.exp(g))[:, None]  # SFPU exp, fp32
    qk = torch.stack([q, k], dim=1)  # [H, 2, K] as one 2x32 tile per K tile
    r = mm_dev(qk, S, b_bits=s_bits)  # [H, 2, V]
    rq, rk = r[:, 0], r[:, 1]
    vread = f32(e * rk)
    u = f32(beta[:, None] * f32(v - vread))  # SFPU: sub then mul, fp32 each
    S_dec = f32(e[:, :, None] * S)  # SFPU decay inside the update acquire, fp32
    S_new = mm_dev(k[:, :, None], u[:, None, :], b_bits=u_bits, seed=S_dec)  # rank-1 accumulate (kT in0, u in1)
    if sfpu_dot:
        qk_dot = f32((q.to(torch.float64) * k.to(torch.float64)).sum(-1)).to(torch.float32)  # exact-ish SFPU dot
    else:
        qk_dot = mm_dev(q[:, None, :], k[:, :, None])[:, 0, 0]  # 2x32 . 32x1 on the FPU
    o = f32(f32(e * rq) + f32(qk_dot[:, None] * u))
    return S_new, o


class Lazy:
    """two_row with a running scale: S_true = c * S_stored. The scale is a per-head fp32 scalar."""

    THRESH = 2.0**-60

    def __init__(self, S, c=None):
        self.S = S.clone()
        self.c = torch.ones(S.shape[0], dtype=torch.float32) if c is None else c.clone()

    def step(self, q, k, v, g, beta):
        e = f32(torch.exp(g))
        qk = torch.stack([q, k], dim=1)
        r = f32(self.c[:, None, None] * mm_dev(qk, self.S))  # scale the two rows, not S
        rq, rk = r[:, 0], r[:, 1]
        vread = f32(e[:, None] * rk)
        u = f32(beta[:, None] * f32(v - vread))
        c_new = f32(self.c * e)
        self.S = mm_dev(k[:, :, None], f32(u / c_new[:, None])[:, None, :], seed=self.S)
        self.c = c_new
        renorm = self.c < self.THRESH
        if renorm.any():
            self.S[renorm] = f32(self.c[renorm][:, None, None] * self.S[renorm])
            self.c[renorm] = 1.0
        qk_dot = mm_dev(q[:, None, :], k[:, :, None])[:, 0, 0]
        o = f32(f32(e[:, None] * rq) + f32(qk_dot[:, None] * u))
        return o

    def true_state(self):
        return f32(self.c[:, None, None] * self.S)


# ---- runners ----------------------------------------------------------------------------------------------------------
def make_inputs(T, g_lo, g_hi, seed):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32)
    q, k = r(T, HV, K), r(T, HV, K)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    q = q * K**-0.5
    v = r(T, HV, V)
    beta = torch.rand(T, HV, generator=gen)
    g = g_lo + (g_hi - g_lo) * torch.rand(T, HV, generator=gen)
    S0 = 0.05 * r(HV, K, V)
    return q, k, v, g, beta, S0


def pcc(a, b):
    a = a.double().flatten() - a.double().mean()
    b = b.double().flatten() - b.double().mean()
    return float(a @ b / (a.norm() * b.norm()))


def run_form(form, x, T, calls=1):
    """Run T tokens as `calls` chained calls of T/calls tokens each. Returns (o [T,H,V], final state true value,
    per-token metrics vs fp64)."""
    q, k, v, g, beta, S0 = x
    S_ref = S0.double()
    o_all, met = [], []
    per_call = T // calls
    state = S0.clone()
    lazy = None
    for c in range(calls):
        if form == "lazy_persist":
            lazy = lazy if lazy is not None else Lazy(state)
        elif form == "lazy_fold":
            lazy = Lazy(state)  # a fresh scale every call: the folded state is the only thing carried
        for t in range(c * per_call, (c + 1) * per_call):
            S_ref, o_ref = step_ref64(S_ref, q[t], k[t], v[t], g[t], beta[t])
            if form == "direct":
                state, o = step_direct(state, q[t], k[t], v[t], g[t], beta[t])
            elif form == "two_row":
                state, o = step_two_row(state, q[t], k[t], v[t], g[t], beta[t])
            elif form == "two_row_sfpu_dot":
                state, o = step_two_row(state, q[t], k[t], v[t], g[t], beta[t], sfpu_dot=True)
            elif form == "two_row_S10":
                state, o = step_two_row(state, q[t], k[t], v[t], g[t], beta[t], s_bits=10, u_bits=10)
            elif form in ("lazy_persist", "lazy_fold"):
                o = lazy.step(q[t], k[t], v[t], g[t], beta[t])
                state = lazy.true_state()
            else:
                raise ValueError(form)
            o_all.append(o)
            met.append(
                (
                    pcc(o_ref, o),
                    pcc(S_ref, state),
                    float((S_ref - state.double()).abs().max() / S_ref.abs().max()),
                )
            )
        if form == "lazy_fold":
            lazy = None  # the state leaves the call folded (c = 1)
    return torch.stack(o_all), state, met


def main():
    out = []
    P = lambda *a: (print(*a), out.append(" ".join(str(s) for s in a)))
    FORMS = ["direct", "two_row", "two_row_sfpu_dot", "two_row_S10", "lazy_persist", "lazy_fold"]
    for name, g_lo, g_hi in (("decay g in [-2, 0]", -2.0, 0.0), ("heavy decay g in [-6, -4]", -6.0, -4.0)):
        x = make_inputs(64, g_lo, g_hi, seed=7)
        P(f"\n== 64 tokens, HV {HV}, K = V = {K}, {name}: PCC and max rel error vs fp64 after 1 / 8 / 32 / 64 tokens")
        P(f"{'form':18} {'pcc o (1/8/32/64)':44} {'pcc S':44} {'max rel S (1/8/32/64)':40}")
        for form in FORMS:
            _, _, met = run_form(form, x, 64)
            pick = [met[i] for i in (0, 7, 31, 63)]
            P(
                f"{form:18} "
                + " ".join(f"{m[0]:.7f}" for m in pick).ljust(44)
                + " "
                + " ".join(f"{m[1]:.7f}" for m in pick).ljust(44)
                + " "
                + " ".join(f"{m[2]:.1e}" for m in pick)
            )
        P(f"\n-- call-shape identity, {name}: one T = 8 call vs 8 chained T = 1 calls (torch.equal on o and the state)")
        for form in FORMS:
            o8, s8, _ = run_form(form, x, 8, calls=1)
            o1, s1, _ = run_form(form, x, 8, calls=8)
            eq_o, eq_s = torch.equal(o8, o1), torch.equal(s8, s1)
            d = (s8 - s1).abs().max().item()
            P(f"{form:18} o equal: {str(eq_o):5}  state equal: {str(eq_s):5}  max |dS| {d:.2e}")
    # the scale's underflow horizon: how often lazy renormalises
    P(
        "\n-- lazy scale: tokens until c < 2^-60 at mean g of -1, -3, -5:",
        [round(60 * math.log(2) / m) for m in (1, 3, 5)],
    )
    return out


if __name__ == "__main__":
    lines = main()
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write("\n".join(lines) + "\n")
