# SPDX-License-Identifier: Apache-2.0
"""Phase-1 de-risk: chunked-parallel gated delta-rule prefill, validated on CPU
against the EXACT sequential recurrence the current TT prefill kernel computes
(deltanet_prefill_compute.cpp):

    S_mid = decay * S_prev          # per-token scalar decay, broadcast over [Dk,Dv]
    mem   = k @ S_mid               # [Dv]   (k:[Dk], S_mid:[Dk,Dv])
    delta = beta * (v - mem)        # [Dv]
    S     = S_mid + outer(k, delta) # [Dk,Dv] rank-1 update
    o     = q @ S                   # [Dv]   (uses the UPDATED state)

Chunked form (per head, chunk length C, entering state S0; d_i = prod_{0..i} decay):
    A_{jm} = beta_j (d_j/d_m)(k_j . k_m)         for m<j  (strictly lower-tri)
    U      = (I + A)^{-1} [ beta_j (v_j - d_j (k_j^T S0)) ]
    O_i    = d_i (q_i^T S0) + sum_{j<=i} (d_i/d_j)(q_i . k_j) u_j
    S_new  = d_{C-1} S0 + sum_j (d_{C-1}/d_j) k_j u_j^T

Run:  /opt/venv/bin/python3 models/demos/qwen36_27b/t3k/chunk_delta_ref.py
"""
import torch

torch.manual_seed(0)


def seq_rec(q, k, v, beta, decay, S0):
    """Ground-truth sequential recurrence (matches the TT prefill kernel)."""
    S = S0.clone()
    O = []
    for i in range(q.shape[0]):
        Smid = decay[i] * S
        mem = k[i] @ Smid                       # [Dv]
        delta = beta[i] * (v[i] - mem)          # [Dv]
        S = Smid + torch.outer(k[i], delta)     # [Dk,Dv]
        O.append(q[i] @ S)                      # uses UPDATED state
    return torch.stack(O), S


def chunk_rec(q, k, v, beta, decay, S0, C):
    """Chunked-parallel equivalent. Matmul-heavy; S/C sequential chunk steps."""
    Stot, Dk = q.shape
    Dv = v.shape[1]
    S = S0.clone()
    O = torch.zeros(Stot, Dv, dtype=q.dtype)
    eye = torch.eye(C, dtype=q.dtype)
    for c0 in range(0, Stot, C):
        c1 = min(c0 + C, Stot)
        n = c1 - c0
        qc, kc, vc = q[c0:c1], k[c0:c1], v[c0:c1]
        bc, gc = beta[c0:c1], decay[c0:c1]
        d = torch.cumprod(gc, 0)                       # d_i = prod_{0..i} decay  [n]
        rj_m = d[:, None] / d[None, :]                 # d_j / d_m
        KK = kc @ kc.T                                 # [n,n]
        A = torch.tril(bc[:, None] * rj_m * KK, -1)    # strictly lower-tri
        RHS = bc[:, None] * (vc - d[:, None] * (kc @ S))   # [n,Dv]
        U = torch.linalg.solve(eye[:n, :n] + A, RHS)   # [n,Dv]
        M = torch.tril(rj_m * (qc @ kc.T), 0)          # inclusive lower-tri
        O[c0:c1] = d[:, None] * (qc @ S) + M @ U
        coef = d[-1] / d                               # d_{n-1}/d_j  [n]
        S = d[-1] * S + (kc * coef[:, None]).T @ U     # [Dk,Dv]
    return O, S


def inv_unitri_neumann(L):
    """(I+N)^-1 for N strictly-lower nilpotent (N^n=0), via repeated-squaring
    Neumann product: (I-N)(I+N^2)(I+N^4)... — EXACT, ~log2(n) squarings.
    This is the tile-level recipe the chunked kernel will use for the 32x32 inverse."""
    n = L.shape[0]
    I = torch.eye(n, dtype=L.dtype)
    N = L - I                       # strictly lower-tri, nilpotent
    inv = I - N                     # (I - N)
    P = N                           # P = N^(2^k); start k=0 -> we already used (I-N), next factor (I+N^2)
    k = 1
    while (1 << k) < n:
        P = P @ P                   # N^(2^k)
        inv = inv @ (I + P)
        k += 1
    return inv


def chunk_rec_tiled_inv(q, k, v, beta, decay, S0, C=32):
    """Same as chunk_rec but the (I+A)^-1 uses the Neumann-product inverse the
    kernel will use (validates the on-tile inverse recipe). C=32 -> single tile A."""
    Stot, Dk = q.shape
    Dv = v.shape[1]
    S = S0.clone()
    O = torch.zeros(Stot, Dv, dtype=q.dtype)
    for c0 in range(0, Stot, C):
        c1 = min(c0 + C, Stot)
        n = c1 - c0
        qc, kc, vc = q[c0:c1], k[c0:c1], v[c0:c1]
        bc, gc = beta[c0:c1], decay[c0:c1]
        d = torch.cumprod(gc, 0)
        rj_m = d[:, None] / d[None, :]
        A = torch.tril(bc[:, None] * rj_m * (kc @ kc.T), -1)
        IA = torch.eye(n, dtype=q.dtype) + A
        U = inv_unitri_neumann(IA) @ (bc[:, None] * (vc - d[:, None] * (kc @ S)))
        M = torch.tril(rj_m * (qc @ kc.T), 0)
        O[c0:c1] = d[:, None] * (qc @ S) + M @ U
        coef = d[-1] / d
        S = d[-1] * S + (kc * coef[:, None]).T @ U
    return O, S


def chunk_rec_factored(q, k, v, beta, decay, S0, C=32, bf16=False):
    """Memory-safe form: fold per-token decay d (relative to chunk start) into the
    keys/queries so the C×C masks are CONSTANT (tril) — no per-chunk ratio tiles.
    A = tril_strict( (beta*d * k) @ (k/d)^T );  M = tril_incl( (d*q) @ (k/d)^T ).
    Optionally round the folded factors to bf16 to gauge on-device precision."""
    def b16(x):
        return x.bfloat16().float() if bf16 else x
    Stot, Dk = q.shape
    Dv = v.shape[1]
    S = S0.clone()
    O = torch.zeros(Stot, Dv, dtype=torch.float32)
    for c0 in range(0, Stot, C):
        c1 = min(c0 + C, Stot)
        n = c1 - c0
        qc, kc, vc = q[c0:c1], k[c0:c1], v[c0:c1]
        bc, gc = beta[c0:c1], decay[c0:c1]
        d = torch.cumprod(gc, 0)                # in (0,1], relative to chunk start
        dinv = 1.0 / d
        Kdec = b16(bc[:, None] * d[:, None] * kc)     # [n,Dk]
        Kinv = b16(dinv[:, None] * kc)                # [n,Dk]  (= (k/d))
        Qd = b16(d[:, None] * qc)
        A = torch.tril(b16(Kdec @ Kinv.T), -1)        # strict lower
        kS0 = kc @ S
        RHS = bc[:, None] * (vc - d[:, None] * kS0)
        U = torch.linalg.solve(torch.eye(n) + A, RHS)
        M = torch.tril(b16(Qd @ Kinv.T), 0)
        O[c0:c1] = d[:, None] * (qc @ S) + M @ U
        dlast = d[-1]
        S = dlast * S + dlast * (Kinv.T @ U)          # coef_j = dlast/d_j = dlast*dinv_j
    return O, S


def _b(x):  # round to bf16 (simulate every ttnn op output)
    return x.bfloat16().float()


def chunk_rec_devbf16(q, k, v, beta, decay, S0, C=32, form="factored"):
    """Device-faithful: round EVERY intermediate to bf16, as ttnn does.
    form='factored' uses (1/d)-amplified keys; form='ratio' uses bounded ratio
    matrices (raw O(1) keys). Confirms which survives device bf16."""
    Stot, Dk = q.shape; Dv = v.shape[1]
    S = S0.clone(); O = torch.zeros(Stot, Dv)
    for c0 in range(0, Stot, C):
        c1 = min(c0 + C, Stot); n = c1 - c0
        qc, kc, vc = q[c0:c1], k[c0:c1], v[c0:c1]
        bc, gc = beta[c0:c1], decay[c0:c1]
        d = torch.cumprod(gc, 0)
        kS0 = _b(_b(kc) @ _b(S)); qS0 = _b(_b(qc) @ _b(S))
        if form == "factored":
            Kd = _b((bc * d)[:, None] * kc); Ki = _b((1.0 / d)[:, None] * kc); Qd = _b(d[:, None] * qc)
            A = torch.tril(_b(Kd @ Ki.T), -1)
            Mfull = torch.tril(_b(Qd @ Ki.T), 0)
            KiT_U_coef = Ki                                  # state uses Ki, dlast factor
            statefac = d[-1]
        else:  # ratio
            rj = torch.tril(d[:, None] / d[None, :], 0)      # bounded <=1
            A = torch.tril(_b(bc[:, None] * rj * _b(_b(kc) @ _b(kc).T)), -1)
            Mfull = torch.tril(_b(rj * _b(_b(qc) @ _b(kc).T)), 0)
            coef = d[-1] / d                                 # bounded <=1
            KiT_U_coef = _b(coef[:, None] * kc); statefac = d[-1]
        inv = torch.eye(n)
        P = A
        invm = torch.eye(n) - A
        for _ in range(4):
            P = _b(P @ P); invm = _b(invm @ (torch.eye(n) + P))
        rhs = _b(bc[:, None] * (vc - _b(d[:, None] * kS0)))
        U = _b(invm @ rhs)
        O[c0:c1] = _b(_b(d[:, None] * qS0) + _b(Mfull @ U))
        S = _b(statefac * _b(S + _b(KiT_U_coef.T @ U)))
    return O, S


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


if __name__ == "__main__":
    Dk, Dv = 128, 128
    # realistic gates: decay in (0,1) (exp(-exp(A_log)*softplus(.))), beta in (0,1) (sigmoid)
    for S in [251, 1024, 4096]:
        q = torch.randn(S, Dk) / (Dk ** 0.5)        # q is 1/sqrt(Dk) scaled (kernel convention)
        k = torch.nn.functional.normalize(torch.randn(S, Dk), dim=-1)  # l2-normed keys
        v = torch.randn(S, Dv)
        beta = torch.sigmoid(torch.randn(S))
        decay = torch.exp(-torch.exp(torch.randn(S) * 0.5) * torch.nn.functional.softplus(torch.randn(S)))
        S0 = torch.randn(Dk, Dv) * 0.1
        O_ref, Sf_ref = seq_rec(q, k, v, beta, decay, S0)
        for C in [32, 64]:
            O_c, Sf_c = chunk_rec(q, k, v, beta, decay, S0, C)
            print(f"S={S:5d} C={C:3d} (linalg.solve): PCC(O)={pcc(O_ref, O_c):.8f}  PCC(Sfinal)={pcc(Sf_ref, Sf_c):.8f}  "
                  f"maxabs(O)={(O_ref-O_c).abs().max():.2e}")
        # kernel-faithful: Neumann-product 32x32 inverse, C=32
        O_t, Sf_t = chunk_rec_tiled_inv(q, k, v, beta, decay, S0, C=32)
        print(f"S={S:5d} C= 32 (Neumann tile-inv): PCC(O)={pcc(O_ref, O_t):.8f}  PCC(Sfinal)={pcc(Sf_ref, Sf_t):.8f}  "
              f"maxabs(O)={(O_ref-O_t).abs().max():.2e}")
        # factored (fold decay into keys; CONST masks; 256K-memory-safe), fp32 + bf16
        O_f, Sf_f = chunk_rec_factored(q, k, v, beta, decay, S0, C=32, bf16=False)
        O_b, Sf_b = chunk_rec_factored(q, k, v, beta, decay, S0, C=32, bf16=True)
        print(f"S={S:5d} C= 32 (factored fp32)   : PCC(O)={pcc(O_ref, O_f):.8f}  PCC(Sfinal)={pcc(Sf_ref, Sf_f):.8f}")
        print(f"S={S:5d} C= 32 (factored bf16)   : PCC(O)={pcc(O_ref, O_b):.8f}  PCC(Sfinal)={pcc(Sf_ref, Sf_b):.8f}  "
              f"maxabs(O)={(O_ref-O_b).abs().max():.2e}")
        # device-faithful bf16 (round EVERY intermediate): factored vs ratio
        Of, _ = chunk_rec_devbf16(q, k, v, beta, decay, S0, C=32, form="factored")
        Or, _ = chunk_rec_devbf16(q, k, v, beta, decay, S0, C=32, form="ratio")
        print(f"S={S:5d} C= 32 (DEVbf16 factored): PCC(O)={pcc(O_ref, Of):.8f}")
        print(f"S={S:5d} C= 32 (DEVbf16 ratio)   : PCC(O)={pcc(O_ref, Or):.8f}")
