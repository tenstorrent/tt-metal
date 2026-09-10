# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked (WY-form) Kimi Delta Attention prefill recurrence in plain fp32 ttnn ops.

Replaces ``ttnn.experimental.kda`` recurrent_chunk_scan for this model: that kernel's face-blocked polynomial inverse and
exp(G_i)*exp(-G_j) factorisation overflow to inf on Kimi-Linear-48B's gate/key statistics (layer 25 within the first 256
tokens, every layer once the chunk-cumulative decay approaches -160). This path (a) keeps every decay factor in fp32,
(b) centres the per-chunk cumulative gate so all exp() arguments stay within +-C*|g_min|/2 (<= 40 with C=32 and the -2.5
clamp), (c) solves (I + A) X = B by block forward substitution instead of materialising the inverse (see
_solve_unit_lower), and (d) forms
the cumulative gate with ttnn.cumsum (fp32-exact), never a matmul (Tensix matmuls round fp32 inputs to ~bf16; that error
compounds through every later chunk, everything else is per-chunk noise). Torch twin
with the derivation: ``reference/kda_chunked_ref.py`` (matches the exact recurrence to 1e-7 on CPU).

Per token: q,k <- l2norm; q *= K**-0.5; S <- S*exp(g_t)[:,None]; delta = beta*(v - k^T S); S += k (x) delta; o = q^T S.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
_DEBUG = __import__("os").environ.get("KIMI_CHUNKED_DEBUG") == "1"


def _dbg(n, **ts):
    parts = []
    for name, t in ts.items():
        if t is None:
            continue
        h = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        fin = h[torch.isfinite(h)]
        parts.append(
            f"{name}:max={fin.abs().max().item() if fin.numel() else float('nan'):.3e}"
            + (f",nonfinite={int((~torch.isfinite(h)).sum())}" if not torch.isfinite(h).all() else "")
        )
    print(f"[chunk {n}] " + " ".join(parts), flush=True)


@dataclass
class ChunkedKDAConstants:
    """Per-(mesh, heads, chunk) constants: causal masks and identity replicated over the TP-local heads."""

    lower_incl: ttnn.Tensor  # [1,H,C,C] j <= i
    lower_strict: ttnn.Tensor  # [1,H,C,C] j < i
    eye: ttnn.Tensor  # [1,H,C,C]
    blockdiag: ttnn.Tensor  # [1,H,C,C] 1 inside the diagonal block x block squares (strictly lower part is what matters)
    compute: ttnn.DeviceComputeKernelConfig
    chunk: int
    heads: int
    block: int  # forward-substitution block: (I+A) is inverted exactly only inside these blocks


def make_constants(mesh_device, heads: int, chunk: int = 32, block: int = 4) -> ChunkedKDAConstants:
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None

    def dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, chunk, chunk).repeat(1, heads, 1, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
            mesh_mapper=mapper,
        )

    ones = torch.ones(chunk, chunk)
    bd = torch.zeros(chunk, chunk)
    for b0 in range(0, chunk, block):
        bd[b0 : b0 + block, b0 : b0 + block] = 1.0
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return ChunkedKDAConstants(
        lower_incl=dev(torch.tril(ones)),
        lower_strict=dev(torch.tril(ones, diagonal=-1)),
        eye=dev(torch.eye(chunk)),
        blockdiag=dev(bd),
        compute=compute,
        chunk=chunk,
        heads=heads,
        block=block,
    )


def _heads_major(x: ttnn.Tensor, heads: int, dtype=ttnn.float32) -> ttnn.Tensor:
    """[1,T,H*D] -> [1,H,T,D] (fp32)."""
    T = x.shape[1]
    D = x.shape[-1] // heads
    if x.dtype != dtype:
        x = ttnn.typecast(x, dtype, memory_config=DRAM)
    x = ttnn.reshape(x, (1, T, heads, D))
    return ttnn.permute(x, (0, 2, 1, 3), memory_config=DRAM)


def _l2norm(x: ttnn.Tensor, extra_scale: float) -> ttnn.Tensor:
    """x / ||x|| along the last dim, times ``extra_scale``: rms_norm(x) = x*sqrt(D)/||x||."""
    D = x.shape[-1]
    y = ttnn.rms_norm(x, epsilon=1e-6 / D, memory_config=DRAM)  # eps on the mean matches sum-eps 1e-6
    return ttnn.multiply(y, extra_scale / (D**0.5), memory_config=DRAM)


def _mm(a, b, compute):
    return ttnn.matmul(a, b, memory_config=DRAM, dtype=ttnn.float32, compute_kernel_config=compute)


def _neumann_inverse(A: ttnn.Tensor, k: ChunkedKDAConstants, nilpotency: int) -> ttnn.Tensor:
    """(I + A)^-1 for strictly lower-triangular A [1,H,C,C] with A^nilpotency = 0: prod_j (I + (-A)^(2^j)), exact."""
    N = ttnn.neg(A, memory_config=DRAM)
    P = ttnn.add(k.eye, N, memory_config=DRAM)
    steps = 1
    while steps < nilpotency:
        N2 = _mm(N, N, k.compute)
        ttnn.deallocate(N)
        N = N2
        F = ttnn.add(k.eye, N, memory_config=DRAM)
        P2 = _mm(P, F, k.compute)
        ttnn.deallocate(P)
        ttnn.deallocate(F)
        P = P2
        steps *= 2
    ttnn.deallocate(N)
    return P


def _solve_unit_lower(A: ttnn.Tensor, B: ttnn.Tensor, k: ChunkedKDAConstants) -> ttnn.Tensor:
    """X = (I + A)^-1 B for strictly lower-triangular A [1,H,C,C], B [1,H,C,R], by BLOCK forward substitution.

    Materialising the full inverse is numerically wrong here: with beta ~ 1 and anti-aligned keys its entries grow like
    2^(C-1), and Tensix matmuls round fp32 inputs to ~bf16, so T@B cancels 1e8-scale terms to O(1) garbage (layers 8 and
    25 of Kimi-Linear-48B blow up to inf within 256 tokens). Forward substitution only ever multiplies O(1) solved rows by
    |A_ij| <= 1 -- the same arithmetic the exact per-token recurrence performs. Blocks of ``k.block`` rows are inverted
    exactly (Neumann, amplification <= 2^(block-1) = 8) and the blocks are chained: X <- T_bd (B - A_off X), where after
    iteration m the first m+1 blocks are final and later blocks are provisional but unused."""
    A_bd = ttnn.multiply(A, k.blockdiag, memory_config=DRAM)
    A_off = ttnn.subtract(A, A_bd, memory_config=DRAM)
    T_bd = _neumann_inverse(A_bd, k, k.block)
    ttnn.deallocate(A_bd)
    X = _mm(T_bd, B, k.compute)
    for _ in range(k.chunk // k.block - 1):
        R = ttnn.subtract(B, _mm(A_off, X, k.compute), memory_config=DRAM)
        ttnn.deallocate(X)
        X = _mm(T_bd, R, k.compute)
        ttnn.deallocate(R)
    ttnn.deallocate(A_off)
    ttnn.deallocate(T_bd)
    return X


def chunked_kda_prefill_ttnn(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    gate: ttnn.Tensor,
    beta: ttnn.Tensor,
    state: ttnn.Tensor,
    consts: ChunkedKDAConstants,
    *,
    scale: float | None = None,
    output_dtype=ttnn.bfloat16,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """q,k [1,T,H*K] bf16; v [1,T,H*V] bf16; gate [1,T,H*K] fp32 (log decay, clamped so that C*|g| <= ~80); beta [1,T,H] fp32;
    state [1,H,K,V] fp32 (read only). Returns (new_state [1,H,K,V] fp32, output [H,T,V] ``output_dtype``) -- the layout
    ``ttnn.experimental.kda.sigmoid_gated_rms_norm`` consumes."""
    H, C = consts.heads, consts.chunk
    T = q.shape[1]
    assert T % C == 0, (T, C)
    K = q.shape[-1] // H
    V = v.shape[-1] // H
    scale = K**-0.5 if scale is None else scale
    cc = consts.compute

    qh = _l2norm(_heads_major(q, H), scale)  # [1,H,T,K]
    kh = _l2norm(_heads_major(k, H), 1.0)
    vh = _heads_major(v, H)  # [1,H,T,V]
    gh = _heads_major(gate, H)  # [1,H,T,K]
    bh = ttnn.permute(ttnn.reshape(beta, (1, T, H, 1)), (0, 2, 1, 3), memory_config=DRAM)  # [1,H,T,1]
    S = state if state.dtype == ttnn.float32 else ttnn.typecast(state, ttnn.float32)
    S_owned = False
    outs = []
    for n in range(T // C):
        s0, s1 = n * C, (n + 1) * C
        qc = ttnn.slice(qh, (0, 0, s0, 0), (1, H, s1, K), memory_config=DRAM)
        kc = ttnn.slice(kh, (0, 0, s0, 0), (1, H, s1, K), memory_config=DRAM)
        vc = ttnn.slice(vh, (0, 0, s0, 0), (1, H, s1, V), memory_config=DRAM)
        gc = ttnn.slice(gh, (0, 0, s0, 0), (1, H, s1, K), memory_config=DRAM)
        bc = ttnn.slice(bh, (0, 0, s0, 0), (1, H, s1, 1), memory_config=DRAM)
        # inclusive cumsum over the chunk [1,H,C,K]. MUST be ttnn.cumsum: Tensix matmuls round fp32 inputs to ~bf16, and an
        # error in the cumulative decay compounds across every later chunk (matmul-cumsum: 4.7e-2 abs; cumsum: 1e-5).
        G = ttnn.cumsum(gc, dim=2, dtype=ttnn.float32, memory_config=DRAM)
        # last row of the (non-increasing) cumsum = its minimum over the chunk rows -> no tile-unaligned slice needed
        c = ttnn.multiply(ttnn.min(G, dim=2, keepdim=True, memory_config=DRAM), 0.5, memory_config=DRAM)  # [1,H,1,K]
        Gc = ttnn.subtract(G, c, memory_config=DRAM)
        ttnn.deallocate(G)
        eGc = ttnn.exp(Gc, memory_config=DRAM)
        enGc = ttnn.exp(ttnn.neg(Gc, memory_config=DRAM), memory_config=DRAM)
        ttnn.deallocate(Gc)
        Qt = ttnn.multiply(qc, eGc, memory_config=DRAM)
        Kl = ttnn.multiply(kc, eGc, memory_config=DRAM)
        Kr = ttnn.multiply(kc, enGc, memory_config=DRAM)
        ttnn.deallocate(eGc)
        ttnn.deallocate(enGc)
        bKl = ttnn.multiply(Kl, bc, memory_config=DRAM)
        ttnn.deallocate(Kl)
        KrT = ttnn.permute(Kr, (0, 1, 3, 2), memory_config=DRAM)  # [1,H,K,C]
        ttnn.deallocate(Kr)
        A = ttnn.multiply(_mm(bKl, KrT, cc), consts.lower_strict, memory_config=DRAM)  # [1,H,C,C]
        bv = ttnn.multiply(vc, bc, memory_config=DRAM)
        rhs = ttnn.concat([bv, bKl], dim=3, memory_config=DRAM)  # [1,H,C,V+K]: solve both right-hand sides at once
        ttnn.deallocate(bv)
        ttnn.deallocate(bKl)
        X = _solve_unit_lower(A, rhs, consts)
        if _DEBUG:
            _dbg(n, A=A, X=X, KrT=KrT, Qt=Qt, S=S)
        ttnn.deallocate(A)
        ttnn.deallocate(rhs)
        U = ttnn.slice(X, (0, 0, 0, 0), (1, H, C, V), memory_config=DRAM)  # [1,H,C,V]
        W = ttnn.slice(X, (0, 0, 0, V), (1, H, C, V + K), memory_config=DRAM)  # [1,H,C,K]
        ttnn.deallocate(X)
        ec = ttnn.permute(ttnn.exp(c, memory_config=DRAM), (0, 1, 3, 2), memory_config=DRAM)  # [1,H,K,1]
        ttnn.deallocate(c)
        S0c = ttnn.multiply(S, ec, memory_config=DRAM)  # rows scaled by exp(c)
        WS = _mm(W, S0c, cc)
        Delta = ttnn.subtract(U, WS, memory_config=DRAM)  # [1,H,C,V]
        if _DEBUG:
            _dbg(n, U=U, W=W, WS=WS, S0c=S0c, ec=ec, Delta=Delta)
        ttnn.deallocate(WS)
        ttnn.deallocate(U)
        ttnn.deallocate(W)
        Aqk = ttnn.multiply(_mm(Qt, KrT, cc), consts.lower_incl, memory_config=DRAM)
        O = ttnn.add(_mm(Qt, S0c, cc), _mm(Aqk, Delta, cc), memory_config=DRAM)  # [1,H,C,V]
        ttnn.deallocate(Aqk)
        ttnn.deallocate(Qt)
        ttnn.deallocate(S0c)
        outs.append(O)
        # S1 = exp(2c) * S + exp(c) * (Kr^T @ Delta)
        upd = ttnn.multiply(_mm(KrT, Delta, cc), ec, memory_config=DRAM)
        ec2 = ttnn.multiply(ec, ec, memory_config=DRAM)
        S_new = ttnn.add(ttnn.multiply(S, ec2, memory_config=DRAM), upd, memory_config=DRAM)
        if _DEBUG:
            _dbg(n, O=O, upd=upd, S_new=S_new)
        for t_ in (upd, ec2, ec, KrT, Delta, qc, kc, vc, gc, bc):
            ttnn.deallocate(t_)
        if S_owned:
            ttnn.deallocate(S)
        S, S_owned = S_new, True
    for t_ in (qh, kh, vh, gh, bh):
        ttnn.deallocate(t_)
    out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=DRAM)  # [1,H,T,V]
    if len(outs) > 1:
        for o in outs:
            ttnn.deallocate(o)
    if out.dtype != output_dtype:
        out_c = ttnn.typecast(out, output_dtype, memory_config=DRAM)
        ttnn.deallocate(out)
        out = out_c
    out = ttnn.reshape(out, (H, T, V))
    if not S_owned:  # T == 0 cannot happen (T % C == 0 and T >= C), kept for symmetry
        S = ttnn.clone(S, memory_config=DRAM)
    return S, out
