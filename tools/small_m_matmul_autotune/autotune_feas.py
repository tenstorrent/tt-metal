# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Feasibility + geometry mirror of the CURRENT C++ picker, re-derived from source 2026-08-03.

CRITICAL: this must track small_m_matmul_plan.hpp::compute_cb_sizes and _config.cpp::pick_plan EXACTLY.
The previous mirror was stale w.r.t. the c_2 wrap fix (it still used cb2 = 2*out_blk under reduce-scatter,
where the shipped code now uses 2*ceil(out_blk/Pk)). That staleness made the mirror reject configs the
picker accepts, which is how a heuristic validated on a restricted candidate set got deployed over a wider
one and regressed 4-34%. Enumeration ranges below are the picker's OWN loops, not a sweep's subset.
"""
cdiv = lambda a, b: -(-a // b)
rup = lambda a, b: cdiv(a, b) * b
TB, TF = 2048, 4096
kCb1Depth, kCb7Depth = 4, 2
kL1Budget = 1440 * 1024
kNumBanks, kMinCores, kMaxCores = 8, 16, 104


def nt_width_shard_feasible(Nt):
    return 7 * cdiv(Nt, 8) < Nt


def rscatter_selected(Pk, Kt, Mblk, N_sub):
    rs_T = Mblk * N_sub
    feasible = (Pk > 1) and (rs_T >= Pk)
    mc = cdiv(rs_T, Pk)
    return feasible and Pk >= 4 and N_sub >= 2 and (Kt <= 64 or (Pk <= 6 and mc >= 2))


def l1_bytes(Pk, Kt, Mblk, Kslice, nsb, kb):
    out = Mblk * nsb
    rs = rscatter_selected(Pk, Kt, Mblk, nsb)
    cb0 = Mblk * Kslice
    cb1 = kCb1Depth * kb * nsb
    cb3 = out  # fp32
    cb7 = kCb7Depth * out if (Pk > 1 and not rs) else 0
    cb2 = 2 * cdiv(out, Pk) if rs else 2 * out  # <-- the c_2 fix
    rs_slice = cdiv(out, Pk) if rs else 0
    cb8 = cb9 = 2 * rs_slice
    bf16 = cb0 + cb1 + cb2 + cb7 + cb8 + cb9
    return bf16 * TB + cb3 * TF


def geo(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb):
    """None if infeasible, else the geometry dict the cost models consume."""
    if not nt_width_shard_feasible(Nt):
        return None
    cores = kNumBanks * Pk * Ns * Sm
    if cores < kMinCores or cores > kMaxCores:
        return None
    Ktl = rup(cdiv(Kt, Pk), kb * kNumBanks)
    wasteK = Pk * Ktl / Kt - 1.0
    if wasteK > 0.20:
        return None
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, kNumBanks)
    Nown = cdiv(Nband, Ns)
    if nsb > Nown:
        return None
    Nbpc = cdiv(Nown, nsb)
    wasteN = kNumBanks * Ns * Nbpc * nsb / Nt - 1.0
    if wasteN > 0.20:
        return None
    if l1_bytes(Pk, Kt, Mblk, Ktl, nsb, kb) > kL1Budget:
        return None
    return dict(
        cores=cores,
        Ktl=Ktl,
        Mblk=Mblk,
        Nown=Nown,
        Nbpc=Nbpc,
        wasteK=wasteK,
        wasteN=wasteN,
        rs=1 if rscatter_selected(Pk, Kt, Mblk, nsb) else 0,
        out_tiles=Mblk * Nown,
        sub_area=Mblk * nsb,
    )


def enumerate_full(Mt, Kt, Nt):
    """EXACTLY the picker's enumeration: Pk 1..12, Ns 1..6, Sm 1..Mt, kb {1,2,4,8}, nsb 1..Nown."""
    Nband = cdiv(Nt, kNumBanks)
    out = []
    for Pk in range(1, 13):
        for Ns in range(1, 7):
            Nown = cdiv(Nband, Ns)
            for Sm in range(1, Mt + 1):
                for kb in (1, 2, 4, 8):
                    for nsb in range(1, Nown + 1):
                        g = geo(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb)
                        if g:
                            out.append(((Pk, Ns, Sm, kb, nsb), g))
    return out
