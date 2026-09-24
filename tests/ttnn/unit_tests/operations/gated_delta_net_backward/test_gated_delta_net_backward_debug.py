# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug harness for gated_delta_net_backward.  DO NOT DELETE.

The op's real debugging lever is not DEVICE_PRINT: every intermediate the
device produces (Tinv, kcd, P, the decay vectors, v_corr, u, c, S, dS, v_new,
dv_new) is written to a real DRAM scratch tensor, so it can simply be read back
with `ttnn.to_torch` and compared block by block against a host mirror of the
same schedule.  `device_algorithm` below IS that mirror — it follows the
kernels' block order exactly (chunked, Neumann-inverted, three stages) and is
itself pinned against the golden float64 oracle by
`test_device_algorithm_matches_oracle`.

So a numerical failure localizes in one run: whichever scratch block diverges
first is the phase that is wrong.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn

import sys

from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

# The package re-exports the function under the module's own name, so reach the
# module (which owns the `_LAST_SCRATCH` debug hook) through sys.modules.
gdn = sys.modules["ttnn.operations.gated_delta_net_backward.gated_delta_net_backward"]

from eval.golden_tests.gated_delta_net_backward.helpers import pytorch_gated_delta_net_backward


# ---------------------------------------------------------------------------
# Host mirror of the device schedule
# ---------------------------------------------------------------------------


def device_algorithm(q, k, v, g, beta, do, dht=None, initial_state=None, chunk_size=64, scale=None):
    """Exactly the block schedule the kernels run, in float64 torch."""
    dt = torch.float64
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    if scale is None:
        scale = K**-0.5
    NC = (T + C - 1) // C
    L = NC * C
    pad = L - T

    def chunkify(x, last):
        x = x.transpose(1, 2).contiguous().to(dt)
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return x.reshape(B, H, NC, C, last)

    def chunkify_vec(x):
        x = x.transpose(1, 2).contiguous().to(dt)
        if pad:
            x = F.pad(x, (0, pad))
        return x.reshape(B, H, NC, C)

    qc = chunkify(q, K) * scale
    kc = chunkify(k, K)
    vc = chunkify(v, V)
    doc = chunkify(do, V)
    gc = chunkify_vec(g)
    bc = chunkify_vec(beta)

    eye = torch.eye(C, dtype=dt)
    lt_ones = torch.tril(torch.ones(C, C, dtype=dt))
    strict_tril = torch.tril(torch.ones(C, C, dtype=dt), -1)
    ut_ones = torch.triu(torch.ones(C, C, dtype=dt))
    sut_ones = torch.triu(torch.ones(C, C, dtype=dt), 1)
    m = math.ceil(math.log2(C))

    P_ = {}
    for b in range(B):
        for h in range(H):
            for i in range(NC):
                gi, bi = gc[b, h, i], bc[b, h, i]
                kb = kc[b, h, i] * bi[:, None]
                vb_ = vc[b, h, i] * bi[:, None]
                decay = lt_ones @ gi
                rmg = sut_ones @ gi
                gam = decay.exp()
                w = rmg.exp()
                dc1 = decay + rmg
                Gam = dc1[0].exp()
                X = decay[:, None].expand(C, C)
                Y = X.T
                Lm = (X - Y + (lt_ones - 1.0) * 1e4).exp()
                A = -((kb @ kc[b, h, i].T) * Lm) * strict_tril
                Pw = A.clone()
                Tinv = eye + A
                for _ in range(1, m):
                    Pw = Pw @ Pw
                    Tinv = Tinv + Tinv @ Pw
                U = kb * gam[:, None]
                kcd = Tinv @ U
                Q = qc[b, h, i] * gam[:, None]
                Pm = kc[b, h, i] * w[:, None]
                intra = (qc[b, h, i] @ kc[b, h, i].T) * Lm
                u = intra.T @ doc[b, h, i]
                c = Q.T @ doc[b, h, i]
                v_corr = Tinv @ vb_
                P_[(b, h, i)] = dict(
                    Tinv=Tinv,
                    kcd=kcd,
                    vcorr=v_corr,
                    Pm=Pm,
                    u=u,
                    c=c,
                    Lm=Lm,
                    decay=decay,
                    gam=gam,
                    Gam=Gam,
                    w=w,
                    kb=kb,
                    vb=vb_,
                    intra=intra,
                    dc1=dc1,
                    U=U,
                )

    S_store, dS_store, vnew_store, dvnew_store = {}, {}, {}, {}
    dh0 = None
    for b in range(B):
        for h in range(H):
            S = torch.zeros(K, V, dtype=dt) if initial_state is None else initial_state[b, h].to(dt)
            for i in range(NC):
                p = P_[(b, h, i)]
                S_store[(b, h, i)] = S.clone()
                vnew = p["vcorr"] - p["kcd"] @ S
                vnew_store[(b, h, i)] = vnew
                S = S * p["Gam"] + p["Pm"].T @ vnew
            dS = torch.zeros(K, V, dtype=dt) if dht is None else dht[b, h].to(dt)
            for i in range(NC - 1, -1, -1):
                p = P_[(b, h, i)]
                dS_store[(b, h, i)] = dS.clone()
                dvnew = p["u"] + p["Pm"] @ dS
                dvnew_store[(b, h, i)] = dvnew
                dS = p["Gam"] * dS + p["c"] - p["kcd"].T @ dvnew
            if initial_state is not None:
                if dh0 is None:
                    dh0 = torch.zeros(B, H, K, V, dtype=dt)
                dh0[b, h] = dS

    dq = torch.zeros(B, H, NC, C, K, dtype=dt)
    dk = torch.zeros(B, H, NC, C, K, dtype=dt)
    dv = torch.zeros(B, H, NC, C, V, dtype=dt)
    dg = torch.zeros(B, H, NC, C, dtype=dt)
    dbeta = torch.zeros(B, H, NC, C, dtype=dt)
    for b in range(B):
        for h in range(H):
            for i in range(NC):
                p = P_[(b, h, i)]
                Si, dSo = S_store[(b, h, i)], dS_store[(b, h, i)]
                vnew, dvnew = vnew_store[(b, h, i)], dvnew_store[(b, h, i)]
                Lm, gam, w, Gam = p["Lm"], p["gam"], p["w"], p["Gam"]
                qt, kt_, vt_ = qc[b, h, i], kc[b, h, i], vc[b, h, i]
                dot = doc[b, h, i]
                U = p["U"]
                dQ = dot @ Si.T
                M = (dot @ vnew.T) * Lm
                dP = vnew @ dSo.T
                ndkcd = dvnew @ Si.T
                dGam = (dSo * Si).sum()
                dvb = p["Tinv"].T @ dvnew
                ndU = p["Tinv"].T @ ndkcd
                d_attn = dvnew @ p["vb"].T - ndkcd @ U.T
                dAn = -((p["Tinv"].T @ d_attn @ p["Tinv"].T) * strict_tril)
                W = dAn * Lm
                dkb = W @ kt_ - ndU * gam[:, None]
                dq[b, h, i] = scale * (dQ * gam[:, None] + M @ kt_)
                dk[b, h, i] = M.T @ qt + dP * w[:, None] + W.T @ p["kb"] + dkb * bc[b, h, i][:, None]
                dv[b, h, i] = dvb * bc[b, h, i][:, None]
                dbeta[b, h, i] = (dvb * vt_).sum(-1) + (dkb * kt_).sum(-1)
                dgg = (dQ * qt).sum(-1) * gam - (ndU * U).sum(-1)
                dww = (dP * kt_).sum(-1) * w
                R = M * (qt @ kt_.T) + W * (p["kb"] @ kt_.T)
                e = R.sum(-1) - R.sum(-2) + dgg
                dg[b, h, i] = ut_ones @ e + sut_ones.T @ dww + dGam * Gam

    def unchunk(x, last):
        return x.reshape(B, H, L, last)[:, :, :T].transpose(1, 2).contiguous()

    def unchunk_vec(x):
        return x.reshape(B, H, L)[:, :, :T].transpose(1, 2).contiguous()

    return (
        (unchunk(dq, K), unchunk(dk, K), unchunk(dv, V), unchunk_vec(dg), unchunk_vec(dbeta), dh0),
        P_,
        dict(S=S_store, dS=dS_store, vnew=vnew_store, dvnew=dvnew_store),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SHAPE = (1, 32, 1, 32, 32)
CHUNK = 32


def make_inputs(shape, seed=0, g_scale=0.5):
    B, T, H, K, V = shape
    torch.manual_seed(seed)

    def l2(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    return dict(
        q=l2(torch.randn(B, T, H, K, dtype=torch.float64)),
        k=l2(torch.randn(B, T, H, K, dtype=torch.float64)),
        v=torch.randn(B, T, H, V, dtype=torch.float64),
        g=F.logsigmoid(torch.randn(B, T, H, dtype=torch.float64)) * g_scale,
        beta=torch.rand(B, T, H, dtype=torch.float64),
        do=torch.randn(B, T, H, V, dtype=torch.float64),
    )


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    if not a.any() and not b.any():
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def read_block(flat, base_tile, rows, cols):
    """Reconstruct a [rows*32, cols*32] block from the flat [N*32, 32] scratch."""
    out = torch.zeros(rows * 32, cols * 32, dtype=torch.float64)
    for r in range(rows):
        for c in range(cols):
            t = base_tile + r * cols + c
            out[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32] = flat[t * 32 : (t + 1) * 32, :].double()
    return out


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,chunk",
    [((1, 32, 1, 32, 32), 32), ((1, 128, 2, 64, 64), 32), ((1, 100, 2, 64, 64), 64)],
    ids=["tiny", "multi", "ragged"],
)
def test_device_algorithm_matches_oracle(shape, chunk):
    """The host mirror must equal the float64 autograd oracle, or it cannot be
    used to localize a device bug."""
    ref = make_inputs(shape)
    expected = pytorch_gated_delta_net_backward(
        ref["q"], ref["k"], ref["v"], ref["g"], ref["beta"], ref["do"], chunk_size=chunk
    )
    got, _, _ = device_algorithm(ref["q"], ref["k"], ref["v"], ref["g"], ref["beta"], ref["do"], chunk_size=chunk)
    for name, a, e in zip(("dq", "dk", "dv", "dg", "dbeta"), got, expected):
        assert pcc(a, e) > 0.99999999, f"{name}: host mirror diverges from the oracle (PCC {pcc(a, e)})"


def test_scratch_blocks(device):
    """Compare every prep / scan scratch block the device wrote against the
    host mirror, in schedule order.  The first miss is the broken phase."""
    shape, chunk = SHAPE, CHUNK
    B, T, H, K, V = shape
    ref = make_inputs(shape)

    def dev(t):
        return ttnn.from_torch(
            t.to(torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    gated_delta_net_backward(
        dev(ref["q"]),
        dev(ref["k"]),
        dev(ref["v"]),
        dev(ref["g"]),
        dev(ref["beta"]),
        dev(ref["do"]),
        chunk_size=chunk,
    )
    sc = gdn._LAST_SCRATCH
    flat = ttnn.to_torch(sc["sc"])
    smap, geo, Vb = sc["map"], sc["geo"], sc["Vb"]
    Ct, Kt, Vt, NC = geo.Ct, geo.Kt, geo.Vt, geo.NC

    _, P_, scan = device_algorithm(ref["q"], ref["k"], ref["v"], ref["g"], ref["beta"], ref["do"], chunk_size=chunk)

    misses = []
    for bh in range(B * H):
        b, h = bh // H, bh % H
        for i in range(NC):
            it = bh * NC + i
            p = P_[(b, h, i)]
            checks = [
                ("Tinv", read_block(flat, smap.base_attn + it * smap.st_attn, Ct, Ct), p["Tinv"]),
                ("kcd", read_block(flat, smap.base_kcd + it * smap.st_kcd, Ct, Kt), p["kcd"]),
                ("P", read_block(flat, smap.base_p + it * smap.st_p, Ct, Kt), p["Pm"]),
                ("vcorr", read_block(flat, smap.base_vcorr + it * smap.st_vcorr, Ct, Vt), p["vcorr"]),
                ("u", read_block(flat, smap.base_u + it * smap.st_u, Ct, Vt), p["u"]),
                ("c", read_block(flat, smap.base_c + it * smap.st_c, Kt, Vt), p["c"]),
                ("S", read_block(flat, smap.base_s + it * smap.st_s, Kt, Vt), scan["S"][(b, h, i)]),
                ("dS", read_block(flat, smap.base_ds + it * smap.st_ds, Kt, Vt), scan["dS"][(b, h, i)]),
                ("vnew", read_block(flat, smap.base_vnew + it * smap.st_vnew, Ct, Vt), scan["vnew"][(b, h, i)]),
                ("dvnew", read_block(flat, smap.base_dvnew + it * smap.st_dvnew, Ct, Vt), scan["dvnew"][(b, h, i)]),
            ]
            vecs = read_block(flat, smap.base_vec + it * smap.st_vec, 4, 1)
            checks += [
                ("decay", vecs[0 * 32 * Ct : 1 * 32 * Ct, 0:1], p["decay"][:, None]),
                ("beta_vec", vecs[1 * 32 * Ct : 2 * 32 * Ct, 0:1], None),
                ("dc1", vecs[2 * 32 * Ct : 3 * 32 * Ct, 0:1], p["dc1"][:, None]),
                ("w", vecs[3 * 32 * Ct : 4 * 32 * Ct, 0:1], p["w"][:, None]),
            ]
            for name, got_blk, exp_blk in checks:
                if exp_blk is None:
                    continue
                g2 = got_blk[: exp_blk.shape[0], : exp_blk.shape[1]]
                md = float((g2 - exp_blk).abs().max())
                mag = max(1.0, float(exp_blk.abs().max()))
                # Scale-relative, and no PCC on the decay columns: `decay` is a
                # cumulative sum and `dc1` is a CONSTANT column, so PCC there
                # measures rounding noise rather than structure.  The band is the
                # device's tf32 storage floor for float32 CBs -- the pack format
                # for a Float32 CB is Tf32 (10 mantissa bits), which is the
                # dominant error term in dg and is documented in the op file.
                const_col = name in ("decay", "dc1")
                bad = md > 3e-3 * mag
                if not const_col:
                    bad = bad or pcc(g2, exp_blk) < 0.999
                if bad:
                    misses.append(
                        f"item({b},{h},{i}) {name}: PCC {pcc(g2, exp_blk):.6f} "
                        f"maxdiff {md:.4g} (|exp|max {mag:.4g})"
                    )
    print("\n".join(misses) if misses else "ALL SCRATCH BLOCKS MATCH")
    assert not misses, "first divergences:\n" + "\n".join(misses[:8])
