"""Is dg's error made in stage G, or already present in the scratch stage G reads?

Recompute dg on the host from the DEVICE's own scratch blocks.  If that matches
the device's dg, the error is upstream (prep/scan); if it matches the reference,
stage G's arithmetic is where it is made.
"""
import sys

import torch
import ttnn

sys.path.insert(0, "/localdev/mnedeljkovic/tt-metal/tests/ttnn/unit_tests/operations/gated_delta_net_backward")
from test_gated_delta_net_backward_debug import device_algorithm, pcc, read_block

from eval.golden_tests.gated_delta_net_backward.helpers import make_reference_inputs
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

gdn = sys.modules["ttnn.operations.gated_delta_net_backward.gated_delta_net_backward"]


def rel_rms(got, exp):
    d = exp.double().pow(2).mean().sqrt()
    return float((got.double() - exp.double()).pow(2).mean().sqrt() / d)


def test_bisect(device):
    shape, chunk = (1, 128, 2, 64, 64), 32
    B, T, H, K, V = shape
    ref = make_reference_inputs(shape, state_mode="with_h0_and_dht", g_scale=8.0)
    exp, P_, scan = device_algorithm(
        ref["q"],
        ref["k"],
        ref["v"],
        ref["g"],
        ref["beta"],
        ref["do"],
        dht=ref["dht"],
        initial_state=ref["h0"],
        chunk_size=chunk,
    )

    def dev(t):
        return ttnn.from_torch(
            t.to(torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    got = gated_delta_net_backward(
        dev(ref["q"]),
        dev(ref["k"]),
        dev(ref["v"]),
        dev(ref["g"]),
        dev(ref["beta"]),
        dev(ref["do"]),
        dht=dev(ref["dht"]),
        initial_state=dev(ref["h0"]),
        chunk_size=chunk,
    )
    sc = gdn._LAST_SCRATCH
    flat = ttnn.to_torch(sc["sc"]).double()
    smap, geo = sc["map"], sc["geo"]
    Ct, Kt, Vt, NC, C = geo.Ct, geo.Kt, geo.Vt, geo.NC, chunk
    scale = K**-0.5

    lt = torch.tril(torch.ones(C, C, dtype=torch.float64))
    st = torch.tril(torch.ones(C, C, dtype=torch.float64), -1)
    ut = torch.triu(torch.ones(C, C, dtype=torch.float64))
    sut = torch.triu(torch.ones(C, C, dtype=torch.float64), 1)

    import os

    exact_decay = os.environ.get("GDN_EXACT_DECAY", "0") == "1"
    exact_state = os.environ.get("GDN_EXACT_STATE", "0") == "1"
    dg_host = torch.zeros(B, H, NC, C, dtype=torch.float64)
    for bh in range(B * H):
        b, h = bh // H, bh % H
        for i in range(NC):
            it = bh * NC + i
            rb = lambda base, stride, r, c: read_block(flat, base + it * stride, r, c)
            Tinv = rb(smap.base_attn, smap.st_attn, Ct, Ct)
            vecs = rb(smap.base_vec, smap.st_vec, 3, 1)
            decay = vecs[: 32 * Ct, 0]
            beta = vecs[32 * Ct : 64 * Ct, 0]
            dc1 = vecs[64 * Ct : 96 * Ct, 0]
            Si = rb(smap.base_s, smap.st_s, Kt, Vt)[:K, :V]
            dSo = rb(smap.base_ds, smap.st_ds, Kt, Vt)[:K, :V]
            vnew = rb(smap.base_vnew, smap.st_vnew, Ct, Vt)[:, :V]
            dvnew = rb(smap.base_dvnew, smap.st_dvnew, Ct, Vt)[:, :V]
            p = P_[(b, h, i)]
            if exact_decay:
                decay = p["decay"]
                dc1 = p["dc1"]
            if os.environ.get("GDN_EXACT_TINV", "0") == "1":
                Tinv = p["Tinv"]
            if exact_state:
                Si = scan["S"][(b, h, i)]
                dSo = scan["dS"][(b, h, i)]
                vnew = scan["vnew"][(b, h, i)]
                dvnew = scan["dvnew"][(b, h, i)]
            qt = p["Tinv"].new_tensor(0)  # placeholder
            # inputs (exact, from the host side)
            sl = slice(i * C, (i + 1) * C)
            qt = ref["q"][b, sl, h, :].double() * scale
            kt_ = ref["k"][b, sl, h, :].double()
            vt_ = ref["v"][b, sl, h, :].double()
            dot = ref["do"][b, sl, h, :].double()
            gam = decay.exp()
            w = (dc1 - decay).exp()
            Gam = dc1[0].exp()
            X = decay[:, None].expand(C, C)
            Lm = (X - X.T + (lt - 1.0) * 1e4).exp()
            kb = kt_ * beta[:, None]
            U = kb * gam[:, None]
            dQ = dot @ Si.T
            M = (dot @ vnew.T) * Lm
            dP = vnew @ dSo.T
            ndkcd = dvnew @ Si.T
            dGam = (dSo * Si).sum()
            ndU = Tinv.T @ ndkcd
            d_attn = dvnew @ (vt_ * beta[:, None]).T - ndkcd @ U.T
            dAn = -((Tinv.T @ d_attn @ Tinv.T) * st)
            W = dAn * Lm
            dgg = (dQ * qt).sum(-1) * gam - (ndU * U).sum(-1)
            dww = (dP * kt_).sum(-1) * w
            R = M * (qt @ kt_.T) + W * (kb @ kt_.T)
            e = (R - R.T).sum(-1) + dgg
            dg_host[b, h, i] = ut @ e + sut.T @ dww + dGam * Gam

    dg_host = dg_host.reshape(B, H, NC * C)[:, :, :T].transpose(1, 2).contiguous()
    dg_dev = ttnn.to_torch(got[3]).double()
    print(
        f"DGBISECT  host-from-device-scratch vs reference : pcc={pcc(dg_host, exp[3]):.6f} relrms={rel_rms(dg_host, exp[3]):.5f}"
    )
    print(
        f"DGBISECT  device                   vs reference : pcc={pcc(dg_dev, exp[3]):.6f} relrms={rel_rms(dg_dev, exp[3]):.5f}"
    )
    print(
        f"DGBISECT  device vs host-from-device-scratch    : pcc={pcc(dg_dev, dg_host):.6f} relrms={rel_rms(dg_dev, dg_host):.5f}"
    )
