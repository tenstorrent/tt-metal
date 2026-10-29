# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: ttnn.experimental.kda.gdn_decode_step vs a torch reference at every per-device value-head count
(TP=8: Nv=6, TP=4: Nv=12, TP=1: Nv=48 -> the per-head scalars span two tiles and a|b span three), single device.

  pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_nv_scratch.py -s

Bit-identity across kernel changes (TP=4 / TP=8 must stay byte-exact): dump (out, state, hist) per test from the
reference build, then check against the new build with the same seeds:
  GDN_GOLDEN_DIR=/path GDN_GOLDEN_MODE=dump  pytest ... -k "tp4 or tp8"
  GDN_GOLDEN_DIR=/path GDN_GOLDEN_MODE=check pytest ... -k "tp4 or tp8"
"""
import os
import re

import pytest
import torch

import ttnn

Dk = Dv = 128
CFGS = [
    pytest.param((6, 2), id="tp8"),
    pytest.param((12, 4), id="tp4"),
    pytest.param((48, 16), id="tp1"),
]


def _dims(Nv, Nk):
    KD, VD = Nk * Dk, Nv * Dv
    C = 2 * KD + VD  # [q | k | v]
    az = C + VD  # a|b column offset (after z)
    W = az + -(-2 * Nv // 32) * 32  # a|b padded to whole tiles
    return KD, VD, C, az, W


def _golden(request, **tensors):
    """GDN_GOLDEN_MODE=dump: save the device results; =check: assert torch.equal against the saved ones."""
    d, mode = os.environ.get("GDN_GOLDEN_DIR"), os.environ.get("GDN_GOLDEN_MODE")
    if not d or not mode:
        return
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    p = os.path.join(d, name + ".pt")
    if mode == "dump":
        os.makedirs(d, exist_ok=True)
        torch.save({k: v.clone().cpu() for k, v in tensors.items()}, p)
        print(f"golden dumped: {p}")
    elif mode == "check":
        ref = torch.load(p)
        for k, v in tensors.items():
            same = torch.equal(ref[k], v.cpu())
            print(f"golden {k}: bit-identical={same}")
            assert same, f"{name}: {k} differs from the golden dump (max|d|={(ref[k].float() - v.float()).abs().max()})"


def _reference(Nv, Nk, q, k, v, beta, g, h, w, scale, l2_eps=1e-6, norm_eps=1e-6):
    """q,k: [Nk,Dk]; v: [Nv,Dv]; beta,g: [Nv]; h: [Nv,Dk,Dv]; w: [Dv]. Returns (out [Nv*Dv], h_new)."""
    rf = Nv // Nk
    q = q.repeat_interleave(rf, dim=0)
    k = k.repeat_interleave(rf, dim=0)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + l2_eps) * scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + l2_eps)
    h = h * torch.exp(g)[:, None, None]
    v_read = torch.einsum("hk,hkv->hv", kn, h)
    delta = beta[:, None] * (v - v_read)
    h = h + torch.einsum("hk,hv->hkv", kn, delta)
    o = torch.einsum("hk,hkv->hv", qn, h)
    on = o / torch.sqrt((o * o).mean(-1, keepdim=True) + norm_eps) * w[None, :]
    return on.reshape(-1), h


def _reference_conv(Nv, Nk, hist, new_row, taps, dtb, nea, h, w, scale, l2_eps=1e-6, norm_eps=1e-6):
    """Fused-conv mode. hist: 3 x [C] rows (oldest first), new_row: [W] = [q|k|v|z|a|b], taps: 4 x [C].
    Returns (gated [Nv*Dv], h_new, new_hist (4 rows cs0..cs3 after the shift))."""
    KD, VD, C, az, _ = _dims(Nv, Nk)
    conv = hist[0] * taps[0] + hist[1] * taps[1] + hist[2] * taps[2] + new_row[:C] * taps[3]
    conv = torch.nn.functional.silu(conv)
    q, k, v = conv[:KD].reshape(Nk, Dk), conv[KD : 2 * KD].reshape(Nk, Dk), conv[2 * KD : C].reshape(Nv, Dv)
    z = new_row[C:az]
    a = new_row[az : az + Nv]
    b = new_row[az + Nv : az + 2 * Nv]
    beta = torch.sigmoid(b)
    g = nea * torch.nn.functional.softplus(a + dtb, beta=1.0, threshold=20.0)
    out, h_new = _reference(Nv, Nk, q, k, v, beta, g, h, w, scale, l2_eps, norm_eps)
    gated = out * torch.nn.functional.silu(z)
    return gated, h_new, [hist[0], hist[1], hist[2], new_row[:C]]


def _pack_rows_user(Nv, Nk, rows, b, both=False):
    """4 torch [C] rows -> [Nv, 4, 32, 32] packed head tiles for user row b (chunk c in row 2c + (b & 1))."""
    KD = Nk * Dk
    rf = Nv // Nk
    out = torch.zeros(Nv, 4, 32, 32, dtype=torch.bfloat16)
    for h in range(Nv):
        hk = h // rf
        for j, r in enumerate(rows):
            r = r.reshape(-1).to(torch.bfloat16)
            chunks = torch.cat(
                [
                    r[hk * Dk : (hk + 1) * Dk],
                    r[KD + hk * Dk : KD + (hk + 1) * Dk],
                    r[2 * KD + h * Dv : 2 * KD + (h + 1) * Dv],
                ]
            ).reshape(-1, 32)
            n = chunks.shape[0]
            for par in (0, 1) if both else (b & 1,):
                out[h, j, par : 2 * n + par : 2, :] = chunks
    return out


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize("cfg", CFGS)
@pytest.mark.parametrize("scalar_dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_gdn_decode_step_plain(device, request, cfg, scalar_dtype):
    """Plain variant (FUSED=1): B=1, qkv post conv+silu, beta/g scalars [1,1,Nv]; 3 chained steps."""
    Nv, Nk = cfg
    KD, VD, C, _, _ = _dims(Nv, Nk)
    torch.manual_seed(100 + Nv)
    scale = Dk**-0.5
    h = (0.05 * torch.randn(Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h.reshape(1, Nv, Dk, Dv), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    h_ref = h.clone()
    outs, states = [], []
    for step in range(3):
        q = (0.5 * torch.randn(Nk, Dk)).bfloat16()
        k = (0.5 * torch.randn(Nk, Dk)).bfloat16()
        v = (0.5 * torch.randn(Nv, Dv)).bfloat16()
        beta = torch.sigmoid(torch.randn(Nv)).float()
        g = (-0.5 * torch.rand(Nv)).float()
        if scalar_dtype == ttnn.bfloat16:
            beta, g = beta.bfloat16().float(), g.bfloat16().float()
        out_ref, h_ref = _reference(Nv, Nk, q.float(), k.float(), v.float(), beta, g, h_ref, w.float(), scale)
        qkv_row = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)]).reshape(1, 1, C)
        out = ttnn.experimental.kda.gdn_decode_step(
            to_dev(qkv_row, ttnn.bfloat16),
            to_dev(beta.reshape(1, 1, Nv), scalar_dtype),
            to_dev(g.reshape(1, 1, Nv), scalar_dtype),
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=scale,
            output_dtype=ttnn.float32,
        )
        out_t = ttnn.to_torch(out).reshape(-1)
        h_t = ttnn.to_torch(state).reshape(Nv, Dk, Dv)
        outs.append(out_t.clone())
        states.append(h_t.clone())
        p_out, p_h = _pcc(out_t, out_ref), _pcc(h_t, h_ref)
        d_out = (out_t - out_ref).abs().max().item()
        d_h = (h_t - h_ref).abs().max().item()
        print(f"Nv={Nv} step {step}: out pcc={p_out:.6f} max|d|={d_out:.4e}  state pcc={p_h:.6f} max|d|={d_h:.4e}")
        assert p_out > 0.999 and p_h > 0.9999, (p_out, p_h)
    _golden(request, out=torch.stack(outs), state=torch.stack(states))


@pytest.mark.parametrize("cfg", CFGS)
@pytest.mark.parametrize("B", [1, 2, 4, 8, 16, 32])
def test_gdn_decode_step_conv_batched(device, request, cfg, B):
    """Fused-conv variant (FUSED=2): Bmax = 32 state/history; B active users in rows/slots 0..B-1; slots >= B must
    be untouched; the packed history shift must be exact."""
    Nv, Nk = cfg
    KD, VD, C, az, W = _dims(Nv, Nk)
    torch.manual_seed(1000 * Nv + B)
    Bmax = 32
    scale = Dk**-0.5
    h_all = (0.05 * torch.randn(Bmax, Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    dtb = (0.1 * torch.randn(Nv)).bfloat16()
    nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
    cs_all = [[(0.5 * torch.randn(C)).bfloat16() for _ in range(4)] for _ in range(Bmax)]  # per user: cs0..cs3
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h_all.clone(), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    taps_dev = to_dev(_pack_rows_user(Nv, Nk, taps, 0, both=True), ttnn.bfloat16)
    hist_host = torch.stack([_pack_rows_user(Nv, Nk, cs_all[b], b) for b in range(Bmax)])  # [Bmax, Nv, 4, 32, 32]
    hist_dev = to_dev(hist_host.clone(), ttnn.bfloat16)
    rows = (0.5 * torch.randn(B, W)).bfloat16()
    rows[:, az + 2 * Nv :] = 0
    refs = []
    h_ref = h_all.clone()
    new_hist = hist_host.clone()
    for b in range(B):
        gated_ref, h_b, nh = _reference_conv(
            Nv,
            Nk,
            [t.float() for t in cs_all[b][1:]],
            rows[b].float(),
            [t.float() for t in taps],
            dtb.float(),
            nea.float(),
            h_all[b],
            w.float(),
            scale,
        )
        refs.append(gated_ref)
        h_ref[b] = h_b
        new_hist[b] = _pack_rows_user(Nv, Nk, [t.bfloat16() for t in nh], b)
    out = ttnn.experimental.kda.gdn_decode_step(
        to_dev(rows.reshape(1, B, W), ttnn.bfloat16),
        to_dev(dtb, ttnn.bfloat16),
        to_dev(nea, ttnn.bfloat16),
        state,
        w_dev,
        Nv,
        Nk,
        Dk,
        Dv,
        scale=scale,
        output_dtype=ttnn.float32,
        conv_hist=hist_dev,
        conv_taps=taps_dev,
        qkvz_dim=az,
    )
    out_t = ttnn.to_torch(out).reshape(B, -1)
    h_t = ttnn.to_torch(state)
    hist_t = ttnn.to_torch(hist_dev)
    worst_out = min(_pcc(out_t[b], refs[b]) for b in range(B))
    worst_state = min(_pcc(h_t[b], h_ref[b]) for b in range(B))
    untouched = torch.equal(h_t[B:], h_all[B:]) and torch.equal(hist_t[B:], hist_host[B:]) if B < Bmax else True
    hist_ok = torch.equal(hist_t[:B], new_hist[:B])
    max_d = max((out_t[b] - refs[b]).abs().max().item() for b in range(B))
    max_dh = max((h_t[b] - h_ref[b]).abs().max().item() for b in range(B))
    print(
        f"Nv={Nv} B={B}: worst out pcc={worst_out:.6f} max|d|={max_d:.3e} worst state pcc={worst_state:.6f} "
        f"max|dstate|={max_dh:.3e} hist shift exact={hist_ok} untouched={untouched}"
    )
    assert worst_out > 0.999 and worst_state > 0.9999 and hist_ok and untouched, (
        worst_out,
        worst_state,
        hist_ok,
        untouched,
    )
    _golden(request, out=out_t, state=h_t, hist=hist_t)


def test_gdn_decode_step_program_cache_two_nv(device):
    """Nv=12 (TP=4) and Nv=48 (TP=1) fused-conv programs in one process must be distinct cache entries and both
    correct (Nv is part of the program hash)."""
    B = 8
    n0 = device.num_program_cache_entries()
    for Nv, Nk in ((12, 4), (48, 16)):
        KD, VD, C, az, W = _dims(Nv, Nk)
        torch.manual_seed(7 + Nv)
        scale = Dk**-0.5
        h_all = (0.05 * torch.randn(B, Nv, Dk, Dv)).float()
        w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
        taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
        dtb = (0.1 * torch.randn(Nv)).bfloat16()
        nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
        cs_all = [[(0.5 * torch.randn(C)).bfloat16() for _ in range(4)] for _ in range(B)]
        to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        state = to_dev(h_all.clone(), ttnn.float32)
        hist_dev = to_dev(torch.stack([_pack_rows_user(Nv, Nk, cs_all[b], b) for b in range(B)]), ttnn.bfloat16)
        rows = (0.5 * torch.randn(B, W)).bfloat16()
        rows[:, az + 2 * Nv :] = 0
        for _ in range(2):  # second call hits the cache
            out = ttnn.experimental.kda.gdn_decode_step(
                to_dev(rows.reshape(1, B, W), ttnn.bfloat16),
                to_dev(dtb, ttnn.bfloat16),
                to_dev(nea, ttnn.bfloat16),
                state,
                to_dev(w, ttnn.bfloat16),
                Nv,
                Nk,
                Dk,
                Dv,
                scale=scale,
                output_dtype=ttnn.float32,
                conv_hist=hist_dev,
                conv_taps=to_dev(_pack_rows_user(Nv, Nk, taps, 0, both=True), ttnn.bfloat16),
                qkvz_dim=az,
            )
        # step 1 of the reference from the original state (device ran two steps; compare the first user's out of
        # step 2 with a two-step reference instead)
        h_ref = h_all.clone()
        cs = [[t.float() for t in cs_all[b]] for b in range(B)]
        for _ in range(2):
            for b in range(B):
                gated_ref, h_ref[b], nh = _reference_conv(
                    Nv, Nk, cs[b][1:], rows[b].float(), [t.float() for t in taps], dtb.float(), nea.float(),
                    h_ref[b], w.float(), scale,
                )
                cs[b] = nh
                if b == 0:
                    ref0 = gated_ref
        out_t = ttnn.to_torch(out).reshape(B, -1)
        p = _pcc(out_t[0], ref0)
        ps = _pcc(ttnn.to_torch(state), h_ref)
        print(f"program-cache Nv={Nv}: out pcc={p:.6f} state pcc={ps:.6f} entries={device.num_program_cache_entries()}")
        assert p > 0.999 and ps > 0.9999, (Nv, p, ps)
    print(f"program cache entries added: {device.num_program_cache_entries() - n0}")
    assert device.num_program_cache_entries() - n0 >= 2
