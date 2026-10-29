# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_decode_step vs a torch reference of the GDN decode step (single device).

  pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step.py -s
"""
import pytest
import torch

import ttnn

Nv, Nk, Dk, Dv = 6, 2, 128, 128
KD, VD = Nk * Dk, Nv * Dv


def _reference(q, k, v, beta, g, h, w, scale, l2_eps=1e-6, norm_eps=1e-6):
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


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize("steps", [3])
def test_gdn_decode_step(device, steps):
    torch.manual_seed(0)
    scale = Dk**-0.5
    h = (0.05 * torch.randn(Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h.reshape(1, Nv, Dk, Dv), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    h_ref = h.clone()
    for step in range(steps):
        q = (0.5 * torch.randn(Nk, Dk)).bfloat16()
        k = (0.5 * torch.randn(Nk, Dk)).bfloat16()
        v = (0.5 * torch.randn(Nv, Dv)).bfloat16()
        beta = torch.sigmoid(torch.randn(Nv)).float()
        g = (-0.5 * torch.rand(Nv)).float()
        out_ref, h_ref = _reference(q.float(), k.float(), v.float(), beta, g, h_ref, w.float(), scale)
        qkv_row = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)]).reshape(1, 1, 2 * KD + VD)
        out = ttnn.experimental.kda.gdn_decode_step(
            to_dev(qkv_row, ttnn.bfloat16),
            to_dev(beta.reshape(1, 1, Nv), ttnn.float32),
            to_dev(g.reshape(1, 1, Nv), ttnn.float32),
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
        p_out, p_h = _pcc(out_t, out_ref), _pcc(h_t, h_ref)
        d_out = (out_t - out_ref).abs().max().item()
        d_h = (h_t - h_ref).abs().max().item()
        print(f"step {step}: out pcc={p_out:.6f} max|d|={d_out:.4e}  state pcc={p_h:.6f} max|d|={d_h:.4e}")
        assert p_out > 0.999 and p_h > 0.9999, (p_out, p_h)


def _reference_conv(hist, new_row, taps, dtb, nea, h, w, scale, l2_eps=1e-6, norm_eps=1e-6):
    """Fused-conv mode reference. hist: 3 x [C] rows (oldest first), new_row: [W] = [q|k|v|z|a|b], taps: 4 x [C].
    Returns (gated [Nv*Dv], h_new, new_hist (list of 4 rows: cs0..cs3 after the shift))."""
    C = 2 * KD + VD
    conv = hist[0] * taps[0] + hist[1] * taps[1] + hist[2] * taps[2] + new_row[:C] * taps[3]
    conv = torch.nn.functional.silu(conv)
    q, k, v = conv[:KD].reshape(Nk, Dk), conv[KD : 2 * KD].reshape(Nk, Dk), conv[2 * KD : C].reshape(Nv, Dv)
    z = new_row[C : C + VD]
    a = new_row[C + VD : C + VD + Nv]
    b = new_row[C + VD + Nv : C + VD + 2 * Nv]
    beta = torch.sigmoid(b)
    g = nea * torch.nn.functional.softplus(a + dtb, beta=1.0, threshold=20.0)
    out, h_new = _reference(q, k, v, beta, g, h, w, scale, l2_eps, norm_eps)
    gated = out * torch.nn.functional.silu(z)
    new_hist = [hist[0], hist[1], hist[2], new_row[:C]]  # cs0 <- old cs1, ..., cs3 <- new qkv (hist holds old cs1..cs3)
    return gated, h_new, new_hist


def _pack_rows(rows):
    """4 torch [C] rows (oldest first) -> [Nv, 4, 32, 32] packed head tiles (row c = channel chunk c of [q|k|v])."""
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
            )
            n = chunks.numel() // 32
            out[h, j, 0 : 2 * n : 2, :] = chunks.reshape(-1, 32)  # chunk c in row 2c
    return out


@pytest.mark.parametrize("fp32_state", [False, True])
@pytest.mark.parametrize("fast_mode", [False, True])
@pytest.mark.parametrize("steps", [3])
def test_gdn_decode_step_conv(device, steps, fast_mode, fp32_state):
    if fp32_state and not fast_mode:
        pytest.skip("fp32_state needs fast_mode")
    torch.manual_seed(1)
    scale = Dk**-0.5
    C = 2 * KD + VD
    W = C + VD + 32  # [q|k|v | z | a b (padded to a tile)]
    az = C + VD
    h = (0.05 * torch.randn(Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    dtb = (0.1 * torch.randn(Nv)).bfloat16()
    nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
    cs = [(0.5 * torch.randn(C)).bfloat16() for _ in range(4)]  # cs0 (oldest) .. cs3 (newest)
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h.reshape(1, Nv, Dk, Dv), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    taps_dev = to_dev(_pack_rows(taps), ttnn.bfloat16)
    hist_dev = to_dev(_pack_rows(cs).unsqueeze(0), ttnn.bfloat16)  # [Bmax=1, Nv, 4, 32, 32]
    h_ref = h.clone()
    cs_ref = [t.float() for t in cs]
    for step in range(steps):
        row = (0.5 * torch.randn(W)).bfloat16()
        row[az + 2 * Nv :] = 0
        gated_ref, h_ref, new_hist = _reference_conv(
            cs_ref[1:], row.float(), [t.float() for t in taps], dtb.float(), nea.float(), h_ref, w.float(), scale
        )
        out = ttnn.experimental.kda.gdn_decode_step(
            to_dev(row.reshape(1, 1, W), ttnn.bfloat16),
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
            fast_mode=fast_mode,
            fp32_state=fp32_state,
        )
        out_t = ttnn.to_torch(out).reshape(-1)
        h_t = ttnn.to_torch(state).reshape(Nv, Dk, Dv)
        hist_t = ttnn.to_torch(hist_dev)
        p_out, p_h = _pcc(out_t, gated_ref), _pcc(h_t, h_ref)
        hist_ok = torch.equal(hist_t[0], _pack_rows([t.bfloat16() for t in new_hist]))
        print(
            f"step {step}: out pcc={p_out:.6f} max|d|={(out_t - gated_ref).abs().max().item():.4e}  "
            f"state pcc={p_h:.6f} max|d|={(h_t - h_ref).abs().max().item():.4e}  packed-history shift exact={hist_ok}"
        )
        state_thr = 0.999 if fast_mode else 0.9999
        assert p_out > 0.999 and p_h > state_thr and hist_ok, (p_out, p_h, hist_ok)
        cs_ref = new_hist


def _pack_rows_user(rows, b, both=False):
    """4 torch [C] rows -> [Nv, 4, 32, 32] packed head tiles for user row b (chunk c in row 2c + (b & 1))."""
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


@pytest.mark.parametrize("fp32_state", [False, True])
@pytest.mark.parametrize("fast_mode", [False, True])
@pytest.mark.parametrize("B", [1, 2, 4, 8, 16, 32])
def test_gdn_decode_step_conv_batched(device, B, fast_mode, fp32_state):
    if fp32_state and not fast_mode:
        pytest.skip("fp32_state needs fast_mode")
    """Bmax = 32 state/history; B active users in rows/slots 0..B-1; slots >= B must be untouched."""
    torch.manual_seed(10 + B)
    Bmax = 32
    scale = Dk**-0.5
    C = 2 * KD + VD
    W = C + VD + 32
    az = C + VD
    h_all = (0.05 * torch.randn(Bmax, Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    dtb = (0.1 * torch.randn(Nv)).bfloat16()
    nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
    cs_all = [[(0.5 * torch.randn(C)).bfloat16() for _ in range(4)] for _ in range(Bmax)]  # per user: cs0..cs3
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h_all.clone(), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    taps_dev = to_dev(_pack_rows_user(taps, 0, both=True), ttnn.bfloat16)
    hist_host = torch.stack([_pack_rows_user(cs_all[b], b) for b in range(Bmax)])  # [Bmax, Nv, 4, 32, 32]
    hist_dev = to_dev(hist_host.clone(), ttnn.bfloat16)
    rows = (0.5 * torch.randn(B, W)).bfloat16()
    rows[:, az + 2 * Nv :] = 0
    # reference per user
    refs = []
    h_ref = h_all.clone()
    new_hist = hist_host.clone()
    for b in range(B):
        gated_ref, h_b, nh = _reference_conv(
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
        new_hist[b] = _pack_rows_user([t.bfloat16() for t in nh], b)
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
        fast_mode=fast_mode,
        fp32_state=fp32_state,
    )
    out_t = ttnn.to_torch(out).reshape(B, -1)
    h_t = ttnn.to_torch(state)
    hist_t = ttnn.to_torch(hist_dev)
    worst_out = min(_pcc(out_t[b], refs[b]) for b in range(B))
    worst_state = min(_pcc(h_t[b], h_ref[b]) for b in range(B))
    untouched = torch.equal(h_t[B:], h_all[B:]) and torch.equal(hist_t[B:], hist_host[B:]) if B < Bmax else True
    hist_ok = torch.equal(hist_t[:B], new_hist[:B])
    max_d = max((out_t[b] - refs[b]).abs().max().item() for b in range(B))
    print(
        f"B={B}: worst out pcc={worst_out:.6f} max|d|={max_d:.3e} worst state pcc={worst_state:.6f} hist shift exact={hist_ok} untouched={untouched}"
    )
    state_thr = 0.999 if fast_mode else 0.9999
    assert worst_out > 0.999 and worst_state > state_thr and hist_ok and untouched, (
        worst_out,
        worst_state,
        hist_ok,
        untouched,
    )


@pytest.mark.parametrize("B", [1, 8, 16, 32])
def test_gdn_decode_step_conv_timing(device, B):
    """Rough host-measured time per call (fused-conv, fast_mode=True); not a correctness test."""
    import time

    torch.manual_seed(0)
    Bmax = 32
    scale = Dk**-0.5
    C = 2 * KD + VD
    W = C + VD + 32
    az = C + VD
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev((0.05 * torch.randn(Bmax, Nv, Dk, Dv)).float(), ttnn.float32)
    w_dev = to_dev((1.0 + 0.1 * torch.randn(Dv)).bfloat16(), ttnn.bfloat16)
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    taps_dev = to_dev(_pack_rows_user(taps, 0, both=True), ttnn.bfloat16)
    hist_dev = to_dev(torch.stack([_pack_rows_user(taps, b) for b in range(Bmax)]), ttnn.bfloat16)
    dtb = to_dev((0.1 * torch.randn(Nv)).bfloat16(), ttnn.bfloat16)
    nea = to_dev((-torch.exp(0.2 * torch.randn(Nv))).bfloat16(), ttnn.bfloat16)
    rows = (0.5 * torch.randn(B, W)).bfloat16()
    rows[:, az + 2 * Nv :] = 0
    x = to_dev(rows.reshape(1, B, W), ttnn.bfloat16)

    def run():
        return ttnn.experimental.kda.gdn_decode_step(
            x,
            dtb,
            nea,
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
            fast_mode=True,
        )

    for _ in range(3):
        run()
    ttnn.synchronize_device(device)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        run()
    ttnn.synchronize_device(device)
    print(f"TIMING B={B}: {(time.perf_counter() - t0) / n * 1e6:.1f} us/call (host-measured, avg of {n})")


def _run_value_blocks_chain(device, B, fp32_state, out_dtype, value_blocks, steps, seed):
    """`steps` chained fused-conv fast_mode decode steps from a fixed seed; returns (outs, state, hist) on host."""
    torch.manual_seed(seed)
    Bmax = 32
    scale = Dk**-0.5
    C = 2 * KD + VD
    W = C + VD + 32
    az = C + VD
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev((0.05 * torch.randn(Bmax, Nv, Dk, Dv)).float(), ttnn.float32)
    w_dev = to_dev((1.0 + 0.1 * torch.randn(Dv)).bfloat16(), ttnn.bfloat16)
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    taps_dev = to_dev(_pack_rows_user(taps, 0, both=True), ttnn.bfloat16)
    cs_all = [[(0.5 * torch.randn(C)).bfloat16() for _ in range(4)] for _ in range(Bmax)]
    hist_dev = to_dev(torch.stack([_pack_rows_user(cs_all[b], b) for b in range(Bmax)]), ttnn.bfloat16)
    dtb = to_dev((0.1 * torch.randn(Nv)).bfloat16(), ttnn.bfloat16)
    nea = to_dev((-torch.exp(0.2 * torch.randn(Nv))).bfloat16(), ttnn.bfloat16)
    vb_kw = {} if value_blocks == 1 else dict(value_blocks=value_blocks)
    outs = []
    for _ in range(steps):
        rows = (0.5 * torch.randn(B, W)).bfloat16()
        rows[:, az + 2 * Nv :] = 0
        out = ttnn.experimental.kda.gdn_decode_step(
            to_dev(rows.reshape(1, B, W), ttnn.bfloat16),
            dtb,
            nea,
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=scale,
            output_dtype=out_dtype,
            conv_hist=hist_dev,
            conv_taps=taps_dev,
            qkvz_dim=az,
            fast_mode=True,
            fp32_state=fp32_state,
            **vb_kw,
        )
        outs.append(ttnn.to_torch(out).reshape(B, -1))
        ttnn.deallocate(out)
    return outs, ttnn.to_torch(state), ttnn.to_torch(hist_dev)


@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("fp32_state", [True, False])
@pytest.mark.parametrize("B", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("vb", [2, 4])
def test_value_blocks_bit_identical_decode(device, vb, B, fp32_state, out_dtype):
    """value_blocks = vb vs 1 (QWEN36_GDN_STEP_VALUE_BLOCKS): torch.equal on every step's out, the full rec_state
    (incl. untouched slots >= B) and the packed history, over 3 chained steps. B = 32 resolves to 1 block on a
    <= 110-core grid (12 groups x 6 heads x 2 > grid) -- then it is the original program and trivially equal."""
    steps = 3
    seed = 1234 + B
    ref = _run_value_blocks_chain(device, B, fp32_state, out_dtype, 1, steps, seed)
    got = _run_value_blocks_chain(device, B, fp32_state, out_dtype, vb, steps, seed)
    for i in range(steps):
        assert torch.equal(ref[0][i], got[0][i]), f"value_blocks={vb} B={B} step {i}: out differs"
    assert torch.equal(ref[1], got[1]), f"value_blocks={vb} B={B}: rec_state differs"
    assert torch.equal(ref[2], got[2]), f"value_blocks={vb} B={B}: packed history differs"
    print(f"value_blocks {vb} == 1 at B={B} fp32_state={fp32_state} out={out_dtype}: torch.equal (out, state, hist)")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("B", [1, 8])
def test_value_blocks_trace_replay_decode(device, B):
    """Eager step 0 + trace capture (records only) + 2 replays at value_blocks = 4 equal 3 eager value_blocks = 1
    steps (semaphores re-initialized per launch; no hang)."""
    steps = 3
    seed = 99 + B
    ref = _run_value_blocks_chain(device, B, True, ttnn.bfloat16, 1, steps, seed)
    torch.manual_seed(seed)
    Bmax = 32
    scale = Dk**-0.5
    C = 2 * KD + VD
    W = C + VD + 32
    az = C + VD
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev((0.05 * torch.randn(Bmax, Nv, Dk, Dv)).float(), ttnn.float32)
    w_dev = to_dev((1.0 + 0.1 * torch.randn(Dv)).bfloat16(), ttnn.bfloat16)
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    taps_dev = to_dev(_pack_rows_user(taps, 0, both=True), ttnn.bfloat16)
    cs_all = [[(0.5 * torch.randn(C)).bfloat16() for _ in range(4)] for _ in range(Bmax)]
    hist_dev = to_dev(torch.stack([_pack_rows_user(cs_all[b], b) for b in range(Bmax)]), ttnn.bfloat16)
    dtb = to_dev((0.1 * torch.randn(Nv)).bfloat16(), ttnn.bfloat16)
    nea = to_dev((-torch.exp(0.2 * torch.randn(Nv))).bfloat16(), ttnn.bfloat16)
    row_list = []
    for _ in range(steps):
        rows = (0.5 * torch.randn(B, W)).bfloat16()
        rows[:, az + 2 * Nv :] = 0
        row_list.append(rows)
    x = to_dev(row_list[0].reshape(1, B, W), ttnn.bfloat16)

    def run():
        return ttnn.experimental.kda.gdn_decode_step(
            x,
            dtb,
            nea,
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=scale,
            output_dtype=ttnn.bfloat16,
            conv_hist=hist_dev,
            conv_taps=taps_dev,
            qkvz_dim=az,
            fast_mode=True,
            fp32_state=True,
            value_blocks=4,
        )

    out0 = run()  # eager step 0 (compiles the program)
    outs = [ttnn.to_torch(out0).reshape(B, -1)]
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_t = run()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for i in range(1, steps):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(row_list[i].reshape(1, B, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x
        )
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        outs.append(ttnn.to_torch(out_t).reshape(B, -1))
    ttnn.release_trace(device, tid)
    for i in range(steps):
        assert torch.equal(ref[0][i], outs[i]), f"trace value_blocks=4 B={B} step {i}: out differs from eager vb=1"
    assert torch.equal(ref[1], ttnn.to_torch(state)), f"trace value_blocks=4 B={B}: rec_state differs"
    assert torch.equal(ref[2], ttnn.to_torch(hist_dev)), f"trace value_blocks=4 B={B}: packed history differs"


@pytest.mark.parametrize("vb", [1, 2, 4])
@pytest.mark.parametrize("B", [1, 8])
def test_value_blocks_timing_decode(device, B, vb):
    """Host-measured time per call at value_blocks 1 / 2 / 4 (report only; use Tracy for the device kernel time)."""
    import time

    for i in range(2):
        t0 = time.perf_counter() if i else None
        _run_value_blocks_chain(device, B, True, ttnn.bfloat16, vb, 10, 5)
        ttnn.synchronize_device(device)
    print(f"TIMING value_blocks={vb} B={B}: {(time.perf_counter() - t0) / 10 * 1e6:.1f} us/call (incl. host I/O)")
