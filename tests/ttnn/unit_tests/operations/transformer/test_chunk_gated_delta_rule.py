# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.transformer.chunk_gated_delta_rule — the C++ chunked Gated Delta Rule forward, optionally
also returning the intermediates a training backward consumes:

    output_intermediates=False (default): (o, final_state | None)        — the original contract
    output_intermediates=True:            (o, final_state, h, v_new, g_cumsum, A)

checked against a float64 oracle (flash-linear-attention chunk_gated_delta_rule_fwd semantics).
Also pins:
* the default call is unchanged: a 2-tuple, bit-identical o / final_state to the intermediates mode;
* shapes with more (head, chunk) prep work-items than compute cores (B*HV*NC > grid). Those used to
  corrupt `o` through a CB ring-straddle in the prep kernel's scratch buffers.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn


def _oracle(q, k, v, g, beta, *, initial_state=None, chunk_size=64, scale=None):
    """float64 chunked gated delta rule forward + intermediates.

    q, k [B,T,H,K]; v [B,T,HV,V]; g, beta [B,T,HV]; HV = G*H (GQA: q/k head hv // G).
    Returns o [B,T,HV,V], final_state [B,HV,K,V], h [B,NC,HV,K,V] (state entering each chunk),
    v_new [B,T,HV,V], g_cumsum [B,T,HV], A [B,T,HV,C] (rows of the per-chunk UT inverse).
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    C = chunk_size
    scale = K**-0.5 if scale is None else scale
    f64 = torch.float64
    q = q.to(f64).repeat_interleave(G, dim=2).transpose(1, 2)
    k = k.to(f64).repeat_interleave(G, dim=2).transpose(1, 2)
    v = v.to(f64).transpose(1, 2)
    g, beta = g.to(f64).transpose(1, 2), beta.to(f64).transpose(1, 2)
    NC = math.ceil(T / C)
    pad = NC * C - T
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    q = q * scale
    kb, vb = k * beta[..., None], v * beta[..., None]
    ch = lambda x: x.reshape(B, HV, NC, C, x.shape[-1])
    q_c, k_c, kb_c, vb_c = ch(q), ch(k), ch(kb), ch(vb)
    decay = g.reshape(B, HV, NC, C).cumsum(-1)
    incl = torch.tril(torch.ones(C, C, dtype=torch.bool))
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool), diagonal=-1)
    diff = decay[..., :, None] - decay[..., None, :]
    L = torch.where(incl, diff, torch.full_like(diff, -math.inf)).exp()
    A_strict = (-(kb_c @ k_c.transpose(-1, -2)) * L).masked_fill(~strict, 0.0)
    eye = torch.eye(C, dtype=f64)
    Tinv = torch.linalg.solve_triangular(eye - A_strict, eye.expand_as(A_strict), upper=False, unitriangular=True)
    v_corr, kcd = Tinv @ vb_c, Tinv @ (kb_c * decay.exp()[..., None])
    S = torch.zeros(B, HV, K, V, dtype=f64) if initial_state is None else initial_state.to(f64).clone()
    h = torch.empty(B, HV, NC, K, V, dtype=f64)
    o = torch.empty(B, HV, NC, C, V, dtype=f64)
    vn = torch.empty(B, HV, NC, C, V, dtype=f64)
    for i in range(NC):
        h[:, :, i] = S
        d = decay[:, :, i]
        vn[:, :, i] = v_corr[:, :, i] - kcd[:, :, i] @ S
        o[:, :, i] = (q_c[:, :, i] * d.exp()[..., None]) @ S + (
            (q_c[:, :, i] @ k_c[:, :, i].transpose(-1, -2)) * L[:, :, i]
        ) @ vn[:, :, i]
        dl = d[..., -1]
        S = (
            S * dl.exp()[..., None, None]
            + (k_c[:, :, i] * (dl[..., None] - d).exp()[..., None]).transpose(-1, -2) @ vn[:, :, i]
        )
    tok = lambda x: x.reshape(B, HV, NC * C, x.shape[-1])[:, :, :T].transpose(1, 2)
    gcum = decay.reshape(B, HV, NC * C)[:, :, :T].transpose(1, 2)
    return tok(o), S, h.permute(0, 2, 1, 3, 4), tok(vn), gcum, tok(Tinv)


NAMES = ("o", "final_state", "h", "v_new", "g_cumsum", "A")


def _inputs(B, T, H, HV, K, V, with_h0, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = {
        "q": l2(rn(B, T, H, K)),
        "k": l2(rn(B, T, H, K)),
        "v": rn(B, T, HV, V),
        "g": F.logsigmoid(rn(B, T, HV)),
        "beta": torch.rand(B, T, HV, generator=gen, dtype=torch.float64),
        "initial_state": rn(B, HV, K, V) * 0.1 if with_h0 else None,
    }
    # Round to what the op actually computes on: q/k/v in bfloat16 (FLA Triton dtypes) whatever the
    # input dtype; g/beta/state at the input dtype, widened losslessly to fp32 on device.
    in_t = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    for n in ("q", "k", "v"):
        x[n] = x[n].to(torch.bfloat16).double()
    for n in ("g", "beta", "initial_state"):
        if x[n] is not None:
            x[n] = x[n].to(in_t).double()
    return x


def _dev(x, device, dtype):
    return {
        n: (
            None
            if t is None
            else ttnn.from_torch(
                t.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        for n, t in x.items()
    }


def _check(name, got, exp, pcc_min=0.999, rms_max=0.02):
    a = ttnn.to_torch(got).double().reshape(exp.shape).flatten()
    e = exp.double().flatten()
    assert torch.isfinite(a).all(), f"{name}: non-finite"
    if e.std().item() < 1e-30:  # constant reference (e.g. h with NC == 1 and no h0): PCC undefined
        err = (a - e).abs().max().item()
        assert err <= 1e-6, f"{name}: constant reference, max abs err {err:.3e}"
        return 1.0, 0.0
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    rms = ((a - e).pow(2).mean().sqrt() / e.std()).item()
    print(f"PCC {name:<28s} pcc={pcc:.7f} rel_rms={rms:.3e}")  # visible with -s
    assert pcc >= pcc_min and rms <= rms_max, f"{name}: pcc={pcc:.6f} rel_rms={rms:.3e} (need {pcc_min}/{rms_max})"
    return pcc, rms


# (B, T, H, HV, K, V, chunk)
CASES = [
    (1, 64, 1, 1, 64, 64, 64),  # single chunk
    (1, 128, 2, 2, 64, 128, 64),  # wide V
    (1, 100, 2, 2, 64, 64, 64),  # ragged tail (T % C == 36)
    (2, 64, 4, 4, 64, 64, 32),  # multi-batch, chunk 32
    (1, 200, 2, 2, 32, 64, 32),  # K = 32, ragged
    (1, 256, 2, 4, 128, 128, 64),  # GQA: HV = 2 * H
    (1, 256, 4, 4, 128, 128, 64),
    (1, 1024, 8, 8, 128, 128, 64),  # 128 prep items > 110 cores (the fixed ring-straddle bug)
]
IDS = [f"B{c[0]}_T{c[1]}_H{c[2]}_HV{c[3]}_K{c[4]}_V{c[5]}_c{c[6]}" for c in CASES]


@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_with_intermediates(case, dtype, with_h0, device):
    B, T, H, HV, K, V, C = case
    x = _inputs(B, T, H, HV, K, V, with_h0, dtype)
    exp = _oracle(x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], chunk_size=C)
    d = _dev(x, device, dtype)
    out = ttnn.transformer.chunk_gated_delta_rule(
        d["q"],
        d["k"],
        d["v"],
        d["g"],
        d["beta"],
        initial_state=d["initial_state"],
        chunk_size=C,
        output_intermediates=True,
    )
    assert len(out) == 6
    NC = math.ceil(T / C)
    shapes = {
        "o": [B, T, HV, V],
        "final_state": [B, HV, K, V],
        "h": [B, NC, HV, K, V],
        "v_new": [B, T, HV, V],
        "g_cumsum": [B, T, HV],
        "A": [B, T, HV, C],
    }
    for name, t, e in zip(NAMES, out, exp):
        assert list(t.shape) == shapes[name], f"{name}: shape {list(t.shape)} != {shapes[name]}"
        _check(name, t, e)
    # h[:, 0] IS the initial state (a DRAM copy of the fp32-widened input: bit-exact), or exactly 0.
    h0 = ttnn.to_torch(out[2]).double()[:, 0]
    if with_h0:
        assert torch.equal(h0, x["initial_state"]), f"h[:,0] != h0 (max err {(h0 - x['initial_state']).abs().max()})"
    else:
        assert h0.abs().max().item() == 0.0


@pytest.mark.parametrize("case", [CASES[2], CASES[5]], ids=[IDS[2], IDS[5]])
def test_head_major(case, device):
    B, T, H, HV, K, V, C = case
    x = _inputs(B, T, H, HV, K, V, True, ttnn.bfloat16, seed=1)
    exp = _oracle(x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], chunk_size=C)
    d = _dev(x, device, ttnn.bfloat16)
    o, s, h, vn, gc, A = ttnn.transformer.chunk_gated_delta_rule(
        d["q"],
        d["k"],
        d["v"],
        d["g"],
        d["beta"],
        initial_state=d["initial_state"],
        chunk_size=C,
        output_head_major=True,
        output_intermediates=True,
    )
    BH, NC = B * HV, math.ceil(T / C)
    hm = lambda e: e.transpose(1, 2).reshape(BH, T, *e.shape[3:])  # [B,T,HV,*] -> [BH,T,*]
    assert list(o.shape) == [BH, T, V] and list(vn.shape) == [BH, T, V]
    assert list(A.shape) == [BH, T, C] and list(gc.shape) == [BH, T] and list(h.shape) == [BH, NC, K, V]
    _check("o", o, hm(exp[0]))
    _check("final_state", s, exp[1])
    _check("h", h, exp[2].transpose(1, 2).reshape(BH, NC, K, V))
    _check("v_new", vn, hm(exp[3]))
    _check("g_cumsum", gc, exp[4].transpose(1, 2).reshape(BH, T))
    _check("A", A, hm(exp[5]))


@pytest.mark.parametrize("case", [CASES[2], CASES[7]], ids=[IDS[2], IDS[7]])
def test_default_unchanged(case, device):
    """Default (output_intermediates=False) keeps the original contract: a 2-tuple (o, final_state|None),
    bit-identical o / final_state to output_intermediates=True."""
    B, T, H, HV, K, V, C = case
    x = _inputs(B, T, H, HV, K, V, True, ttnn.bfloat16, seed=2)
    exp = _oracle(x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], chunk_size=C)
    d = _dev(x, device, ttnn.bfloat16)
    args = (d["q"], d["k"], d["v"], d["g"], d["beta"])
    res2 = ttnn.transformer.chunk_gated_delta_rule(
        *args, initial_state=d["initial_state"], output_final_state=True, chunk_size=C
    )
    assert isinstance(res2, tuple) and len(res2) == 2
    o2, s2 = res2
    res6 = ttnn.transformer.chunk_gated_delta_rule(
        *args, initial_state=d["initial_state"], chunk_size=C, output_intermediates=True
    )
    assert isinstance(res6, tuple) and len(res6) == 6
    assert torch.equal(ttnn.to_torch(o2), ttnn.to_torch(res6[0]))
    assert torch.equal(ttnn.to_torch(s2), ttnn.to_torch(res6[1]))
    _check("o (default)", o2, exp[0])
    _check("final_state (default)", s2, exp[1])
    # output_final_state still defaults to False -> (o, None), exactly as before
    o_only, none = ttnn.transformer.chunk_gated_delta_rule(*args, chunk_size=C)
    assert none is None
    # output_intermediates=True always returns the final state, even with output_final_state=False
    res = ttnn.transformer.chunk_gated_delta_rule(*args, chunk_size=C, output_intermediates=True)
    assert len(res) == 6 and all(t is not None for t in res)


@pytest.mark.parametrize("B,T,H,HV,K", [(1, 256, 2, 4, 128), (1, 4096, 4, 8, 128)], ids=["T256_H2_HV4", "T4096_H4_HV8"])
def test_flat_qkv_path(B, T, H, HV, K, device):
    """OPT-A/B flat path (qwen36 prefill): q/k/v arrive FLAT token-major [B, T, H*K] and UN-normalized;
    the prep kernel L2-normalizes q/k in-kernel (chunk 32 only). Exercises the scratch-CB quantum on
    the QK_NORM branch, and the intermediates on the flat path."""
    C, V = 32, K
    x = _inputs(B, T, H, HV, K, V, True, ttnn.bfloat16, seed=3)
    gen = torch.Generator().manual_seed(4)
    mag = lambda *s: (torch.rand(*s, generator=gen, dtype=torch.float64) + 0.5)
    q_raw = (x["q"] * mag(B, T, H, 1)).to(torch.bfloat16).double()  # un-normalize
    k_raw = (x["k"] * mag(B, T, H, 1)).to(torch.bfloat16).double()
    l2 = lambda t: t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    exp = _oracle(l2(q_raw), l2(k_raw), x["v"], x["g"], x["beta"], initial_state=x["initial_state"], chunk_size=C)
    d = _dev(
        {**x, "q": q_raw.reshape(B, T, H * K), "k": k_raw.reshape(B, T, H * K), "v": x["v"].reshape(B, T, HV * V)},
        device,
        ttnn.bfloat16,
    )
    o2, s2 = ttnn.transformer.chunk_gated_delta_rule(
        d["q"],
        d["k"],
        d["v"],
        d["g"],
        d["beta"],
        initial_state=d["initial_state"],
        output_final_state=True,
        chunk_size=C,
    )
    _check("o (flat, 2-output API)", o2, exp[0])
    _check("final_state (flat, 2-output API)", s2, exp[1])
    out = ttnn.transformer.chunk_gated_delta_rule(
        d["q"],
        d["k"],
        d["v"],
        d["g"],
        d["beta"],
        initial_state=d["initial_state"],
        chunk_size=C,
        output_intermediates=True,
    )
    for name, t, e in zip(NAMES, out, exp):
        _check(f"{name} (flat)", t, e)
