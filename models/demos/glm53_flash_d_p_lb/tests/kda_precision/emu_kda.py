"""CPU emulation of the device KDA recurrence's precision effects on one layer's real inputs.

Inputs: the device's captured attn_norm (x_00..x_10.pt of a capture dir) through the reference projections (fp32) ->
q, k, v (conv + SiLU), g (-5 sigmoid), beta. The recurrence in the device's chunk form (C = 32, direct sequential
scan, no SP / groups) with selectable roundings; the baseline is the same chunk form in float64.

Effects (comma list per variant):
  kd16 / qd16       kd / q_decay stored bf16 (KDA_PREP_OUTPUT_BF16_MASK bits 1, 2)
  a16               alpha = exp(G_last) stored bf16 (bit 5)
  atf               alpha read through a TF32 source register (truncated) once (the prep's transpose or the scan's
                    mul_bcast_cols srcB)
  sdec              the scan's decay multiply reads the fp32 state through a TF32 source register (truncated)
  sadd              the scan's update add reads both fp32 operands through TF32 (truncated)
  vsub              v_beta - kd @ S read through TF32 (truncated)
  mm                matmul operands that are fp32 (S, v_new, intra, t_inv, k_dec_t) rounded to TF32 (nearest)
  mmt               the same, truncated
  rne               every TF32 read above rounds to nearest instead of truncating
  g16               the gate (sigmoid) rounded to bf16 before the -5 scale (the device's decay projection output)

    python emu_kda.py <capture dir> <layer> [variants ;-separated]
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.golden import hf_path, load_spec
from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig, l2norm, load_layer
from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

torch.set_num_threads(16)
C = 32


def tf32(x, rne=False):
    """fp32 -> TF32 (10 explicit mantissa bits), truncated (default) or round-to-nearest-even."""
    i = x.float().contiguous().view(torch.int32)
    if rne:
        lsb = (i >> 13) & 1
        i = i + 0xFFF + lsb
    i = i & ~0x1FFF
    return i.view(torch.float32).to(x.dtype)


def bf16(x):
    return x.float().to(torch.bfloat16).to(x.dtype)


def inputs(cap, layer, n_chunks=11):
    spec = load_spec()
    path = hf_path(spec)
    cfg = GlmConfig.from_json(str(Path(path) / "config.json"))
    w = load_layer(WeightLoader(path), cfg, layer, torch.float32, eager_experts=False)
    a = "self_attn."
    conv_w = torch.cat([w[f"{a}{n}_conv1d.weight"] for n in "qkv"])[:, 0].float()
    h, d = cfg.linear_num_heads, cfg.linear_head_dim
    qs, ks, vs, gs, bs = [], [], [], [], []
    tail = torch.zeros(cfg.linear_conv_kernel - 1, 3 * h * d)
    for c in range(n_chunks):
        x = torch.load(Path(cap) / f"x_{c:02d}.pt")[layer].float()
        s = x.shape[0]
        qkv = torch.cat([F.linear(x, w[a + f"{n}_proj.weight"]) for n in "qkv"], dim=-1)
        window = torch.cat([tail, qkv])
        conv = F.silu(F.conv1d(window.t().unsqueeze(0), conv_w.unsqueeze(1), groups=window.shape[1])[0].t())
        tail = window[-(cfg.linear_conv_kernel - 1) :]
        q, k, v = (t.reshape(s, h, d).transpose(0, 1).float() for t in conv.split(h * d, dim=-1))
        raw_g = F.linear(F.linear(x, w[a + "f_a_proj.weight"]), w[a + "f_b_proj.weight"]).float()
        raw_g = (raw_g + w[a + "dt_bias"].float()).view(s, h, d)
        sig = torch.sigmoid(w[a + "A_log"].float().exp().view(1, h, 1) * raw_g)
        beta = torch.sigmoid(F.linear(x, w[a + "b_proj.weight"])).float()
        qs.append(l2norm(q) * d**-0.5)
        ks.append(l2norm(k))
        vs.append(v)
        gs.append(sig.transpose(0, 1))
        bs.append(beta.t())
    return (torch.cat(t, dim=1) for t in (qs, ks, vs, gs, bs))  # [H, T, D] / [H, T]


def _terms(k, q, v, g, beta, c0, eff, dtype, tr):
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)
    incl = torch.tril(torch.ones(C, C, dtype=torch.bool), 0)
    eye = torch.eye(C, dtype=dtype)
    H = k.shape[0]
    sl = slice(c0, c0 + C)
    kc, qc, vc, gc, bc = k[:, sl], q[:, sl], v[:, sl], g[:, sl], beta[:, sl]
    G = gc.double().cumsum(1).to(dtype)  # [H, C, K]
    Gl = G[:, -1:]  # [H, 1, K]
    dec = (G.unsqueeze(2) - G.unsqueeze(1)).exp()  # [H, i, j, K] exp(G_i - G_j)
    A = (bc.unsqueeze(-1) * kc).unsqueeze(2).mul(dec).mul(kc.unsqueeze(1)).sum(-1)
    A = A.masked_fill(~strict, 0)
    t_inv = torch.linalg.solve_triangular(eye + A, eye.expand(H, C, C), upper=False)
    del dec
    kd = bc.unsqueeze(-1) * kc * G.exp()
    kdt = (kc * (Gl - G).exp()).transpose(1, 2)  # [H, K, C]
    alpha = Gl.exp().transpose(1, 2)  # [H, K, 1]
    vb = bc.unsqueeze(-1) * vc
    if "kd16" in eff:
        kd = bf16(kd)
    if "a16" in eff:
        alpha = bf16(alpha)
    if "atf" in eff:
        alpha = tr(alpha)
    if "atf2" in eff:  # the prep: exp(G_last/2) copied through srcA, squared, transposed through srcA
        a = (Gl / 2).exp().transpose(1, 2)
        alpha = tr(tr(a) * tr(a))
        if "a16" in eff:
            alpha = bf16(alpha)
    d = -torch.expm1(Gl).transpose(1, 2)
    return {"kd": kd, "kdt": kdt, "alpha": alpha, "d": d, "vb": vb, "t_inv": t_inv}


def _step(S, t, eff, tr, mmq):
    vb = t["vb"]
    if S.shape[-1] != vb.shape[-1]:  # the summary's paired [B | A + B] state: both paths take the same values
        vb = torch.cat([vb] * (S.shape[-1] // vb.shape[-1]), dim=-1)
    P = t["kd"] @ mmq(S)
    D = (tr(vb) - tr(P)) if "vsub" in eff else (vb - P)
    vn = mmq(t["t_inv"]) @ mmq(D)
    U = mmq(t["kdt"]) @ mmq(vn)
    if "fixd" in eff:  # proposed: d = 1 - alpha in fp32; S - S*d on the FPU (TF32 operands), the sums in fp32
        return S - tr(S) * tr(t["d"]) + U
    Sd = (tr(S) * t["alpha"]) if "sdec" in eff else (S * t["alpha"])
    return (tr(Sd) + tr(U)) if "sadd" in eff else (Sd + U)


def run(q, k, v, sig, beta, eff, dtype=torch.float64, snap=()):
    """The device's chunk form over 32-token chunks. Default: sequential. With "grp": the device's SP2 grouped path for
    chunk-aligned 5120 starts: per SP rank (2560 rows) 5 groups of 16 chunks summarized as affine maps (zero / identity
    seeds; A = (A + B) - B, "sumx": both read through TF32; "sum16": A, B stored bf16), composed per rank ("pack16":
    the rank's [A | B] packed bf16), and the carry = rank 1 o rank 0 applied to the state (select_final_carry's
    unsplit case)."""
    eff = set(eff)
    rne = "rne" in eff
    H, T, K = k.shape
    V = v.shape[-1]
    sig = bf16(sig) if "g16" in eff else sig
    g = (-5.0 * sig.double()).to(dtype)
    q, k, v, beta = (t.to(dtype) for t in (q, k, v, beta))
    S = torch.zeros(H, K, V, dtype=dtype)
    tr = (lambda x: tf32(x, rne)) if dtype == torch.float32 else (lambda x: x)
    mmq = (lambda x: tf32(x, not ("mmt" in eff) or rne)) if ("mm" in eff or "mmt" in eff) else (lambda x: x)
    snaps = {}
    if "grp" not in eff:
        for c0 in range(0, T, C):
            S = _step(S, _terms(k, q, v, g, beta, c0, eff, dtype, tr), eff, tr, mmq)
            if c0 + C in snap:
                snaps[c0 + C] = S.clone()
        return S, snaps
    eye = torch.eye(K, dtype=dtype).expand(H, K, K)
    chunk, ranks, groups = 5120, 2, 5
    rows = chunk // ranks
    gchunks = rows // C // groups
    for s0 in range(0, T, chunk):
        rank_maps = []
        for r in range(ranks):
            Ar, Br = eye.clone(), torch.zeros(H, K, V, dtype=dtype)
            for gi in range(groups):
                pair = torch.cat([torch.zeros(H, K, V, dtype=dtype), eye.clone()], dim=-1)  # [B | A + B]
                for j in range(gchunks):
                    c0 = s0 + r * rows + (gi * gchunks + j) * C
                    pair = _step(pair, _terms(k, q, v, g, beta, c0, eff, dtype, tr), eff, tr, mmq)
                B, ApB = pair[..., :V], pair[..., V:]
                A = (tr(ApB) - tr(B)) if "sumx" in eff else (ApB - B)
                if "sum16" in eff:
                    A, B = bf16(A), bf16(B)
                Ar, Br = mmq(A) @ mmq(Ar), mmq(A) @ mmq(Br) + B
            if "pack16" in eff:
                Ar, Br = bf16(Ar), bf16(Br)
            rank_maps.append((Ar, Br))
        for Ar, Br in rank_maps:
            S = mmq(Ar) @ mmq(S) + Br
        if s0 + chunk in snap:
            snaps[s0 + chunk] = S.clone()
    return S, snaps


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


if __name__ == "__main__":
    cap, layer = sys.argv[1], int(sys.argv[2])
    variants = sys.argv[3].split(";") if len(sys.argv) > 3 else ["", "a16", "atf", "sdec", "sadd", "a16,sdec,sadd"]
    t = time.time()
    q, k, v, sig, beta = inputs(cap, layer)
    print(f"[emu] L{layer} inputs {time.time() - t:.0f}s, T {k.shape[1]}", flush=True)
    snaps = (5120, 25600, 51200, 56320)
    snaps = tuple(s for s in snaps if s <= k.shape[1])
    ex, exs = run(q, k, v, sig, beta, [], torch.float64, snaps)
    torch.save({"exact64": ex.float()}, Path(cap) / f"emu_exact64_L{layer}.pt")
    # slowest channels: per head and key row, the mean per-token gate (the decay rate)
    rate = (5.0 * sig.double()).mean(1)  # [H, K]
    try:
        dev = {p: torch.load(Path(cap) / f"state_{p}.pt")[layer]["kda_recurrent"].double() for p in snaps}
    except Exception:
        dev = {}
    for p in snaps:
        if p in dev:
            d = dev[p].reshape(exs[p].shape)
            hr = [rel(d[h], exs[p][h]) for h in range(d.shape[0])]
            print(
                f"[emu] device @ {p}: rel {rel(d, exs[p]):.4f} ratio {float(d.norm() / exs[p].norm()):.4f} "
                f"heads max {max(hr):.4f} med {sorted(hr)[32]:.4f}",
                flush=True,
            )
    for var in variants:
        t = time.time()
        eff = [e for e in var.split(",") if e]
        S, ss = run(q, k, v, sig, beta, eff, torch.float32, snaps)
        line = []
        for p in snaps:
            e, s = exs[p], ss[p]
            hr = [rel(s[h], e[h]) for h in range(s.shape[0])]
            line.append(f"@{p} rel {rel(s, e):.4f} ratio {float(s.norm() / e.norm()):.4f} hmax {max(hr):.4f}")
        print(f"[emu] L{layer} {var or 'fp32':22s} " + " | ".join(line) + f"  ({time.time() - t:.0f}s)", flush=True)
