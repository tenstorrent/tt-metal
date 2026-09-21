"""DFlash2 ttnn drafter bring-up — PCC each component vs the torch golden (dflash2_golden.npz).

Runs on a single-device (1x1) mesh with full (non-sharded) tensors, matching the torch golden.
Gate dispatch: python dflash2_ttnn_bringup.py A1   (A1..A7)

Env: DFLASH_W (drafter weights dir), DFLASH_GOLDEN (npz from dflash2_golden.py).
"""
import glob
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

import ttnn
from models.common.utility_functions import comp_pcc

W = os.environ.get("DFLASH_W", "/home/ttuser/experiments/qwen36_27b/dflash_weights")
GOLDEN = os.environ.get("DFLASH_GOLDEN", "/home/ttuser/experiments/qwen36_27b/profiles/dflash2_golden.npz")
PCC = 0.99
H = 5120

CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def lin(*a, **k):
    return ttnn.linear(*a, compute_kernel_config=CKC, **k)


def mm(*a, **k):
    return ttnn.matmul(*a, compute_kernel_config=CKC, **k)


def rms(x, weight, epsilon=1e-6):
    return ttnn.rms_norm(x, weight=weight, epsilon=epsilon, compute_kernel_config=CKC)


_SD = None


def sd():
    global _SD
    if _SD is None:
        _SD = {}
        for f in sorted(glob.glob(f"{W}/*.safetensors")):
            _SD.update(load_file(f))
    return _SD


_DT = ttnn.float32 if os.environ.get("DFLASH_FP32") else ttnn.bfloat16


def to_dev(t, md, layout=ttnn.TILE_LAYOUT, dtype=None):
    if t.ndim == 1:
        t = t.reshape(1, -1)
    return ttnn.from_torch(t, dtype=dtype or _DT, layout=layout, device=md, mesh_mapper=ttnn.ReplicateTensorToMesh(md))


def from_dev(x, md):
    return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[:1]


def check(name, golden, computed):
    passing, msg = comp_pcc(torch.as_tensor(golden).float(), computed.float(), PCC)
    print(f"[{name}] PCC {msg}  ->  {'PASS' if passing else 'FAIL'}")
    return passing


# ------------------------------------------------------------------ gates
def gate_A1(md, g):
    """fc(25600->5120) + hidden_norm RMSNorm."""
    s = sd()
    fc_w = to_dev(s["fc.weight"].T.contiguous(), md)  # (25600,5120)
    norm_w = to_dev(s["hidden_norm.weight"], md)  # (1,5120)
    x = to_dev(torch.from_numpy(g["target_hidden_cat"]), md)  # (1,C,25600)
    h = lin(x, fc_w)  # (1,C,5120)
    ok_fc = check("A1.fc", g["fc_out"], from_dev(h, md))
    h = rms(h, weight=norm_w, epsilon=1e-6)
    ok_norm = check("A1.hidden_norm", g["hidden_norm_out"], from_dev(h, md))
    return ok_fc and ok_norm


def _expansion_matrix(groups=320, gsz=16, H=5120):
    E = torch.zeros(groups, H)
    for g in range(groups):
        E[g, g * gsz : (g + 1) * gsz] = 1.0
    return E


def _shift_matrix(L):
    """S (1,L,L) lower shift: (S@x)[t]=x[t-1], row 0 = 0."""
    S = torch.zeros(1, L, L)
    for t in range(1, L):
        S[0, t, t - 1] = 1.0
    return S


def _gdc(values, kp, base0, base1, E, S, sel, L, H=5120):
    """One grouped-dynamic causal conv. values=(1,L,H) to convolve; kp=kproj(input)=(1,L,1280);
    base0/base1=(1,1,H) static taps for branch `sel`; E=(320,H) group->channel expansion; S=shift."""
    off = sel * 640
    dyn0 = ttnn.slice(kp, [0, 0, off], [1, L, off + 320])  # (1,L,320) tap0
    dyn1 = ttnn.slice(kp, [0, 0, off + 320], [1, L, off + 640])  # (1,L,320) tap1
    c0 = ttnn.add(lin(dyn0, E), base0)  # (1,L,H)
    c1 = ttnn.add(lin(dyn1, E), base1)
    v1 = mm(S, values)  # causal shift
    return ttnn.add(ttnn.mul(c0, values), ttnn.mul(c1, v1))


def gate_A2(md, g):
    """GroupedDynamicCausalConv (layer0 attention_conv): prepare + finish vs torch reference."""
    import sys as _sys

    _sys.path.insert(0, os.environ.get("DFLASH_REPO", "/tmp/dflash_src"))
    from dflash.model import GroupedDynamicCausalConv

    s = sd()
    pfx = "layers.0.attention_conv"
    ref = GroupedDynamicCausalConv(H, 2, 16).to(torch.float32).eval()
    ref.load_state_dict(
        {
            "base_kernel": s[f"{pfx}.base_kernel"].float(),
            "kernel_projection.weight": s[f"{pfx}.kernel_projection.weight"].float(),
        }
    )
    x_t = torch.from_numpy(g["noise_embedding"]).float()  # (1,8,5120)
    L = x_t.shape[1]
    with torch.no_grad():
        cx_t, saved_t = ref.prepare(x_t)
        # finish is applied to a DIFFERENT tensor (attn output); use a synthetic y for the test.
        torch.manual_seed(1)
        y_t = torch.randn_like(x_t) * 0.1
        fx_t = ref.finish(y_t, saved_t)

    E = to_dev(_expansion_matrix(), md)
    kp_w = to_dev(s[f"{pfx}.kernel_projection.weight"].T.contiguous(), md)
    bk = s[f"{pfx}.base_kernel"].float()
    b00 = to_dev(bk[0, 0].reshape(1, 1, H), md)
    b01 = to_dev(bk[0, 1].reshape(1, 1, H), md)
    b10 = to_dev(bk[1, 0].reshape(1, 1, H), md)
    b11 = to_dev(bk[1, 1].reshape(1, 1, H), md)
    x = to_dev(x_t, md)
    y = to_dev(y_t, md)
    S = to_dev(_shift_matrix(L), md)
    kp = lin(x, kp_w)  # kproj(input) once
    cx = _gdc(x, kp, b00, b01, E, S, 0, L)  # prepare (base[0], dyn sel=0)
    fx = _gdc(y, kp, b10, b11, E, S, 1, L)  # finish  (base[1], dyn sel=1)
    ok1 = check("A2.prepare", cx_t, from_dev(cx, md))
    ok2 = check("A2.finish", fx_t, from_dev(fx, md))
    return ok1 and ok2


def _rotate_half(x):
    sh = list(x.shape)
    d = sh[-1] // 2
    x1 = ttnn.slice(x, [0] * len(sh), sh[:-1] + [d])
    x2 = ttnn.slice(x, [0] * (len(sh) - 1) + [d], sh)
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _apply_rope(x, cos, sin):
    return ttnn.add(ttnn.mul(x, cos), ttnn.mul(_rotate_half(x), sin))


def gate_A3(md, g):
    """Draft cross+self attention (layer0): q=block, kv=[ctx++block], bidirectional GQA, dual-pos RoPE."""
    s = sd()
    p = "layers.0.self_attn"
    qw = to_dev(s[f"{p}.q_proj.weight"].T.contiguous(), md)
    kw = to_dev(s[f"{p}.k_proj.weight"].T.contiguous(), md)
    vw = to_dev(s[f"{p}.v_proj.weight"].T.contiguous(), md)
    ow = to_dev(s[f"{p}.o_proj.weight"].T.contiguous(), md)
    qn = to_dev(s[f"{p}.q_norm.weight"], md)
    kn = to_dev(s[f"{p}.k_norm.weight"], md)
    hidden = to_dev(torch.from_numpy(g["attn0_hidden"]), md)
    ctx = to_dev(torch.from_numpy(g["attn0_ctx"]), md)
    cos = to_dev(torch.from_numpy(g["attn0_cos"]), md)
    sin = to_dev(torch.from_numpy(g["attn0_sin"]), md)
    L = g["attn0_hidden"].shape[1]
    C = g["attn0_ctx"].shape[1]
    NH, NKV, HD = 32, 8, 128
    q = rms(ttnn.reshape(lin(hidden, qw), (1, L, NH, HD)), weight=qn, epsilon=1e-6)
    q = ttnn.transpose(q, 1, 2)  # (1,32,L,128)
    k = ttnn.concat([lin(ctx, kw), lin(hidden, kw)], dim=1)
    k = rms(ttnn.reshape(k, (1, C + L, NKV, HD)), weight=kn, epsilon=1e-6)
    k = ttnn.transpose(k, 1, 2)  # (1,8,C+L,128)
    v = ttnn.concat([lin(ctx, vw), lin(hidden, vw)], dim=1)
    v = ttnn.transpose(ttnn.reshape(v, (1, C + L, NKV, HD)), 1, 2)  # (1,8,C+L,128)
    cos4 = ttnn.reshape(cos, (1, 1, C + L, HD))
    sin4 = ttnn.reshape(sin, (1, 1, C + L, HD))
    cosq = ttnn.slice(cos4, [0, 0, C, 0], [1, 1, C + L, HD])
    sinq = ttnn.slice(sin4, [0, 0, C, 0], [1, 1, C + L, HD])
    q = _apply_rope(q, cosq, sinq)
    k = _apply_rope(k, cos4, sin4)
    o = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=HD**-0.5, compute_kernel_config=CKC
    )
    o = ttnn.reshape(ttnn.transpose(o, 1, 2), (1, L, NH * HD))  # (1,L,4096)
    o = lin(o, ow)  # (1,L,5120)
    return check("A3.attn", g["attn0_out"], from_dev(o, md))


def _load_layer(s, i, md):
    p = f"layers.{i}"
    lw = {
        "in_ln": to_dev(s[f"{p}.input_layernorm.weight"], md),
        "post_ln": to_dev(s[f"{p}.post_attention_layernorm.weight"], md),
    }
    for k in ("q", "k", "v", "o"):
        lw[k] = to_dev(s[f"{p}.self_attn.{k}_proj.weight"].T.contiguous(), md)
    lw["qn"] = to_dev(s[f"{p}.self_attn.q_norm.weight"], md)
    lw["kn"] = to_dev(s[f"{p}.self_attn.k_norm.weight"], md)
    for tag, cp in (("ac", f"{p}.attention_conv"), ("mc", f"{p}.mlp_conv")):
        lw[f"{tag}_kp"] = to_dev(s[f"{cp}.kernel_projection.weight"].T.contiguous(), md)
        bk = s[f"{cp}.base_kernel"].float()
        lw[f"{tag}_b00"] = to_dev(bk[0, 0].reshape(1, 1, H), md)
        lw[f"{tag}_b01"] = to_dev(bk[0, 1].reshape(1, 1, H), md)
        lw[f"{tag}_b10"] = to_dev(bk[1, 0].reshape(1, 1, H), md)
        lw[f"{tag}_b11"] = to_dev(bk[1, 1].reshape(1, 1, H), md)
    for k in ("gate", "up", "down"):
        lw[k] = to_dev(s[f"{p}.mlp.{k}_proj.weight"].T.contiguous(), md)
    return lw


def _attn(hidden, ctx, cos, sin, lw, L, C):
    NH, NKV, HD = 32, 8, 128
    q = rms(ttnn.reshape(lin(hidden, lw["q"]), (1, L, NH, HD)), weight=lw["qn"], epsilon=1e-6)
    q = ttnn.transpose(q, 1, 2)
    k = ttnn.concat([lin(ctx, lw["k"]), lin(hidden, lw["k"])], dim=1)
    k = rms(ttnn.reshape(k, (1, C + L, NKV, HD)), weight=lw["kn"], epsilon=1e-6)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(
        ttnn.reshape(ttnn.concat([lin(ctx, lw["v"]), lin(hidden, lw["v"])], dim=1), (1, C + L, NKV, HD)), 1, 2
    )
    cos4 = ttnn.reshape(cos, (1, 1, C + L, HD))
    sin4 = ttnn.reshape(sin, (1, 1, C + L, HD))
    cosq = ttnn.slice(cos4, [0, 0, C, 0], [1, 1, C + L, HD])
    sinq = ttnn.slice(sin4, [0, 0, C, 0], [1, 1, C + L, HD])
    q = _apply_rope(q, cosq, sinq)
    k = _apply_rope(k, cos4, sin4)
    o = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=HD**-0.5, compute_kernel_config=CKC
    )
    return lin(ttnn.reshape(ttnn.transpose(o, 1, 2), (1, L, NH * HD)), lw["o"])


def _mlp(x, lw):
    return lin(ttnn.mul(lin(x, lw["gate"], activation="silu"), lin(x, lw["up"])), lw["down"])


def _layer(hidden, ctx, cos, sin, lw, E, S, L, C):
    res = hidden
    h = rms(hidden, weight=lw["in_ln"], epsilon=1e-6)
    kp = lin(h, lw["ac_kp"])
    hc = _gdc(h, kp, lw["ac_b00"], lw["ac_b01"], E, S, 0, L)
    a = _attn(hc, ctx, cos, sin, lw, L, C)
    a = _gdc(a, kp, lw["ac_b10"], lw["ac_b11"], E, S, 1, L)
    h = ttnn.add(res, a)
    res = h
    h2 = rms(h, weight=lw["post_ln"], epsilon=1e-6)
    kpm = lin(h2, lw["mc_kp"])
    h2c = _gdc(h2, kpm, lw["mc_b00"], lw["mc_b01"], E, S, 0, L)
    m = _mlp(h2c, lw)
    m = _gdc(m, kpm, lw["mc_b10"], lw["mc_b11"], E, S, 1, L)
    return ttnn.add(res, m)


def gate_A4(md, g):
    """One full draft layer (layer0) vs golden layer0_out."""
    s = sd()
    lw = _load_layer(s, 0, md)
    L = g["noise_embedding"].shape[1]
    C = g["hidden_norm_out"].shape[1]
    E = to_dev(_expansion_matrix(), md)
    S = to_dev(_shift_matrix(L), md)
    hidden = to_dev(torch.from_numpy(g["noise_embedding"]), md)
    ctx = to_dev(torch.from_numpy(g["hidden_norm_out"]), md)
    cos = to_dev(torch.from_numpy(g["attn0_cos"]), md)
    sin = to_dev(torch.from_numpy(g["attn0_sin"]), md)
    out = _layer(hidden, ctx, cos, sin, lw, E, S, L, C)
    return check("A4.layer0", g["layer0_out"], from_dev(out, md))


def gate_A5(md, g):
    """Full 5-layer draft forward + final norm vs golden draft_hidden."""
    s = sd()
    layers = [_load_layer(s, i, md) for i in range(5)]
    norm_w = to_dev(s["norm.weight"], md)
    L = g["noise_embedding"].shape[1]
    C = g["hidden_norm_out"].shape[1]
    E = to_dev(_expansion_matrix(), md)
    S = to_dev(_shift_matrix(L), md)
    h = to_dev(torch.from_numpy(g["noise_embedding"]), md)
    ctx = to_dev(torch.from_numpy(g["hidden_norm_out"]), md)
    cos = to_dev(torch.from_numpy(g["attn0_cos"]), md)
    sin = to_dev(torch.from_numpy(g["attn0_sin"]), md)
    ok = True
    for i in range(5):
        h = _layer(h, ctx, cos, sin, layers[i], E, S, L, C)
        ok = check(f"A5.layer{i}", g[f"layer{i}_out"], from_dev(h, md)) and ok
    h = rms(h, weight=norm_w, epsilon=1e-6)
    ok = check("A5.draft_hidden_full", g["draft_hidden_full"], from_dev(h, md)) and ok
    return ok


def _draft_hidden(md, g, s):
    layers = [_load_layer(s, i, md) for i in range(5)]
    norm_w = to_dev(s["norm.weight"], md)
    L = g["noise_embedding"].shape[1]
    C = g["hidden_norm_out"].shape[1]
    E = to_dev(_expansion_matrix(), md)
    S = to_dev(_shift_matrix(L), md)
    h = to_dev(torch.from_numpy(g["noise_embedding"]), md)
    ctx = to_dev(torch.from_numpy(g["hidden_norm_out"]), md)
    cos = to_dev(torch.from_numpy(g["attn0_cos"]), md)
    sin = to_dev(torch.from_numpy(g["attn0_sin"]), md)
    for i in range(5):
        h = _layer(h, ctx, cos, sin, layers[i], E, S, L, C)
    return rms(h, weight=norm_w, epsilon=1e-6), L


def gate_A6(md, g):
    """Full drafter -> CandidateSelector tokens (the real proposal path), ttnn vs golden on real hidden."""
    import sys as _sys

    _sys.path.insert(0, os.environ.get("DFLASH_REPO", "/tmp/dflash_src"))
    from dflash.model import CandidateSelector
    from safetensors import safe_open

    class Cfg:
        selector_rank = 256
        selector_top_k = 16
        hidden_size = 5120
        vocab_size = 248320

    s = sd()
    sel = CandidateSelector(Cfg()).float().eval()
    sel.load_state_dict(
        {
            "predecessor_codebook.weight": s["candidate_selector.predecessor_codebook"].float(),
            "successor_codebook.weight": s["candidate_selector.successor_codebook"].float(),
            "hidden_projection.weight": s["candidate_selector.hidden_projection.weight"].float(),
        }
    )
    h, L = _draft_hidden(md, g, s)
    dh = from_dev(h, md)[:, 1 - L :, :].float()
    gh = torch.from_numpy(g["draft_hidden"]).float()
    real = np.load(os.environ["DFLASH_REAL"])
    anchor = torch.tensor([int(real["anchor"])])
    shard = "/home/ttuser/experiments/qwen36_27b/model_volume/weights/Qwen3.6-27B/model-00008-of-00015.safetensors"
    with safe_open(shard, framework="pt") as f:
        lm = f.get_tensor("lm_head.weight").float()
    with torch.no_grad():
        gt = sel.select(gh, gh @ lm.T, anchor, 0.0)[0][0]
        tt = sel.select(dh, dh @ lm.T, anchor, 0.0)[0][0]
    match = int((gt == tt).sum())
    print(f"[A6] selector golden={gt.tolist()}")
    print(f"[A6] selector ttnn  ={tt.tolist()}  match={match}/{L - 1}")
    return match >= (L - 1)


def gate_A7(md, g):
    """Greedy draft-token match (the real gate): does ttnn's draft_hidden yield the same lm_head
    argmax tokens as the golden? A heuristic drafter tolerates hidden noise iff tokens don't flip."""
    from safetensors import safe_open

    s = sd()
    h, L = _draft_hidden(md, g, s)
    dh = from_dev(h, md)[:, 1 - L :, :].float()  # (1,7,5120) ttnn draft positions
    gh = torch.from_numpy(g["draft_hidden"]).float()  # (1,7,5120) golden
    shard = f"{os.environ.get('TGT_W','/home/ttuser/experiments/qwen36_27b/model_volume/weights/Qwen3.6-27B')}/model-00008-of-00015.safetensors"
    with safe_open(shard, framework="pt") as f:
        lm = f.get_tensor("lm_head.weight").float()  # (248320,5120)
    gt = (gh @ lm.T).argmax(-1)[0]
    tt = (dh @ lm.T).argmax(-1)[0]
    match = int((gt == tt).sum())
    print(f"[A7] greedy tokens golden={gt.tolist()}")
    print(f"[A7] greedy tokens ttnn  ={tt.tolist()}  match={match}/{L - 1}")
    return match >= (L - 1)  # require all 7 to match


GATES = {"A1": gate_A1, "A2": gate_A2, "A3": gate_A3, "A4": gate_A4, "A5": gate_A5, "A6": gate_A6, "A7": gate_A7}


def main():
    gate = sys.argv[1] if len(sys.argv) > 1 else "A1"
    g = dict(np.load(GOLDEN))
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        ok = GATES[gate](md, g)
    finally:
        ttnn.close_mesh_device(md)
    print(f"\n=== gate {gate}: {'GREEN' if ok else 'RED'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
