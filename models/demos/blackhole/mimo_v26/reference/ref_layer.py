"""Independent CPU oracle for one MiMo-V2.6-Flash decoder layer, written from the spec (config + modeling_mimo_v2.py semantics), NOT by calling
the HF module. Verified against the HF module in tests/test_ref_vs_hf.py. dtype: torch.float32 or torch.bfloat16 (matmuls in that dtype; norm, softmax,
router in fp32 like the HF code)."""
import torch
import torch.nn.functional as F

from .loader import is_swa

D = 4096
NH, HQK, HV = 64, 192, 128
ROPE_DIM = 64  # int(192 * 0.334)
WINDOW = 128
V_SCALE = 0.707
EPS = 1e-6
TOPK, NEXP = 8, 256


def rms_norm(x, w, eps=EPS):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(x.dtype)


def rope_tables(S, theta, device="cpu", start=0):
    pos = torch.arange(start, start + S, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM))
    f = pos[:, None] * inv[None]
    e = torch.cat([f, f], -1)
    return e.cos(), e.sin()  # [S,64] fp32


def rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-b, a], -1)


def apply_rope_partial(x, cos, sin):
    """x [B,H,S,192]: rotate the FIRST 64 dims (split-half rotate_half), pass the other 128 through."""
    r = x[..., :ROPE_DIM]
    r = r * cos.to(x.dtype) + rotate_half(r) * sin.to(x.dtype)
    return torch.cat([r, x[..., ROPE_DIM:]], -1)


def attn_mask(Sq, Sk, swa, q0=0):
    """bool [Sq,Sk] True=visible. query position i=q0+row, key j: j<=i; SWA also j>i-128 (window 128 INCLUDING self)."""
    i = torch.arange(q0, q0 + Sq)[:, None]
    j = torch.arange(Sk)[None]
    m = j <= i
    if swa:
        m &= j > i - WINDOW
    return m


def attention(x, sd, i, kv_past=None, q0=0):
    """x [B,S,D] already normed. Returns (out [B,S,D], k [B,nkv,S,192] post-rope, v [B,nkv,S,128] already * 0.707)."""
    swa = is_swa(i)
    nkv = 8 if swa else 4
    B, S, _ = x.shape
    qkv = x @ sd["self_attn.qkv_proj.weight"].T
    q, k, v = qkv.split([NH * HQK, nkv * HQK, nkv * HV], -1)
    q = q.view(B, S, NH, HQK).transpose(1, 2)
    k = k.view(B, S, nkv, HQK).transpose(1, 2)
    v = v.view(B, S, nkv, HV).transpose(1, 2) * V_SCALE
    cos, sin = rope_tables(S, 1e4 if swa else 1e7, start=q0)
    q, k = apply_rope_partial(q, cos, sin), apply_rope_partial(k, cos, sin)
    kf, vf = k, v
    if kv_past is not None:
        kf, vf = torch.cat([kv_past[0], k], 2), torch.cat([kv_past[1], v], 2)
    rep = NH // nkv
    ke, ve = kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1)  # q head h -> kv head h // rep
    s = (q @ ke.transpose(2, 3)).float() * HQK**-0.5
    m = attn_mask(S, kf.shape[2], swa, q0)
    s = s.masked_fill(~m, float("-inf"))
    if swa:
        sink = sd["self_attn.attention_sink_bias"].float().view(1, NH, 1, 1).expand(B, NH, S, 1)
        s = torch.cat([s, sink], -1)  # raw logit, added AFTER the 192**-0.5 scaling
    p = F.softmax(s - s.max(-1, keepdim=True).values, -1)
    if swa:
        p = p[..., :-1]
    o = (p.to(x.dtype) @ ve).transpose(1, 2).reshape(B, S, NH * HV)
    return o @ sd["self_attn.o_proj.weight"].T, kf if kv_past is None else k, vf if kv_past is None else v


def router(x, sd):
    """x [T,D]. fp32 sigmoid scores; experts chosen on score+bias, weights from the UNBIASED scores renormalised to sum 1 (no scaling)."""
    logits = x.float() @ sd["mlp.gate.weight"].float().T
    s = logits.sigmoid()
    sel = s + sd["mlp.gate.e_score_correction_bias"].float()
    top = sel.topk(TOPK + 1, -1)
    idx = top.indices[:, :TOPK]
    w = s.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    gap = (
        top.values[:, TOPK - 1] - top.values[:, TOPK]
    )  # margin between the 8th and 9th ranked expert (near-tie indicator)
    return idx, w, gap


def moe(x, sd):
    T = x.shape[0]
    idx, w, gap = router(x, sd)
    out = torch.zeros(T, D, dtype=torch.float32)
    flat = idx.flatten()
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=NEXP).tolist()
    tok = order // TOPK
    wsorted = w.flatten()[order]
    p = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        t = tok[p : p + c]
        xe = x[t]
        g = xe @ sd[f"mlp.experts.{e}.gate_proj.weight"].T
        u = xe @ sd[f"mlp.experts.{e}.up_proj.weight"].T
        y = (F.silu(g) * u) @ sd[f"mlp.experts.{e}.down_proj.weight"].T
        out.index_add_(0, t, y.float() * wsorted[p : p + c, None])
        p += c
    return out.to(x.dtype), idx, w, gap


def dense(x, sd):
    return (F.silu(x @ sd["mlp.gate_proj.weight"].T) * (x @ sd["mlp.up_proj.weight"].T)) @ sd["mlp.down_proj.weight"].T


@torch.no_grad()
def layer_forward(h, sd, i, kv_past=None, q0=0):
    """h [B,S,D] (dtype of the weights). Returns dict: attn_in, attn_out, ffn_in, ffn_out, h_out, k, v, routing_idx, routing_w, route_gap."""
    r = {}
    r["attn_in"] = rms_norm(h, sd["input_layernorm.weight"])
    r["attn_out"], r["k"], r["v"] = attention(r["attn_in"], sd, i, kv_past, q0)
    h1 = h + r["attn_out"]
    r["ffn_in"] = rms_norm(h1, sd["post_attention_layernorm.weight"])
    if i == 0:
        r["ffn_out"] = dense(r["ffn_in"], sd)
    else:
        B, S, _ = h.shape
        o, idx, w, gap = moe(r["ffn_in"].reshape(-1, D), sd)
        r["ffn_out"] = o.view(B, S, D)
        r["routing_idx"], r["routing_w"], r["route_gap"] = idx, w, gap
    r["h_out"] = h1 + r["ffn_out"]
    return r
