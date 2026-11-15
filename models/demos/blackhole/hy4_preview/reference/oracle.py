# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fp32 CPU oracle for Hy4-Preview (HYV4ForCausalLM), written from config.json + tensor shapes + the
DeepSeek-V3.2 (DSA) / IndexCache / DeepSeek-V4 (hyper-connection) designs: the checkpoint ships no
modeling code. Every formula that could not be confirmed from files is a field of `Conv` (default =
the value taken) and is listed in RESULT.md as NEEDS-CONFIRMATION.

Shapes: x4 [B, S, hc=4, D] is the iHC residual stream; sublayer inputs are [B, S, D].
All math in float32; weights converted bf16 -> fp32 on load.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.hy4_preview.reference.loader import Hy4Checkpoint, split_gate_up


@dataclass(frozen=True)
class Conv:
    """Unconfirmed conventions. Defaults are the choices taken; variants are for end-to-end A/B."""

    rope_interleave: bool = True  # main attention RoPE: adjacent-pair (DeepSeek-V3 HF) vs half-split (neox)
    gate_up: str = "chunk"  # fused expert layout: chunk [gate;up] | interleave | swap
    # swiglu_limit handling: all = clamp(up,+-L) and clamp(gate,max=L) in routed+shared experts (DSV4 Expert);
    # none | routed | shared | gate (gate max only) | up (up +-L only) | all+dense (also layer-0 MLP)
    clamp_mode: str = "routed"
    attn_gate_in: str = "normed"  # input of linear_gate: normed (sublayer input after input_layernorm) | raw
    attn_gate_act: str = "sigmoid"
    hc_norm_eps: str = "rms"  # eps inside the hc mixes rsqrt: rms (config rms_norm_eps) | hc (hc_eps)
    hc_pre_eps: bool = True  # pre = sigmoid(..) + hc_eps
    idx_rope_first: bool = (
        False  # indexer RoPE on first 64 of 128 dims (GLM-DSA) vs last 64 (DSV4): indexer_check says LAST
    )
    idx_rope_interleave: bool = True  # indexer RoPE adjacent pair (weak evidence, indexer_check) vs half-split
    idx_ln_eps: float = 1e-6  # indexer k_norm is a LayerNorm with bias
    sink_in_denominator: bool = True  # learnable sink = extra logit column in the softmax, no value
    mtp_h_source: str = "normed"  # hidden fed to MTP: normed (post model.norm; CONFIRMED by variant_ab: agree 0.77 vs 0.0) | hc_head (pre-norm)
    mtp_cat: str = "e_h"  # eh_proj input order: cat(enorm(emb), hnorm(h)) | cat(hnorm(h), enorm(emb))


# ---------------------------------------------------------------- primitives
def rms_norm(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()


def layer_norm(x, w, b, eps):
    return F.layer_norm(x.float(), x.shape[-1:], w.float(), b.float(), eps)


def rope_cos_sin(S, dim, theta, start=0):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(start, start + S, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()  # [S, dim/2]


def apply_rope(x, cos, sin, interleave):
    """x [..., S, (H), dim] with S at dim 1 of [B,S,H,dim] or [B,S,dim]; cos/sin [S, dim/2]."""
    shape = [1, cos.shape[0]] + [1] * (x.dim() - 3) + [cos.shape[1]]
    c, s = cos.view(shape), sin.view(shape)
    if interleave:
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat((a * c - b * s, a * s + b * c), dim=-1)


# ---------------------------------------------------------------- iHC
def hc_mixes(x4, fn, scale, base, cfg, conv, n_out):
    """x4 [B,S,hc,D] -> raw mixes [B,S,n_out] before activation: linear(x_flat * rsqrt(mean(x_flat^2)+eps))."""
    xf = x4.flatten(2).float()
    eps = cfg.norm_eps if conv.hc_norm_eps == "rms" else cfg.hc_eps
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return F.linear(xf * r, fn.float())


def hc_pre_post(x4, hp, cfg, conv):
    """iHC (identity residual mix): pre [B,S,hc] collapses the streams, post [B,S,hc] expands the sublayer out."""
    m = hc_mixes(x4, hp["hc_fn"], hp["hc_scale"], hp["hc_base"], cfg, conv, 2 * cfg.hc_mult)
    h = cfg.hc_mult
    pre = torch.sigmoid(m[..., :h] * hp["hc_scale"][0] + hp["hc_base"][:h])
    if conv.hc_pre_eps:
        pre = pre + cfg.hc_eps
    post = cfg.hc_magnitude * torch.sigmoid(m[..., h : 2 * h] * hp["hc_scale"][1] + hp["hc_base"][h : 2 * h])
    return pre, post


def hc_collapse(x4, pre):
    return (pre.unsqueeze(-1) * x4.float()).sum(2)


def hc_expand(x4, y, post):
    """identity residual: stream_i <- stream_i + post_i * y"""
    return x4.float() + post.unsqueeze(-1) * y.unsqueeze(2)


def hc_head(x4, g, cfg, conv):
    xf = x4.flatten(2).float()
    eps = cfg.norm_eps if conv.hc_norm_eps == "rms" else cfg.hc_eps
    m = F.linear(xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps), g["model.hc_head.hc_head_fn"])
    w = torch.sigmoid(m * g["model.hc_head.hc_head_scale"][0] + g["model.hc_head.hc_head_base"]) + cfg.hc_eps
    return (w.unsqueeze(-1) * x4.float()).sum(2)


# ---------------------------------------------------------------- attention
def indexer_topk(h, qr, sd, cfg, conv, cos_i, sin_i):
    """DSA lightning indexer (full layers). h [B,S,D] sublayer input, qr [B,S,q_lora] (post q_a_layernorm).
    Returns idx [B,S,k] (k = min(topk,S)), -1 where the causal prefix is shorter than k, plus scores."""
    B, S, _ = h.shape
    nh, d, rd = cfg.idx_heads, cfg.idx_dim, cfg.rope
    q = F.linear(qr, sd["self_attn.indexer.wq_b.weight"]).view(B, S, nh, d)
    k = layer_norm(
        F.linear(h, sd["self_attn.indexer.wk.weight"]),
        sd["self_attn.indexer.k_norm.weight"],
        sd["self_attn.indexer.k_norm.bias"],
        conv.idx_ln_eps,
    )
    w = F.linear(h, sd["self_attn.indexer.weights_proj.weight"]) * (nh**-0.5) * (d**-0.5)

    def rot(t):
        sl = slice(0, rd) if conv.idx_rope_first else slice(d - rd, d)
        t = t.clone()
        t[..., sl] = apply_rope(t[..., sl], cos_i, sin_i, conv.idx_rope_interleave)
        return t

    q, k = rot(q), rot(k)  # (the V3.2 Hadamard rotation of q,k leaves q.k invariant in fp32: omitted)
    topk = min(cfg.idx_topk, S)
    out = torch.empty(B, S, topk, dtype=torch.int64)
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    for b in range(B):
        for s0 in range(0, S, 256):
            s1 = min(S, s0 + 256)
            sc = torch.einsum("shd,td->sht", q[b, s0:s1], k[b]).relu_()
            sc = (sc * w[b, s0:s1].unsqueeze(-1)).sum(1)  # [s,S]
            sc = sc.masked_fill(~causal[s0:s1], float("-inf"))
            v, ix = sc.topk(topk, dim=-1)
            ix = torch.where(torch.isfinite(v), ix, torch.full_like(ix, -1))
            out[b, s0:s1] = ix
    return out


def attention(h, sd, cfg, conv, topk_idx, probe=None):
    """Gated MLA + sparse selection + learnable sink. h [B,S,D] = input_layernorm(collapsed stream).
    topk_idx [B,S,k] (position ids, -1 = unused) or None for full causal (S <= index_topk)."""
    B, S, D = h.shape
    H, nope, rope, vd = cfg.n_heads, cfg.nope, cfg.rope, cfg.v_dim
    qk = nope + rope
    qa = rms_norm(F.linear(h, sd["self_attn.q_a_proj.weight"]), sd["self_attn.q_a_layernorm.weight"], cfg.norm_eps)
    q = F.linear(qa, sd["self_attn.q_b_proj.weight"]).view(B, S, H, qk)
    kv = F.linear(h, sd["self_attn.kv_a_proj_with_mqa.weight"])
    lat, k_pe = kv[..., : cfg.kv_lora], kv[..., cfg.kv_lora :]
    lat = rms_norm(lat, sd["self_attn.kv_a_layernorm.weight"], cfg.norm_eps)
    kvb = F.linear(lat, sd["self_attn.kv_b_proj.weight"]).view(B, S, H, nope + vd)
    k_nope, v = kvb[..., :nope], kvb[..., nope:]
    cos, sin = rope_cos_sin(S, rope, cfg.rope_theta)
    q_pe = apply_rope(q[..., nope:], cos, sin, conv.rope_interleave)
    k_pe = apply_rope(k_pe, cos, sin, conv.rope_interleave)
    qf = torch.cat([q[..., :nope], q_pe], -1)
    kf = torch.cat([k_nope, k_pe.unsqueeze(2).expand(B, S, H, rope)], -1)
    sink = sd["self_attn.learnable_sink_param"].float()  # [H]
    scale = qk**-0.5
    o = torch.empty(B, S, H, vd)
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    for b in range(B):
        for s0 in range(0, S, 256):
            s1 = min(S, s0 + 256)
            sc = torch.einsum("shd,thd->hst", qf[b, s0:s1], kf[b]) * scale  # [H,s,S]
            allowed = causal[s0:s1]
            if topk_idx is not None:
                sel = torch.zeros(s1 - s0, S + 1, dtype=torch.bool)  # column S = trash for unused (-1) slots
                ix = topk_idx[b, s0:s1]
                sel.scatter_(1, torch.where(ix >= 0, ix, torch.full_like(ix, S)), True)
                sel = sel[:, :S]
                allowed = allowed & sel
            sc = sc.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            if conv.sink_in_denominator:
                full = torch.cat([sc, sink.view(H, 1, 1).expand(H, s1 - s0, 1)], -1)
                p = full.softmax(-1)[..., :S]
            else:
                p = sc.softmax(-1)
            if probe is not None:
                probe(b, s0, s1, p)
            o[b, s0:s1] = torch.einsum("hst,thd->shd", p, v[b])
    o = o.reshape(B, S, H * vd)
    g_in = h if conv.attn_gate_in == "normed" else None
    return o, g_in


def attn_block(h, hraw, sd, cfg, conv, topk_idx, probe=None):
    o, g_in = attention(h, sd, cfg, conv, topk_idx, probe)
    g_in = h if g_in is not None else hraw
    gate = F.linear(g_in, sd["self_attn.linear_gate.weight"])
    gate = torch.sigmoid(gate) if conv.attn_gate_act == "sigmoid" else gate
    return F.linear(o * gate, sd["self_attn.o_proj.weight"])


# ---------------------------------------------------------------- MLP / MoE
def swiglu(gate, up, limit, mode, site):
    """site in routed | shared | dense. See Conv.clamp_mode."""
    on = {
        "all": site in ("routed", "shared"),
        "all+dense": True,
        "routed": site == "routed",
        "shared": site == "shared",
        "gate": site in ("routed", "shared"),
        "up": site in ("routed", "shared"),
        "none": False,
    }[mode]
    if on and limit > 0:
        if mode != "gate":
            up = up.clamp(-limit, limit)
        if mode != "up":
            gate = gate.clamp(max=limit)
    return F.silu(gate) * up


def dense_mlp(h, sd, cfg, conv):
    g = F.linear(h, sd["mlp.gate_proj.weight"])
    u = F.linear(h, sd["mlp.up_proj.weight"])
    return F.linear(swiglu(g, u, cfg.swiglu_limit, conv.clamp_mode, "dense"), sd["mlp.down_proj.weight"])


def shared_expert(h, sd, cfg, conv):
    g = F.linear(h, sd["mlp.shared_experts.gate_proj.weight"])
    u = F.linear(h, sd["mlp.shared_experts.up_proj.weight"])
    return F.linear(
        swiglu(g, u, cfg.swiglu_limit, conv.clamp_mode, "shared"), sd["mlp.shared_experts.down_proj.weight"]
    )


def router(x, sd, cfg):
    """x [N,D] -> (weights [N,k] fp32, indices [N,k]). sigmoid scores; e_score_correction_bias picks, does not scale;
    n_group=topk_group=1 so no group masking; norm_topk_prob; x routed_scaling_factor."""
    scores = torch.sigmoid(F.linear(x.float(), sd["mlp.gate.weight"].float()))
    sel = (scores + sd["mlp.gate.e_score_correction_bias"].float()).topk(cfg.topk, dim=-1)[1]
    w = scores.gather(1, sel)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * cfg.routed_scale, sel


def moe_multi(ck, layer, hs, sds, cfg, convs):
    """MoE over several variant streams that share the same expert weights (each read once).
    hs: list of [B,S,D]; returns list of [B,S,D]."""
    shapes = [h.shape for h in hs]
    xs = [h.reshape(-1, h.shape[-1]).float() for h in hs]
    rts = [router(x, sd, cfg) for x, sd in zip(xs, sds)]
    ys = [torch.zeros_like(x) for x in xs]
    used = sorted(set(torch.cat([r[1].flatten() for r in rts]).tolist()))
    p = ck.layer_prefix(layer) + "mlp.experts."
    for e in used:
        gu = ck.get(p + "gate_up_proj", rows=e).float()
        dn = ck.get(p + "down_proj", rows=e).float()
        splits = {}
        for x, (w, sel), y, conv in zip(xs, rts, ys, convs):
            tok, slot = torch.where(sel == e)
            if tok.numel() == 0:
                continue
            if conv.gate_up not in splits:
                splits[conv.gate_up] = split_gate_up(gu, conv.gate_up)
            wg, wu = splits[conv.gate_up]
            xt = x[tok]
            a = swiglu(xt @ wg.T, xt @ wu.T, cfg.swiglu_limit, conv.clamp_mode, "routed")
            y.index_add_(0, tok, (a @ dn.T) * w[tok, slot].unsqueeze(-1))
    outs = []
    for y, x, sd, conv, shp in zip(ys, xs, sds, convs, shapes):
        outs.append((y + shared_expert(x, sd, cfg, conv)).view(shp))
    return outs, [r[1].view(*s[:2], -1) for r, s in zip(rts, shapes)]


# ---------------------------------------------------------------- layers
def hc_params(sd, which):
    p = f"hc_{which}_layer.hc_pre."
    return {"hc_fn": sd[p + "hc_fn"], "hc_scale": sd[p + "hc_scale"], "hc_base": sd[p + "hc_base"]}


def decoder_layers_multi(ck, layer, x4s, convs, idx_states, taps=None):
    """One iHC decoder layer for several variant streams sharing weights. x4s: list [B,S,4,D].
    idx_states: list of dicts {'idx': last full-indexer top-k}. Returns new x4s. taps: optional dict to fill."""
    cfg = ck.cfg
    sd = ck.layer_state(layer)
    dense = cfg.is_dense(layer)
    full = cfg.is_full_indexer(layer)
    # attention sublayer
    hs, raws, pres, posts = [], [], [], []
    for x4, conv in zip(x4s, convs):
        hp = hc_params(sd, "attn")
        pre, post = hc_pre_post(x4, hp, cfg, conv)
        raw = hc_collapse(x4, pre)
        hs.append(rms_norm(raw, sd["input_layernorm.weight"], cfg.norm_eps))
        raws.append(raw)
        posts.append(post)
    attn_outs = []
    for h, raw, conv, st in zip(hs, raws, convs, idx_states):
        S = h.shape[1]
        if full:
            qa = rms_norm(
                F.linear(h, sd["self_attn.q_a_proj.weight"]), sd["self_attn.q_a_layernorm.weight"], cfg.norm_eps
            )
            cos_i, sin_i = rope_cos_sin(S, cfg.rope, cfg.rope_theta)
            st["idx"] = indexer_topk(h, qa, sd, cfg, conv, cos_i, sin_i)
        idx = st.get("idx")
        idx = idx if (idx is not None and S > cfg.idx_topk) else None  # S <= topk: every causal key selected
        attn_outs.append(attn_block(h, raw, sd, cfg, conv, idx))
    x4s = [hc_expand(x4, o, p) for x4, o, p in zip(x4s, attn_outs, posts)]
    if taps is not None:
        taps["after_attn"] = [x.clone() for x in x4s]
    # mlp sublayer
    hs, posts = [], []
    for x4, conv in zip(x4s, convs):
        pre, post = hc_pre_post(x4, hc_params(sd, "mlp"), cfg, conv)
        hs.append(rms_norm(hc_collapse(x4, pre), sd["post_attention_layernorm.weight"], cfg.norm_eps))
        posts.append(post)
    if dense:
        ys = [dense_mlp(h, sd, cfg, c) for h, c in zip(hs, convs)]
        routes = None
    else:
        ys, routes = moe_multi(ck, layer, hs, [sd] * len(hs), cfg, convs)
    x4s = [hc_expand(x4, y, p) for x4, y, p in zip(x4s, ys, posts)]
    if taps is not None:
        taps["routes"] = routes
        taps["mlp_in"] = hs
    return x4s


def mtp_forward(ck, h_last, tok_next, conv, g=None, embed=None, zero_pos0=False):
    """MTP layer (model.mtp_layers.0): predicts token t+2 from h_t and token t+1.
    h_last [B,S,D] (main-model hidden per conv.mtp_h_source), tok_next [B,S] ids of tokens t+1.
    The layer has no iHC tensors: plain residual stream. Returns hidden after final_layernorm [B,S,D]."""
    cfg = ck.cfg
    sd = ck.layer_state(-1)
    emb = embed[tok_next].float()
    if zero_pos0:  # vLLM mtp.py: inputs_embeds zeroed where positions == 0 (position 0 = row 0 of the sequence)
        emb[:, 0] = 0
    e, hh = rms_norm(emb, sd["enorm.weight"], cfg.norm_eps), rms_norm(h_last, sd["hnorm.weight"], cfg.norm_eps)
    x = F.linear(torch.cat([e, hh] if conv.mtp_cat == "e_h" else [hh, e], -1), sd["eh_proj.weight"])
    S = x.shape[1]
    hn = rms_norm(x, sd["input_layernorm.weight"], cfg.norm_eps)
    qa = rms_norm(F.linear(hn, sd["self_attn.q_a_proj.weight"]), sd["self_attn.q_a_layernorm.weight"], cfg.norm_eps)
    cos_i, sin_i = rope_cos_sin(S, cfg.rope, cfg.rope_theta)
    idx = indexer_topk(hn, qa, sd, cfg, conv, cos_i, sin_i) if S > cfg.idx_topk else None
    x = x + attn_block(hn, hn, sd, cfg, conv, idx)
    hn = rms_norm(x, sd["post_attention_layernorm.weight"], cfg.norm_eps)
    y, _ = moe_multi(ck, -1, [hn], [sd], cfg, [conv])
    x = x + y[0]
    return rms_norm(x, sd["final_layernorm.weight"], cfg.norm_eps)


class Hy4Oracle:
    """Layer-streaming full model: weights of one layer in RAM at a time (<= ~3 GB fp32 + expert reads)."""

    def __init__(self, ckpt=None):
        self.ck = Hy4Checkpoint(ckpt) if ckpt else Hy4Checkpoint()
        self.cfg = self.ck.cfg
        self._embed = None
        self._lm_head = None

    @property
    def embed(self):
        if self._embed is None:
            self._embed = self.ck.get("model.embed_tokens.weight")
        return self._embed

    def embed_ids(self, ids):  # ids [B,S] -> stream [B,S,4,D]
        e = self.embed[ids].float()
        return e.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)

    def logits(self, hidden_normed):
        if self._lm_head is None:
            self._lm_head = self.ck.get("lm_head.weight")
        return F.linear(hidden_normed.float(), self._lm_head.float())

    def head(self, x4, conv):
        g = {k: v.float() for k, v in self.ck.global_state().items()}
        h = hc_head(x4, g, self.cfg, conv)
        return h, rms_norm(h, g["model.norm.weight"], self.cfg.norm_eps)
