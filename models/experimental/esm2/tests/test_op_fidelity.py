# SPDX-License-Identifier: MIT
"""Per-op device fidelity table on the real smoke input (2 layers + head).

The CPU rounding sim — even with the device-observed gelu truncation —
predicts hidden NRMSE ~1.6e-2 while the device measures 5.04e-2, and the
excess is already ~3x at depth 2 (sweep a7ee11b5: d2 8.98e-3). Remaining
suspects are ttnn SFPU approximations (exp in softmax, rsqrt in
layer_norm) whose FUNCTION error (not rounding) the sim never modeled.

This test builds the real backend for 2 sliced layers, replays the exact
device op sequence with an instrumented loop (every intermediate pulled to
host), computes the identical intermediates in fp32 (reference math —
v1 had a twin bug: rotary tables used raw freqs instead of cos/sin,. ln1 and q_pre were exact, isolating
the bug to the missing .cos()/.sin()), and prints a per-op whole-tensor
RMS-ratio table. The first op diverging beyond the ~1e-3 rounding scale is
the coherent source. Smoke input has 0 pads => attention mask is a no-op.
K is compared transposed ([B,H,D,L], the split helper's layout).
"""

from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from tt.esm2.config import Esm2TTConfig
from tt.esm2.loader import load_canonical_weights
from tt.esm2.reference_layers import position_ids_from_input_ids, rotate_half

NLAYER = 2


def rms_ratio(a, b):
    a = torch.as_tensor(a).double().reshape(-1)
    b = torch.as_tensor(b).double().reshape(-1)
    return float((a - b).norm() / b.norm())


def slice_weights(w, depth):
    return {k: v for k, v in w.items() if not (k.startswith("layers.") and int(k.split(".")[1]) >= depth)}


def twin_forward(cfg, w, ids, am):
    """fp32 reference math, dumping every intermediate for NLAYER layers."""
    d = {}
    eps = cfg.layer_norm_eps
    H, D = cfg.num_attention_heads, cfg.head_dim
    qs = float(D) ** -0.5
    emb = w["embeddings.word_embeddings.weight"]
    h = emb[ids]
    is_m = ids.eq(cfg.mask_token_id)
    h = h.masked_fill(is_m.unsqueeze(-1), 0.0)
    ratio = is_m.sum(-1).float() / am.sum(-1).float()
    x = h * (1.0 - cfg.mask_ratio_train) / (1.0 - ratio[:, None, None])
    inv_freq = 1.0 / (cfg.rotary_base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    pos = position_ids_from_input_ids(ids, cfg.pad_token_id)
    freqs = pos.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cat([freqs.cos()] * 2, -1).unsqueeze(1)  # [B,1,L,D]
    sin = torch.cat([freqs.sin()] * 2, -1).unsqueeze(1)
    for i in range(NLAYER):
        p = f"layers.{i}."
        ln1 = F.layer_norm(x, (cfg.hidden_size,), w[p + "ln_attn.weight"], w[p + "ln_attn.bias"], eps)
        q_pre = F.linear(ln1, w[p + "attn.q.weight"], w[p + "attn.q.bias"]) * qs
        k_pre = F.linear(ln1, w[p + "attn.k.weight"], w[p + "attn.k.bias"])
        v_pre = F.linear(ln1, w[p + "attn.v.weight"], w[p + "attn.v.bias"])
        qkv = torch.cat([F.linear(ln1, w[p + "attn.q.weight"], w[p + "attn.q.bias"]) * qs, k_pre, v_pre], dim=-1)
        B, L, _ = q_pre.shape
        q = q_pre.view(B, L, H, D).transpose(1, 2)
        k = k_pre.view(B, L, H, D).transpose(1, 2)
        v = v_pre.view(B, L, H, D).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = torch.matmul(q, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        o = torch.matmul(probs, v).transpose(1, 2).reshape(B, L, H * D)
        ao = F.linear(o, w[p + "attn_out.weight"], w[p + "attn_out.bias"])
        xmid = x + ao
        ln2 = F.layer_norm(xmid, (cfg.hidden_size,), w[p + "ln_ffn.weight"], w[p + "ln_ffn.bias"], eps)
        f1 = F.linear(ln2, w[p + "ffn1.weight"], w[p + "ffn1.bias"])
        g = F.gelu(f1)
        f2 = F.linear(g, w[p + "ffn2.weight"], w[p + "ffn2.bias"])
        xout = xmid + f2
        d[i] = dict(
            ln1=ln1,
            qkv=qkv,
            q=q,
            k=k,
            q_pre=q,
            v=v,
            scores=scores,
            probs=probs,
            o=o,
            ao=ao,
            xmid=xmid,
            ln2=ln2,
            f1=f1,
            g=g,
            f2=f2,
            xout=xout,
        )
        x = xout
    hidden = F.layer_norm(x, (cfg.hidden_size,), w["final_ln.weight"], w["final_ln.bias"], eps)
    g1 = F.linear(hidden, w["lm.dense.weight"], w["lm.dense.bias"])
    g2 = F.layer_norm(F.gelu(g1), (cfg.hidden_size,), w["lm.ln.weight"], w["lm.ln.bias"], eps)
    logits = F.linear(g2, emb) + w["lm.bias"]
    d["head"] = dict(hidden=hidden, g1=g1, g2g=F.gelu(g1), g2=g2, logits=logits)
    return d, x


def main():
    dnp = np.load("/input/inputs.npz")
    ids_np, am_np = dnp["short__input_ids"], dnp["short__attention_mask"]
    with open("/weights/config.json") as f:
        cfg0 = Esm2TTConfig.from_dict(json.load(f))
    wall = load_canonical_weights("/weights", cfg0)
    cfg = dataclasses.replace(cfg0, num_hidden_layers=NLAYER)
    w = slice_weights(wall, NLAYER)
    ids = torch.from_numpy(ids_np)
    am = torch.from_numpy(am_np)

    twin, _ = twin_forward(cfg, w, ids, am)

    # sanity: twin must agree with the trusted Esm2Model end-to-end
    from tt.esm2.reference_layers import Esm2Model

    ref = Esm2Model(cfg, weights=w).eval()
    with torch.no_grad():
        lr, hr = ref(ids, am)
    print(
        f"twin-vs-Esm2Model sanity: logits {rms_ratio(twin['head']['logits'], lr):.2e} "
        f"hidden {rms_ratio(twin['head']['hidden'], hr):.2e} (expect <1e-6)",
        flush=True,
    )

    from tt.esm2.ttnn_backend import TtnnEsm2

    import ttnn

    rows = []
    device = ttnn.open_device(device_id=0)
    try:
        tt = TtnnEsm2(cfg, w, device=device, precision="bf16").build()
        ttnn = tt.ttnn
        host = tt._host
        x = tt._dev(tt.host_embedding(ids, am), dtype=tt.stream_dtype)
        q_tables, k_tables = tt.rotary_tensors(ids)
        mask = tt.mask_tensor(am)
        for i in range(NLAYER):
            lyr = tt.layer_ops[i]
            eps = cfg.layer_norm_eps
            a = ttnn.layer_norm(tt._cast(x, tt.dtype), epsilon=eps, weight=lyr["ln_a_w"], bias=lyr["ln_a_b"])
            qkv = ttnn.linear(a, lyr["qkv_w"], bias=lyr["qkv_b"])
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=cfg.num_attention_heads)
            q = tt._rotary_apply(q, *q_tables)
            k = tt._rotary_apply_t(k, *k_tables)
            scores = ttnn.matmul(q, k)
            scores = ttnn.add(scores, mask)
            probs = ttnn.softmax(scores, dim=-1)
            o = ttnn.matmul(probs, v)
            o = ttnn.transformer.concatenate_heads(o)
            ao = ttnn.linear(o, lyr["ao_w"], bias=lyr["ao_b"])
            xmid = ttnn.add(x, tt._cast(ao, tt.stream_dtype))
            z = ttnn.layer_norm(tt._cast(xmid, tt.dtype), epsilon=eps, weight=lyr["ln_f_w"], bias=lyr["ln_f_b"])
            f1 = ttnn.linear(z, lyr["f1_w"], bias=lyr["f1_b"])
            g = ttnn.gelu(f1)
            f2 = ttnn.linear(g, lyr["f2_w"], bias=lyr["f2_b"])
            xout = ttnn.add(xmid, tt._cast(f2, tt.stream_dtype))
            dev = dict(
                ln1=host(a),
                qkv=host(qkv),
                q=host(q),
                kT=host(k),
                v=host(v),
                scores=host(scores),
                probs=host(probs),
                o=host(o),
                ao=host(ao),
                xmid=host(xmid),
                ln2=host(z),
                f1=host(f1),
                g=host(g),
                f2=host(f2),
                xout=host(xout),
            )
            t = twin[i]
            pairs = [
                ("ln1", t["ln1"]),
                ("qkv", t["qkv"]),
                ("q", t["q"]),
                ("kT", t["k"].transpose(-1, -2).contiguous()),
                ("v", t["v"]),
                ("scores", t["scores"]),
                ("probs", t["probs"]),
                ("o", t["o"]),
                ("ao", t["ao"]),
                ("xmid", t["xmid"]),
                ("ln2", t["ln2"]),
                ("f1", t["f1"]),
                ("g", t["g"]),
                ("f2", t["f2"]),
                ("xout", t["xout"]),
            ]
            for key, tv in pairs:
                rows.append((f"L{i}.{key}", rms_ratio(dev[key], tv)))
            x = xout
        hidden = ttnn.layer_norm(tt._cast(x, tt.dtype), epsilon=eps, weight=tt.final_w, bias=tt.final_b)
        g1 = ttnn.linear(hidden, tt.lm_d_w, bias=tt.lm_d_b)
        g2g = ttnn.gelu(g1)
        g2 = ttnn.layer_norm(g2g, epsilon=eps, weight=tt.lm_l_w, bias=tt.lm_l_b)
        logits = ttnn.linear(g2, tt.dec_w, bias=tt.lm_bias)
        th = twin["head"]
        for key, dv, tv in (
            ("hidden", hidden, th["hidden"]),
            ("g1", g1, th["g1"]),
            ("g2g", g2g, th["g2g"]),
            ("g2", g2, th["g2"]),
            ("logits", logits, th["logits"]),
        ):
            rows.append((f"head.{key}", rms_ratio(host(dv), tv)))
    finally:
        ttnn.close_device(device)

    print(f"{'tensor':14s} {'RMS ratio':>10s}   (modeled rounding scale ~1e-3)", flush=True)
    for name, v in rows:
        print(f"{name:14s} {v:10.3e}", flush=True)
    print("OP_FIDELITY_DONE", flush=True)


if __name__ == "__main__":
    main()
