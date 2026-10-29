# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B checkpoint -> the per-chip weights of the resident decode model (see model.py).

TP over n chips: chip c holds GDN key heads [4c, 4c+4) and value heads [12c, 12c+12), attention query heads
[6c, 6c+6) and KV head c, MLP columns [c*I/n, (c+1)*I/n) and lm_head vocab columns [c*V/n, (c+1)*V/n).
The (1 + w) RMSNorm weights are folded into the rows of the projection that follows them.

load(lazy=True) (the default) defers the big matrices (Deferred): the resident model reads each layer's from the
checkpoint as it places it, so the host holds about one layer's at a time instead of the whole model (~50 GB).
"""
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from qwen36_resident.common import Deferred
from qwen36_resident.model import DK, DV, HD, NK, NQ, NV, is_attn

PREFIX = "model.language_model."


def checkpoint_dir(model=None, revision=None):
    """a local checkpoint directory: HF_MODEL (a path or a repo id) else Qwen/Qwen3.6-27B, from the HF cache"""
    model = model or os.environ.get("HF_MODEL", "Qwen/Qwen3.6-27B")
    if os.path.isdir(model):
        return model
    return snapshot_download(model, revision=revision, local_files_only=True)


class Checkpoint:
    def __init__(self, path=None):
        self.path = path or checkpoint_dir()
        self.index = json.load(open(os.path.join(self.path, "model.safetensors.index.json")))["weight_map"]
        self.config = json.load(open(os.path.join(self.path, "config.json")))["text_config"]

    def get(self, name):
        # a handle per read (~1 ms): an open handle keeps the mmap'd pages it has read in this process's RSS,
        # which over a whole load is the whole checkpoint (~54 GB); get_tensor returns a copy
        with safe_open(os.path.join(self.path, self.index[name]), framework="pt") as f:
            return f.get_tensor(name)

    def layer(self, i, name):
        return self.get(f"{PREFIX}layers.{i}.{name}")


def _fold(norm_w, w_t):
    """rows of w_t [K, N] scaled by (1 + norm_w) [K], as bf16"""
    return ((1.0 + norm_w.float())[:, None] * w_t.float()).bfloat16()


def _deferred(key, fn, n):
    return [Deferred(key, fn, c) for c in range(n)]


def gdn_proj(ck, i, d):
    """per-chip GDN projections of layer i: [q | k | v | z | a | b] columns (the input rmsnorm folded in)"""
    ln = ck.layer(i, "input_layernorm.weight")
    qkv = ck.layer(i, "linear_attn.in_proj_qkv.weight")  # [2 KD + VD, H]: q | k | v
    z = ck.layer(i, "linear_attn.in_proj_z.weight")
    a = ck.layer(i, "linear_attn.in_proj_a.weight")
    b = ck.layer(i, "linear_attn.in_proj_b.weight")
    kd = NK * DK
    out = []
    for c in range(d.n):
        q_rows = slice(c * d.gq, (c + 1) * d.gq)
        k_rows = slice(kd + c * d.gq, kd + (c + 1) * d.gq)
        v_rows = slice(2 * kd + c * d.gv, 2 * kd + (c + 1) * d.gv)
        heads = slice(c * d.nv, (c + 1) * d.nv)
        rows = torch.cat([qkv[q_rows], qkv[k_rows], qkv[v_rows], z[c * d.gv : (c + 1) * d.gv], a[heads], b[heads]])
        Wq = torch.zeros(rows.shape[1], d.g_cols, dtype=torch.bfloat16)
        Wq[:, : rows.shape[0]] = _fold(ln, rows.T)
        out.append(Wq)
    return out


def gdn_layer(ck, i, d, lazy=False):
    """per-chip GDN mixer weights of layer i (the input rmsnorm folded into the projection)"""
    conv = ck.layer(i, "linear_attn.conv1d.weight")[:, 0]  # [conv_dim, 4], tap 0 = oldest input
    dt = ck.layer(i, "linear_attn.dt_bias").float()
    neg_a = -ck.layer(i, "linear_attn.A_log").float().exp()
    kd = NK * DK
    Wqs = _deferred(("gdn", i), lambda: gdn_proj(ck, i, d), d.n) if lazy else gdn_proj(ck, i, d)
    chips = []
    for c in range(d.n):
        q_rows = slice(c * d.gq, (c + 1) * d.gq)
        k_rows = slice(kd + c * d.gq, kd + (c + 1) * d.gq)
        v_rows = slice(2 * kd + c * d.gv, 2 * kd + (c + 1) * d.gv)
        heads = slice(c * d.nv, (c + 1) * d.nv)
        taps = torch.cat([conv[q_rows], conv[k_rows], conv[v_rows]]).float().bfloat16().float()
        chips.append(dict(Wq=Wqs[c], taps=taps, dt=dt[heads], neg_a=neg_a[heads]))
    return dict(chips=chips, norm_w=ck.layer(i, "linear_attn.norm.weight").float().bfloat16().float())


def attn_proj(ck, i, d):
    """per-chip attention projections of layer i: [q heads | gates | k | v] columns"""
    ln = ck.layer(i, "input_layernorm.weight")
    qg = ck.layer(i, "self_attn.q_proj.weight").reshape(NQ, 2, HD, -1)  # per head [q | gate]
    k = ck.layer(i, "self_attn.k_proj.weight")
    v = ck.layer(i, "self_attn.v_proj.weight")
    out = []
    for c in range(d.n):
        hs = slice(c * d.h, (c + 1) * d.h)
        rows = torch.cat(
            [
                qg[hs, 0].reshape(d.h * HD, -1),
                qg[hs, 1].reshape(d.h * HD, -1),
                k[c * HD : (c + 1) * HD],
                v[c * HD : (c + 1) * HD],
            ]
        )
        Wq = torch.zeros(rows.shape[1], d.a_cols, dtype=torch.bfloat16)
        Wq[:, : rows.shape[0]] = _fold(ln, rows.T)
        out.append(Wq)
    return out


def attn_layer(ck, i, d, lazy=False):
    """per-chip attention mixer weights of layer i: [q heads | gates | k | v] columns"""
    Wqs = _deferred(("attn", i), lambda: attn_proj(ck, i, d), d.n) if lazy else attn_proj(ck, i, d)
    wq = (1.0 + ck.layer(i, "self_attn.q_norm.weight").float()).bfloat16().float()
    wk = (1.0 + ck.layer(i, "self_attn.k_norm.weight").float()).bfloat16().float()
    return dict(chips=[dict(Wq=W) for W in Wqs], wq=wq, wk=wk)


def out_proj(ck, i, d, attn):
    w = ck.layer(i, "self_attn.o_proj.weight" if attn else "linear_attn.out_proj.weight")  # [H, V]
    return [w[:, c * d.gv : (c + 1) * d.gv].T.contiguous() for c in range(d.n)]


def mlp(ck, i, d):
    ln = ck.layer(i, "post_attention_layernorm.weight")
    g, u, dn = (ck.layer(i, f"mlp.{p}_proj.weight") for p in ("gate", "up", "down"))
    out = []
    for c in range(d.n):
        cols = slice(c * d.ic, (c + 1) * d.ic)
        out.append(dict(G=_fold(ln, g[cols].T), U=_fold(ln, u[cols].T), D=dn[:, cols].T.contiguous()))
    return out


def mlp_gate_up(ck, i, d):
    """per chip the [G | U] columns of layer i's MLP (what the resident model streams as one entry)"""
    ln = ck.layer(i, "post_attention_layernorm.weight")
    g, u = (ck.layer(i, f"mlp.{p}_proj.weight") for p in ("gate", "up"))
    cols = lambda c: slice(c * d.ic, (c + 1) * d.ic)
    return [torch.cat([_fold(ln, g[cols(c)].T), _fold(ln, u[cols(c)].T)], dim=1) for c in range(d.n)]


def mlp_down(ck, i, d):
    dn = ck.layer(i, "mlp.down_proj.weight")
    return [dn[:, c * d.ic : (c + 1) * d.ic].T.contiguous() for c in range(d.n)]


def mlp_lazy(ck, i, d):
    """mlp() with Deferred matrices; GU: the [G | U] columns (G and U are None)"""
    gu = _deferred(("gu", i), lambda: mlp_gate_up(ck, i, d), d.n)
    dn = _deferred(("down", i), lambda: mlp_down(ck, i, d), d.n)
    return [dict(G=None, U=None, GU=gu[c], D=dn[c]) for c in range(d.n)]


def lm_head(ck, d):
    ln = ck.get(f"{PREFIX}norm.weight")
    w = ck.get("lm_head.weight")  # [V, H]
    per = w.shape[0] // d.n
    heads = []
    for c in range(d.n):
        h = torch.zeros(w.shape[1], d.vocab, dtype=torch.bfloat16)
        h[:, :per] = _fold(ln, w[c * per : (c + 1) * per].T)
        heads.append(h)
    return heads, per


def load(ck, d, layers, interval, with_head=True, lazy=True):
    """the resident model's weight dict for the first `layers` layers (every layer its own copy); lazy: the big
    matrices Deferred (see the module docstring)"""
    w = dict(gdn=[], attn=[], mlp=[], out=[])
    for i in range(layers):
        attn = is_attn(i, interval)
        (w["attn"] if attn else w["gdn"]).append(attn_layer(ck, i, d, lazy) if attn else gdn_layer(ck, i, d, lazy))
        if lazy:
            w["out"].append(_deferred(("out", i), lambda i=i, attn=attn: out_proj(ck, i, d, attn), d.n))
            w["mlp"].append(mlp_lazy(ck, i, d))
        else:
            w["out"].append(out_proj(ck, i, d, attn))
            w["mlp"].append(mlp(ck, i, d))
    if with_head:
        if lazy:
            w["head"] = _deferred(("head",), lambda: lm_head(ck, d)[0], d.n)
            w["vocab_per_chip"] = ck.config["vocab_size"] // d.n
        else:
            w["head"], w["vocab_per_chip"] = lm_head(ck, d)
    return w


def embedding(ck):
    return ck.get(f"{PREFIX}embed_tokens.weight")
