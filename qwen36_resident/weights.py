# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B checkpoint -> the per-chip weights of the resident decode model (see model.py).

TP over n chips: chip c holds GDN key heads [4c, 4c+4) and value heads [12c, 12c+12), attention query heads
[6c, 6c+6) and KV head c, MLP columns [c*I/n, (c+1)*I/n) and lm_head vocab columns [c*V/n, (c+1)*V/n).
The (1 + w) RMSNorm weights are folded into the rows of the projection that follows them.
"""
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

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
        self._files = {}

    def get(self, name):
        f = self.index[name]
        if f not in self._files:
            self._files[f] = safe_open(os.path.join(self.path, f), framework="pt")
        return self._files[f].get_tensor(name)

    def layer(self, i, name):
        return self.get(f"{PREFIX}layers.{i}.{name}")


def _fold(norm_w, w_t):
    """rows of w_t [K, N] scaled by (1 + norm_w) [K], as bf16"""
    return ((1.0 + norm_w.float())[:, None] * w_t.float()).bfloat16()


def gdn_layer(ck, i, d):
    """per-chip GDN mixer weights of layer i (the input rmsnorm folded into the projection)"""
    ln = ck.layer(i, "input_layernorm.weight")
    qkv = ck.layer(i, "linear_attn.in_proj_qkv.weight")  # [2 KD + VD, H]: q | k | v
    z = ck.layer(i, "linear_attn.in_proj_z.weight")
    a = ck.layer(i, "linear_attn.in_proj_a.weight")
    b = ck.layer(i, "linear_attn.in_proj_b.weight")
    conv = ck.layer(i, "linear_attn.conv1d.weight")[:, 0]  # [conv_dim, 4], tap 0 = oldest input
    dt = ck.layer(i, "linear_attn.dt_bias").float()
    neg_a = -ck.layer(i, "linear_attn.A_log").float().exp()
    kd, vd = NK * DK, NV * DV
    chips = []
    for c in range(d.n):
        q_rows = slice(c * d.gq, (c + 1) * d.gq)
        k_rows = slice(kd + c * d.gq, kd + (c + 1) * d.gq)
        v_rows = slice(2 * kd + c * d.gv, 2 * kd + (c + 1) * d.gv)
        heads = slice(c * d.nv, (c + 1) * d.nv)
        rows = torch.cat([qkv[q_rows], qkv[k_rows], qkv[v_rows], z[c * d.gv : (c + 1) * d.gv], a[heads], b[heads]])
        Wq = torch.zeros(rows.shape[1], d.g_cols, dtype=torch.bfloat16)
        Wq[:, : rows.shape[0]] = _fold(ln, rows.T)
        taps = torch.cat([conv[q_rows], conv[k_rows], conv[v_rows]]).float().bfloat16().float()
        chips.append(dict(Wq=Wq, taps=taps, dt=dt[heads], neg_a=neg_a[heads]))
    return dict(chips=chips, norm_w=ck.layer(i, "linear_attn.norm.weight").float().bfloat16().float())


def attn_layer(ck, i, d):
    """per-chip attention mixer weights of layer i: [q heads | gates | k | v] columns"""
    ln = ck.layer(i, "input_layernorm.weight")
    qg = ck.layer(i, "self_attn.q_proj.weight").reshape(NQ, 2, HD, -1)  # per head [q | gate]
    k = ck.layer(i, "self_attn.k_proj.weight")
    v = ck.layer(i, "self_attn.v_proj.weight")
    chips = []
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
        chips.append(dict(Wq=Wq))
    wq = (1.0 + ck.layer(i, "self_attn.q_norm.weight").float()).bfloat16().float()
    wk = (1.0 + ck.layer(i, "self_attn.k_norm.weight").float()).bfloat16().float()
    return dict(chips=chips, wq=wq, wk=wk)


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


def load(ck, d, layers, interval, with_head=True):
    """the resident model's weight dict for the first `layers` layers (every layer its own copy)"""
    w = dict(gdn=[], attn=[], mlp=[], out=[])
    for i in range(layers):
        attn = is_attn(i, interval)
        (w["attn"] if attn else w["gdn"]).append(attn_layer(ck, i, d) if attn else gdn_layer(ck, i, d))
        w["out"].append(out_proj(ck, i, d, attn))
        w["mlp"].append(mlp(ck, i, d))
    if with_head:
        w["head"], w["vocab_per_chip"] = lm_head(ck, d)
    return w


def embedding(ck):
    return ck.get(f"{PREFIX}embed_tokens.weight")
