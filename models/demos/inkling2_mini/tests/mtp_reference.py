# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Torch (fp32) reference of the Inkling-2-Mini DSpark drafter, from `mtp.safetensors` + the main checkpoint.

Same math as tt/mtp.py and the CPU acceptance study (E9, port of vLLM `inkling2/nvidia/dspark.py`).
`Reference(ckpt).draft(seq, feats, p)` -> 7 greedy draft tokens for positions p+1..p+7, given the anchor seq[p]
and feats [>= p, 4, 2304] (taps of positions 0..p-1). shortcut: fp32 throughout (vLLM rounds some steps to bf16).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

V = 200058


class Reference:
    def __init__(self, ckpt="/work/models/Inkling-2-Mini"):
        ckpt = Path(ckpt)
        mc = json.loads((ckpt / "config.json").read_text())["mtp_config"]
        self.H, self.NH, self.NKV, self.D = (
            mc["hidden_size"],
            mc["num_attention_heads"],
            mc["num_key_value_heads"],
            mc["head_dim"],
        )
        self.eps, self.theta, self.win, self.blk, self.R = (
            mc["rms_norm_eps"],
            mc["rope_theta"],
            mc["sliding_window_size"],
            mc["block_size"],
            mc["markov_rank"],
        )
        self.local, self.nl = set(mc["local_layer_ids"]), mc["num_hidden_layers"]
        with safe_open(ckpt / "mtp.safetensors", "pt") as f:
            self.W = {k: f.get_tensor(k).float() for k in f.keys()}
        with safe_open(ckpt / "model-00001-of-00001.safetensors", "pt") as f:
            self.embed = f.get_tensor("model.llm.embed.weight")
            self.embed_norm = f.get_tensor("model.llm.embed_norm.weight").float()
            self.unembed = f.get_tensor("model.llm.unembed.weight")[:V].float()
        self.alpha = torch.softmax(self.W["fuse_weight"], -1)
        self.fold_w = self.unembed @ self.W["markov_head.hidden_out.weight"]
        self.fold_b = self.unembed @ self.W["markov_head.hidden_out.bias"]
        self.inv = 1.0 / self.theta ** (torch.arange(0, self.D, 2).float() / self.D)
        self._kv, self._kv_n = None, 0

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, pos):
        ang = pos.float()[:, None] * self.inv[None]
        c, s = ang.cos()[:, None], ang.sin()[:, None]
        h = self.D // 2
        return torch.cat([x[..., :h] * c - x[..., h:] * s, x[..., h:] * c + x[..., :h] * s], -1)

    def mix(self, feats, row, w):
        return self.rms((feats.float() * self.alpha[row][None, :, None]).sum(1), w)

    def L(self, i, n):
        return self.W[f"layers.{i}.{n}"]

    def ctx_kv(self, feats):
        pos = torch.arange(feats.shape[0])
        out = []
        for i in range(self.nl):
            c = self.mix(feats, i, self.L(i, "dflare_norm.weight"))
            k = self.rms(
                (c @ self.L(i, "self_attn.k_proj_ctx.weight").T).view(-1, self.NKV, self.D),
                self.L(i, "self_attn.k_norm_ctx.weight"),
            )
            v = (c @ self.L(i, "self_attn.v_proj_ctx.weight").T).view(-1, self.NKV, self.D)
            out.append((self.rope(k, pos), v))
        return out

    @torch.no_grad()
    def draft(self, seq, feats, p, kv=None):
        kv = kv or self.ctx_kv(feats[:p])
        B, H, NH, NKV, D = self.blk, self.H, self.NH, self.NKV, self.D
        pos = torch.arange(p, p + B)
        x = self.W["mask_embed"].expand(B, H).clone()
        x[0] = self.rms(self.embed[seq[p]].float(), self.embed_norm)
        res = x
        for i in range(self.nl):
            h = self.rms(res, self.L(i, "input_layernorm.weight"))
            q = self.rope(
                self.rms(
                    (h @ self.L(i, "self_attn.q_proj.weight").T).view(B, NH, D), self.L(i, "self_attn.q_norm.weight")
                ),
                pos,
            )
            k = self.rope(
                self.rms(
                    (h @ self.L(i, "self_attn.k_proj.weight").T).view(B, NKV, D), self.L(i, "self_attn.k_norm.weight")
                ),
                pos,
            )
            v = (h @ self.L(i, "self_attn.v_proj.weight").T).view(B, NKV, D)
            lo = max(0, p - self.win) if i in self.local else 0
            K = torch.cat([kv[i][0][lo:p], k]).repeat_interleave(NH // NKV, 1)
            Vv = torch.cat([kv[i][1][lo:p], v]).repeat_interleave(NH // NKV, 1)
            a = torch.softmax(torch.einsum("qhd,khd->hqk", q, K) / math.sqrt(D), -1)
            o = torch.einsum("hqk,khd->qhd", a, Vv).reshape(B, NH * D)
            res = res + o @ self.L(i, "self_attn.o_proj.weight").T
            h = self.rms(res, self.L(i, "post_attention_layernorm.weight"))
            g = h @ self.L(i, "mlp.gate_proj.weight").T
            res = (
                res
                + (torch.nn.functional.silu(g) * (h @ self.L(i, "mlp.up_proj.weight").T))
                @ self.L(i, "mlp.down_proj.weight").T
            )
        hid = self.rms(res, self.W["norm.weight"])[1:]
        base = hid @ self.unembed.T
        f_rnn = self.mix(feats[p - 1 : p], -1, self.W["markov_head.dflare_norm.weight"])[0]
        Wr, br = self.W["markov_head.rnn_proj.weight"], self.W["markov_head.rnn_proj.bias"]
        state, prev, out = torch.zeros(self.R), seq[p], []
        for i in range(B - 1):
            raw = br + Wr @ torch.cat([state, self.W["markov_head.markov_w1.weight"][prev], hid[i], f_rnn])
            gate, cand, o = raw.split(self.R)
            gate = torch.sigmoid(gate)
            state = gate * state + (1 - gate) * torch.tanh(cand)
            logits = 30.0 * torch.tanh((base[i] + self.fold_w @ torch.tanh(o) + self.fold_b) / 30.0)
            prev = int(logits.argmax())
            out.append(prev)
        return out
