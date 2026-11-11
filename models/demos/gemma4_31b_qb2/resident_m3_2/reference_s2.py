# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference of one resident decoder layer with real attention (step 2), from the same quantized weights as the
device (reference.py) and the KV cache read back from the device.

Per chip c (local q heads 8, sliding kv heads 4 / full kv head 1), M rows at positions p .. p + M - 1:
  qkv = bf16(h @ Wqkv_c);  q = bf16(rope(bf16(rmsnorm(q) w_q)));
  sliding: k = bf16(rope(bf16(rmsnorm(k) w_k))), v = bf16(rmsnorm(v));  full: v = bf16(rmsnorm(kv)),
  k = bf16(rope(bf16(v w_k)));  row r attends (scale 1, fp32 softmax) to positions p + r - 1023 .. p + r (sliding,
  ring of 9 pages, positions mod 1152) / 0 .. p + r (full), keys and values from the cache (the new rows as the
  device wrote them: the K / V write is checked separately against k, v above);
  o_c = bf16(attn);  then the step-1 reference's residual math.
"""
import torch

from models.demos.gemma4_31b_qb2.resident_m3_2.common import (
    E_DOWN,
    E_GU,
    E_O_F,
    E_O_S,
    E_QKV_F,
    E_QKV_S,
    IC,
    N_CHIPS,
    Q_F,
    Q_S,
    is_full,
)
from models.demos.gemma4_31b_qb2.resident_m3_2.reference import Reference, bf16, gelu_tanh, rms

KINDS = ("sliding_attention", "full_attention")


def rope(x, cos, sin):
    """x [..., M, D] with cos / sin [M, D]: rotate-half (HF apply_rotary_pos_emb)"""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos + rot * sin


class Reference2(Reference):
    def __init__(self, cache, layers, rope_tables, norms):
        """rope_tables: kind -> (cos, sin) float [positions, D] (bf16 values); norms: [layers] of (w_q, w_k)"""
        super().__init__(cache, layers)
        self.rope_tables, self.norms = rope_tables, norms

    def layer(self, l, x, p, caches, tables):
        """x [M, HIDDEN] -> (x', info); caches[c] = (K, V) float [pages, H, 128, D] after the device's step"""
        full = is_full(l)
        kind = KINDS[int(full)]
        M = x.shape[0]
        w = self.w[l]
        D, H = (512, 1) if full else (256, 4)
        qkv_w, o_w = (w[E_QKV_F], w[E_O_F]) if full else (w[E_QKV_S], w[E_O_S])
        cos, sin = (t[p : p + M].float() for t in self.rope_tables[kind])
        wq, wk = (t.float() for t in self.norms[l])
        h = bf16(rms(x) * self.pre(l, 0))
        info = {"o": [], "k": [], "v": [], "k_dev": [], "v_dev": []}
        parts = []
        for c in range(N_CHIPS):
            y = self.mm(h, qkv_w[c])
            qn = (Q_F if full else Q_S)
            q = y[:, :qn].reshape(M, 8, D).transpose(0, 1)  # [8, M, D]
            q = bf16(rope(bf16(rms(q) * wq), cos, sin))
            if full:
                kv = y[:, qn:].reshape(M, 1, D).transpose(0, 1)
                v = bf16(rms(kv))
                k = bf16(rope(bf16(v * wk), cos, sin))
            else:
                k = y[:, qn : qn + 4 * D].reshape(M, 4, D).transpose(0, 1)
                v = y[:, qn + 4 * D :].reshape(M, 4, D).transpose(0, 1)
                k = bf16(rope(bf16(rms(k) * wk), cos, sin))
                v = bf16(rms(v))
            K, V = caches[c]
            o = torch.zeros(8, M, D)
            kd, vd = torch.zeros(H, M, D), torch.zeros(H, M, D)
            for r in range(M):
                pos = p + r
                lo = 0 if full else max(0, pos - 1023)
                ps = torch.arange(lo, pos + 1)
                if full:
                    pages = tables["full_attention"][ps // 128]
                else:
                    pages = tables["sliding_attention"][(ps % 1152) // 128]
                Ks = K[pages, :, ps % 128]  # [n, H, D]
                Vs = V[pages, :, ps % 128]
                kd[:, r], vd[:, r] = Ks[-1], Vs[-1]
                for hq in range(8):
                    hk = 0 if full else hq // 2
                    s = Ks[:, hk] @ q[hq, r]
                    o[hq, r] = torch.softmax(s, 0) @ Vs[:, hk]
            o = bf16(o.transpose(0, 1).reshape(M, 8 * D))
            info["o"].append(o)
            info["k"].append(k)
            info["v"].append(v)
            info["k_dev"].append(kd)
            info["v_dev"].append(vd)
            parts.append(self.mm(o, o_w[c]))
        y = bf16(sum(parts))
        pa, pf = (t.float() for t in self.vectors["post"][l])
        x = bf16(x + pa * rms(y))
        h = bf16(rms(x) * self.pre(l, 1))
        parts = []
        for c in range(N_CHIPS):
            gu = self.mm(h, w[E_GU][c])
            a = bf16(gelu_tanh(gu[:, :IC]) * gu[:, IC:])
            parts.append(self.mm(a, w[E_DOWN][c]))
        y = bf16(sum(parts))
        x = bf16((x + pf * rms(y)) * self.vectors["scalar"][l])
        return x, info


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


class HFLayer:
    """one Gemma4 decoder layer from the checkpoint's bf16 weights, fp32 math, no tensor parallel and no quantization
    (HF Gemma4TextDecoderLayer semantics), for one row at position pos given the cache history"""

    def __init__(self, ck, l):
        g = lambda n: ck.layer(l, n).float()  # noqa: E731
        self.full = is_full(l)
        self.w = {n: g(n) for n in (
            "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight",
            "mlp.up_proj.weight", "mlp.down_proj.weight")}
        if not self.full:
            self.w["self_attn.v_proj.weight"] = g("self_attn.v_proj.weight")
        self.scalar = float(ck.layer(l, "layer_scalar").float().item())

    def __call__(self, x, khist, vhist, cos, sin, parts=None):
        """x [HIDDEN]; khist / vhist [kv heads, n, D] (positions before this one, oldest first); cos / sin [D];
        parts (a dict) gets the attention output o [32 heads x D], the post-attention residual and the MLP output"""
        w = self.w
        D = 512 if self.full else 256
        h = rms(x) * w["input_layernorm.weight"]
        q = (w["self_attn.q_proj.weight"] @ h).reshape(32, D)
        kraw = (w["self_attn.k_proj.weight"] @ h).reshape(-1, D)
        vraw = kraw if self.full else (w["self_attn.v_proj.weight"] @ h).reshape(-1, D)
        q = rope(rms(q) * w["self_attn.q_norm.weight"], cos, sin)
        k = rope(rms(kraw) * w["self_attn.k_norm.weight"], cos, sin)
        v = rms(vraw)
        K = torch.cat([khist.float(), k[:, None]], dim=1)
        V = torch.cat([vhist.float(), v[:, None]], dim=1)
        group = 32 // K.shape[0]
        o = torch.stack([torch.softmax(K[hq // group] @ q[hq], 0) @ V[hq // group] for hq in range(32)])
        y = w["self_attn.o_proj.weight"] @ o.reshape(-1)
        x = x + rms(y) * w["post_attention_layernorm.weight"]
        if parts is not None:
            parts.update(o=o.reshape(-1), x_mid=x.clone(), attn_y=y, q=q, k=k, v=v)
        h = rms(x) * w["pre_feedforward_layernorm.weight"]
        a = gelu_tanh(w["mlp.gate_proj.weight"] @ h) * (w["mlp.up_proj.weight"] @ h)
        y = w["mlp.down_proj.weight"] @ a
        return (x + rms(y) * w["post_feedforward_layernorm.weight"]) * self.scalar
