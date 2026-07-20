# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the XTTS-v2 conditioning branch (Block 1): conditioning encoder +
Perceiver resampler. Mirrors coqui TTS.tts.layers.tortoise (ConditioningEncoder /
AttentionBlock / QKVAttentionLegacy) + TTS.tts.layers.xtts.perceiver_encoder
(PerceiverResampler), op-for-op, so it can be validated against captured coqui goldens
and used as the reference for the TTNN port.

Flow (get_style_emb):
    mel [1, 80, T]
      -> ConditioningEncoder: Conv1d(80->1024,k1) + 6x AttentionBlock  -> enc [1, 1024, T]
      -> permute -> [1, T, 1024] -> PerceiverResampler(32 latents)      -> [1, 32, 1024]

AttentionBlock: x_norm = GroupNorm(32,1024, fp32); qkv=Conv1d(1024->3072,k1)(x_norm);
  16-head QKV attention (scale 1/sqrt(sqrt(64)), non-causal over T); proj=Conv1d(1024->1024);
  out = x_norm + proj  (residual on the *normed* input).
Perceiver layer: cross-attn (latents attend to cat[latents, frames], 8 heads x 64,
  scale 1/8) + residual; GEGLU FFN (1024->5460 -> GEGLU -> 2730 -> 1024) + residual.
Final RMSNorm: F.normalize(x, dim=-1) * sqrt(1024) * gamma.
"""

import math
import os

import torch
import torch.nn.functional as F

from models.experimental.xtts_v2.reference.xtts_gpt_ref import DEFAULT_CKPT, load_full_state

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "cond")

N_MEL = 80
DIM = 1024
ENC_HEADS = 16
ENC_BLOCKS = 6
GN_GROUPS = 32
PERC_LATENTS = 32
PERC_HEADS = 8
PERC_DIM_HEAD = 64
PERC_DEPTH = 2


def load_cond_state(ckpt_path=DEFAULT_CKPT):
    """Conditioning-branch weights, keyed relative to each submodule (encoder keys like
    'init.weight'/'attn.0.*'; perceiver keys like 'latents'/'layers.0.*'/'norm.gamma')."""
    full = load_full_state(ckpt_path)
    out = {}
    for k, v in full.items():
        for pref in ("gpt.conditioning_encoder.", "gpt.conditioning_perceiver."):
            if k.startswith(pref):
                out[k[len(pref) :]] = v.float()
    return out


class CondReference:
    def __init__(self, ckpt_path=DEFAULT_CKPT):
        self.w = load_cond_state(ckpt_path)

    # -- conditioning encoder --------------------------------------------------
    def _qkv_attention(self, qkv):  # qkv [1, 3072, T] -> [1, 1024, T]
        bs, width, length = qkv.shape
        ch = width // (3 * ENC_HEADS)  # 64
        q, k, v = qkv.reshape(bs * ENC_HEADS, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    def _attn_block(self, x, i):  # x [1,1024,T]
        w = self.w
        x_norm = F.group_norm(x, GN_GROUPS, w[f"attn.{i}.norm.weight"], w[f"attn.{i}.norm.bias"])
        qkv = F.conv1d(x_norm, w[f"attn.{i}.qkv.weight"], w[f"attn.{i}.qkv.bias"])
        a = self._qkv_attention(qkv)
        h = F.conv1d(a, w[f"attn.{i}.proj_out.weight"], w[f"attn.{i}.proj_out.bias"])
        return x_norm + h

    def conditioning_encoder(self, mel):  # mel [1,80,T] -> [1,1024,T]
        w = self.w
        h = F.conv1d(mel, w["init.weight"], w["init.bias"])
        for i in range(ENC_BLOCKS):
            h = self._attn_block(h, i)
        return h

    # -- perceiver resampler ---------------------------------------------------
    def _perc_attn(self, latents, frames, i):  # latents [1,32,1024], frames [1,T,1024]
        w = self.w
        context = torch.cat([latents, frames], dim=1)  # cross_attn_include_queries
        q = latents @ w[f"layers.{i}.0.to_q.weight"].t()  # [1,32,512]
        kv = context @ w[f"layers.{i}.0.to_kv.weight"].t()  # [1,32+T,1024]
        k, v = kv.chunk(2, dim=-1)

        def heads(t):
            b, n, _ = t.shape
            return t.reshape(b, n, PERC_HEADS, PERC_DIM_HEAD).permute(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        scale = PERC_DIM_HEAD**-0.5
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)  # [1,8,32,64]
        out = out.permute(0, 2, 1, 3).reshape(1, PERC_LATENTS, PERC_HEADS * PERC_DIM_HEAD)
        return out @ w[f"layers.{i}.0.to_out.weight"].t()  # [1,32,1024]

    def _perc_ff(self, x, i):  # x [1,32,1024]
        w = self.w
        h = x @ w[f"layers.{i}.1.0.weight"].t() + w[f"layers.{i}.1.0.bias"]  # [1,32,5460]
        a, g = h.chunk(2, dim=-1)
        h = a * F.gelu(g)  # GEGLU -> [1,32,2730]
        return h @ w[f"layers.{i}.1.2.weight"].t() + w[f"layers.{i}.1.2.bias"]  # [1,32,1024]

    def perceiver(self, frames):  # frames [1,T,1024] -> [1,32,1024]
        w = self.w
        latents = w["latents"].unsqueeze(0)  # [1,32,1024]
        for i in range(PERC_DEPTH):
            latents = self._perc_attn(latents, frames, i) + latents
            latents = self._perc_ff(latents, i) + latents
        return F.normalize(latents, dim=-1) * math.sqrt(DIM) * w["norm.gamma"]

    def get_style_emb(self, mel):  # mel [1,80,T] -> (enc_out [1,1024,T], perc_out [1,32,1024])
        with torch.no_grad():
            enc = self.conditioning_encoder(mel)
            perc = self.perceiver(enc.permute(0, 2, 1))
        return enc, perc


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


if __name__ == "__main__":
    ref = CondReference()
    mel = torch.load(os.path.join(GOLDEN_DIR, "mel_in.pt"))
    enc_g = torch.load(os.path.join(GOLDEN_DIR, "enc_out.pt"))
    perc_g = torch.load(os.path.join(GOLDEN_DIR, "perc_out.pt"))
    enc, perc = ref.get_style_emb(mel)
    print(f"enc  {tuple(enc.shape)}  PCC vs coqui = {_pcc(enc, enc_g):.6f}")
    print(f"perc {tuple(perc.shape)} PCC vs coqui = {_pcc(perc, perc_g):.6f}")
