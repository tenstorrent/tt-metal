# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the XTTS-v2 conditioning branch (Block 1): conditioning encoder + Perceiver
resampler. Self-contained op-for-op (torch + F only), written from the coqui source:
  - tortoise/autoregressive.py: ConditioningEncoder
  - tortoise/arch_utils.py:     AttentionBlock, QKVAttentionLegacy, GroupNorm32/normalization
  - xtts/perceiver_encoder.py:  PerceiverResampler, Attention, Attend, RMSNorm, FeedForward
  - tortoise/transformer.py:    GEGLU

Block boundary (GPT.get_style_emb with use_perceiver_resampler=True):

    mel [1, 80, T]
      -> ConditioningEncoder: Conv1d(80->1024,k1) + 6x AttentionBlock   -> enc [1, 1024, T]
      -> enc.permute(0,2,1) [1, T, 1024] -> PerceiverResampler(32)       -> perc [1, 32, 1024]

`perc` is the gpt_cond_latent that conditions the GPT (Block 3); get_style_emb returns its
transpose [1, 1024, 32].

Op details worth noting:
  AttentionBlock: x_norm=GroupNorm(32,1024, fp32); qkv=Conv1d(1024->3072,k1)(x_norm);
    16-head legacy attn (scale 1/sqrt(sqrt(64)), non-causal); proj=Conv1d(1024->1024);
    out = x_norm + proj   (residual on the NORMED input; tortoise_norm=False).
  Perceiver layer: cross-attn (latents attend to cat[latents, frames], 8 heads x 64,
    scale 1/8) + residual; GEGLU FFN (1024->5460 -> GEGLU -> 2730 -> 1024) + residual.
  Final RMSNorm: F.normalize(x, dim=-1) * sqrt(1024) * gamma.

Run (needs repo root on PYTHONPATH so the checkpoint loader imports):
    PYTHONPATH=<repo> python models/experimental/xtts_v2/reference/xtts_cond_ref.py
"""

import math
import os

import torch
import torch.nn.functional as F

from models.experimental.xtts_v2.reference.xtts_gpt_ref import DEFAULT_CKPT, load_full_state, pcc

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "cond")

# ---- config (from coqui config.json + gpt.py construction) ----
N_MEL = 80
DIM = 1024
ENC_HEADS = 16  # ConditioningEncoder(num_attn_heads=heads), heads=16
ENC_BLOCKS = 6
GN_GROUPS = 32  # normalization(1024) -> GroupNorm32(32, 1024)
GN_EPS = 1e-5
PERC_LATENTS = 32
PERC_HEADS = 8
PERC_DIM_HEAD = 64
PERC_DEPTH = 2


def load_cond_state(ckpt_path=DEFAULT_CKPT):
    """Return (encoder_weights, perceiver_weights), each keyed relative to its submodule."""
    full = load_full_state(ckpt_path)
    enc, perc = {}, {}
    for k, v in full.items():
        if k.startswith("gpt.conditioning_encoder."):
            enc[k[len("gpt.conditioning_encoder.") :]] = v.float()
        elif k.startswith("gpt.conditioning_perceiver."):
            perc[k[len("gpt.conditioning_perceiver.") :]] = v.float()
    return enc, perc


# --------------------------------------------------------------------------------------
# Conditioning encoder
# --------------------------------------------------------------------------------------
def _group_norm(x, weight, bias):
    # coqui GroupNorm32 runs in fp32 then casts back.
    return F.group_norm(x.float(), GN_GROUPS, weight, bias, GN_EPS).type(x.dtype)


def _qkv_attention(qkv, n_heads=ENC_HEADS):
    # qkv [b, 3*C, T] -> [b, C, T]; matches QKVAttentionLegacy.
    bs, width, length = qkv.shape
    ch = width // (3 * n_heads)  # 64
    q, k, v = qkv.reshape(bs * n_heads, ch * 3, length).split(ch, dim=1)
    scale = 1 / math.sqrt(math.sqrt(ch))
    weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # scale on both q and k
    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    a = torch.einsum("bts,bcs->bct", weight, v)
    return a.reshape(bs, -1, length)


def _attn_block(x, w, i):  # x [b, 1024, T]
    x_norm = _group_norm(x, w[f"attn.{i}.norm.weight"], w[f"attn.{i}.norm.bias"])
    qkv = F.conv1d(x_norm, w[f"attn.{i}.qkv.weight"], w[f"attn.{i}.qkv.bias"])
    h = _qkv_attention(qkv)
    h = F.conv1d(h, w[f"attn.{i}.proj_out.weight"], w[f"attn.{i}.proj_out.bias"])
    return x_norm + h  # residual on the normed input (tortoise_norm=False)


def conditioning_encoder(mel, enc_w):  # mel [b, 80, T] -> [b, 1024, T]
    h = F.conv1d(mel, enc_w["init.weight"], enc_w["init.bias"])
    for i in range(ENC_BLOCKS):
        h = _attn_block(h, enc_w, i)
    return h


# --------------------------------------------------------------------------------------
# Perceiver resampler
# --------------------------------------------------------------------------------------
def _heads(t):  # [b, n, h*d] -> [b, h, n, d]
    b, n, _ = t.shape
    return t.reshape(b, n, PERC_HEADS, PERC_DIM_HEAD).permute(0, 2, 1, 3)


def _perc_attn(latents, frames, pw, i):  # latents [b,32,1024], frames [b,T,1024]
    context = torch.cat([latents, frames], dim=1)  # cross_attn_include_queries=True
    q = latents @ pw[f"layers.{i}.0.to_q.weight"].t()  # [b,32,512]
    kv = context @ pw[f"layers.{i}.0.to_kv.weight"].t()  # [b,32+T,1024]
    k, v = kv.chunk(2, dim=-1)
    q, k, v = _heads(q), _heads(k), _heads(v)
    scale = PERC_DIM_HEAD**-0.5  # 1/8
    sim = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    attn = sim.softmax(dim=-1)
    out = torch.einsum("bhij,bhjd->bhid", attn, v)  # [b,8,32,64]
    out = out.permute(0, 2, 1, 3).reshape(latents.shape[0], PERC_LATENTS, PERC_HEADS * PERC_DIM_HEAD)  # [b,32,512]
    return out @ pw[f"layers.{i}.0.to_out.weight"].t()  # [b,32,1024]


def _perc_ff(x, pw, i):  # GEGLU FFN, x [b,32,1024]
    h = x @ pw[f"layers.{i}.1.0.weight"].t() + pw[f"layers.{i}.1.0.bias"]  # [b,32,5460]
    a, gates = h.chunk(2, dim=-1)
    h = a * F.gelu(gates)  # GEGLU -> [b,32,2730]
    return h @ pw[f"layers.{i}.1.2.weight"].t() + pw[f"layers.{i}.1.2.bias"]  # [b,32,1024]


def perceiver(frames, pw):  # frames [b,T,1024] -> [b,32,1024]
    b = frames.shape[0]
    latents = pw["latents"].unsqueeze(0).expand(b, -1, -1)  # proj_context is Identity (dim==dim_context)
    for i in range(PERC_DEPTH):
        latents = _perc_attn(latents, frames, pw, i) + latents
        latents = _perc_ff(latents, pw, i) + latents
    return F.normalize(latents, dim=-1) * math.sqrt(DIM) * pw["norm.gamma"]  # final RMSNorm


@torch.no_grad()
def get_style_emb(mel, enc_w, perc_w):
    """mel [b,80,T] -> (enc [b,1024,T], perc [b,32,1024]).  perc = gpt_cond_latent for Block 3."""
    enc = conditioning_encoder(mel, enc_w)
    perc = perceiver(enc.permute(0, 2, 1), perc_w)
    return enc, perc


def make_synthetic_mel(n_frames=128, seed=0):
    """Deterministic mel [1, 80, T]. Content is irrelevant for op-for-op validation (both
    reference and coqui compute the same thing); GroupNorm normalizes the input anyway."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, N_MEL, n_frames, generator=g)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=GOLDEN_DIR)
    ap.add_argument("--n-frames", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[ref] loading conditioning weights from {args.ckpt}")
    enc_w, perc_w = load_cond_state(args.ckpt)
    mel = make_synthetic_mel(args.n_frames)
    enc, perc = get_style_emb(mel, enc_w, perc_w)
    print(f"[ref] mel {tuple(mel.shape)} -> enc {tuple(enc.shape)} -> perc {tuple(perc.shape)}")
    print(f"[ref] perc mean={perc.mean().item():.5f} std={perc.std().item():.5f}")

    torch.save(mel, os.path.join(args.out, "mel_in.pt"))
    torch.save(enc, os.path.join(args.out, "enc_out.pt"))
    torch.save(perc, os.path.join(args.out, "perc_out.pt"))  # == gpt_cond_latent [1,32,1024]
    torch.save({"n_frames": args.n_frames, "dim": DIM, "latents": PERC_LATENTS}, os.path.join(args.out, "meta.pt"))
    print(f"[ref] wrote conditioning goldens to {args.out}")


if __name__ == "__main__":
    main()
