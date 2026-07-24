# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Self-contained functional reference for the Qwen3-ASR AuT audio encoder.

Mirrors `Qwen3ASRAudioEncoder.forward` exactly, but as plain functional torch
driven by a weights dict loaded from the HF safetensors. No `qwen_asr` / HF
modeling dependency. This is the spec the ttnn port mirrors 1:1, and the oracle
for the Phase-2 PCC test (compared against the saved golden .npy).

Config (Qwen3-ASR-1.7B audio_config): num_mel_bins=128, d_model=1024,
encoder_layers=24, heads=16 (head_dim=64), ffn=4096, downsample_hidden_size=480,
output_dim=2048, n_window=50, n_window_infer=800, max_source_positions=1500.
"""
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

CFG = dict(
    num_mel_bins=128,
    d_model=1024,
    encoder_layers=24,
    heads=16,
    ffn=4096,
    ds_hidden=480,
    output_dim=2048,
    n_window=50,
    n_window_infer=800,
    max_source_positions=1500,
    ln_eps=1e-5,
)
PREFIX = "thinker.audio_tower."


def load_audio_tower_weights(snap_dir=None, dtype=torch.float32):
    if snap_dir is None:
        base = "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"
        snap_dir = os.path.join(base, os.listdir(base)[0])
    w = {}
    for f in sorted(glob.glob(snap_dir + "/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.startswith(PREFIX):
                    w[k[len(PREFIX) :]] = h.get_tensor(k).to(dtype)
    return w


def sinusoids(length, channels, max_timescale=10000.0):
    log_inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-log_inc * torch.arange(channels // 2).float())
    t = torch.arange(length)[:, None].float() * inv[None, :]
    return torch.cat([torch.sin(t), torch.cos(t)], dim=1)


def feat_out_len(L):
    leave = L % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (L // 100) * 13


def build_cu_seqlens(feature_len, n_window=50, n_window_infer=800, per_chunk_after_cnn=13):
    aftercnn = feat_out_len(feature_len)
    window_aftercnn = per_chunk_after_cnn * (n_window_infer // (n_window * 2))
    lens = []
    lens += [window_aftercnn] * (aftercnn // window_aftercnn)
    rem = aftercnn % window_aftercnn
    if rem:
        lens.append(rem)
    cu = torch.tensor([0] + lens).cumsum(0).to(torch.int32)
    return cu, aftercnn


def attn_mask_from_cu(cu, seqlen, dtype=torch.float32):
    m = torch.full((1, 1, seqlen, seqlen), torch.finfo(dtype).min, dtype=dtype)
    for i in range(1, len(cu)):
        a, b = int(cu[i - 1]), int(cu[i])
        m[..., a:b, a:b] = 0
    return m


def conv_frontend(mel, w):
    """mel (num_mel, T) -> (n_chunks, 13, d_model). Mirrors the chunked conv2d path."""
    nm, T = mel.shape
    n_window = CFG["n_window"]
    chunk = n_window * 2  # 100
    n_chunks = math.ceil(T / chunk)
    # split along time into chunks of 100 (last may be shorter), pad to 100
    pieces = []
    for i in range(n_chunks):
        seg = mel[:, i * chunk : (i + 1) * chunk]
        if seg.shape[1] < chunk:
            seg = F.pad(seg, (0, chunk - seg.shape[1]))
        pieces.append(seg)
    x = torch.stack(pieces, 0).unsqueeze(1)  # (n_chunks,1,128,100)
    x = F.gelu(F.conv2d(x, w["conv2d1.weight"], w["conv2d1.bias"], stride=2, padding=1))
    x = F.gelu(F.conv2d(x, w["conv2d2.weight"], w["conv2d2.bias"], stride=2, padding=1))
    x = F.gelu(F.conv2d(x, w["conv2d3.weight"], w["conv2d3.bias"], stride=2, padding=1))
    b, c, fr, t = x.shape  # (n_chunks,480,16,13)
    x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * fr)  # (n_chunks,13,7680)
    x = F.linear(x, w["conv_out.weight"])  # no bias -> (n_chunks,13,1024)
    return x


def ln(x, w, name, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), w[f"{name}.weight"], w[f"{name}.bias"], eps)


def encoder_layer(x, w, li, mask):
    """x (S, d_model). Pre-LN attention + FFN, block-diagonal attention via mask."""
    p = f"layers.{li}."
    S, D = x.shape
    H, hd = CFG["heads"], CFG["d_model"] // CFG["heads"]
    residual = x
    h = ln(x, w, p + "self_attn_layer_norm")
    q = F.linear(h, w[p + "self_attn.q_proj.weight"], w[p + "self_attn.q_proj.bias"])
    k = F.linear(h, w[p + "self_attn.k_proj.weight"], w[p + "self_attn.k_proj.bias"])
    v = F.linear(h, w[p + "self_attn.v_proj.weight"], w[p + "self_attn.v_proj.bias"])
    q = q.view(S, H, hd).transpose(0, 1)  # (H,S,hd)
    k = k.view(S, H, hd).transpose(0, 1)
    v = v.view(S, H, hd).transpose(0, 1)
    scale = hd**-0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale + mask[0]  # (H,S,S)
    probs = torch.softmax(scores, dim=-1)
    o = torch.matmul(probs, v).transpose(0, 1).reshape(S, D)
    o = F.linear(o, w[p + "self_attn.out_proj.weight"], w[p + "self_attn.out_proj.bias"])
    x = residual + o
    residual = x
    h = ln(x, w, p + "final_layer_norm")
    h = F.gelu(F.linear(h, w[p + "fc1.weight"], w[p + "fc1.bias"]))
    h = F.linear(h, w[p + "fc2.weight"], w[p + "fc2.bias"])
    return residual + h


def encode(mel, w, return_stages=False, windowed=False):
    """mel (num_mel, T) -> audio embeds (S, output_dim).

    windowed=False (default): full bidirectional attention over the whole sequence.
      This matches the CPU/sdpa reference EXACTLY (the encoder passes attention_mask=None
      to the layers; cu_seqlens only takes effect on the flash_attention_2 varlen path).
      For our <=60s segments (S up to ~800 post-cnn tokens) full attention is both exact
      and cheap on TT.
    windowed=True: block-diagonal attention from cu_seqlens (matches the FA2 deployment;
      an efficiency path for long-form / streaming, NOT needed for segment ASR)."""
    T = mel.shape[1]
    conv = conv_frontend(mel, w)  # (n_chunks,13,1024)
    n_chunks, per_chunk = conv.shape[0], conv.shape[1]
    pe = sinusoids(CFG["max_source_positions"], CFG["d_model"])[:per_chunk]  # (13,1024)
    conv = conv + pe.unsqueeze(0)
    x = conv.reshape(n_chunks * per_chunk, CFG["d_model"])  # (S,1024)
    S = x.shape[0]
    if windowed:
        cu, _ = build_cu_seqlens(T, per_chunk_after_cnn=per_chunk)
    else:
        cu = torch.tensor([0, S], dtype=torch.int32)
    mask = attn_mask_from_cu(cu, S, x.dtype)
    stages = {"conv_out": conv.detach().clone()}
    for li in range(CFG["encoder_layers"]):
        x = encoder_layer(x, w, li, mask)
        if li == 0:
            stages["enc_layer0"] = x.detach().clone()
    x = ln(x, w, "ln_post")
    stages["ln_post"] = x.detach().clone()
    x = F.linear(x, w["proj1.weight"], w["proj1.bias"])
    x = F.gelu(x)
    x = F.linear(x, w["proj2.weight"], w["proj2.bias"])
    stages["proj2"] = x.detach().clone()
    return (x, stages) if return_stages else x


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


if __name__ == "__main__":
    G = "/home/ttuser/ttwork/qwen3_asr_golden"
    w = load_audio_tower_weights()
    mel = torch.from_numpy(np.load(f"{G}/input_features.npy")).float()
    print(f"mel {tuple(mel.shape)}  cu_seqlens={build_cu_seqlens(mel.shape[1])[0].tolist()}")
    emb, stages = encode(mel, w, return_stages=True)
    # conv_out golden is (n_chunks,13,1024) before flatten; compare flattened
    g_conv = torch.from_numpy(np.load(f"{G}/conv2d1.npy"))  # only sanity (shape)
    checks = {
        "conv_out (pre-PE? no: ours has PE)": None,
        "enc_layer0": (stages["enc_layer0"], np.load(f"{G}/enc_layer0.npy")),
        "ln_post": (stages["ln_post"], np.load(f"{G}/ln_post.npy")),
        "proj2/audio_tower": (stages["proj2"], np.load(f"{G}/audio_tower.npy")),
    }
    for name, pair in checks.items():
        if pair is None:
            continue
        ours, gold = pair
        gold = torch.from_numpy(gold).float()
        print(f"PCC {name:22s} = {pcc(ours, gold):.6f}  (ours {tuple(ours.shape)} vs {tuple(gold.shape)})")
