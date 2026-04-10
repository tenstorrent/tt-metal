"""TT-Metal (ttnn) implementation of DUSt3R layers.

Each function takes a device + torch weights/inputs and returns a torch tensor
on host so the test harness can compute PCC against the reference.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import ttnn


def _t2d(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def patch_embed(img: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device):
    """Patch embedding via im2col + matmul (equivalent to stride=kernel Conv2d).

    img: (B, 3, H, W)      weight: (1024, 3, 16, 16)      bias: (1024,)
    returns (B, N, 1024) on host, where N = (H/16) * (W/16).
    """
    B, C, H, W = img.shape
    p = 16
    hp, wp = H // p, W // p
    N = hp * wp

    # im2col on host: (B, C, H, W) -> (B, N, C*p*p)
    patches = img.unfold(2, p, p).unfold(3, p, p)  # (B, C, hp, wp, p, p)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, hp, wp, C, p, p)
    patches = patches.reshape(B, N, C * p * p)

    # Flatten weight to match: Conv2d weight (E, C, p, p) -> (C*p*p, E)
    w_flat = weight.reshape(weight.shape[0], -1).t().contiguous()  # (C*p*p, E)

    tt_patches = ttnn.from_torch(patches, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_w = ttnn.from_torch(w_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(bias.reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.matmul(tt_patches, tt_w)
    out = ttnn.add(out, tt_b)

    out_torch = ttnn.to_torch(out)  # (B, N, E)
    return out_torch


# ---------- RoPE (host-side, identical to reference) ----------

def _rope_apply(tokens: torch.Tensor, pos: torch.Tensor, base: float = 100.0) -> torch.Tensor:
    """Apply DUSt3R 2D RoPE100 on host. tokens: (B, H, N, Dh), pos: (B, N, 2)."""
    B, H, N, Dh = tokens.shape
    assert Dh % 2 == 0
    D = Dh // 2  # per-axis dim
    y = tokens[..., :D]
    x = tokens[..., D:]

    def apply_1d(t, p1d):
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        seq = torch.arange(int(p1d.max()) + 1, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", seq, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(t.dtype)
        cos = emb.cos()[p1d][:, None, :, :]
        sin = emb.sin()[p1d][:, None, :, :]
        t1, t2 = t.chunk(2, dim=-1)
        rotated = torch.cat((-t2, t1), dim=-1)
        return t * cos + rotated * sin

    y = apply_1d(y, pos[..., 0])
    x = apply_1d(x, pos[..., 1])
    return torch.cat((y, x), dim=-1)


# ---------- Encoder block ----------

def encoder_block(
    x: torch.Tensor,
    pos: torch.Tensor,
    weights: dict,
    device,
    heads: int = 16,
) -> torch.Tensor:
    """Single ViT-L encoder block on TT.

    x: (B, N, 1024) torch      pos: (B, N, 2) torch
    weights: dict with keys norm1/2.{weight,bias}, attn.qkv/proj.{weight,bias},
             mlp.fc1/fc2.{weight,bias}
    returns (B, N, 1024) torch
    """
    B, N, D = x.shape
    dh = D // heads

    tt_x = _t2d(x, device)

    # --- norm1 + qkv ---
    g1 = _t2d(weights["norm1.weight"].reshape(1, 1, -1), device)
    b1 = _t2d(weights["norm1.bias"].reshape(1, 1, -1), device)
    tt_n1 = ttnn.layer_norm(tt_x, weight=g1, bias=b1, epsilon=1e-6)

    qkv_w = _t2d(weights["attn.qkv.weight"].t().contiguous(), device)
    qkv_b = _t2d(weights["attn.qkv.bias"].reshape(1, 1, -1), device)
    tt_qkv = ttnn.matmul(tt_n1, qkv_w)
    tt_qkv = ttnn.add(tt_qkv, qkv_b)

    # Pull qkv to host to apply RoPE and attention (simple + correct).
    qkv_host = ttnn.to_torch(tt_qkv)  # (B, N, 3D)
    qkv_host = qkv_host.reshape(B, N, 3, heads, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv_host[0], qkv_host[1], qkv_host[2]  # (B, H, N, dh)
    q = _rope_apply(q, pos)
    k = _rope_apply(k, pos)

    # Attention on device via bmm+softmax.
    tt_q = _t2d(q.reshape(B * heads, N, dh), device)
    tt_k = _t2d(k.transpose(-2, -1).reshape(B * heads, dh, N).contiguous(), device)
    tt_v = _t2d(v.reshape(B * heads, N, dh), device)
    tt_scores = ttnn.matmul(tt_q, tt_k)
    tt_scores = ttnn.multiply(tt_scores, 1.0 / math.sqrt(dh))
    tt_attn = ttnn.softmax(tt_scores, dim=-1)
    tt_ctx = ttnn.matmul(tt_attn, tt_v)  # (B*H, N, dh)
    ctx_host = ttnn.to_torch(tt_ctx).reshape(B, heads, N, dh).transpose(1, 2).reshape(B, N, D)

    tt_ctx2 = _t2d(ctx_host, device)
    pw = _t2d(weights["attn.proj.weight"].t().contiguous(), device)
    pb = _t2d(weights["attn.proj.bias"].reshape(1, 1, -1), device)
    tt_proj = ttnn.matmul(tt_ctx2, pw)
    tt_proj = ttnn.add(tt_proj, pb)

    tt_x = ttnn.add(tt_x, tt_proj)

    # --- norm2 + mlp ---
    g2 = _t2d(weights["norm2.weight"].reshape(1, 1, -1), device)
    b2 = _t2d(weights["norm2.bias"].reshape(1, 1, -1), device)
    tt_n2 = ttnn.layer_norm(tt_x, weight=g2, bias=b2, epsilon=1e-6)

    w1 = _t2d(weights["mlp.fc1.weight"].t().contiguous(), device)
    b1f = _t2d(weights["mlp.fc1.bias"].reshape(1, 1, -1), device)
    tt_h = ttnn.matmul(tt_n2, w1)
    tt_h = ttnn.add(tt_h, b1f)
    tt_h = ttnn.gelu(tt_h)

    w2 = _t2d(weights["mlp.fc2.weight"].t().contiguous(), device)
    b2f = _t2d(weights["mlp.fc2.bias"].reshape(1, 1, -1), device)
    tt_m = ttnn.matmul(tt_h, w2)
    tt_m = ttnn.add(tt_m, b2f)

    tt_out = ttnn.add(tt_x, tt_m)
    return ttnn.to_torch(tt_out)
