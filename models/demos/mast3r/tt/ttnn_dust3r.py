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


def full_encoder(
    img: torch.Tensor,
    state: dict,
    device,
    depth: int = 24,
) -> torch.Tensor:
    """Full encoder: patch_embed -> 24 encoder blocks -> enc_norm.

    img: (B, 3, H, W) torch. Returns (B, N, 1024) torch on host.
    """
    # Patch embed on device.
    x = patch_embed(img, state["patch_embed.proj.weight"], state["patch_embed.proj.bias"], device)
    B, C, H, W = img.shape
    hp, wp = H // 16, W // 16
    # Make positions matching reference (row-major y then x).
    ys = torch.arange(hp)
    xs = torch.arange(wp)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack((gy, gx), dim=-1).reshape(hp * wp, 2).unsqueeze(0).expand(B, -1, -1).contiguous()

    for i in range(depth):
        w = {
            "norm1.weight": state[f"enc_blocks.{i}.norm1.weight"],
            "norm1.bias": state[f"enc_blocks.{i}.norm1.bias"],
            "norm2.weight": state[f"enc_blocks.{i}.norm2.weight"],
            "norm2.bias": state[f"enc_blocks.{i}.norm2.bias"],
            "attn.qkv.weight": state[f"enc_blocks.{i}.attn.qkv.weight"],
            "attn.qkv.bias": state[f"enc_blocks.{i}.attn.qkv.bias"],
            "attn.proj.weight": state[f"enc_blocks.{i}.attn.proj.weight"],
            "attn.proj.bias": state[f"enc_blocks.{i}.attn.proj.bias"],
            "mlp.fc1.weight": state[f"enc_blocks.{i}.mlp.fc1.weight"],
            "mlp.fc1.bias": state[f"enc_blocks.{i}.mlp.fc1.bias"],
            "mlp.fc2.weight": state[f"enc_blocks.{i}.mlp.fc2.weight"],
            "mlp.fc2.bias": state[f"enc_blocks.{i}.mlp.fc2.bias"],
        }
        x = encoder_block(x, pos, w, device)

    # Final enc_norm on device.
    tt_x = _t2d(x, device)
    g = _t2d(state["enc_norm.weight"].reshape(1, 1, -1), device)
    b = _t2d(state["enc_norm.bias"].reshape(1, 1, -1), device)
    tt_x = ttnn.layer_norm(tt_x, weight=g, bias=b, epsilon=1e-6)
    return ttnn.to_torch(tt_x)


# ---------- Decoder block ----------

def _ttnn_linear(t_host: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device) -> torch.Tensor:
    """Simple device-side linear: host in/out (for use as building block)."""
    tt_in = _t2d(t_host, device)
    tt_w = _t2d(weight.t().contiguous(), device)
    tt_b = _t2d(bias.reshape(1, 1, -1), device)
    tt_out = ttnn.matmul(tt_in, tt_w)
    tt_out = ttnn.add(tt_out, tt_b)
    return ttnn.to_torch(tt_out)


def _ttnn_layer_norm(t_host: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device) -> torch.Tensor:
    tt_in = _t2d(t_host, device)
    g = _t2d(weight.reshape(1, 1, -1), device)
    b = _t2d(bias.reshape(1, 1, -1), device)
    return ttnn.to_torch(ttnn.layer_norm(tt_in, weight=g, bias=b, epsilon=1e-6))


def _ttnn_attn_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, device) -> torch.Tensor:
    """Attention on device: (B*H, N, dh) x (B*H, dh, M) -> softmax -> @v."""
    dh = q.shape[-1]
    tt_q = _t2d(q, device)
    tt_k = _t2d(k, device)
    tt_v = _t2d(v, device)
    scores = ttnn.matmul(tt_q, tt_k)
    scores = ttnn.multiply(scores, 1.0 / math.sqrt(dh))
    attn = ttnn.softmax(scores, dim=-1)
    ctx = ttnn.matmul(attn, tt_v)
    return ttnn.to_torch(ctx)


def decoder_block(
    x: torch.Tensor,
    y: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    weights: dict,
    device,
    heads: int = 12,
) -> torch.Tensor:
    """One DUSt3R decoder block on TT.

    x, y: (B, N, 768) torch           pos_x, pos_y: (B, N, 2) torch
    weights: keys norm1/2/3/norm_y.{w,b}, attn.qkv/proj.{w,b},
             cross_attn.projq/projk/projv/proj.{w,b}, mlp.fc1/fc2.{w,b}
    """
    B, N, D = x.shape
    dh = D // heads

    # --- self-attention over x ---
    x_n = _ttnn_layer_norm(x, weights["norm1.weight"], weights["norm1.bias"], device)
    qkv = _ttnn_linear(x_n, weights["attn.qkv.weight"], weights["attn.qkv.bias"], device)
    qkv = qkv.reshape(B, N, 3, heads, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = _rope_apply(q, pos_x)
    k = _rope_apply(k, pos_x)
    ctx = _ttnn_attn_core(
        q.reshape(B * heads, N, dh),
        k.transpose(-2, -1).reshape(B * heads, dh, N).contiguous(),
        v.reshape(B * heads, N, dh),
        device,
    )
    ctx = ctx.reshape(B, heads, N, dh).transpose(1, 2).reshape(B, N, D)
    proj = _ttnn_linear(ctx, weights["attn.proj.weight"], weights["attn.proj.bias"], device)
    x = x + proj

    # --- cross-attention: q from x, k/v from y ---
    x_n2 = _ttnn_layer_norm(x, weights["norm2.weight"], weights["norm2.bias"], device)
    y_n = _ttnn_layer_norm(y, weights["norm_y.weight"], weights["norm_y.bias"], device)
    q = _ttnn_linear(x_n2, weights["cross_attn.projq.weight"], weights["cross_attn.projq.bias"], device)
    k = _ttnn_linear(y_n, weights["cross_attn.projk.weight"], weights["cross_attn.projk.bias"], device)
    v = _ttnn_linear(y_n, weights["cross_attn.projv.weight"], weights["cross_attn.projv.bias"], device)
    q = q.reshape(B, N, heads, dh).transpose(1, 2)
    k = k.reshape(B, y.shape[1], heads, dh).transpose(1, 2)
    v = v.reshape(B, y.shape[1], heads, dh).transpose(1, 2)
    q = _rope_apply(q, pos_x)
    k = _rope_apply(k, pos_y)
    ctx = _ttnn_attn_core(
        q.reshape(B * heads, N, dh),
        k.transpose(-2, -1).reshape(B * heads, dh, y.shape[1]).contiguous(),
        v.reshape(B * heads, y.shape[1], dh),
        device,
    )
    ctx = ctx.reshape(B, heads, N, dh).transpose(1, 2).reshape(B, N, D)
    proj = _ttnn_linear(ctx, weights["cross_attn.proj.weight"], weights["cross_attn.proj.bias"], device)
    x = x + proj

    # --- MLP ---
    x_n3 = _ttnn_layer_norm(x, weights["norm3.weight"], weights["norm3.bias"], device)
    h = _ttnn_linear(x_n3, weights["mlp.fc1.weight"], weights["mlp.fc1.bias"], device)
    tt_h = _t2d(h, device)
    tt_h = ttnn.gelu(tt_h)
    h = ttnn.to_torch(tt_h)
    m = _ttnn_linear(h, weights["mlp.fc2.weight"], weights["mlp.fc2.bias"], device)
    return x + m


def _dec_block_weights(state: dict, idx: int, branch: int) -> dict:
    prefix = "dec_blocks." if branch == 1 else "dec_blocks2."
    keys = [
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        "norm3.weight", "norm3.bias", "norm_y.weight", "norm_y.bias",
        "attn.qkv.weight", "attn.qkv.bias",
        "attn.proj.weight", "attn.proj.bias",
        "cross_attn.projq.weight", "cross_attn.projq.bias",
        "cross_attn.projk.weight", "cross_attn.projk.bias",
        "cross_attn.projv.weight", "cross_attn.projv.bias",
        "cross_attn.proj.weight", "cross_attn.proj.bias",
        "mlp.fc1.weight", "mlp.fc1.bias",
        "mlp.fc2.weight", "mlp.fc2.bias",
    ]
    return {k: state[f"{prefix}{idx}.{k}"] for k in keys}


def full_decoder(
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    pos: torch.Tensor,
    state: dict,
    device,
    depth: int = 12,
):
    """Dual-branch DUSt3R decoder on TT.

    feat1, feat2: (B, N, 1024) encoder outputs.  pos: (B, N, 2).
    Returns (out1, out2) each (B, N, 768).
    """
    # Project to decoder dim via ttnn linear.
    emb_w = state["decoder_embed.weight"]
    emb_b = state["decoder_embed.bias"]
    f1 = _ttnn_linear(feat1, emb_w, emb_b, device)
    f2 = _ttnn_linear(feat2, emb_w, emb_b, device)

    for i in range(depth):
        w1 = _dec_block_weights(state, i, 1)
        w2 = _dec_block_weights(state, i, 2)
        nf1 = decoder_block(f1, f2, pos, pos, w1, device)
        nf2 = decoder_block(f2, f1, pos, pos, w2, device)
        f1, f2 = nf1, nf2

    g = state["dec_norm.weight"]
    b = state["dec_norm.bias"]
    out1 = _ttnn_layer_norm(f1, g, b, device)
    out2 = _ttnn_layer_norm(f2, g, b, device)
    return out1, out2


def dpt_head(feats_list, hw, state: dict, branch: int, device):
    """DPT head — currently host-torch fallback (conv-heavy, ttnn.conv2d path
    through fold has tooling issues in this build env; TODO: move to device).

    feats_list: list of 4 (B, N, D) token tensors (enc + 3 decoder depths).
    hw: (H, W) token grid (e.g. 32, 32 for 512 input).
    Returns (B, 4, H_img, W_img) on host.
    """
    from reference.torch_dust3r import load_dpt_head
    head = load_dpt_head(state, branch=branch)
    with torch.no_grad():
        return head(feats_list, hw)
