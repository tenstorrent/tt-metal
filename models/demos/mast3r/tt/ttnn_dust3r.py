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

_ROPE_CACHE: dict = {}


def _rope_cos_sin(D: int, max_pos: int, base: float, dtype):
    key = (D, max_pos, base, dtype)
    c = _ROPE_CACHE.get(key)
    if c is None:
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        seq = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", seq, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        c = (emb.cos(), emb.sin())
        _ROPE_CACHE[key] = c
    return c


def _rope_apply(tokens: torch.Tensor, pos: torch.Tensor, base: float = 100.0) -> torch.Tensor:
    """Apply DUSt3R 2D RoPE100 on host. tokens: (B, H, N, Dh), pos: (B, N, 2)."""
    B, H, N, Dh = tokens.shape
    assert Dh % 2 == 0
    D = Dh // 2
    max_pos = int(pos.max().item()) + 1
    cos, sin = _rope_cos_sin(D, max_pos, base, tokens.dtype)

    y = tokens[..., :D]
    x = tokens[..., D:]

    def apply_1d(t, p1d, cos_t, sin_t):
        c = cos_t[p1d][:, None, :, :]
        s = sin_t[p1d][:, None, :, :]
        t1, t2 = t.chunk(2, dim=-1)
        rotated = torch.cat((-t2, t1), dim=-1)
        return t * c + rotated * s

    y = apply_1d(y, pos[..., 0], cos, sin)
    x = apply_1d(x, pos[..., 1], cos, sin)
    return torch.cat((y, x), dim=-1)


# ---------- Encoder block (device-resident) ----------

def _preload_enc_block_weights(state: dict, i: int, device) -> dict:
    """Upload weights for one encoder block.

    Linear weights use BFLOAT8_B (tile-level block-float8) for compute speedup,
    while biases and LayerNorm params stay in BFLOAT16 for numeric margin.
    """
    def mat8(t):
        return ttnn.from_torch(t.t().contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt = {}
    tt["g1"] = _t2d(state[f"enc_blocks.{i}.norm1.weight"].reshape(1, 1, -1), device)
    tt["b1"] = _t2d(state[f"enc_blocks.{i}.norm1.bias"].reshape(1, 1, -1), device)
    tt["g2"] = _t2d(state[f"enc_blocks.{i}.norm2.weight"].reshape(1, 1, -1), device)
    tt["b2"] = _t2d(state[f"enc_blocks.{i}.norm2.bias"].reshape(1, 1, -1), device)
    tt["qkv_w"] = mat8(state[f"enc_blocks.{i}.attn.qkv.weight"])
    tt["qkv_b"] = _t2d(state[f"enc_blocks.{i}.attn.qkv.bias"].reshape(1, 1, -1), device)
    tt["pw"] = mat8(state[f"enc_blocks.{i}.attn.proj.weight"])
    tt["pb"] = _t2d(state[f"enc_blocks.{i}.attn.proj.bias"].reshape(1, 1, -1), device)
    tt["w1"] = mat8(state[f"enc_blocks.{i}.mlp.fc1.weight"])
    tt["b1f"] = _t2d(state[f"enc_blocks.{i}.mlp.fc1.bias"].reshape(1, 1, -1), device)
    tt["w2"] = mat8(state[f"enc_blocks.{i}.mlp.fc2.weight"])
    tt["b2f"] = _t2d(state[f"enc_blocks.{i}.mlp.fc2.bias"].reshape(1, 1, -1), device)
    return tt


def encoder_block_device_pre(
    tt_x,
    pos: torch.Tensor,
    tt_w: dict,
    device,
    heads: int = 16,
):
    """Encoder block with pre-uploaded device weights (no per-call upload)."""
    shape = tt_x.shape
    B, N, D = int(shape[0]), int(shape[1]), int(shape[2])
    dh = D // heads

    tt_n1 = ttnn.layer_norm(tt_x, weight=tt_w["g1"], bias=tt_w["b1"], epsilon=1e-6)
    tt_qkv = ttnn.linear(tt_n1, tt_w["qkv_w"], bias=tt_w["qkv_b"])

    # Split qkv on device — v stays on device, only q and k get downloaded for RoPE.
    tt_q_part = ttnn.slice(tt_qkv, [0, 0, 0], [B, N, D])
    tt_k_part = ttnn.slice(tt_qkv, [0, 0, D], [B, N, 2 * D])
    tt_v_part = ttnn.slice(tt_qkv, [0, 0, 2 * D], [B, N, 3 * D])
    tt_v = ttnn.reshape(tt_v_part, (B, N, heads, dh))
    tt_v = ttnn.permute(tt_v, (0, 2, 1, 3))  # (B, H, N, dh)

    qk_host = ttnn.to_torch(ttnn.concat([tt_q_part, tt_k_part], dim=-1))  # (B, N, 2D)
    q = qk_host[..., :D].reshape(B, N, heads, dh).permute(0, 2, 1, 3)
    k = qk_host[..., D:].reshape(B, N, heads, dh).permute(0, 2, 1, 3)
    q = _rope_apply(q, pos)
    k = _rope_apply(k, pos)

    tt_q = _t2d(q, device)
    tt_k = _t2d(k, device)
    tt_ctx = ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False)
    tt_ctx = ttnn.transformer.concatenate_heads(tt_ctx)  # (B, N, D)
    tt_proj = ttnn.linear(tt_ctx, tt_w["pw"], bias=tt_w["pb"])

    tt_x = ttnn.add(tt_x, tt_proj)

    tt_n2 = ttnn.layer_norm(tt_x, weight=tt_w["g2"], bias=tt_w["b2"], epsilon=1e-6)
    tt_h = ttnn.linear(tt_n2, tt_w["w1"], bias=tt_w["b1f"])
    tt_h = ttnn.gelu(tt_h)
    tt_m = ttnn.linear(tt_h, tt_w["w2"], bias=tt_w["b2f"])

    tt_x = ttnn.add(tt_x, tt_m)
    return tt_x


def encoder_block_device(
    tt_x,  # ttnn tensor on device, (B, N, 1024)
    pos: torch.Tensor,  # (B, N, 2) host
    weights: dict,
    device,
    heads: int = 16,
):
    """Encoder block that keeps x on device end-to-end.

    Only the attention q/k get briefly round-tripped to host for RoPE
    (ttnn.experimental.rotary_embedding doesn't directly match DUSt3R's 2D
     RoPE100 — TODO implement device RoPE).
    """
    # Read shape from x before work (tt_x shape == torch shape).
    shape = tt_x.shape
    B, N, D = int(shape[0]), int(shape[1]), int(shape[2])
    dh = D // heads

    # --- norm1 + qkv on device, keep residual ---
    g1 = _t2d(weights["norm1.weight"].reshape(1, 1, -1), device)
    b1 = _t2d(weights["norm1.bias"].reshape(1, 1, -1), device)
    tt_n1 = ttnn.layer_norm(tt_x, weight=g1, bias=b1, epsilon=1e-6)

    qkv_w = _t2d(weights["attn.qkv.weight"].t().contiguous(), device)
    qkv_b = _t2d(weights["attn.qkv.bias"].reshape(1, 1, -1), device)
    tt_qkv = ttnn.linear(tt_n1, qkv_w, bias=qkv_b)

    # Round-trip qkv once for RoPE + attention (keeps attn on device for the bmm).
    qkv_host = ttnn.to_torch(tt_qkv).reshape(B, N, 3, heads, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv_host[0], qkv_host[1], qkv_host[2]
    q = _rope_apply(q, pos)
    k = _rope_apply(k, pos)

    tt_q = _t2d(q.reshape(B * heads, N, dh), device)
    tt_k = _t2d(k.transpose(-2, -1).reshape(B * heads, dh, N).contiguous(), device)
    tt_v = _t2d(v.reshape(B * heads, N, dh), device)
    tt_scores = ttnn.matmul(tt_q, tt_k)
    tt_scores = ttnn.multiply(tt_scores, 1.0 / math.sqrt(dh))
    tt_attn = ttnn.softmax(tt_scores, dim=-1)
    tt_ctx = ttnn.matmul(tt_attn, tt_v)  # (B*H, N, dh)

    # Permute heads/tokens back to (B, N, D) on host then re-upload for proj.
    ctx_host = ttnn.to_torch(tt_ctx).reshape(B, heads, N, dh).transpose(1, 2).reshape(B, N, D)
    tt_ctx2 = _t2d(ctx_host, device)
    pw = _t2d(weights["attn.proj.weight"].t().contiguous(), device)
    pb = _t2d(weights["attn.proj.bias"].reshape(1, 1, -1), device)
    tt_proj = ttnn.linear(tt_ctx2, pw, bias=pb)

    tt_x = ttnn.add(tt_x, tt_proj)

    # --- norm2 + mlp fully on device ---
    g2 = _t2d(weights["norm2.weight"].reshape(1, 1, -1), device)
    b2 = _t2d(weights["norm2.bias"].reshape(1, 1, -1), device)
    tt_n2 = ttnn.layer_norm(tt_x, weight=g2, bias=b2, epsilon=1e-6)

    w1 = _t2d(weights["mlp.fc1.weight"].t().contiguous(), device)
    b1f = _t2d(weights["mlp.fc1.bias"].reshape(1, 1, -1), device)
    tt_h = ttnn.linear(tt_n2, w1, bias=b1f)
    tt_h = ttnn.gelu(tt_h)

    w2 = _t2d(weights["mlp.fc2.weight"].t().contiguous(), device)
    b2f = _t2d(weights["mlp.fc2.bias"].reshape(1, 1, -1), device)
    tt_m = ttnn.linear(tt_h, w2, bias=b2f)

    tt_x = ttnn.add(tt_x, tt_m)
    return tt_x


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

    # Pre-upload all encoder block weights once (cached on module as a simple memo).
    cache_key = id(state)
    if not hasattr(full_encoder, "_cache") or full_encoder._cache_key != cache_key:
        full_encoder._cache = [_preload_enc_block_weights(state, i, device) for i in range(depth)]
        full_encoder._cache_key = cache_key

    tt_x = _t2d(x, device)
    for i in range(depth):
        tt_x = encoder_block_device_pre(tt_x, pos, full_encoder._cache[i], device)

    # Final enc_norm on device.
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


def _preload_dec_block_weights(state: dict, idx: int, branch: int, device) -> dict:
    prefix = "dec_blocks." if branch == 1 else "dec_blocks2."
    def w(k):
        return state[f"{prefix}{idx}.{k}"]
    def vec(t):
        return _t2d(t.reshape(1, 1, -1), device)
    def mat(t):
        return ttnn.from_torch(t.t().contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    return {
        "g1": vec(w("norm1.weight")), "b1v": vec(w("norm1.bias")),
        "g2": vec(w("norm2.weight")), "b2v": vec(w("norm2.bias")),
        "g3": vec(w("norm3.weight")), "b3v": vec(w("norm3.bias")),
        "gy": vec(w("norm_y.weight")), "byv": vec(w("norm_y.bias")),
        "qkv_w": mat(w("attn.qkv.weight")), "qkv_b": vec(w("attn.qkv.bias")),
        "proj_w": mat(w("attn.proj.weight")), "proj_b": vec(w("attn.proj.bias")),
        "cq_w": mat(w("cross_attn.projq.weight")), "cq_b": vec(w("cross_attn.projq.bias")),
        "ck_w": mat(w("cross_attn.projk.weight")), "ck_b": vec(w("cross_attn.projk.bias")),
        "cv_w": mat(w("cross_attn.projv.weight")), "cv_b": vec(w("cross_attn.projv.bias")),
        "cp_w": mat(w("cross_attn.proj.weight")), "cp_b": vec(w("cross_attn.proj.bias")),
        "fc1_w": mat(w("mlp.fc1.weight")), "fc1_b": vec(w("mlp.fc1.bias")),
        "fc2_w": mat(w("mlp.fc2.weight")), "fc2_b": vec(w("mlp.fc2.bias")),
    }


def decoder_block_device_pre(
    tt_x,  # device (B, N, 768)
    tt_y,  # device (B, N, 768)
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    tt_w: dict,
    device,
    heads: int = 12,
):
    shape = tt_x.shape
    B, N, D = int(shape[0]), int(shape[1]), int(shape[2])
    dh = D // heads

    # --- self-attention (SDPA, v stays on device) ---
    tt_n1 = ttnn.layer_norm(tt_x, weight=tt_w["g1"], bias=tt_w["b1v"], epsilon=1e-6)
    tt_qkv = ttnn.linear(tt_n1, tt_w["qkv_w"], bias=tt_w["qkv_b"])
    tt_q_part = ttnn.slice(tt_qkv, [0, 0, 0], [B, N, D])
    tt_k_part = ttnn.slice(tt_qkv, [0, 0, D], [B, N, 2 * D])
    tt_v_part = ttnn.slice(tt_qkv, [0, 0, 2 * D], [B, N, 3 * D])
    tt_v = ttnn.reshape(tt_v_part, (B, N, heads, dh))
    tt_v = ttnn.permute(tt_v, (0, 2, 1, 3))
    qk_host = ttnn.to_torch(ttnn.concat([tt_q_part, tt_k_part], dim=-1))
    q = qk_host[..., :D].reshape(B, N, heads, dh).permute(0, 2, 1, 3)
    k = qk_host[..., D:].reshape(B, N, heads, dh).permute(0, 2, 1, 3)
    q = _rope_apply(q, pos_x)
    k = _rope_apply(k, pos_x)
    tt_q = _t2d(q, device)
    tt_k = _t2d(k, device)
    tt_ctx = ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False)
    tt_ctx = ttnn.permute(tt_ctx, (0, 2, 1, 3))
    tt_ctx = ttnn.reshape(tt_ctx, (B, N, D))
    tt_proj = ttnn.linear(tt_ctx, tt_w["proj_w"], bias=tt_w["proj_b"])
    tt_x = ttnn.add(tt_x, tt_proj)

    # --- cross-attention (SDPA) ---
    tt_n2 = ttnn.layer_norm(tt_x, weight=tt_w["g2"], bias=tt_w["b2v"], epsilon=1e-6)
    tt_yn = ttnn.layer_norm(tt_y, weight=tt_w["gy"], bias=tt_w["byv"], epsilon=1e-6)
    tt_qc = ttnn.linear(tt_n2, tt_w["cq_w"], bias=tt_w["cq_b"])
    tt_kc = ttnn.linear(tt_yn, tt_w["ck_w"], bias=tt_w["ck_b"])
    tt_vc = ttnn.linear(tt_yn, tt_w["cv_w"], bias=tt_w["cv_b"])
    M = int(tt_y.shape[1])
    # Keep v on device; only download q + k for RoPE.
    tt_v = ttnn.reshape(tt_vc, (B, M, heads, dh))
    tt_v = ttnn.permute(tt_v, (0, 2, 1, 3))
    qk_host = ttnn.to_torch(ttnn.concat([tt_qc, tt_kc], dim=-1))
    q = qk_host[..., :D].reshape(B, N, heads, dh).permute(0, 2, 1, 3)
    k = qk_host[..., D:].reshape(B, M, heads, dh).permute(0, 2, 1, 3)
    q = _rope_apply(q, pos_x)
    k = _rope_apply(k, pos_y)
    tt_q = _t2d(q, device)
    tt_k = _t2d(k, device)
    tt_ctx = ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False)
    tt_ctx = ttnn.transformer.concatenate_heads(tt_ctx)
    tt_cproj = ttnn.linear(tt_ctx, tt_w["cp_w"], bias=tt_w["cp_b"])
    tt_x = ttnn.add(tt_x, tt_cproj)

    # --- MLP ---
    tt_n3 = ttnn.layer_norm(tt_x, weight=tt_w["g3"], bias=tt_w["b3v"], epsilon=1e-6)
    tt_h = ttnn.linear(tt_n3, tt_w["fc1_w"], bias=tt_w["fc1_b"])
    tt_h = ttnn.gelu(tt_h)
    tt_m = ttnn.linear(tt_h, tt_w["fc2_w"], bias=tt_w["fc2_b"])
    tt_x = ttnn.add(tt_x, tt_m)
    return tt_x


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
    # Pre-upload decoder weights once (memoized on module).
    cache_key = ("dec", id(state))
    if getattr(full_decoder, "_cache_key", None) != cache_key:
        full_decoder._w1 = [_preload_dec_block_weights(state, i, 1, device) for i in range(depth)]
        full_decoder._w2 = [_preload_dec_block_weights(state, i, 2, device) for i in range(depth)]
        full_decoder._emb_w = _t2d(state["decoder_embed.weight"].t().contiguous(), device)
        full_decoder._emb_b = _t2d(state["decoder_embed.bias"].reshape(1, 1, -1), device)
        full_decoder._dnorm_g = _t2d(state["dec_norm.weight"].reshape(1, 1, -1), device)
        full_decoder._dnorm_b = _t2d(state["dec_norm.bias"].reshape(1, 1, -1), device)
        full_decoder._cache_key = cache_key

    tt_f1 = ttnn.linear(_t2d(feat1, device), full_decoder._emb_w, bias=full_decoder._emb_b)
    tt_f2 = ttnn.linear(_t2d(feat2, device), full_decoder._emb_w, bias=full_decoder._emb_b)

    taps1: list[torch.Tensor] = []
    taps2: list[torch.Tensor] = []
    tap_layers = (0, 6, 11)
    for i in range(depth):
        nf1 = decoder_block_device_pre(tt_f1, tt_f2, pos, pos, full_decoder._w1[i], device)
        nf2 = decoder_block_device_pre(tt_f2, tt_f1, pos, pos, full_decoder._w2[i], device)
        tt_f1, tt_f2 = nf1, nf2
        if i in tap_layers:
            taps1.append(ttnn.to_torch(tt_f1))
            taps2.append(ttnn.to_torch(tt_f2))

    out1 = ttnn.to_torch(ttnn.layer_norm(tt_f1, weight=full_decoder._dnorm_g, bias=full_decoder._dnorm_b, epsilon=1e-6))
    out2 = ttnn.to_torch(ttnn.layer_norm(tt_f2, weight=full_decoder._dnorm_g, bias=full_decoder._dnorm_b, epsilon=1e-6))
    return out1, out2, taps1, taps2


def dust3r_forward(img1: torch.Tensor, img2: torch.Tensor, state: dict, device):
    """Full end-to-end DUSt3R forward on TT (with host DPT fallback)."""
    enc1 = full_encoder(img1, state, device)
    enc2 = full_encoder(img2, state, device)
    B, C, H, W = img1.shape
    hp, wp = H // 16, W // 16
    ys = torch.arange(hp)
    xs = torch.arange(wp)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack((gy, gx), dim=-1).reshape(hp * wp, 2).unsqueeze(0).expand(B, -1, -1).contiguous()

    _, _, taps1, taps2 = full_decoder(enc1, enc2, pos, state, device)
    feats1 = [enc1.float(), taps1[0].float(), taps1[1].float(), taps1[2].float()]
    feats2 = [enc2.float(), taps2[0].float(), taps2[1].float(), taps2[2].float()]

    # Two independent DPT heads — run concurrently to overlap host conv work.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(dpt_head, feats1, (hp, wp), state, 1, device)
        f2 = pool.submit(dpt_head, feats2, (hp, wp), state, 2, device)
        out1 = f1.result()
        out2 = f2.result()
    return out1, out2


_DPT_HEAD_CACHE: dict = {}


def dpt_head(feats_list, hw, state: dict, branch: int, device):
    """DPT head — host-torch fallback in bfloat16 + torch.compile for speed."""
    from reference.torch_dust3r import load_dpt_head
    key = (id(state), branch, "bf16c", hw)
    head = _DPT_HEAD_CACHE.get(key)
    if head is None:
        base = load_dpt_head(state, branch=branch).to(torch.bfloat16).eval()
        try:
            compiled = torch.compile(base, mode="reduce-overhead", dynamic=False)
            # Warm up compilation with dummy input matching shape.
            with torch.no_grad():
                dummy = [f.to(torch.bfloat16) for f in feats_list]
                compiled(dummy, hw)
                compiled(dummy, hw)
            head = compiled
        except Exception:
            head = base
        _DPT_HEAD_CACHE[key] = head
    feats_bf16 = [f.to(torch.bfloat16) for f in feats_list]
    with torch.no_grad():
        return head(feats_bf16, hw).float()
