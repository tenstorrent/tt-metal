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


_PATCH_EMBED_CACHE: dict = {}


def patch_embed(img: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device):
    """Host-returning patch_embed (used by per-layer tests).

    Wraps the on-device version and downloads to host.
    """
    tt_out = patch_embed_device(img, weight, bias, device)
    out_torch = ttnn.to_torch(tt_out).reshape(1, -1, weight.shape[0])
    return out_torch


def patch_embed_device(img: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device):
    """Patch embedding on device via host im2col + ttnn.linear.

    img: (B, 3, H, W) host. weight: (1024, 3, 16, 16). bias: (1024,).
    Returns ttnn tensor of shape (B, N, 1024) on device.
    """
    B, C, H, W = img.shape
    p = weight.shape[2]
    hp, wp = H // p, W // p
    N = hp * wp

    cache_key = (id(weight), id(device))
    cached = _PATCH_EMBED_CACHE.get(cache_key)
    if cached is None:
        w_flat = weight.reshape(weight.shape[0], -1).t().contiguous()  # (C*p*p, E)
        tt_w = ttnn.from_torch(w_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(bias.reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _PATCH_EMBED_CACHE[cache_key] = (tt_w, tt_b)
    else:
        tt_w, tt_b = cached

    # im2col on host (cheap), upload to device, matmul on device — output stays on device.
    patches = img.unfold(2, p, p).unfold(3, p, p)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B, N, C * p * p)
    tt_patches = ttnn.from_torch(patches, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.linear(tt_patches, tt_w, bias=tt_b)
    return tt_out


# ---------- RoPE (host-side, identical to reference) ----------

_ROPE_CACHE: dict = {}
_POS_MAX_CACHE: dict = {}
_ROPE_LUT_CACHE: dict = {}
_DEVICE_ROPE_CACHE: dict = {}


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


def _pos_max_plus_one(pos: torch.Tensor) -> int:
    # pos is reused across all encoder/decoder blocks every inference; the
    # `.item()` host sync was repeating ~96 times per call. Memoise on object
    # id (positions are constructed once per dust3r_forward).
    key = id(pos)
    v = _POS_MAX_CACHE.get(key)
    if v is None:
        v = int(pos.max().item()) + 1
        _POS_MAX_CACHE[key] = v
    return v


def _rope_lut_host(pos: torch.Tensor, D: int, base: float, dtype):
    """Pre-indexed combined cos/sin LUTs of shape (B, 1, N, 2D).

    First D channels are vertical-pos cos/sin, last D are horizontal-pos.
    Indexing cos[pos[..., 0]] etc. happens once per inference instead of
    every RoPE call (~96× per inference for encoder + decoder).
    """
    key = (id(pos), D, base, dtype)
    lut = _ROPE_LUT_CACHE.get(key)
    if lut is not None:
        return lut
    max_pos = _pos_max_plus_one(pos)
    cos, sin = _rope_cos_sin(D, max_pos, base, dtype)
    cy = cos[pos[..., 0]][:, None, :, :]
    sy = sin[pos[..., 0]][:, None, :, :]
    cx = cos[pos[..., 1]][:, None, :, :]
    sx = sin[pos[..., 1]][:, None, :, :]
    cos_full = torch.cat((cy, cx), dim=-1).contiguous()  # (B, 1, N, 2D)
    sin_full = torch.cat((sy, sx), dim=-1).contiguous()
    lut = (cos_full, sin_full)
    _ROPE_LUT_CACHE[key] = lut
    return lut


def _rope_apply(tokens: torch.Tensor, pos: torch.Tensor, base: float = 100.0) -> torch.Tensor:
    """Apply DUSt3R 2D RoPE100 on host. tokens: (B, H, N, Dh), pos: (B, N, 2).

    Single multiply-add over the full Dh, with rotate-half applied independently
    to the y-half (first Dh//2) and x-half (last Dh//2) of each head.
    """
    Dh = tokens.shape[-1]
    D = Dh // 2
    cos_full, sin_full = _rope_lut_host(pos, D, base, tokens.dtype)

    # rotate_half_split: split into 4 quarters [a, b, c, d] (each Dh//4) → [-b, a, -d, c].
    Q = D // 2
    a = tokens[..., :Q]
    b = tokens[..., Q:2 * Q]
    c = tokens[..., 2 * Q:3 * Q]
    d = tokens[..., 3 * Q:]
    rotated = torch.cat((-b, a, -d, c), dim=-1)
    return tokens * cos_full + rotated * sin_full


def _rope_lut_device(pos: torch.Tensor, dh: int, device, base: float = 100.0):
    """cos_full / sin_full of shape (B, 1, N, dh) uploaded to device — cached
    per (pos object id, dh) so we pay one upload per inference.
    """
    key = (id(pos), dh, base, id(device))
    cached = _DEVICE_ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    cos_full, sin_full = _rope_lut_host(pos, dh // 2, base, torch.bfloat16)
    tt_cos = ttnn.from_torch(cos_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_sin = ttnn.from_torch(sin_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _DEVICE_ROPE_CACHE[key] = (tt_cos, tt_sin)
    return tt_cos, tt_sin


def _rope_device(tt_t, tt_cos, tt_sin, B: int, H: int, N: int, dh: int):
    """Apply DUSt3R 2D RoPE on device. tt_t: (B, H, N, dh) — split y/x halves,
    rotate each half independently via the [-b, a, -d, c] pattern.
    """
    Q = dh // 4
    a = ttnn.slice(tt_t, [0, 0, 0, 0],         [B, H, N, Q])
    b = ttnn.slice(tt_t, [0, 0, 0, Q],         [B, H, N, 2 * Q])
    c = ttnn.slice(tt_t, [0, 0, 0, 2 * Q],     [B, H, N, 3 * Q])
    d = ttnn.slice(tt_t, [0, 0, 0, 3 * Q],     [B, H, N, 4 * Q])
    rotated = ttnn.concat([ttnn.neg(b), a, ttnn.neg(d), c], dim=-1)
    return ttnn.add(ttnn.mul(tt_t, tt_cos), ttnn.mul(rotated, tt_sin))


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

    # Single fused op replaces 3 slices + reshape + permute + concat — v stays on device.
    tt_q, tt_k, tt_v = ttnn.transformer.split_query_key_value_and_split_heads(
        tt_qkv, num_heads=heads, transpose_key=False
    )
    # On-device 2D RoPE — eliminates the host roundtrip for q/k.
    tt_cos, tt_sin = _rope_lut_device(pos, dh, device)
    tt_q = _rope_device(tt_q, tt_cos, tt_sin, B, heads, N, dh)
    tt_k = _rope_device(tt_k, tt_cos, tt_sin, B, heads, N, dh)
    tt_ctx = ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False)
    tt_ctx = ttnn.transformer.concatenate_heads(tt_ctx)  # (B, N, D)
    tt_proj = ttnn.linear(tt_ctx, tt_w["pw"], bias=tt_w["pb"])

    tt_x = ttnn.add(tt_x, tt_proj)

    tt_n2 = ttnn.layer_norm(tt_x, weight=tt_w["g2"], bias=tt_w["b2"], epsilon=1e-6)
    tt_h = ttnn.linear(tt_n2, tt_w["w1"], bias=tt_w["b1f"], activation="gelu")
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
    return_device: bool = False,
):
    """Full encoder: patch_embed -> 24 encoder blocks -> enc_norm.

    img: (B, 3, H, W) torch. Returns (B, N, 1024) — host torch by default,
    device tensor when ``return_device=True`` (used by dust3r_forward to feed
    the decoder without a round-trip).
    """
    # Patch embed on device, output stays device-resident.
    B, C, H, W = img.shape
    hp, wp = H // 16, W // 16
    tt_x = patch_embed_device(img, state["patch_embed.proj.weight"], state["patch_embed.proj.bias"], device)
    # Make positions matching reference (row-major y then x).
    ys = torch.arange(hp)
    xs = torch.arange(wp)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack((gy, gx), dim=-1).reshape(hp * wp, 2).unsqueeze(0).expand(B, -1, -1).contiguous()

    # Pre-upload all encoder block weights once (cached on module as a simple memo).
    cache_key = id(state)
    if not hasattr(full_encoder, "_cache") or full_encoder._cache_key != cache_key:
        full_encoder._cache = [_preload_enc_block_weights(state, i, device) for i in range(depth)]
        full_encoder._enc_norm_g = _t2d(state["enc_norm.weight"].reshape(1, 1, -1), device)
        full_encoder._enc_norm_b = _t2d(state["enc_norm.bias"].reshape(1, 1, -1), device)
        full_encoder._cache_key = cache_key

    for i in range(depth):
        tt_x = encoder_block_device_pre(tt_x, pos, full_encoder._cache[i], device)

    tt_x = ttnn.layer_norm(tt_x, weight=full_encoder._enc_norm_g, bias=full_encoder._enc_norm_b, epsilon=1e-6)
    if return_device:
        return tt_x
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
    # Fuse cross-attn K and V projections into one linear so we can use
    # split_query_key_value_and_split_heads(qc, kv_input_tensor=kvc, ...).
    ck_w_t = w("cross_attn.projk.weight")
    cv_w_t = w("cross_attn.projv.weight")
    ckv_w_torch = torch.cat([ck_w_t, cv_w_t], dim=0)            # (2*D, D) (out, in)
    ckv_b_torch = torch.cat([w("cross_attn.projk.bias"), w("cross_attn.projv.bias")], dim=0)
    return {
        "g1": vec(w("norm1.weight")), "b1v": vec(w("norm1.bias")),
        "g2": vec(w("norm2.weight")), "b2v": vec(w("norm2.bias")),
        "g3": vec(w("norm3.weight")), "b3v": vec(w("norm3.bias")),
        "gy": vec(w("norm_y.weight")), "byv": vec(w("norm_y.bias")),
        "qkv_w": mat(w("attn.qkv.weight")), "qkv_b": vec(w("attn.qkv.bias")),
        "proj_w": mat(w("attn.proj.weight")), "proj_b": vec(w("attn.proj.bias")),
        "cq_w": mat(w("cross_attn.projq.weight")), "cq_b": vec(w("cross_attn.projq.bias")),
        "ckv_w": mat(ckv_w_torch), "ckv_b": vec(ckv_b_torch),
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
    tt_q, tt_k, tt_v = ttnn.transformer.split_query_key_value_and_split_heads(
        tt_qkv, num_heads=heads, transpose_key=False
    )
    tt_cos, tt_sin = _rope_lut_device(pos_x, dh, device)
    tt_q = _rope_device(tt_q, tt_cos, tt_sin, B, heads, N, dh)
    tt_k = _rope_device(tt_k, tt_cos, tt_sin, B, heads, N, dh)
    tt_ctx = ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False)
    tt_ctx = ttnn.transformer.concatenate_heads(tt_ctx)  # 1 op vs permute+reshape
    tt_proj = ttnn.linear(tt_ctx, tt_w["proj_w"], bias=tt_w["proj_b"])
    tt_x = ttnn.add(tt_x, tt_proj)

    # --- cross-attention (SDPA) ---
    # Fused K+V linear + split_qkv reduces from 3 linears + 5 reshape/permute ops
    # to 2 linears + 1 split.
    tt_n2 = ttnn.layer_norm(tt_x, weight=tt_w["g2"], bias=tt_w["b2v"], epsilon=1e-6)
    tt_yn = ttnn.layer_norm(tt_y, weight=tt_w["gy"], bias=tt_w["byv"], epsilon=1e-6)
    tt_qc = ttnn.linear(tt_n2, tt_w["cq_w"], bias=tt_w["cq_b"])
    tt_kvc = ttnn.linear(tt_yn, tt_w["ckv_w"], bias=tt_w["ckv_b"])  # (B, M, 2*D)
    tt_q, tt_k, tt_v = ttnn.transformer.split_query_key_value_and_split_heads(
        tt_qc, kv_input_tensor=tt_kvc, num_heads=heads, transpose_key=False
    )
    tt_cos_q, tt_sin_q = _rope_lut_device(pos_x, dh, device)
    tt_cos_k, tt_sin_k = _rope_lut_device(pos_y, dh, device)
    M = int(tt_y.shape[1])
    tt_q = _rope_device(tt_q, tt_cos_q, tt_sin_q, B, heads, N, dh)
    tt_k = _rope_device(tt_k, tt_cos_k, tt_sin_k, B, heads, M, dh)
    tt_ctx = ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False)
    tt_ctx = ttnn.transformer.concatenate_heads(tt_ctx)
    tt_cproj = ttnn.linear(tt_ctx, tt_w["cp_w"], bias=tt_w["cp_b"])
    tt_x = ttnn.add(tt_x, tt_cproj)

    # --- MLP ---
    tt_n3 = ttnn.layer_norm(tt_x, weight=tt_w["g3"], bias=tt_w["b3v"], epsilon=1e-6)
    tt_h = ttnn.linear(tt_n3, tt_w["fc1_w"], bias=tt_w["fc1_b"], activation="gelu")
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
    compute_norm: bool = True,
):
    """Dual-branch DUSt3R decoder on TT.

    feat1, feat2: (B, N, 1024) encoder outputs.  pos: (B, N, 2).
    Returns (out1, out2) each (B, N, 768).

    When ``compute_norm=False`` (the dust3r_forward path), the final dec_norm
    + downloads are skipped — DPT only consumes tap-layer features.
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

    # feat1/feat2 may already be device tensors (from full_encoder return_device=True)
    # — accept either form so we can avoid the encoder→decoder round-trip.
    tt_in1 = feat1 if isinstance(feat1, ttnn.Tensor) else _t2d(feat1, device)
    tt_in2 = feat2 if isinstance(feat2, ttnn.Tensor) else _t2d(feat2, device)
    tt_f1 = ttnn.linear(tt_in1, full_decoder._emb_w, bias=full_decoder._emb_b)
    tt_f2 = ttnn.linear(tt_in2, full_decoder._emb_w, bias=full_decoder._emb_b)

    # Hold tap tensors on device through the loop, batch all downloads at the
    # end so we pay one sync wave instead of three mid-loop syncs per branch.
    dev_taps1: list = []
    dev_taps2: list = []
    tap_layers = (0, 6, 11)
    for i in range(depth):
        nf1 = decoder_block_device_pre(tt_f1, tt_f2, pos, pos, full_decoder._w1[i], device)
        nf2 = decoder_block_device_pre(tt_f2, tt_f1, pos, pos, full_decoder._w2[i], device)
        tt_f1, tt_f2 = nf1, nf2
        if i in tap_layers:
            dev_taps1.append(tt_f1)
            dev_taps2.append(tt_f2)

    if compute_norm:
        tt_out1 = ttnn.layer_norm(tt_f1, weight=full_decoder._dnorm_g, bias=full_decoder._dnorm_b, epsilon=1e-6)
        tt_out2 = ttnn.layer_norm(tt_f2, weight=full_decoder._dnorm_g, bias=full_decoder._dnorm_b, epsilon=1e-6)
        out1 = ttnn.to_torch(tt_out1)
        out2 = ttnn.to_torch(tt_out2)
    else:
        out1 = out2 = None
    taps1 = [ttnn.to_torch(t) for t in dev_taps1]
    taps2 = [ttnn.to_torch(t) for t in dev_taps2]
    return out1, out2, taps1, taps2


def dust3r_forward(img1: torch.Tensor, img2: torch.Tensor, state: dict, device):
    """Full end-to-end DUSt3R forward on TT (with host DPT fallback)."""
    # Keep encoder outputs on device for direct decoder use; download once at end.
    tt_enc1 = full_encoder(img1, state, device, return_device=True)
    tt_enc2 = full_encoder(img2, state, device, return_device=True)
    B, C, H, W = img1.shape
    hp, wp = H // 16, W // 16
    ys = torch.arange(hp)
    xs = torch.arange(wp)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack((gy, gx), dim=-1).reshape(hp * wp, 2).unsqueeze(0).expand(B, -1, -1).contiguous()

    _, _, taps1, taps2 = full_decoder(tt_enc1, tt_enc2, pos, state, device, compute_norm=False)
    # Now download enc tensors for DPT head input.
    enc1 = ttnn.to_torch(tt_enc1)
    enc2 = ttnn.to_torch(tt_enc2)
    feats1 = [enc1, taps1[0], taps1[1], taps1[2]]
    feats2 = [enc2, taps2[0], taps2[1], taps2[2]]

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
    """DPT head — host-torch bfloat16 + channels_last memory layout for fast convs."""
    from reference.torch_dust3r import load_dpt_head
    key = (id(state), branch, "bf16cl", hw)
    head = _DPT_HEAD_CACHE.get(key)
    if head is None:
        head = (
            load_dpt_head(state, branch=branch)
            .to(torch.bfloat16)
            .to(memory_format=torch.channels_last)
            .eval()
        )
        _DPT_HEAD_CACHE[key] = head
    feats_bf16 = [f.to(torch.bfloat16) for f in feats_list]
    with torch.no_grad():
        return head(feats_bf16, hw).float()
