# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import ttnn


def tt_patch_embed(pixel_values, weight, bias, device):
    """Patch embedding via Conv2d projection.

    Runs the patch embedding conv on CPU using PyTorch for correctness,
    then converts the result to a ttnn tensor on device.

    # TODO: Replace CPU conv with ttnn.conv2d for on-device execution.

    Args:
        pixel_values: torch tensor (B, 3, 1008, 1008) - input images in NCHW format.
        weight: torch tensor (1024, 3, 14, 14) - conv weight.
        bias: None or torch tensor - no bias in SAM3 patch embed.
        device: ttnn device.

    Returns:
        ttnn tensor (B, 72, 72, 1024) in TILE_LAYOUT on device.
    """
    # Run conv2d on CPU with PyTorch for correctness.
    # Input: (B, 3, 1008, 1008), kernel: 14x14, stride: 14 -> output: (B, 1024, 72, 72)
    with torch.no_grad():
        output = F.conv2d(pixel_values, weight, bias=bias, stride=14, padding=0)

    # Permute from NCHW (B, 1024, 72, 72) to NHWC (B, 72, 72, 1024)
    output = output.permute(0, 2, 3, 1).contiguous()

    # Convert to ttnn tensor on device in TILE_LAYOUT
    tt_output = ttnn.from_torch(output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_output


def get_patch_embed_params(vit_model):
    """Extract patch embed weights from PyTorch ViT model.

    Args:
        vit_model: PyTorch ViT model containing patch_embed.proj (Conv2d).

    Returns:
        dict with keys 'weight' and 'bias' as torch tensors (or None for bias).
    """
    return {
        "weight": vit_model.patch_embed.proj.weight.data,
        "bias": vit_model.patch_embed.proj.bias.data if vit_model.patch_embed.proj.bias is not None else None,
    }


# ---------------------------------------------------------------------------
# Window partition / unpartition (torch fallback)
# ---------------------------------------------------------------------------


def tt_window_partition(x, window_size):
    """Partition a (B, H, W, C) ttnn tensor into non-overlapping windows.

    Falls back to torch for the reshape/pad logic, then converts back.

    # TODO: Implement natively in ttnn to avoid device round-trip.

    Args:
        x: ttnn tensor of shape (B, H, W, C) in TILE_LAYOUT on device.
        window_size: int, window side length.

    Returns:
        windows: ttnn tensor (B*num_windows, window_size, window_size, C) in TILE_LAYOUT on device.
        (Hp, Wp): padded height and width before partition.
    """
    device = x.device()
    x_torch = ttnn.to_torch(x).float()
    B, H, W, C = x_torch.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x_torch = F.pad(x_torch, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x_torch = x_torch.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x_torch.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)

    tt_windows = ttnn.from_torch(windows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_windows, (Hp, Wp)


def tt_window_unpartition(windows, window_size, pad_hw, hw):
    """Reverse window partition and remove padding.

    Falls back to torch for the reshape logic, then converts back.

    # TODO: Implement natively in ttnn to avoid device round-trip.

    Args:
        windows: ttnn tensor (B*num_windows, window_size, window_size, C) in TILE_LAYOUT.
        window_size: int, window side length.
        pad_hw: tuple (Hp, Wp), padded height and width.
        hw: tuple (H, W), original height and width before padding.

    Returns:
        x: ttnn tensor (B, H, W, C) in TILE_LAYOUT on device.
    """
    device = windows.device()
    w_torch = ttnn.to_torch(windows).float()

    Hp, Wp = pad_hw
    H, W = hw
    C = w_torch.shape[-1]
    B = w_torch.shape[0] // (Hp * Wp // window_size // window_size)

    x = w_torch.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_x


# ---------------------------------------------------------------------------
# RoPE helper (torch fallback)
# ---------------------------------------------------------------------------


def _apply_rope_torch(q, k, freqs_cis):
    """Apply 2D axial rotary position encoding to Q and K tensors.

    Uses torch complex arithmetic, matching the reference apply_rotary_enc.

    # TODO: Implement RoPE natively in ttnn.

    Args:
        q: torch tensor (B, num_heads, L, head_dim).
        k: torch tensor (B, num_heads, L, head_dim).
        freqs_cis: torch complex tensor (L, head_dim//2).

    Returns:
        (q_out, k_out): torch tensors with RoPE applied, same shape as input.
    """
    xq_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Broadcast freqs_cis to match (B, num_heads, L, head_dim//2)
    ndim = xq_.ndim
    shape = [1] * (ndim - 2) + list(freqs_cis.shape)
    freqs = freqs_cis.view(*shape)

    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
    return xq_out.to(q.dtype), xk_out.to(k.dtype)


# ---------------------------------------------------------------------------
# ViT Attention
# ---------------------------------------------------------------------------


def _apply_rope_ttnn(q, cos, sin):
    """Apply half-half RoPE on device: q * cos + rotate_half(q) * sin."""
    rh = ttnn.experimental.rotate_half(q)
    return ttnn.add(ttnn.multiply(q, cos), ttnn.multiply(rh, sin))


def tt_vit_attention(
    x,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    num_heads,
    rope_cos=None,
    rope_sin=None,
    freqs_cis=None,  # unused (kept for API compatibility)
    device=None,
):
    """Multi-head attention for SAM3 ViT — fully on-device.

    Requires QKV weights pre-permuted to half-half layout for Q and K
    (see _permute_qkv_for_rope in preprocessing). RoPE cos/sin are precomputed
    per attention layer with shape (1, 1, L, head_dim).
    """
    qkv = ttnn.linear(x, qkv_weight, bias=qkv_bias)  # (B, L, dim*3)

    tt_q, tt_k, tt_v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False
    )  # each (B, nH, L, head_dim)

    if rope_cos is not None:
        tt_q = _apply_rope_ttnn(tt_q, rope_cos, rope_sin)
        tt_k = _apply_rope_ttnn(tt_k, rope_cos, rope_sin)

    head_dim = tt_q.shape[-1]
    L = tt_q.shape[-2]
    scale = head_dim ** -0.5

    if L >= 1024 and (L % 32) == 0:
        tt_attn_out = ttnn.transformer.scaled_dot_product_attention(
            tt_q, tt_k, tt_v, is_causal=False, scale=scale,
        )
    else:
        tt_kt = ttnn.transpose(tt_k, -2, -1)
        tt_attn = ttnn.matmul(tt_q, tt_kt)
        tt_attn = ttnn.multiply(tt_attn, scale)
        tt_attn = ttnn.softmax(tt_attn, dim=-1)
        tt_attn_out = ttnn.matmul(tt_attn, tt_v)

    tt_attn_out = ttnn.transformer.concatenate_heads(tt_attn_out)  # (B, L, dim)
    output = ttnn.linear(tt_attn_out, proj_weight, bias=proj_bias)
    return output


# ---------------------------------------------------------------------------
# ViT MLP
# ---------------------------------------------------------------------------


def tt_vit_mlp(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, device=None):
    """ViT MLP block: Linear -> GELU -> Linear.

    Args:
        x: ttnn tensor (B, L, dim) in TILE_LAYOUT on device.
        fc1_weight: ttnn tensor (dim, hidden_dim) - preprocessed weight.
        fc1_bias: ttnn tensor (1, 1, hidden_dim) - preprocessed bias.
        fc2_weight: ttnn tensor (hidden_dim, dim) - preprocessed weight.
        fc2_bias: ttnn tensor (1, 1, dim) - preprocessed bias.
        device: ttnn device (unused, for API consistency).

    Returns:
        ttnn tensor (B, L, dim) in TILE_LAYOUT on device.
    """
    hidden = ttnn.linear(x, fc1_weight, bias=fc1_bias)
    hidden = ttnn.gelu(hidden)
    output = ttnn.linear(hidden, fc2_weight, bias=fc2_bias)
    return output


# ---------------------------------------------------------------------------
# ViT Block (Transformer Block)
# ---------------------------------------------------------------------------


def tt_vit_block(
    x,
    block_params,
    num_heads,
    window_size,
    device,
):
    """Single ViT transformer block — fully on-device.

    Takes/returns ttnn tensor (B, H, W, C). No host transfers inside the block.
    """
    tt_x = x
    B, H, W, C = tt_x.shape
    assert window_size == 0 or (H % window_size == 0 and W % window_size == 0), (
        "on-device window partition requires H,W divisible by window_size"
    )

    # LN1 on last dim (C)
    tt_normed = ttnn.layer_norm(
        tt_x,
        weight=block_params["norm1_weight_tt"],
        bias=block_params["norm1_bias_tt"],
        epsilon=1e-5,
    )

    # Window partition → (B', L, C)
    if window_size > 0:
        nW_h, nW_w = H // window_size, W // window_size
        tt_attn_input = ttnn.reshape(tt_normed, [B, nW_h, window_size, nW_w, window_size, C])
        tt_attn_input = ttnn.permute(tt_attn_input, [0, 1, 3, 2, 4, 5])
        tt_attn_input = ttnn.reshape(
            tt_attn_input, [B * nW_h * nW_w, window_size * window_size, C]
        )
    else:
        tt_attn_input = ttnn.reshape(tt_normed, [B, H * W, C])

    tt_attn_out = tt_vit_attention(
        tt_attn_input,
        block_params["attn_qkv_weight"],
        block_params["attn_qkv_bias"],
        block_params["attn_proj_weight"],
        block_params["attn_proj_bias"],
        num_heads=num_heads,
        rope_cos=block_params.get("rope_cos"),
        rope_sin=block_params.get("rope_sin"),
        device=device,
    )

    # Window unpartition → (B, H, W, C)
    if window_size > 0:
        nW_h, nW_w = H // window_size, W // window_size
        tt_attn_out = ttnn.reshape(
            tt_attn_out, [B, nW_h, nW_w, window_size, window_size, C]
        )
        tt_attn_out = ttnn.permute(tt_attn_out, [0, 1, 3, 2, 4, 5])
        tt_attn_out = ttnn.reshape(tt_attn_out, [B, H, W, C])
    else:
        tt_attn_out = ttnn.reshape(tt_attn_out, [B, H, W, C])

    tt_residual1 = ttnn.add(tt_x, tt_attn_out)
    tt_residual1_flat = ttnn.reshape(tt_residual1, [B, H * W, C])

    tt_normed2 = ttnn.layer_norm(
        tt_residual1_flat,
        weight=block_params["norm2_weight_tt"],
        bias=block_params["norm2_bias_tt"],
        epsilon=1e-5,
    )

    tt_mlp_out = tt_vit_mlp(
        tt_normed2,
        block_params["mlp_fc1_weight"],
        block_params["mlp_fc1_bias"],
        block_params["mlp_fc2_weight"],
        block_params["mlp_fc2_bias"],
        device=device,
    )

    tt_output = ttnn.add(tt_residual1_flat, tt_mlp_out)
    return ttnn.reshape(tt_output, [B, H, W, C])


# ---------------------------------------------------------------------------
# Full ViT Backbone
# ---------------------------------------------------------------------------


def tt_vit_backbone(pixel_values, backbone_params, device):
    """Full SAM3 ViT backbone forward pass.

    Args:
        pixel_values: torch tensor (B, 3, 1008, 1008) - input images.
        backbone_params: dict from preprocess_vit_backbone_weights().
        device: ttnn device.

    Returns:
        List of torch tensors [feats] where feats is (B, 1024, 72, 72) in NCHW format.
    """
    import math

    # --- Patch embedding (CPU) ---
    patch_out = tt_patch_embed(
        pixel_values,
        backbone_params["patch_embed"]["weight"],
        backbone_params["patch_embed"]["bias"],
        device,
    )
    # Convert to torch for pos embed and block processing
    x = ttnn.to_torch(patch_out).float()  # (B, 72, 72, 1024)
    B, H, W, C = x.shape

    # --- Absolute position embeddings ---
    if backbone_params.get("pos_embed") is not None:
        # Use the reference get_abs_pos function for correct tiling/interpolation
        from sam3.model.vitdet import get_abs_pos
        pos_embed_2d = get_abs_pos(
            backbone_params["pos_embed"],
            backbone_params.get("pretrain_use_cls_token", True),
            (H, W),
            retain_cls_token=False,
            tiling=backbone_params.get("tile_abs_pos", True),
        )
        x = x + pos_embed_2d

    # --- Pre-LayerNorm ---
    if backbone_params.get("ln_pre_weight") is not None:
        x = torch.nn.functional.layer_norm(
            x, [C], backbone_params["ln_pre_weight"], backbone_params["ln_pre_bias"], eps=1e-5
        )

    # --- Transformer blocks ---
    num_blocks = len(backbone_params["blocks"])
    global_att_blocks = {7, 15, 23, 31}
    last_global = max(global_att_blocks)

    # Move to device once; each block stays on-device. to_torch once after loop.
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for i in range(num_blocks):
        block_params = backbone_params["blocks"][i]
        tt_x = tt_vit_block(
            tt_x,
            block_params["tt_params"],
            num_heads=16,
            window_size=block_params["window_size"],
            device=device,
        )

    x = ttnn.to_torch(tt_x).float()

    # --- Output: convert to NCHW ---
    # x is (B, H, W, C) -> permute to (B, C, H, W)
    feats = x.permute(0, 3, 1, 2)

    return [feats]


# ---------------------------------------------------------------------------
# Weight preprocessing
# ---------------------------------------------------------------------------


def _permute_qkv_for_rope(qkv, num_heads, head_dim):
    """Permute Q and K head_dim entries (along dim 0 — the QKV output dim) from
    pair (r0,i0,r1,i1,...) to half-half (r0,r1,...,i0,i1,...). V is untouched.

    qkv: nn.Linear weight (3*dim, in_dim) or bias (3*dim,) — output dim is 0.
    """
    perm_within = list(range(0, head_dim, 2)) + list(range(1, head_dim, 2))
    perm_within = torch.tensor(perm_within, dtype=torch.long)
    out = qkv.clone()
    dim = num_heads * head_dim
    for offset in (0, dim):  # Q at 0, K at dim, skip V at 2*dim
        for h in range(num_heads):
            rows = slice(offset + h * head_dim, offset + (h + 1) * head_dim)
            out[rows] = qkv[rows].index_select(0, perm_within)
    return out


def _build_rope_cos_sin_tt(freqs_cis):
    """Build half-half cos/sin torch tensors of shape (1, 1, L, head_dim) from
    freqs_cis (L, head_dim//2) complex.
    """
    cos = freqs_cis.real
    sin = freqs_cis.imag
    cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,L,head_dim)
    sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos_full, sin_full


def preprocess_vit_attention_weights(attn_module, num_heads=16, head_dim=64):
    """Preprocess attention weights for fully on-device attention.

    Q and K columns of QKV are permuted to half-half layout so we can use
    ttnn.experimental.rotate_half + multiply/add to apply RoPE. V is untouched.

    cos/sin tensors are precomputed per layer and uploaded to device for RoPE.
    """
    qkv_w = attn_module.qkv.weight.data  # (dim*3, dim)
    qkv_w = _permute_qkv_for_rope(qkv_w, num_heads, head_dim)
    qkv_w = qkv_w.T.contiguous()  # (dim, dim*3) for ttnn linear
    qkv_w_tt = ttnn.from_torch(qkv_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    qkv_b = attn_module.qkv.bias.data  # (dim*3,)
    qkv_b = _permute_qkv_for_rope(qkv_b, num_heads, head_dim)
    qkv_b = qkv_b.reshape(1, 1, -1)
    qkv_b_tt = ttnn.from_torch(qkv_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    proj_w = attn_module.proj.weight.data.T.contiguous()
    proj_w_tt = ttnn.from_torch(proj_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    proj_b = attn_module.proj.bias.data.reshape(1, 1, -1)
    proj_b_tt = ttnn.from_torch(proj_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    rope_cos_tt = None
    rope_sin_tt = None
    if hasattr(attn_module, "freqs_cis") and attn_module.freqs_cis is not None:
        cos_full, sin_full = _build_rope_cos_sin_tt(attn_module.freqs_cis)
        rope_cos_tt = ttnn.from_torch(cos_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        rope_sin_tt = ttnn.from_torch(sin_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    return {
        "qkv_weight": qkv_w_tt,
        "qkv_bias": qkv_b_tt,
        "proj_weight": proj_w_tt,
        "proj_bias": proj_b_tt,
        "rope_cos": rope_cos_tt,
        "rope_sin": rope_sin_tt,
    }


def preprocess_vit_block_weights(block):
    """Preprocess all weights for a single ViT block.

    Args:
        block: PyTorch ViT Block module.

    Returns:
        dict with:
            'window_size': int
            'tt_params': dict of preprocessed ttnn/torch tensors for tt_vit_block
    """
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_common import (
        preprocess_linear_weight,
        preprocess_linear_bias,
    )

    # Attention weights
    attn_params = preprocess_vit_attention_weights(block.attn)

    # MLP weights
    fc1_w = preprocess_linear_weight(block.mlp.fc1.weight.data)
    fc1_b = preprocess_linear_bias(block.mlp.fc1.bias.data)
    fc2_w = preprocess_linear_weight(block.mlp.fc2.weight.data)
    fc2_b = preprocess_linear_bias(block.mlp.fc2.bias.data)

    # LN params as ttnn tensors for on-device LN.
    norm1_w_tt = ttnn.from_torch(
        block.norm1.weight.data.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    norm1_b_tt = ttnn.from_torch(
        block.norm1.bias.data.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    norm2_w_tt = ttnn.from_torch(
        block.norm2.weight.data.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    norm2_b_tt = ttnn.from_torch(
        block.norm2.bias.data.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    tt_params = {
        # Attention (ttnn tensors, not on device yet)
        "attn_qkv_weight": attn_params["qkv_weight"],
        "attn_qkv_bias": attn_params["qkv_bias"],
        "attn_proj_weight": attn_params["proj_weight"],
        "attn_proj_bias": attn_params["proj_bias"],
        "rope_cos": attn_params["rope_cos"],
        "rope_sin": attn_params["rope_sin"],
        "norm1_weight_tt": norm1_w_tt,
        "norm1_bias_tt": norm1_b_tt,
        "norm2_weight_tt": norm2_w_tt,
        "norm2_bias_tt": norm2_b_tt,
        # MLP (ttnn tensors, not on device yet)
        "mlp_fc1_weight": fc1_w,
        "mlp_fc1_bias": fc1_b,
        "mlp_fc2_weight": fc2_w,
        "mlp_fc2_bias": fc2_b,
    }

    return {
        "window_size": block.window_size,
        "tt_params": tt_params,
    }


def move_block_params_to_device(block_params, device):
    """Move ttnn tensors in block params to device.

    Args:
        block_params: dict from preprocess_vit_block_weights().
        device: ttnn device.

    Returns:
        Updated block_params with ttnn tensors on device.
    """
    tt_params = block_params["tt_params"]
    for key in [
        "attn_qkv_weight", "attn_qkv_bias",
        "attn_proj_weight", "attn_proj_bias",
        "rope_cos", "rope_sin",
        "norm1_weight_tt", "norm1_bias_tt",
        "norm2_weight_tt", "norm2_bias_tt",
        "mlp_fc1_weight", "mlp_fc1_bias",
        "mlp_fc2_weight", "mlp_fc2_bias",
    ]:
        if tt_params.get(key) is not None and isinstance(tt_params[key], ttnn.Tensor):
            tt_params[key] = ttnn.to_device(tt_params[key], device)
    return block_params


def preprocess_vit_backbone_weights(vit_model):
    """Preprocess all weights for the full ViT backbone.

    Args:
        vit_model: PyTorch ViT model (sam3.model.vitdet.ViT).

    Returns:
        dict with all preprocessed weights organized by component.
    """
    params = {}

    # Patch embedding (kept as torch tensors for CPU conv)
    params["patch_embed"] = {
        "weight": vit_model.patch_embed.proj.weight.data.clone(),
        "bias": (
            vit_model.patch_embed.proj.bias.data.clone()
            if vit_model.patch_embed.proj.bias is not None
            else None
        ),
    }

    # Position embeddings (torch tensor)
    params["pos_embed"] = (
        vit_model.pos_embed.data.clone()
        if hasattr(vit_model, "pos_embed") and vit_model.pos_embed is not None
        else None
    )
    params["pretrain_use_cls_token"] = getattr(vit_model, "pretrain_use_cls_token", True)
    params["tile_abs_pos"] = getattr(vit_model, "tile_abs_pos", True)
    params["retain_cls_token"] = getattr(vit_model, "retain_cls_token", False)

    # Pre-LayerNorm
    if hasattr(vit_model, "ln_pre") and not isinstance(vit_model.ln_pre, torch.nn.Identity):
        params["ln_pre_weight"] = vit_model.ln_pre.weight.data.clone()
        params["ln_pre_bias"] = vit_model.ln_pre.bias.data.clone()
    else:
        params["ln_pre_weight"] = None
        params["ln_pre_bias"] = None

    # Blocks
    params["blocks"] = []
    for block in vit_model.blocks:
        params["blocks"].append(preprocess_vit_block_weights(block))

    return params


def move_backbone_params_to_device(backbone_params, device):
    """Move all ttnn tensors in backbone params to device.

    Args:
        backbone_params: dict from preprocess_vit_backbone_weights().
        device: ttnn device.

    Returns:
        Updated backbone_params with ttnn tensors on device.
    """
    for block_params in backbone_params["blocks"]:
        move_block_params_to_device(block_params, device)
    return backbone_params
