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


def tt_vit_attention(
    x,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    num_heads,
    rope_cos=None,
    rope_sin=None,
    freqs_cis=None,
    device=None,
):
    """Multi-head attention for SAM3 ViT.

    Args:
        x: ttnn tensor (B, L, dim) in TILE_LAYOUT on device.
        qkv_weight: ttnn tensor (dim, dim*3) - preprocessed fused QKV weight.
        qkv_bias: ttnn tensor (1, 1, dim*3) - preprocessed fused QKV bias.
        proj_weight: ttnn tensor (dim, dim) - preprocessed output projection weight.
        proj_bias: ttnn tensor (1, 1, dim) - preprocessed output projection bias.
        num_heads: int, number of attention heads.
        rope_cos: unused, reserved for future native ttnn RoPE.
        rope_sin: unused, reserved for future native ttnn RoPE.
        freqs_cis: torch complex tensor (L, head_dim//2) for RoPE, or None.
        device: ttnn device.

    Returns:
        ttnn tensor (B, L, dim) in TILE_LAYOUT on device.
    """
    # --- QKV projection ---
    qkv = ttnn.linear(x, qkv_weight, bias=qkv_bias)  # (B, L, dim*3)

    # --- Move to torch for reshape / RoPE / attention ---
    # TODO: Implement reshape, permute, RoPE, and attention natively in ttnn.
    qkv_torch = ttnn.to_torch(qkv).float()
    B, L, _ = qkv_torch.shape
    head_dim = qkv_torch.shape[-1] // (3 * num_heads)

    # Reshape: (B, L, 3, num_heads, head_dim) -> permute to (3, B, num_heads, L, head_dim)
    qkv_torch = qkv_torch.reshape(B, L, 3, num_heads, head_dim)
    q, k, v = qkv_torch.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, num_heads, L, head_dim)

    # Apply RoPE if provided
    if freqs_cis is not None:
        q, k = _apply_rope_torch(q, k, freqs_cis)

    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nH, L, L)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_weights, v)  # (B, nH, L, head_dim)

    # Concatenate heads: (B, L, dim)
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, -1)

    # Convert back to ttnn for output projection
    tt_attn_out = ttnn.from_torch(attn_out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # --- Output projection ---
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
    """Single ViT transformer block.

    Implements: x = x + Attention(LayerNorm(x))
                x = x + MLP(LayerNorm(x))

    With window partition/unpartition for windowed attention blocks.

    Args:
        x: ttnn tensor (B, H, W, C) in TILE_LAYOUT on device.
        block_params: dict with preprocessed weights (from preprocess_vit_block_weights).
        num_heads: int, number of attention heads.
        window_size: int, window size (0 for global attention).
        device: ttnn device.

    Returns:
        ttnn tensor (B, H, W, C) in TILE_LAYOUT on device.
    """
    # --- Pre-attention LayerNorm ---
    # Convert to torch for LN + reshape operations
    # TODO: Implement layer_norm natively in ttnn for 4D tensors
    x_torch = ttnn.to_torch(x).float()
    B, H, W, C = x_torch.shape
    shortcut = x_torch

    # LayerNorm operates on last dim
    ln1_w = block_params["norm1_weight"]
    ln1_b = block_params["norm1_bias"]
    normed = torch.nn.functional.layer_norm(x_torch, [C], ln1_w, ln1_b, eps=1e-5)

    # --- Window partition (if windowed attention) ---
    if window_size > 0:
        # Window partition on CPU
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            normed = F.pad(normed, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        normed = normed.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        normed = normed.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
        # Flatten spatial dims for attention: (B*nW, ws*ws, C)
        attn_input = normed.reshape(normed.shape[0], window_size * window_size, C)
    else:
        Hp, Wp = H, W
        # Flatten spatial dims for attention: (B, H*W, C)
        attn_input = normed.reshape(B, H * W, C)

    # Move to ttnn for linear projections in attention
    tt_attn_input = ttnn.from_torch(
        attn_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # --- Attention ---
    tt_attn_out = tt_vit_attention(
        tt_attn_input,
        block_params["attn_qkv_weight"],
        block_params["attn_qkv_bias"],
        block_params["attn_proj_weight"],
        block_params["attn_proj_bias"],
        num_heads=num_heads,
        freqs_cis=block_params.get("freqs_cis"),
        device=device,
    )

    # Convert attention output back to torch
    attn_out = ttnn.to_torch(tt_attn_out).float()

    # --- Window unpartition ---
    if window_size > 0:
        # Reshape back to windowed spatial
        attn_out = attn_out.reshape(-1, window_size, window_size, C)
        # Unpartition
        nW_h = Hp // window_size
        nW_w = Wp // window_size
        attn_out = attn_out.reshape(B, nW_h, nW_w, window_size, window_size, C)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if Hp > H or Wp > W:
            attn_out = attn_out[:, :H, :W, :].contiguous()
    else:
        attn_out = attn_out.reshape(B, H, W, C)

    # --- Residual connection ---
    x_torch = shortcut + attn_out

    # --- Post-attention LayerNorm + MLP ---
    shortcut2 = x_torch
    ln2_w = block_params["norm2_weight"]
    ln2_b = block_params["norm2_bias"]
    normed2 = torch.nn.functional.layer_norm(x_torch, [C], ln2_w, ln2_b, eps=1e-5)

    # Reshape for MLP: (B, H*W, C)
    mlp_input = normed2.reshape(B, H * W, C)
    tt_mlp_input = ttnn.from_torch(
        mlp_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # MLP
    tt_mlp_out = tt_vit_mlp(
        tt_mlp_input,
        block_params["mlp_fc1_weight"],
        block_params["mlp_fc1_bias"],
        block_params["mlp_fc2_weight"],
        block_params["mlp_fc2_bias"],
        device=device,
    )

    mlp_out = ttnn.to_torch(tt_mlp_out).float()
    mlp_out = mlp_out.reshape(B, H, W, C)

    # --- Residual connection ---
    output = shortcut2 + mlp_out

    # Convert final result back to ttnn
    tt_output = ttnn.from_torch(
        output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    return tt_output


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

    for i in range(num_blocks):
        block_params = backbone_params["blocks"][i]
        window_size = block_params["window_size"]

        # Convert to ttnn
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        tt_x = tt_vit_block(
            tt_x,
            block_params["tt_params"],
            num_heads=16,
            window_size=window_size,
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


def preprocess_vit_attention_weights(attn_module):
    """Extract and preprocess QKV and projection weights from a PyTorch Attention module.

    The reference model has:
        attn_module.qkv: nn.Linear(dim, dim*3, bias=True)
        attn_module.proj: nn.Linear(dim, dim, bias=True)

    Weights are transposed from PyTorch [out, in] to ttnn [in, out] format.
    Biases are reshaped to [1, 1, out] for broadcasting.

    Args:
        attn_module: PyTorch Attention module with .qkv and .proj attributes.

    Returns:
        dict with keys:
            'qkv_weight': ttnn tensor (dim, dim*3) in TILE_LAYOUT.
            'qkv_bias': ttnn tensor (1, 1, dim*3) in TILE_LAYOUT.
            'proj_weight': ttnn tensor (dim, dim) in TILE_LAYOUT.
            'proj_bias': ttnn tensor (1, 1, dim) in TILE_LAYOUT.
            'freqs_cis': torch complex tensor (L, head_dim//2) or None.
    """
    # QKV: original shape [dim*3, dim] -> transpose to [dim, dim*3]
    qkv_w = attn_module.qkv.weight.data.T.contiguous()
    qkv_w = ttnn.from_torch(qkv_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    qkv_b = attn_module.qkv.bias.data.reshape(1, 1, -1)
    qkv_b = ttnn.from_torch(qkv_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Proj: original shape [dim, dim] -> transpose to [dim, dim]
    proj_w = attn_module.proj.weight.data.T.contiguous()
    proj_w = ttnn.from_torch(proj_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    proj_b = attn_module.proj.bias.data.reshape(1, 1, -1)
    proj_b = ttnn.from_torch(proj_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Extract RoPE frequencies if present
    freqs_cis = None
    if hasattr(attn_module, 'freqs_cis') and attn_module.freqs_cis is not None:
        freqs_cis = attn_module.freqs_cis

    return {
        "qkv_weight": qkv_w,
        "qkv_bias": qkv_b,
        "proj_weight": proj_w,
        "proj_bias": proj_b,
        "freqs_cis": freqs_cis,
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

    tt_params = {
        # Attention (ttnn tensors, not on device yet)
        "attn_qkv_weight": attn_params["qkv_weight"],
        "attn_qkv_bias": attn_params["qkv_bias"],
        "attn_proj_weight": attn_params["proj_weight"],
        "attn_proj_bias": attn_params["proj_bias"],
        "freqs_cis": attn_params["freqs_cis"],
        # LayerNorms (kept as torch tensors for CPU fallback)
        "norm1_weight": block.norm1.weight.data.clone(),
        "norm1_bias": block.norm1.bias.data.clone(),
        "norm2_weight": block.norm2.weight.data.clone(),
        "norm2_bias": block.norm2.bias.data.clone(),
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
        "mlp_fc1_weight", "mlp_fc1_bias",
        "mlp_fc2_weight", "mlp_fc2_bias",
    ]:
        if isinstance(tt_params[key], ttnn.Tensor):
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
