# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Standalone PyTorch reference implementations for Molmo2-8B blocks.

These are pure-PyTorch functions that replicate the behavior of each block
without importing from HuggingFace or TTNN. They are used to:
1. Verify correctness of TTNN implementations (generate golden tensors)
2. Serve as readable documentation of each block's math

Conventions:
- All inputs/outputs are float32 unless noted
- State dicts use the SAME key structure as HuggingFace checkpoints
- Seeds must be set by callers for reproducibility (torch.manual_seed(0))
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# ViT blocks
# ---------------------------------------------------------------------------


def vision_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """LayerNorm as used in the Molmo2 ViT (with bias, eps=1e-6)."""
    return F.layer_norm(x, [x.size(-1)], weight=weight, bias=bias, eps=eps)


def vision_attention_forward(
    x: torch.Tensor,
    state_dict: Dict,
    prefix: str,
    num_heads: int = 16,
    head_dim: int = 72,
) -> torch.Tensor:
    """
    Multi-head attention for Molmo2 ViT.

    Separate wq/wk/wv/wo projections, all with bias.
    Bidirectional (no causal mask).

    Args:
        x: [B, N, hidden_dim]
        state_dict: dict containing {prefix}.wq/wk/wv/wo .weight/.bias
        prefix: key prefix, e.g. "model.vision_backbone.image_vit.transformer.resblocks.0.attention"
        num_heads: 16 for Molmo2 ViT
        head_dim: 72 for Molmo2 ViT
    Returns:
        [B, N, hidden_dim]
    """
    B, N, C = x.shape
    scale = head_dim**-0.5

    q = F.linear(x, state_dict[f"{prefix}.wq.weight"], state_dict[f"{prefix}.wq.bias"])
    k = F.linear(x, state_dict[f"{prefix}.wk.weight"], state_dict[f"{prefix}.wk.bias"])
    v = F.linear(x, state_dict[f"{prefix}.wv.weight"], state_dict[f"{prefix}.wv.bias"])

    # Reshape to [B, num_heads, N, head_dim]
    q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, N, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention (bidirectional)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)

    # Merge heads and project
    out = out.transpose(1, 2).contiguous().view(B, N, num_heads * head_dim)
    return F.linear(out, state_dict[f"{prefix}.wo.weight"], state_dict[f"{prefix}.wo.bias"])


def vision_mlp_forward(
    x: torch.Tensor,
    state_dict: Dict,
    prefix: str,
) -> torch.Tensor:
    """
    2-layer GELU MLP as used in Molmo2 ViT.

    Both fc1 and fc2 have bias. Uses QuickGELU (tanh approximation).

    Args:
        x: [B, N, hidden_dim]
        state_dict: dict containing {prefix}.w1/w2 .weight/.bias
        prefix: key prefix, e.g. "model.vision_backbone.image_vit.transformer.resblocks.0.feed_forward"
    Returns:
        [B, N, hidden_dim]
    """
    h = F.linear(x, state_dict[f"{prefix}.w1.weight"], state_dict[f"{prefix}.w1.bias"])
    h = F.gelu(h, approximate="tanh")
    return F.linear(h, state_dict[f"{prefix}.w2.weight"], state_dict[f"{prefix}.w2.bias"])


def vision_block_forward(
    x: torch.Tensor,
    state_dict: Dict,
    layer_num: int,
    num_heads: int = 16,
    head_dim: int = 72,
    layer_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Full ViT block: pre-norm attention + pre-norm MLP.

    Args:
        x: [B, N, 1152]
        state_dict: HuggingFace state dict (full model or sliced)
        layer_num: Block index (0-24)
        num_heads: 16
        head_dim: 72
        layer_norm_eps: 1e-6
    Returns:
        [B, N, 1152]
    """
    base = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}"

    # Attention with pre-norm
    ln1_w = state_dict[f"{base}.attention_norm.weight"]
    ln1_b = state_dict[f"{base}.attention_norm.bias"]
    attn_in = vision_layernorm(x, ln1_w, ln1_b, layer_norm_eps)
    x = x + vision_attention_forward(attn_in, state_dict, f"{base}.attention", num_heads, head_dim)

    # MLP with pre-norm
    ln2_w = state_dict[f"{base}.ffn_norm.weight"]
    ln2_b = state_dict[f"{base}.ffn_norm.bias"]
    mlp_in = vision_layernorm(x, ln2_w, ln2_b, layer_norm_eps)
    x = x + vision_mlp_forward(mlp_in, state_dict, f"{base}.feed_forward")

    return x


def vision_transformer_forward(
    pixel_values: torch.Tensor,
    state_dict: Dict,
    num_layers: int = 25,
    num_heads: int = 16,
    head_dim: int = 72,
    patch_size: int = 14,
    layer_norm_eps: float = 1e-6,
) -> list:
    """
    Full ViT encoder: patch_embed + pos_embed + N blocks.

    Args:
        pixel_values: [B, C, H, W] images (H=W=378 for native)
        state_dict: HuggingFace state dict
        num_layers: 25 (Molmo2 uses first 25 of 27 blocks)
        num_heads, head_dim: 16, 72
        patch_size: 14
        layer_norm_eps: 1e-6
    Returns:
        List of num_layers tensors, each [B, N_patches, 1152]
    """
    prefix = "model.vision_backbone.image_vit"

    # Patch embedding: fold image into patches and apply linear+bias
    B, C, H, W = pixel_values.shape
    x = pixel_values.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size)
    x = F.linear(
        x,
        state_dict[f"{prefix}.patch_embedding.weight"],
        state_dict[f"{prefix}.patch_embedding.bias"],
    )

    # Add positional embedding (native 378x378 → 27x27=729 patches)
    pos_emb = state_dict[f"{prefix}.positional_embedding"]
    if x.shape[1] != pos_emb.shape[0]:
        # Bicubic interpolation for non-native sizes
        n_h = int(math.sqrt(x.shape[1]))
        n_h_ref = int(math.sqrt(pos_emb.shape[0]))
        pos_2d = pos_emb.reshape(n_h_ref, n_h_ref, -1).permute(2, 0, 1).unsqueeze(0)
        pos_2d = F.interpolate(pos_2d.float(), size=(n_h, n_h), mode="bicubic", align_corners=False)
        pos_emb = pos_2d.squeeze(0).permute(1, 2, 0).reshape(-1, pos_emb.shape[-1])
    x = x + pos_emb.unsqueeze(0)

    hidden_states = []
    for i in range(num_layers):
        x = vision_block_forward(x, state_dict, i, num_heads, head_dim, layer_norm_eps)
        hidden_states.append(x.clone())

    return hidden_states


# ---------------------------------------------------------------------------
# Vision Adapter
# ---------------------------------------------------------------------------


def image_pooling_forward(
    features: torch.Tensor,
    pooled_patches_idx: torch.Tensor,
    state_dict: Dict,
    prefix: str = "model.vision_backbone.image_pooling_2d",
    num_heads: int = 16,
    head_dim: int = 72,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cross-attention pooling using pooled_patches_idx neighborhoods.

    Args:
        features: [B, T*N, 2304] multi-scale features (concat of ViT layers 24+18)
        pooled_patches_idx: [B, N_out, K_pool] indices into features dim-1
        state_dict: HuggingFace state dict
        prefix: key prefix for pooling weights
        num_heads: 16
        head_dim: 72
        attn_mask: optional [B, N_out, K_pool] float mask (0=valid, -inf=masked)
    Returns:
        [B, N_out, hidden_dim=1152]
    """
    B, TN, C_in = features.shape
    _, N_out, K_pool = pooled_patches_idx.shape
    scale = head_dim**-0.5

    # Gather: [B, N_out, K_pool, C_in]
    idx_expanded = pooled_patches_idx.unsqueeze(-1).expand(B, N_out, K_pool, C_in)
    gathered = torch.gather(features.unsqueeze(1).expand(B, N_out, TN, C_in), 2, idx_expanded)

    # Query: masked mean over K_pool dim -> [B, N_out, 1, C_in]
    if attn_mask is not None:
        valid = (attn_mask == 0).float().unsqueeze(-1)
        query = (gathered * valid).sum(dim=2, keepdim=True) / valid.sum(dim=2, keepdim=True).clamp(min=1)
    else:
        query = gathered.mean(dim=2, keepdim=True)

    # Flatten to [B*N_out, 1, C_in] and [B*N_out, K_pool, C_in]
    query = query.reshape(B * N_out, 1, C_in)
    kv = gathered.reshape(B * N_out, K_pool, C_in)

    # Q, K, V projections
    q = F.linear(query, state_dict[f"{prefix}.wq.weight"], state_dict[f"{prefix}.wq.bias"])
    k = F.linear(kv, state_dict[f"{prefix}.wk.weight"], state_dict[f"{prefix}.wk.bias"])
    v = F.linear(kv, state_dict[f"{prefix}.wv.weight"], state_dict[f"{prefix}.wv.bias"])

    hidden_dim = q.shape[-1]
    q = q.view(B * N_out, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(B * N_out, K_pool, num_heads, head_dim).transpose(1, 2)
    v = v.view(B * N_out, K_pool, num_heads, head_dim).transpose(1, 2)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        # attn_mask: [B, N_out, K_pool] → [B*N_out, 1, 1, K_pool]
        mask = attn_mask.reshape(B * N_out, 1, 1, K_pool)
        attn_scores = attn_scores + mask
    attn_w = F.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_w, v)  # [B*N_out, num_heads, 1, head_dim]

    out = out.transpose(1, 2).contiguous().view(B * N_out, 1, hidden_dim)
    out = F.linear(out, state_dict[f"{prefix}.wo.weight"], state_dict[f"{prefix}.wo.bias"])
    return out.view(B, N_out, hidden_dim)


def image_projector_forward(
    x: torch.Tensor,
    state_dict: Dict,
    prefix: str = "model.vision_backbone.image_projector",
) -> torch.Tensor:
    """
    SwiGLU projector: 1152 -> 12288 -> 4096.

    No bias on any projection.

    Args:
        x: [..., 1152]
        state_dict: HuggingFace state dict
        prefix: key prefix for projector weights
    Returns:
        [..., 4096]
    """
    gate = F.silu(F.linear(x, state_dict[f"{prefix}.w1.weight"]))
    up = F.linear(x, state_dict[f"{prefix}.w3.weight"])
    return F.linear(gate * up, state_dict[f"{prefix}.w2.weight"])


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm as used in Molmo2 text model (no bias)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def text_attention_forward(
    x: torch.Tensor,
    state_dict: Dict,
    layer_num: int,
    position_ids: torch.Tensor,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    rope_theta: float = 1_000_000.0,
    rms_norm_eps: float = 1e-5,
    prefix: str = "model.transformer.blocks",
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    GQA with QK-norm and RoPE for Molmo2 text model.

    Args:
        x: [B, S, hidden_dim]
        state_dict: HuggingFace state dict
        layer_num: block index (0-35)
        position_ids: [B, S] token position indices
        num_heads: 32
        num_kv_heads: 8
        head_dim: 128
        rope_theta: 1,000,000
        rms_norm_eps: 1e-5
        prefix: key prefix
        kv_cache: optional tuple (k_cache, v_cache) from previous steps
    Returns:
        [B, S, hidden_dim]
    """
    B, S, C = x.shape
    scale = head_dim**-0.5
    p = f"{prefix}.{layer_num}.self_attn"

    # Fused QKV projection (no bias in Molmo2 text attention)
    qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = F.linear(x, state_dict[f"{p}.att_proj.weight"])  # [B, S, qkv_size]
    q, k, v = qkv.split([num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)

    # QK-norm (qwen3-style: per-head RMSNorm)
    q_norm_w = state_dict[f"{p}.q_norm.weight"]  # [head_dim]
    k_norm_w = state_dict[f"{p}.k_norm.weight"]  # [head_dim]
    q = q.view(B, S, num_heads, head_dim)
    k = k.view(B, S, num_kv_heads, head_dim)
    q = rmsnorm(q, q_norm_w, rms_norm_eps)
    k = rmsnorm(k, k_norm_w, rms_norm_eps)

    # Apply RoPE
    q, k = _apply_rope(q, k, position_ids, head_dim, rope_theta)

    # Reshape to [B, heads, S, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

    # GQA: expand KV heads to match Q heads
    groups = num_heads // num_kv_heads
    k = k.unsqueeze(2).expand(B, num_kv_heads, groups, S, head_dim).reshape(B, num_heads, S, head_dim)
    v = v.unsqueeze(2).expand(B, num_kv_heads, groups, S, head_dim).reshape(B, num_heads, S, head_dim)

    # Causal attention
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
    attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
    out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(B, S, num_heads * head_dim)
    return F.linear(out, state_dict[f"{p}.attn_out.weight"])


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to q and k."""
    B, S, n_q, _ = q.shape
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=q.device).float() / half))
    t = position_ids.float()  # [B, S]
    freqs_expanded = torch.einsum("bs,d->bsd", t, freqs)  # [B, S, half]
    cos = freqs_expanded.cos().unsqueeze(2)  # [B, S, 1, half]
    sin = freqs_expanded.sin().unsqueeze(2)

    def rotate(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    q = q * torch.cat([cos, cos], dim=-1) + rotate(q) * torch.cat([sin, sin], dim=-1)
    k = k * torch.cat([cos, cos], dim=-1) + rotate(k) * torch.cat([sin, sin], dim=-1)
    return q, k


def text_mlp_forward(
    x: torch.Tensor,
    state_dict: Dict,
    layer_num: int,
    prefix: str = "model.transformer.blocks",
) -> torch.Tensor:
    """
    SwiGLU MLP for Molmo2 text model.

    ff_proj is fused [up; gate] (first half up, second half gate) → silu(gate) * up → ff_out.
    No bias.

    Args:
        x: [B, S, hidden_dim]
        state_dict: HuggingFace state dict
        layer_num: block index (0-35)
        prefix: key prefix
    Returns:
        [B, S, hidden_dim]
    """
    p = f"{prefix}.{layer_num}.mlp"
    ff_proj = F.linear(x, state_dict[f"{p}.ff_proj.weight"])
    intermediate = ff_proj.shape[-1] // 2
    up, gate = ff_proj.split(intermediate, dim=-1)
    hidden = F.silu(gate) * up
    return F.linear(hidden, state_dict[f"{p}.ff_out.weight"])


def text_block_forward(
    x: torch.Tensor,
    state_dict: Dict,
    layer_num: int,
    position_ids: torch.Tensor,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    rope_theta: float = 1_000_000.0,
    rms_norm_eps: float = 1e-5,
    prefix: str = "model.transformer.blocks",
) -> torch.Tensor:
    """
    Full text decoder block: pre-norm attention + pre-norm MLP.

    Args:
        x: [B, S, 4096]
        state_dict: HuggingFace state dict
        layer_num: block index (0-35)
        position_ids: [B, S]
        ... config params ...
    Returns:
        [B, S, 4096]
    """
    p = f"{prefix}.{layer_num}"

    # Attention with pre-norm
    attn_norm_w = state_dict[f"{p}.attn_norm.weight"]
    attn_in = rmsnorm(x, attn_norm_w, rms_norm_eps)
    x = x + text_attention_forward(
        attn_in,
        state_dict,
        layer_num,
        position_ids,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        rms_norm_eps,
        prefix,
    )

    # MLP with pre-norm
    ff_norm_w = state_dict[f"{p}.ff_norm.weight"]
    mlp_in = rmsnorm(x, ff_norm_w, rms_norm_eps)
    x = x + text_mlp_forward(mlp_in, state_dict, layer_num, prefix)

    return x


# ---------------------------------------------------------------------------
# Golden tensor utilities
# ---------------------------------------------------------------------------


def generate_vision_block_golden(state_dict: Dict, layer_num: int = 0) -> dict:
    """
    Generate reproducible golden inputs/outputs for a ViT block.

    Returns:
        dict with 'input', 'output', 'config' keys
    """
    torch.manual_seed(0)
    x = torch.randn(1, 729, 1152, dtype=torch.float32)
    output = vision_block_forward(x, state_dict, layer_num)
    return {
        "input": x,
        "output": output,
        "config": {"layer_num": layer_num, "batch": 1, "seq": 729, "hidden": 1152},
    }


def generate_image_projector_golden(state_dict: Dict, num_tokens: int = 256) -> dict:
    """
    Generate reproducible golden inputs/outputs for the image projector.

    Returns:
        dict with 'input', 'output', 'config' keys
    """
    torch.manual_seed(0)
    x = torch.randn(num_tokens, 1152, dtype=torch.float32)
    output = image_projector_forward(x, state_dict)
    return {
        "input": x,
        "output": output,
        "config": {"num_tokens": num_tokens, "input_dim": 1152, "output_dim": 4096},
    }


def generate_text_block_golden(state_dict: Dict, layer_num: int = 0, seq_len: int = 128) -> dict:
    """
    Generate reproducible golden inputs/outputs for a text decoder block.

    Returns:
        dict with 'input', 'output', 'config' keys
    """
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, 4096, dtype=torch.float32)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    output = text_block_forward(x, state_dict, layer_num, position_ids)
    return {
        "input": x,
        "output": output,
        "position_ids": position_ids,
        "config": {"layer_num": layer_num, "batch": 1, "seq": seq_len, "hidden": 4096},
    }
