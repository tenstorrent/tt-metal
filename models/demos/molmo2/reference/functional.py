"""Standalone PyTorch reference implementations for each Molmo2-8B block.

All functions operate in float32 for precision. Convert to bfloat16 before PCC comparison with TTNN.
State dicts follow the HuggingFace weight key convention documented in ARCHITECTURE.md.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Text decoder primitives
# ---------------------------------------------------------------------------


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm computed in float32, result cast back to input dtype."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x).to(orig_dtype)


def dual_embedding(input_ids: torch.Tensor, embedding: torch.Tensor, new_embedding: torch.Tensor) -> torch.Tensor:
    """Dual-vocab embedding: concatenate wte.embedding + wte.new_embedding then look up."""
    full_table = torch.cat([embedding, new_embedding], dim=0)
    safe_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
    return F.embedding(safe_ids, full_table)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. q/k shape: [B, n_heads, S, head_dim]; cos/sin: [B, S, head_dim]."""
    cos = cos.unsqueeze(1)  # [B, 1, S, head_dim]
    sin = sin.unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin cache for RoPE (standard, no scaling). Returns [1, S, head_dim]."""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0)  # [1, S]
    else:
        position_ids = position_ids.float()
    freqs = torch.einsum("bi,j->bij", position_ids, inv_freq)  # [B, S, head_dim//2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def text_attention(
    hidden_states: torch.Tensor,
    att_proj_weight: torch.Tensor,
    attn_out_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Text decoder GQA with fused att_proj and Qwen3-style QK-norm.

    Weight layout (from ARCHITECTURE.md):
      att_proj_weight [6144, 4096]: Q=[0:4096], K=[4096:5120], V=[5120:6144]
      q_norm_weight / k_norm_weight: shape [128] (head_dim)
    QK-norm is applied AFTER reshaping to [B, S, n_heads, head_dim].
    """
    B, S, H = hidden_states.shape

    qkv = F.linear(hidden_states, att_proj_weight)  # [B, S, 6144]
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

    # Reshape to [B, S, n_heads, head_dim] before QK-norm (Qwen3 style)
    q = q.view(B, S, num_heads, head_dim)
    k = k.view(B, S, num_kv_heads, head_dim)
    v = v.view(B, S, num_kv_heads, head_dim)

    # Per-head RMSNorm
    q = rmsnorm(q, q_norm_weight, eps=norm_eps)
    k = rmsnorm(k, k_norm_weight, eps=norm_eps)

    # [B, n_heads, S, head_dim] for SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = apply_rope(q, k, cos, sin)

    # GQA: repeat K and V
    n_rep = num_heads // num_kv_heads
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)

    scale = head_dim**-0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, S, S]
    if attention_mask is not None:
        attn = attn + attention_mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)  # [B, n_heads, S, head_dim]

    out = out.transpose(1, 2).contiguous().view(B, S, num_heads * head_dim)
    return F.linear(out, attn_out_weight)


def text_mlp(
    x: torch.Tensor,
    ff_proj_weight: torch.Tensor,
    ff_out_weight: torch.Tensor,
) -> torch.Tensor:
    """Text decoder SwiGLU MLP.

    ff_proj layout [24576, 4096]:
      value (w_up) = first half  [:12288, :]
      gate (w_gate) = second half [12288:, :]
    The MLP computes: ff_out(silu(gate) * value)
    """
    ff = F.linear(x, ff_proj_weight)  # [B, S, 24576]
    value, gate = ff.chunk(2, dim=-1)  # each [B, S, 12288]
    return F.linear(F.silu(gate) * value, ff_out_weight)


def text_decoder_block(
    hidden_states: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    ff_norm_weight: torch.Tensor,
    att_proj_weight: torch.Tensor,
    attn_out_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    ff_proj_weight: torch.Tensor,
    ff_out_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Pre-norm text decoder block (Molmo2DecoderLayer)."""
    # Attention sub-block
    residual = hidden_states
    normed = rmsnorm(hidden_states, attn_norm_weight, eps=norm_eps)
    attn_out = text_attention(
        normed,
        att_proj_weight,
        attn_out_weight,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        attention_mask,
        num_heads,
        num_kv_heads,
        head_dim,
        norm_eps,
    )
    hidden_states = residual + attn_out

    # MLP sub-block
    residual = hidden_states
    normed = rmsnorm(hidden_states, ff_norm_weight, eps=norm_eps)
    hidden_states = residual + text_mlp(normed, ff_proj_weight, ff_out_weight)

    return hidden_states


# ---------------------------------------------------------------------------
# Combined prefill attention mask
# ---------------------------------------------------------------------------


def build_prefill_mask(
    seq_len: int,
    token_type_ids: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the 4D combined causal + image-bidirectional attention mask.

    Args:
        seq_len: sequence length S
        token_type_ids: [B, S] — 1 for image/video tokens, 0 for text
        dtype: output dtype (use float32 for reference; bfloat4_b for TTNN)
        device: target device

    Returns:
        mask [B, 1, S, S] — 0.0 where attention is allowed, -inf where blocked.
        Passed as additive bias to the attention score matrix.
    """
    B = token_type_ids.shape[0]
    device = device or token_type_ids.device

    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1]
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]

    causal_block = kv_idx > q_idx  # [S, S] — True where causal mask blocks

    is_image = token_type_ids == 1  # [B, S]
    is_image_q = is_image.unsqueeze(2)  # [B, S, 1]
    is_image_kv = is_image.unsqueeze(1)  # [B, 1, S]
    image_override = is_image_q & is_image_kv  # [B, S, S] — True for both-image pairs

    block = causal_block.unsqueeze(0) & ~image_override  # [B, S, S]
    mask = torch.where(
        block, torch.tensor(float("-inf"), dtype=dtype, device=device), torch.zeros(1, dtype=dtype, device=device)
    )
    return mask.unsqueeze(1)  # [B, 1, S, S]


# ---------------------------------------------------------------------------
# ViT primitives
# ---------------------------------------------------------------------------


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard LayerNorm with weight and bias."""
    return F.layer_norm(x, weight.shape, weight, bias, eps)


def vit_attention(
    x: torch.Tensor,
    wq_weight: torch.Tensor,
    wq_bias: torch.Tensor,
    wk_weight: torch.Tensor,
    wk_bias: torch.Tensor,
    wv_weight: torch.Tensor,
    wv_bias: torch.Tensor,
    wo_weight: torch.Tensor,
    wo_bias: torch.Tensor,
    num_heads: int = 16,
    head_dim: int = 72,
    inputs_kv: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    float32_attn: bool = True,
) -> torch.Tensor:
    """ViT Multi-Head Attention (no RoPE, no causal mask, all projections have bias).

    For image pooling cross-attention: pass inputs_kv for K/V source.
    float32_attn: cast Q/K to float32 for attention weights (matches model config).
    """
    inputs_k = inputs_kv if inputs_kv is not None else x
    inputs_v = inputs_kv if inputs_kv is not None else x

    N_q = x.shape[1]
    B = x.shape[0]

    xq = F.linear(x, wq_weight, wq_bias)  # [B, N_q, num_heads*head_dim]
    xk = F.linear(inputs_k, wk_weight, wk_bias)  # [B, N_kv, num_heads*head_dim]
    xv = F.linear(inputs_v, wv_weight, wv_bias)  # [B, N_kv, num_heads*head_dim]

    xq = xq.reshape(B, N_q, num_heads, head_dim)
    xk = xk.reshape(B, -1, num_heads, head_dim)
    xv = xv.reshape(B, -1, num_heads, head_dim)

    orig_dtype = xq.dtype
    if float32_attn:
        xq = xq.to(torch.float32)
        xk = xk.to(torch.float32)

    scale = head_dim**-0.5
    # einsum: "bqhd,bkhd->bhqk" (matches HF: ...qhd,...khd->...hqk)
    attn = torch.einsum("bqhd,bkhd->bhqk", xq * scale, xk)
    if attn_mask is not None:
        attn = attn + attn_mask.to(attn.dtype)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(xq.dtype)
    out = torch.einsum("bhqk,bkhd->bqhd", attn, xv.to(xq.dtype))
    out = out.to(orig_dtype)

    out = out.reshape(B, N_q, num_heads * head_dim)
    return F.linear(out, wo_weight, wo_bias)


def vit_mlp(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_bias: torch.Tensor,
) -> torch.Tensor:
    """ViT GELU MLP (gelu_pytorch_tanh) with bias."""
    return F.linear(F.gelu(F.linear(x, w1_weight, w1_bias), approximate="tanh"), w2_weight, w2_bias)


def vit_block(
    x: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    attn_norm_bias: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    ffn_norm_bias: torch.Tensor,
    wq_weight: torch.Tensor,
    wq_bias: torch.Tensor,
    wk_weight: torch.Tensor,
    wk_bias: torch.Tensor,
    wv_weight: torch.Tensor,
    wv_bias: torch.Tensor,
    wo_weight: torch.Tensor,
    wo_bias: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_bias: torch.Tensor,
    num_heads: int = 16,
    head_dim: int = 72,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Single ViT residual block: pre-norm attention + pre-norm MLP."""
    x = x + vit_attention(
        layernorm(x, attn_norm_weight, attn_norm_bias, norm_eps),
        wq_weight,
        wq_bias,
        wk_weight,
        wk_bias,
        wv_weight,
        wv_bias,
        wo_weight,
        wo_bias,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    x = x + vit_mlp(
        layernorm(x, ffn_norm_weight, ffn_norm_bias, norm_eps),
        w1_weight,
        w1_bias,
        w2_weight,
        w2_bias,
    )
    return x


def add_pos_emb(x: torch.Tensor, pos_emb: torch.Tensor, patch_num: Tuple[int, int]) -> torch.Tensor:
    """Add (bicubic-interpolated) positional embedding to patch tokens.

    pos_emb: [729, 1152] learned embedding for 27×27 grid.
    patch_num: (h, w) actual patch grid (may differ from 27×27 for non-standard sizes).
    """
    grid_h = grid_w = int(math.sqrt(pos_emb.shape[0]))
    pe = pos_emb.reshape(grid_h, grid_w, pos_emb.shape[1])  # [27, 27, 1152]
    ph, pw = patch_num
    if pe.shape[0] != ph or pe.shape[1] != pw:
        pe = pe.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 1152, H, W]
        pe = F.interpolate(pe, size=(ph, pw), mode="bicubic", align_corners=False, antialias=True)
        pe = pe.permute(0, 2, 3, 1).squeeze(0)  # [ph, pw, 1152]
    pe = pe.reshape(-1, pos_emb.shape[1])  # [ph*pw, 1152]
    return x + pe.unsqueeze(0).to(x.dtype)  # broadcast over batch


def vit_encode(
    pixel_values: torch.Tensor,
    patch_embed_weight: torch.Tensor,
    patch_embed_bias: torch.Tensor,
    pos_embedding: torch.Tensor,
    resblock_weights: list,
    patch_num: Tuple[int, int] = (27, 27),
    num_heads: int = 16,
    head_dim: int = 72,
    norm_eps: float = 1e-6,
    n_blocks: int = 25,
    capture_layers: Tuple[int, int] = (24, 18),
) -> torch.Tensor:
    """Run ViT encoder and return concatenated features from two intermediate layers.

    Args:
        pixel_values: [B*n_crops, N_patches=729, 588] flattened patch pixels
        resblock_weights: list of per-block weight dicts (length = n_blocks)
        capture_layers: ordered layer indices to concatenate — default (24, 18) matches
            adapter_config.vit_layers = [-3, -9] → [24, 18] (first dim of 2304 output).

    Returns:
        image_features [B*n_crops, 729, 2304] (concat of hidden[24] and hidden[18])
    """
    x = F.linear(pixel_values, patch_embed_weight, patch_embed_bias)  # [B*C, 729, 1152]
    x = add_pos_emb(x, pos_embedding, patch_num)

    captured = {}
    for i, weights in enumerate(resblock_weights):
        x = vit_block(
            x,
            weights["attn_norm_weight"],
            weights["attn_norm_bias"],
            weights["ffn_norm_weight"],
            weights["ffn_norm_bias"],
            weights["wq_weight"],
            weights["wq_bias"],
            weights["wk_weight"],
            weights["wk_bias"],
            weights["wv_weight"],
            weights["wv_bias"],
            weights["wo_weight"],
            weights["wo_bias"],
            weights["w1_weight"],
            weights["w1_bias"],
            weights["w2_weight"],
            weights["w2_bias"],
            num_heads=num_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
        )
        if i in capture_layers:
            captured[i] = x.clone()

    return torch.cat([captured[l] for l in capture_layers], dim=-1)  # [B*C, 729, 2304]


# ---------------------------------------------------------------------------
# Image pooling 2D (cross-attention adapter)
# ---------------------------------------------------------------------------


def image_pooling_2d(
    image_features: torch.Tensor,
    pooled_patches_idx: torch.Tensor,
    wq_weight: torch.Tensor,
    wq_bias: torch.Tensor,
    wk_weight: torch.Tensor,
    wk_bias: torch.Tensor,
    wv_weight: torch.Tensor,
    wv_bias: torch.Tensor,
    wo_weight: torch.Tensor,
    wo_bias: torch.Tensor,
    num_heads: int = 16,
    head_dim: int = 72,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Image pooling 2D adapter cross-attention.

    Args:
        image_features: [B, n_crops, 729, 2304] from vit_encode (two-layer concat)
        pooled_patches_idx: [B, N_pooled, pool_window] patch indices per token (−1 = invalid)

    Returns:
        pooled: [B, N_pooled, 1152]
    """
    B, n_crops, n_patches, dim = image_features.shape
    N_pooled, pool_window = pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]

    valid = pooled_patches_idx >= 0  # [B, N_pooled, pool_window]
    valid_token = valid.any(dim=-1)  # [B, N_pooled]

    # Gather patches for pooling windows
    batch_idx = torch.arange(B, device=image_features.device).view(B, 1, 1).expand(B, N_pooled, pool_window)
    flat_features = image_features.reshape(B, -1, dim)  # [B, n_crops*729, dim]
    clipped_idx = pooled_patches_idx.clamp(min=0)
    to_pool = flat_features[batch_idx, clipped_idx]  # [B, N_pooled, pool_window, dim]
    to_pool = to_pool * valid.to(image_features.dtype).unsqueeze(-1)

    # Reshape to [B*N_pooled, pool_window, dim]
    to_pool_flat = to_pool.reshape(B * N_pooled, pool_window, dim)
    valid_flat = valid.reshape(B * N_pooled, pool_window)

    # Build attn_mask [B*N_pooled, 1, 1, pool_window]
    attn_mask_pool = valid_flat.unsqueeze(1).unsqueeze(1).float()
    # Convert to additive bias: 0 for valid, -inf for invalid
    attn_mask_pool = torch.where(
        valid_flat.unsqueeze(1).unsqueeze(1),
        torch.zeros_like(attn_mask_pool),
        torch.full_like(attn_mask_pool, float("-inf")),
    )

    # Query = masked mean of patches in window
    denom = valid_flat.float().sum(dim=-1).clamp(min=1)
    query = to_pool_flat.sum(dim=1, keepdim=True) / denom.unsqueeze(-1).unsqueeze(-1).to(to_pool_flat.dtype)

    # Cross-attention: Q from query (masked mean), K/V from full window
    pooled = vit_attention(
        query,
        wq_weight,
        wq_bias,
        wk_weight,
        wk_bias,
        wv_weight,
        wv_bias,
        wo_weight,
        wo_bias,
        num_heads=num_heads,
        head_dim=head_dim,
        inputs_kv=to_pool_flat,
        attn_mask=attn_mask_pool,
        float32_attn=True,
    )  # [B*N_pooled, 1, 1152]

    return pooled.reshape(B, N_pooled, -1)  # [B, N_pooled, 1152]


# ---------------------------------------------------------------------------
# Image projector
# ---------------------------------------------------------------------------


def image_projector(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU projector (no bias): w2(act(w1(x)) * w3(x)).

    w1 = gate [12288, 1152], w3 = up [12288, 1152], w2 = down [4096, 12288].
    """
    return F.linear(F.silu(F.linear(x, w1_weight)) * F.linear(x, w3_weight), w2_weight)


# ---------------------------------------------------------------------------
# Full model forward (image prefill)
# ---------------------------------------------------------------------------


def inject_image_features(
    embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    image_features: torch.Tensor,
    image_patch_id: int = 151938,
) -> torch.Tensor:
    """Additively inject image features at image_patch_id positions."""
    is_patch = input_ids.view(-1) == image_patch_id
    result = embeddings.clone()
    result.view(-1, embeddings.shape[-1])[is_patch] += image_features
    return result


def load_vit_resblock_weights(
    state_dict: Dict, block_idx: int, prefix: str = "model.vision_backbone.image_vit.transformer.resblocks"
) -> Dict:
    """Extract weight dict for a single ViT resblock."""
    p = f"{prefix}.{block_idx}."
    return {
        "attn_norm_weight": state_dict[p + "attention_norm.weight"],
        "attn_norm_bias": state_dict[p + "attention_norm.bias"],
        "ffn_norm_weight": state_dict[p + "ffn_norm.weight"],
        "ffn_norm_bias": state_dict[p + "ffn_norm.bias"],
        "wq_weight": state_dict[p + "attention.wq.weight"],
        "wq_bias": state_dict[p + "attention.wq.bias"],
        "wk_weight": state_dict[p + "attention.wk.weight"],
        "wk_bias": state_dict[p + "attention.wk.bias"],
        "wv_weight": state_dict[p + "attention.wv.weight"],
        "wv_bias": state_dict[p + "attention.wv.bias"],
        "wo_weight": state_dict[p + "attention.wo.weight"],
        "wo_bias": state_dict[p + "attention.wo.bias"],
        "w1_weight": state_dict[p + "feed_forward.w1.weight"],
        "w1_bias": state_dict[p + "feed_forward.w1.bias"],
        "w2_weight": state_dict[p + "feed_forward.w2.weight"],
        "w2_bias": state_dict[p + "feed_forward.w2.bias"],
    }


def load_decoder_block_weights(state_dict: Dict, block_idx: int, prefix: str = "model.transformer.blocks") -> Dict:
    """Extract weight dict for a single text decoder block."""
    p = f"{prefix}.{block_idx}."
    return {
        "attn_norm_weight": state_dict[p + "attn_norm.weight"],
        "ff_norm_weight": state_dict[p + "ff_norm.weight"],
        "att_proj_weight": state_dict[p + "self_attn.att_proj.weight"],
        "attn_out_weight": state_dict[p + "self_attn.attn_out.weight"],
        "q_norm_weight": state_dict[p + "self_attn.q_norm.weight"],
        "k_norm_weight": state_dict[p + "self_attn.k_norm.weight"],
        "ff_proj_weight": state_dict[p + "mlp.ff_proj.weight"],
        "ff_out_weight": state_dict[p + "mlp.ff_out.weight"],
    }
