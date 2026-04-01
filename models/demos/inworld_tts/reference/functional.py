"""Standalone PyTorch reference implementation for Inworld TTS codec decoder.

Each function is a self-contained block that takes raw tensors + weights,
matching the official inworld-ai/tts implementation exactly.
CPU boundaries: FSQ dequantize (input) and ISTFT (output).
"""

from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.

    Matches decoder_modules.RMSNorm.
    """
    norm_x = torch.mean(x**2, dim=-1, keepdim=True)
    return x * torch.rsqrt(norm_x + eps) * weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------
def build_rope_cache(n_positions: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Build RoPE cache matching torchtune RotaryPositionalEmbeddings.

    Returns cache of shape [n_positions, dim//2, 2] containing [cos, sin].
    """
    theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    seq_idx = torch.arange(n_positions, dtype=theta.dtype)
    idx_theta = torch.einsum("i, j -> ij", seq_idx, theta).float()  # [n_positions, dim//2]
    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)  # [n_positions, dim//2, 2]
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings matching torchtune RotaryPositionalEmbeddings.forward().

    IMPORTANT: In the Inworld TTS Attention module, q/k have shape [B, H, T, D]
    which is passed to torchtune's RoPE expecting [B, S, NH, HD]. This means
    torchtune treats dim 1 (=H, num_heads) as the position axis.

    Args:
        x: [B, H, T, D] input tensor (H is treated as position axis)
        rope_cache: [n_positions, D//2, 2] precomputed cos/sin cache
    Returns:
        [B, H, T, D] with rotary embeddings applied along dim 1
    """
    # Match torchtune's forward exactly
    seq_len = x.size(1)
    cache = rope_cache[:seq_len]  # [H, D//2, 2]

    # Reshape input: split last dim into pairs
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, H, T, D//2, 2]

    # Reshape cache for broadcasting: [1, H, 1, D//2, 2]
    cache = cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

    # Apply rotation
    x_out = torch.stack(
        [
            xshaped[..., 0] * cache[..., 0] - xshaped[..., 1] * cache[..., 1],
            xshaped[..., 1] * cache[..., 0] + xshaped[..., 0] * cache[..., 1],
        ],
        -1,
    )

    x_out = x_out.flatten(3)
    return x_out.type_as(x)


# ---------------------------------------------------------------------------
# Multi-Head Attention (bidirectional, fused QKV, with RoPE)
# ---------------------------------------------------------------------------
def attention_forward(
    x: torch.Tensor,
    c_attn_weight: torch.Tensor,
    c_proj_weight: torch.Tensor,
    n_heads: int,
    rope_cache: torch.Tensor,
) -> torch.Tensor:
    """Bidirectional MHA with fused QKV and RoPE.

    Matches decoder_modules.Attention exactly.
    Args:
        x: [B, T, dim]
        c_attn_weight: [3*dim, dim] fused QKV weight
        c_proj_weight: [dim, dim] output projection weight
        n_heads: number of attention heads
        rope_cache: [n_positions, head_dim//2, 2] precomputed cos/sin
    Returns:
        [B, T, dim]
    """
    B, T, dim = x.shape
    head_dim = dim // n_heads

    # Fused QKV projection
    qkv = F.linear(x, c_attn_weight)  # [B, T, 3*dim]

    # Reshape to [3, B, H, T, D] matching einops "b t (r h d) -> r b h t d"
    qkv = qkv.view(B, T, 3, n_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Apply RoPE to Q and K (torchtune convention: rotates along dim 1 = heads axis)
    q = apply_rope(q, rope_cache)
    k = apply_rope(k, rope_cache)

    # Scaled dot-product attention (bidirectional: is_causal=False)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    # Merge heads: [B, H, T, D] -> [B, T, dim]
    y = y.permute(0, 2, 1, 3).contiguous().view(B, T, dim)

    # Output projection
    return F.linear(y, c_proj_weight)


# ---------------------------------------------------------------------------
# SiLU MLP (NOT SwiGLU -- no gate projection)
# ---------------------------------------------------------------------------
def mlp_forward(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> torch.Tensor:
    """SiLU MLP: fc1 -> silu -> fc2.

    Matches decoder_modules.MLP (no gate, no bias).
    """
    h = F.linear(x, fc1_weight)
    h = F.silu(h)
    return F.linear(h, fc2_weight)


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm)
# ---------------------------------------------------------------------------
def transformer_block_forward(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    n_heads: int,
    rope_cache: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Single VocosBackbone transformer block.

    Matches decoder_modules.TransformerBlock.
    weights keys: att_norm_weight, c_attn_weight, c_proj_weight,
                  ffn_norm_weight, fc1_weight, fc2_weight
    """
    # Attention with pre-norm
    h = rmsnorm_forward(x, weights["att_norm_weight"], eps)
    h = attention_forward(h, weights["c_attn_weight"], weights["c_proj_weight"], n_heads, rope_cache)
    x = x + h

    # MLP with pre-norm
    h = rmsnorm_forward(x, weights["ffn_norm_weight"], eps)
    h = mlp_forward(h, weights["fc1_weight"], weights["fc2_weight"])
    x = x + h

    return x


# ---------------------------------------------------------------------------
# Swish activation (for ResnetBlock)
# ---------------------------------------------------------------------------
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# ResnetBlock (GroupNorm + swish + Conv1d, with residual)
# ---------------------------------------------------------------------------
def resnet_block_forward(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """ResnetBlock: norm1->swish->conv1->norm2->swish->conv2 + residual.

    Matches decoder_modules.ResnetBlock (in_channels == out_channels case).
    Input x: [B, C, T] (channels-first for conv)
    weights: norm1_{weight,bias}, conv1_{weight,bias},
             norm2_{weight,bias}, conv2_{weight,bias}
    """
    h = F.group_norm(x, 32, weights["norm1_weight"], weights["norm1_bias"], eps=1e-6)
    h = swish(h)
    h = F.conv1d(h, weights["conv1_weight"], weights["conv1_bias"], padding=1)
    h = F.group_norm(h, 32, weights["norm2_weight"], weights["norm2_bias"], eps=1e-6)
    h = swish(h)
    # dropout is no-op at inference
    h = F.conv1d(h, weights["conv2_weight"], weights["conv2_bias"], padding=1)
    return x + h


# ---------------------------------------------------------------------------
# VocosBackbone
# ---------------------------------------------------------------------------
def vocos_backbone_forward(
    x: torch.Tensor,
    weights: Dict,
    n_heads: int = 16,
    pos_emb_dim: int = 64,
    depth: int = 12,
) -> torch.Tensor:
    """Full VocosBackbone forward pass.

    Matches decoder_modules.VocosBackbone.
    Input x: [B, T, 1024] (sequence-first)
    weights: embed_*, prior_net_0_*, prior_net_1_*,
             transformer_{i}_*, final_layer_norm_*, post_net_0_*, post_net_1_*
    """
    B, T, C = x.shape

    # Build RoPE cache -- torchtune convention: n_positions = n_heads (dim 1 of q/k)
    # In Attention, q has shape [B, H, T, D], torchtune uses H as position axis
    rope_cache = build_rope_cache(n_heads, pos_emb_dim).to(x.device)

    # embed: Conv1d(1024, 1024, k=7, padding=3)
    h = x.transpose(1, 2)  # [B, C, T]
    h = F.conv1d(h, weights["embed_weight"], weights["embed_bias"], padding=3)

    # prior_net: 2x ResnetBlock
    for i in range(2):
        prefix = f"prior_net_{i}_"
        rn_weights = {
            "norm1_weight": weights[prefix + "norm1_weight"],
            "norm1_bias": weights[prefix + "norm1_bias"],
            "conv1_weight": weights[prefix + "conv1_weight"],
            "conv1_bias": weights[prefix + "conv1_bias"],
            "norm2_weight": weights[prefix + "norm2_weight"],
            "norm2_bias": weights[prefix + "norm2_bias"],
            "conv2_weight": weights[prefix + "conv2_weight"],
            "conv2_bias": weights[prefix + "conv2_bias"],
        }
        h = resnet_block_forward(h, rn_weights)

    # Transpose for transformer: [B, C, T] -> [B, T, C]
    h = h.transpose(1, 2)

    # 12 transformer blocks
    for i in range(depth):
        prefix = f"transformer_{i}_"
        tf_weights = {
            "att_norm_weight": weights[prefix + "att_norm_weight"],
            "c_attn_weight": weights[prefix + "c_attn_weight"],
            "c_proj_weight": weights[prefix + "c_proj_weight"],
            "ffn_norm_weight": weights[prefix + "ffn_norm_weight"],
            "fc1_weight": weights[prefix + "fc1_weight"],
            "fc2_weight": weights[prefix + "fc2_weight"],
        }
        h = transformer_block_forward(h, tf_weights, n_heads, rope_cache)

    # Transpose for post-conv: [B, T, C] -> [B, C, T]
    h = h.transpose(1, 2)

    # post_net: 2x ResnetBlock
    for i in range(2):
        prefix = f"post_net_{i}_"
        rn_weights = {
            "norm1_weight": weights[prefix + "norm1_weight"],
            "norm1_bias": weights[prefix + "norm1_bias"],
            "conv1_weight": weights[prefix + "conv1_weight"],
            "conv1_bias": weights[prefix + "conv1_bias"],
            "norm2_weight": weights[prefix + "norm2_weight"],
            "norm2_bias": weights[prefix + "norm2_bias"],
            "conv2_weight": weights[prefix + "conv2_weight"],
            "conv2_bias": weights[prefix + "conv2_bias"],
        }
        h = resnet_block_forward(h, rn_weights)

    # Back to [B, T, C] for final LayerNorm
    h = h.transpose(1, 2)
    h = F.layer_norm(h, [C], weights["final_layer_norm_weight"], weights["final_layer_norm_bias"], eps=1e-6)

    return h


# ---------------------------------------------------------------------------
# ISTFTHead (CPU boundary)
# ---------------------------------------------------------------------------
def istft_head_forward(
    x: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    n_fft: int = 1280,
    hop_length: int = 320,
) -> torch.Tensor:
    """ISTFTHead: linear -> split mag/phase -> exp -> complex -> ISTFT.

    Matches decoder_modules.ISTFTHead. Runs on CPU.
    Input x: [B, T, 1024]
    Returns: [B, 1, num_samples]
    """
    win_length = n_fft

    # Linear projection to n_fft + 2
    x_pred = F.linear(x, out_weight, out_bias)  # [B, T, 1282]
    x_pred = x_pred.transpose(1, 2)  # [B, 1282, T]

    # Split into magnitude and phase
    mag, p = x_pred.chunk(2, dim=1)  # each [B, 641, T]

    # Magnitude activation
    mag = torch.exp(mag)
    mag = torch.clamp(mag, max=1e2)

    # Phase to complex
    x_cos = torch.cos(p)
    y_sin = torch.sin(p)
    S = mag * (x_cos + 1j * y_sin)  # [B, 641, T]

    # ISTFT with "same" padding
    pad = (win_length - hop_length) // 2
    window = torch.hann_window(win_length, device=S.device)

    B, N, T_frames = S.shape

    # Inverse FFT
    ifft = torch.fft.irfft(S, n_fft, dim=1, norm="backward")  # [B, n_fft, T]
    ifft = ifft * window[None, :, None]

    # Overlap and Add
    output_size = (T_frames - 1) * hop_length + win_length
    y = F.fold(
        ifft,
        output_size=(1, output_size),
        kernel_size=(1, win_length),
        stride=(1, hop_length),
    )[:, 0, 0, pad:-pad]

    # Window envelope normalization
    window_sq = window.square().expand(1, T_frames, -1).transpose(1, 2)
    window_envelope = F.fold(
        window_sq,
        output_size=(1, output_size),
        kernel_size=(1, win_length),
        stride=(1, hop_length),
    ).squeeze()[pad:-pad]

    y = y / window_envelope

    return y.unsqueeze(1)


# ---------------------------------------------------------------------------
# FSQ Dequantize (CPU boundary)
# ---------------------------------------------------------------------------
def fsq_dequantize(
    indices: torch.Tensor,
    levels: list = None,
) -> torch.Tensor:
    """FSQ dequantize: integer indices -> float embeddings.

    Matches vector_quantize_pytorch.ResidualFSQ.get_output_from_indices.
    This converts flat codebook indices back to the 8-dimensional float
    representation, then projects to the full embedding dim.

    Args:
        indices: [B, 1, T] integer VQ codes (0 to 65535)
        levels: list of level counts per dimension, default [4,4,4,4,4,4,4,4]
    Returns:
        [B, T, 2048] dequantized embeddings
    """
    if levels is None:
        levels = [4, 4, 4, 4, 4, 4, 4, 4]

    levels_t = torch.tensor(levels, device=indices.device)
    num_dims = len(levels)

    # Flatten indices to [B*T] for decomposition
    flat = indices.squeeze(1).long()  # [B, T]
    B, T = flat.shape

    # Decompose flat index into per-dimension indices (reverse of encoding)
    # The FSQ library encodes: index = sum(d_i * prod(levels[i+1:]))
    # We decode in reverse
    dims = []
    remainder = flat
    for i in range(num_dims - 1, -1, -1):
        dims.insert(0, remainder % levels_t[i])
        remainder = remainder // levels_t[i]

    # Stack to get [B, T, num_dims]
    codes = torch.stack(dims, dim=-1).float()

    # Convert from index to float: map [0, L-1] -> [-1, 1]
    half_levels = (levels_t.float() - 1) / 2
    codes = (codes - half_levels) / half_levels  # normalized to [-1, 1]

    # FSQ uses implicit_codebook projection from num_dims to vq_dim
    # In practice, the library's get_output_from_indices reconstructs
    # the quantized vector directly. The 8-dim representation IS the
    # quantized output before the codebook projection.
    # The actual dim=2048 output comes from the library's internal projection.
    # For our reference, we'll use the library directly when weights are available.

    return codes  # [B, T, 8] -- will need library for full 2048-dim


def fsq_dequantize_with_codebook(
    indices: torch.Tensor,
    quantizer,
) -> torch.Tensor:
    """FSQ dequantize using the actual quantizer module.

    This is the preferred method when the quantizer weights are available.
    Args:
        indices: [B, 1, T] integer VQ codes
        quantizer: vector_quantize_pytorch.ResidualFSQ instance
    Returns:
        [B, T, 2048] dequantized embeddings
    """
    # The library expects [B, T, num_quantizers] for indices
    codes = indices.transpose(1, 2)  # [B, T, 1]
    return quantizer.get_output_from_indices(codes)


# ---------------------------------------------------------------------------
# Full Codec Decoder Pipeline
# ---------------------------------------------------------------------------
def codec_decoder_forward(
    vq_codes: torch.Tensor,
    quantizer,
    fc_post_a_weight: torch.Tensor,
    fc_post_a_bias: torch.Tensor,
    backbone_weights: Dict,
    istft_weights: Dict,
    n_heads: int = 16,
    pos_emb_dim: int = 64,
    depth: int = 12,
) -> torch.Tensor:
    """Full codec decoder: FSQ dequant -> fc_post_a -> VocosBackbone -> ISTFTHead.

    Matches decoder.Decoder.forward().
    Args:
        vq_codes: [B, T] or [B, 1, T] integer VQ codes
        quantizer: ResidualFSQ instance for dequantization
        fc_post_a_{weight,bias}: Linear(2048, 1024) weights
        backbone_weights: dict of VocosBackbone weights
        istft_weights: dict with 'out_weight' and 'out_bias'
    Returns:
        [B, 1, num_samples] audio waveform
    """
    # Ensure shape is [B, 1, T]
    if vq_codes.dim() == 2:
        vq_codes = vq_codes.unsqueeze(1)

    # Step 1: FSQ dequantize (CPU boundary)
    vq_codes_t = vq_codes.transpose(1, 2)  # [B, T, 1]
    vq_post_emb = quantizer.get_output_from_indices(vq_codes_t)  # [B, T, 2048]

    # Step 2: fc_post_a projection
    vq_post_emb_fc = F.linear(vq_post_emb, fc_post_a_weight, fc_post_a_bias)  # [B, T, 1024]

    # Step 3: VocosBackbone
    decoder_hidden = vocos_backbone_forward(vq_post_emb_fc, backbone_weights, n_heads, pos_emb_dim, depth)

    # Step 4: ISTFTHead (CPU boundary)
    audio = istft_head_forward(
        decoder_hidden,
        istft_weights["out_weight"],
        istft_weights["out_bias"],
    )

    return audio


# ---------------------------------------------------------------------------
# Weight Extraction Helpers
# ---------------------------------------------------------------------------
def extract_backbone_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract VocosBackbone weights from a codec decoder state_dict.

    Handles both raw Generator state_dict and full Decoder state_dict formats.
    """
    weights = {}

    # Determine prefix based on state_dict format
    prefix = ""
    if any(k.startswith("decoder.backbone.") for k in state_dict):
        prefix = "decoder.backbone."
    elif any(k.startswith("backbone.") for k in state_dict):
        prefix = "backbone."

    # embed Conv1d
    weights["embed_weight"] = state_dict[prefix + "embed.weight"]
    weights["embed_bias"] = state_dict[prefix + "embed.bias"]

    # prior_net: 2 ResnetBlocks
    for i in range(2):
        src = f"{prefix}prior_net.{i}."
        dst = f"prior_net_{i}_"
        weights[dst + "norm1_weight"] = state_dict[src + "norm1.weight"]
        weights[dst + "norm1_bias"] = state_dict[src + "norm1.bias"]
        weights[dst + "conv1_weight"] = state_dict[src + "conv1.weight"]
        weights[dst + "conv1_bias"] = state_dict[src + "conv1.bias"]
        weights[dst + "norm2_weight"] = state_dict[src + "norm2.weight"]
        weights[dst + "norm2_bias"] = state_dict[src + "norm2.bias"]
        weights[dst + "conv2_weight"] = state_dict[src + "conv2.weight"]
        weights[dst + "conv2_bias"] = state_dict[src + "conv2.bias"]

    # Transformer blocks
    for i in range(12):
        src = f"{prefix}transformers.{i}."
        dst = f"transformer_{i}_"
        weights[dst + "att_norm_weight"] = state_dict[src + "att_norm.weight"]
        weights[dst + "c_attn_weight"] = state_dict[src + "att.c_attn.weight"]
        weights[dst + "c_proj_weight"] = state_dict[src + "att.c_proj.weight"]
        weights[dst + "ffn_norm_weight"] = state_dict[src + "ffn_norm.weight"]
        weights[dst + "fc1_weight"] = state_dict[src + "mlp.fc1.weight"]
        weights[dst + "fc2_weight"] = state_dict[src + "mlp.fc2.weight"]

    # post_net: 2 ResnetBlocks
    for i in range(2):
        src = f"{prefix}post_net.{i}."
        dst = f"post_net_{i}_"
        weights[dst + "norm1_weight"] = state_dict[src + "norm1.weight"]
        weights[dst + "norm1_bias"] = state_dict[src + "norm1.bias"]
        weights[dst + "conv1_weight"] = state_dict[src + "conv1.weight"]
        weights[dst + "conv1_bias"] = state_dict[src + "conv1.bias"]
        weights[dst + "norm2_weight"] = state_dict[src + "norm2.weight"]
        weights[dst + "norm2_bias"] = state_dict[src + "norm2.bias"]
        weights[dst + "conv2_weight"] = state_dict[src + "conv2.weight"]
        weights[dst + "conv2_bias"] = state_dict[src + "conv2.bias"]

    # Final LayerNorm
    weights["final_layer_norm_weight"] = state_dict[prefix + "final_layer_norm.weight"]
    weights["final_layer_norm_bias"] = state_dict[prefix + "final_layer_norm.bias"]

    return weights


def extract_istft_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract ISTFTHead weights from a codec decoder state_dict."""
    prefix = ""
    if any(k.startswith("decoder.head.") for k in state_dict):
        prefix = "decoder.head."
    elif any(k.startswith("head.") for k in state_dict):
        prefix = "head."

    return {
        "out_weight": state_dict[prefix + "out.weight"],
        "out_bias": state_dict[prefix + "out.bias"],
    }
