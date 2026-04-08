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
def extract_backbone_weights(state_dict: Dict[str, torch.Tensor], depth: int = 12) -> Dict[str, torch.Tensor]:
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
    for i in range(depth):
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


# ===========================================================================
# Encoder functions
# ===========================================================================


# ---------------------------------------------------------------------------
# Weight Normalization
# ---------------------------------------------------------------------------
def weight_norm_compute(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Compute actual weight from weight_g and weight_v.

    weight = g * v / ||v||
    weight_g: [C_out, 1, 1]
    weight_v: [C_out, C_in, K]
    Returns: [C_out, C_in, K]
    """
    norm = torch.norm(weight_v, dim=[1, 2], keepdim=True)
    return weight_g * weight_v / norm


# ---------------------------------------------------------------------------
# SnakeBeta Activation
# ---------------------------------------------------------------------------
def snake_beta(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x).

    alpha, beta: [C] learnable parameters, broadcast over [B, C, T].
    """
    alpha = alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
    beta = beta.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
    return x + (1.0 / beta) * torch.sin(alpha * x) ** 2


# ---------------------------------------------------------------------------
# Anti-aliased SnakeBeta (Activation1d wrapper)
# ---------------------------------------------------------------------------
def activation1d_forward(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    up_filter: torch.Tensor,
    down_filter: torch.Tensor,
) -> torch.Tensor:
    """Anti-aliased SnakeBeta activation.

    Upsample 2x -> SnakeBeta -> Downsample 2x to avoid aliasing from nonlinearity.

    Args:
        x: [B, C, T]
        alpha: [C] SnakeBeta alpha parameter
        beta: [C] SnakeBeta beta parameter
        up_filter: [1, 1, K] FIR upsampling filter
        down_filter: [1, 1, K] FIR lowpass/downsampling filter
    Returns:
        [B, C, T]
    """
    B, C, T = x.shape

    # Prepare filters for depthwise conv: [C, 1, K]
    up_kernel = up_filter.squeeze(0).expand(C, -1, -1)  # [C, 1, K]
    down_kernel = down_filter.squeeze(0).expand(C, -1, -1)  # [C, 1, K]
    K = up_kernel.shape[-1]

    print(f"Upsampling kernel shape: {up_filter.shape}, Downsampling kernel shape: {down_filter.shape}")
    print(f"Upsampling kernel shape: {up_kernel.shape}, Downsampling kernel shape: {down_kernel.shape}")
    # For even-length FIR filters, use asymmetric padding to preserve length
    pad_left = K // 2
    pad_right = K // 2 - 1 if K % 2 == 0 else K // 2

    # Upsample by 2: insert zeros between samples, then filter
    x_up = torch.zeros(B, C, T * 2, device=x.device, dtype=x.dtype)
    x_up[:, :, ::2] = x
    x_up = F.pad(x_up, (pad_left, pad_right))
    x_up = F.conv1d(x_up, up_kernel * 2.0, groups=C)

    # Apply SnakeBeta activation
    x_act = snake_beta(x_up, alpha, beta)

    # Downsample by 2: lowpass filter then take every 2nd sample
    x_down = F.pad(x_act, (pad_left, pad_right))
    x_down = F.conv1d(x_down, down_kernel, groups=C)
    x_down = x_down[:, :, ::2]

    return x_down


# ---------------------------------------------------------------------------
# Encoder ResidualUnit
# ---------------------------------------------------------------------------
def encoder_residual_unit_forward(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Encoder ResidualUnit: SnakeBeta + WNConv1d(k=7) + SnakeBeta + WNConv1d(k=1) + skip.

    Input x: [B, C, T]
    weights keys:
        act1_alpha, act1_beta, act1_up_filter, act1_down_filter,
        conv1_weight (precomputed from weight_norm), conv1_bias,
        act2_alpha, act2_beta, act2_up_filter, act2_down_filter,
        conv2_weight (precomputed from weight_norm), conv2_bias
    """
    # First activation + conv (k=7, pad=3)
    h = activation1d_forward(
        x,
        weights["act1_alpha"],
        weights["act1_beta"],
        weights["act1_up_filter"],
        weights["act1_down_filter"],
    )
    h = F.conv1d(h, weights["conv1_weight"], weights["conv1_bias"], padding=3)

    # Second activation + conv (k=1, pad=0)
    h = activation1d_forward(
        h,
        weights["act2_alpha"],
        weights["act2_beta"],
        weights["act2_up_filter"],
        weights["act2_down_filter"],
    )
    h = F.conv1d(h, weights["conv2_weight"], weights["conv2_bias"], padding=0)

    return x + h


# ---------------------------------------------------------------------------
# Encoder Block (3 ResidualUnits + downsampling conv)
# ---------------------------------------------------------------------------
def encoder_block_forward(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    stride: int,
) -> torch.Tensor:
    """EncoderBlock: 3 ResidualUnits + SnakeBeta + downsampling WNConv1d.

    Input x: [B, C_in, T]
    Output: [B, C_out, T // stride]
    weights keys: res_{0,1,2}_*, act_alpha, act_beta, act_up_filter, act_down_filter,
                  downsample_weight, downsample_bias
    """
    # 3 residual units
    for i in range(3):
        prefix = f"res_{i}_"
        res_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        x = encoder_residual_unit_forward(x, res_weights)

    # Final activation before downsample
    x = activation1d_forward(
        x,
        weights["act_alpha"],
        weights["act_beta"],
        weights["act_up_filter"],
        weights["act_down_filter"],
    )

    # Downsampling conv: kernel_size = stride * 2, padding = stride // 2 + stride % 2
    # (matching the DAC / EnCodec convention: pad = ceil(k/2) - stride//2 on each side)
    kernel_size = stride * 2
    # Causal-style padding: pad = kernel_size - stride
    pad_total = kernel_size - stride
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    x = F.pad(x, (pad_left, pad_right))
    x = F.conv1d(x, weights["downsample_weight"], weights["downsample_bias"], stride=stride)

    return x


# ---------------------------------------------------------------------------
# AcousticEncoder (CodecEnc)
# ---------------------------------------------------------------------------
def acoustic_encoder_forward(
    waveform: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Full AcousticEncoder forward.

    Input: [B, 1, samples]
    Output: [B, 1024, T] where T = samples / 320

    The encoder has:
    - conv_blocks.0: initial Conv1d(1, 48, k=7) with weight norm
    - conv_blocks.1-5: 5 EncoderBlocks
    - conv_final_block: SnakeBeta(1536) + Conv1d(1536, 1024, k=3) with weight norm
    """
    channels = [48, 96, 192, 384, 768, 1536]
    strides = [2, 2, 4, 4, 5]

    # Initial conv: Conv1d(1, 48, k=7, pad=3) with weight norm
    weight = weight_norm_compute(
        state_dict["conv_blocks.0.weight_g"],
        state_dict["conv_blocks.0.weight_v"],
    )
    bias = state_dict["conv_blocks.0.bias"]
    x = F.conv1d(waveform, weight, bias, padding=3)

    # 5 encoder blocks
    for block_idx in range(5):
        block_prefix = f"conv_blocks.{block_idx + 1}."
        block_weights = _extract_encoder_block_weights(state_dict, block_prefix, channels[block_idx])
        x = encoder_block_forward(x, block_weights, strides[block_idx])
    return x

    # Final block: SnakeBeta(1536) + Conv1d(1536, 1024, k=3) with weight norm
    final_prefix = "conv_final_block."
    x = activation1d_forward(
        x,
        state_dict[final_prefix + "0.act.alpha"],
        state_dict[final_prefix + "0.act.beta"],
        state_dict[final_prefix + "0.upsample.filter"],
        state_dict[final_prefix + "0.downsample.lowpass.filter"],
    )
    final_weight = weight_norm_compute(
        state_dict[final_prefix + "1.weight_g"],
        state_dict[final_prefix + "1.weight_v"],
    )
    final_bias = state_dict[final_prefix + "1.bias"]
    x = F.conv1d(x, final_weight, final_bias, padding=1)

    return x


def _extract_encoder_block_weights(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    in_channels: int,
) -> Dict[str, torch.Tensor]:
    """Extract weights for one EncoderBlock from the acoustic encoder state_dict.

    Actual xcodec2 checkpoint structure per EncoderBlock:
        {prefix}block.{0,1,2}.block.{0,1,2,3}.* -- 3 ResidualUnits
            .block.0 = Activation1d (SnakeBeta): act.alpha, act.beta, upsample.filter, downsample.lowpass.filter
            .block.1 = WNConv1d: weight_g, weight_v, bias
            .block.2 = Activation1d (SnakeBeta): same as block.0
            .block.3 = WNConv1d: weight_g, weight_v, bias
        {prefix}block.3.* -- final SnakeBeta before downsample
            act.alpha, act.beta, upsample.filter, downsample.lowpass.filter
        {prefix}block.4.* -- downsampling WNConv1d
            weight_g, weight_v, bias
    """
    weights = {}

    # 3 ResidualUnits (block.0, block.1, block.2)
    for res_idx in range(3):
        ru = f"{prefix}block.{res_idx}."
        dst = f"res_{res_idx}_"

        # SnakeBeta 1 (block.{i}.block.0)
        act1 = f"{ru}block.0."
        weights[dst + "act1_alpha"] = state_dict[act1 + "act.alpha"]
        weights[dst + "act1_beta"] = state_dict[act1 + "act.beta"]
        weights[dst + "act1_up_filter"] = state_dict[act1 + "upsample.filter"]
        weights[dst + "act1_down_filter"] = state_dict[act1 + "downsample.lowpass.filter"]

        # Conv1d 1 with weight_norm (block.{i}.block.1)
        conv1 = f"{ru}block.1."
        weights[dst + "conv1_weight"] = weight_norm_compute(
            state_dict[conv1 + "weight_g"],
            state_dict[conv1 + "weight_v"],
        )
        weights[dst + "conv1_bias"] = state_dict[conv1 + "bias"]

        # SnakeBeta 2 (block.{i}.block.2)
        act2 = f"{ru}block.2."
        weights[dst + "act2_alpha"] = state_dict[act2 + "act.alpha"]
        weights[dst + "act2_beta"] = state_dict[act2 + "act.beta"]
        weights[dst + "act2_up_filter"] = state_dict[act2 + "upsample.filter"]
        weights[dst + "act2_down_filter"] = state_dict[act2 + "downsample.lowpass.filter"]

        # Conv1d 2 with weight_norm (block.{i}.block.3)
        conv2 = f"{ru}block.3."
        weights[dst + "conv2_weight"] = weight_norm_compute(
            state_dict[conv2 + "weight_g"],
            state_dict[conv2 + "weight_v"],
        )
        weights[dst + "conv2_bias"] = state_dict[conv2 + "bias"]

    # Final SnakeBeta before downsample (block.3)
    act_ds = f"{prefix}block.3."
    weights["act_alpha"] = state_dict[act_ds + "act.alpha"]
    weights["act_beta"] = state_dict[act_ds + "act.beta"]
    weights["act_up_filter"] = state_dict[act_ds + "upsample.filter"]
    weights["act_down_filter"] = state_dict[act_ds + "downsample.lowpass.filter"]

    # Downsampling WNConv1d (block.4)
    ds = f"{prefix}block.4."
    weights["downsample_weight"] = weight_norm_compute(
        state_dict[ds + "weight_g"],
        state_dict[ds + "weight_v"],
    )
    weights["downsample_bias"] = state_dict[ds + "bias"]

    return weights


# ---------------------------------------------------------------------------
# SemanticEncoder
# ---------------------------------------------------------------------------
def semantic_encoder_forward(
    semantic_features: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    prefix: str = "semantic_encoder.",
) -> torch.Tensor:
    """SemanticEncoder forward.

    Input: [B, 1024, T] (from Wav2Vec2-BERT hidden_states[16], transposed)
    Output: [B, 1024, T]

    Architecture:
    - initial_conv: Conv1d(1024, 1024, k=3, pad=1)
    - residual_blocks: N x (ReLU + Conv1d + ReLU + Conv1d + residual)
    - final_conv: Conv1d(1024, 1024, k=3, pad=1)
    """
    # Initial conv
    x = F.conv1d(
        semantic_features,
        state_dict[prefix + "initial_conv.weight"],
        state_dict.get(prefix + "initial_conv.bias"),
        padding=1,
    )

    # Residual blocks: keys are "residual_blocks.{idx}.weight" / "residual_blocks.{idx}.bias"
    # The xcodec2 SemanticEncoder has 2 conv layers in a residual block at indices 1 and 3
    # (0 and 2 are ReLU activations in the Sequential)
    block_idx = 1
    while prefix + f"residual_blocks.{block_idx}.weight" in state_dict:
        res = x
        # ReLU + Conv1d (index block_idx)
        h = F.relu(x)
        h = F.conv1d(
            h,
            state_dict[prefix + f"residual_blocks.{block_idx}.weight"],
            state_dict.get(prefix + f"residual_blocks.{block_idx}.bias"),
            padding=1,
        )
        # ReLU + Conv1d (index block_idx + 2)
        h = F.relu(h)
        h = F.conv1d(
            h,
            state_dict[prefix + f"residual_blocks.{block_idx + 2}.weight"],
            state_dict.get(prefix + f"residual_blocks.{block_idx + 2}.bias"),
            padding=1,
        )
        x = res + h
        block_idx += 4  # skip to next residual block (0:ReLU, 1:Conv, 2:ReLU, 3:Conv)

    # Final conv
    x = F.conv1d(
        x,
        state_dict[prefix + "final_conv.weight"],
        state_dict.get(prefix + "final_conv.bias"),
        padding=1,
    )

    return x


# ---------------------------------------------------------------------------
# Codec Encoder: Fusion + FSQ Quantization
# ---------------------------------------------------------------------------
def codec_encoder_forward(
    acoustic_out: torch.Tensor,
    semantic_out: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    quantizer,
    prefix: str = "",
) -> torch.Tensor:
    """Fusion + FSQ quantization.

    Input: acoustic [B, 1024, T], semantic [B, 1024, T]
    Output: [B, 1, T] integer VQ codes

    Pipeline:
    - Concatenate acoustic + semantic -> [B, 2048, T]
    - fc_prior: Linear(2048, 2048)
    - FSQ quantizer: project_in(2048, 8) -> quantize -> project_out(8, 2048)
    - Output: integer VQ codes
    """
    # Concatenate along channel dimension
    fused = torch.cat([acoustic_out, semantic_out], dim=1)  # [B, 2048, T]

    # Transpose to [B, T, 2048] for linear layers
    fused = fused.transpose(1, 2)  # [B, T, 2048]

    # fc_prior: Linear(2048, 2048)
    fc_weight = state_dict[prefix + "fc_prior.weight"]
    fc_bias = state_dict[prefix + "fc_prior.bias"]
    fused = F.linear(fused, fc_weight, fc_bias)  # [B, T, 2048]

    # FSQ quantize using the quantizer module
    # quantizer expects [B, T, D] and returns (quantized, indices)
    # ResidualFSQ forward: project_in -> quantize -> project_out
    _, indices = quantizer(fused)  # indices: [B, T, num_quantizers]

    # indices shape is [B, T, 1] for single quantizer, convert to [B, 1, T]
    vq_codes = indices.squeeze(-1).unsqueeze(1)  # [B, 1, T]

    return vq_codes


# ---------------------------------------------------------------------------
# Encoder Weight Extraction Helpers
# ---------------------------------------------------------------------------
def extract_acoustic_encoder_weights(
    state_dict: Dict[str, torch.Tensor],
    prefix: str = "encoder.acoustic_encoder.",
) -> Dict[str, torch.Tensor]:
    """Extract AcousticEncoder weights from a full codec state_dict.

    Returns a sub-dict with the prefix stripped so acoustic_encoder_forward
    can use it directly (keys start with "conv_blocks.", "conv_final_block.", etc.).
    """
    result = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            result[k[len(prefix) :]] = v
    return result


def extract_semantic_encoder_weights(
    state_dict: Dict[str, torch.Tensor],
    prefix: str = "SemanticEncoder_module.",
) -> Dict[str, torch.Tensor]:
    """Extract SemanticEncoder weights, stripping the prefix.

    Returns dict with keys like 'initial_conv.weight', 'residual_blocks.1.weight', etc.
    """
    result = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            result[k[len(prefix) :]] = v
    return result


def extract_encoder_fusion_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Extract fusion layer weights (fc_prior) from a full codec state_dict.

    In the xcodec2 checkpoint, keys are 'fc_prior.weight' and 'fc_prior.bias' directly.
    """
    result = {}
    for k, v in state_dict.items():
        if k.startswith("fc_prior."):
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Wav2Vec2-BERT (Conformer Encoder)
# ---------------------------------------------------------------------------


def w2v_feature_projection_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Feature projection: LayerNorm(160) -> Linear(160, 1024).

    Args:
        x: [B, T, 160] mel filterbank features
        state_dict: must contain feature_projection.layer_norm.{weight,bias},
                    feature_projection.projection.{weight,bias}
    Returns:
        [B, T, 1024]
    """
    # LayerNorm
    x = F.layer_norm(
        x,
        [x.shape[-1]],
        state_dict["feature_projection.layer_norm.weight"],
        state_dict["feature_projection.layer_norm.bias"],
        eps=eps,
    )
    # Linear projection
    x = F.linear(
        x,
        state_dict["feature_projection.projection.weight"],
        state_dict["feature_projection.projection.bias"],
    )
    return x


def w2v_ffn_forward(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_bias: torch.Tensor,
) -> torch.Tensor:
    """Feed-forward network: Linear -> SiLU -> Linear.

    Args:
        x: [B, T, 1024]
        w1_weight: [4096, 1024], w1_bias: [4096]
        w2_weight: [1024, 4096], w2_bias: [1024]
    Returns:
        [B, T, 1024]
    """
    h = F.linear(x, w1_weight, w1_bias)
    h = F.silu(h)
    h = F.linear(h, w2_weight, w2_bias)
    return h


def w2v_relative_position_bias(
    query: torch.Tensor,
    distance_embedding_weight: torch.Tensor,
    seq_len: int,
    left_max: int = 64,
    right_max: int = 8,
) -> torch.Tensor:
    """Compute relative key position bias for self-attention.

    The HuggingFace Wav2Vec2BertSdpaSelfAttention computes:
      positions = arange(seq_len)
      distances = positions[:, None] - positions[None, :]  # [T, T]
      distances = distances.clamp(-left_max, right_max) + left_max  # shift to [0, 72]
      pos_embed = distance_embedding(distances)  # [T, T, head_dim]
      pos_bias = einsum("bhld,lrd->bhlr", query, pos_embed) / sqrt(head_dim)

    Args:
        query: [B, H, T, D] query tensor
        distance_embedding_weight: [73, 64] embedding weight (73 = left_max + right_max + 1)
        seq_len: sequence length T
        left_max: max leftward distance (default 64)
        right_max: max rightward distance (default 8)
    Returns:
        [B, H, T, T] position bias to add to attention scores
    """
    head_dim = query.shape[-1]
    positions = torch.arange(seq_len, device=query.device)
    distances = positions[:, None] - positions[None, :]  # [T, T]
    distances = distances.clamp(-left_max, right_max) + left_max  # [T, T] in [0, 72]

    # Lookup embeddings: [T, T, head_dim]
    pos_embed = distance_embedding_weight[distances.long()]

    # einsum: "bhld,lrd->bhlr"
    # query: [B, H, T, D], pos_embed: [T, T, D]
    pos_bias = torch.einsum("bhld,lrd->bhlr", query, pos_embed.to(query.dtype))
    pos_bias = pos_bias / (head_dim**0.5)

    return pos_bias


def w2v_self_attention_forward(
    x: torch.Tensor,
    state_dict_prefix: str,
    state_dict: Dict[str, torch.Tensor],
    seq_len: int,
) -> torch.Tensor:
    """Full multi-head attention with relative position bias.

    Args:
        x: [B, T, 1024] input
        state_dict_prefix: e.g. "encoder.layers.0.self_attn."
        state_dict: contains linear_q/k/v/out weights/biases, distance_embedding.weight
        seq_len: sequence length
    Returns:
        [B, T, 1024]
    """
    B, T, dim = x.shape
    n_heads = 16
    head_dim = dim // n_heads  # 64
    p = state_dict_prefix

    # Q, K, V projections
    q = F.linear(x, state_dict[p + "linear_q.weight"], state_dict[p + "linear_q.bias"])
    k = F.linear(x, state_dict[p + "linear_k.weight"], state_dict[p + "linear_k.bias"])
    v = F.linear(x, state_dict[p + "linear_v.weight"], state_dict[p + "linear_v.bias"])

    # Reshape to [B, H, T, D]
    q = q.view(B, T, n_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(B, T, n_heads, head_dim).permute(0, 2, 1, 3)
    v = v.view(B, T, n_heads, head_dim).permute(0, 2, 1, 3)

    # Attention scores: Q @ K^T / sqrt(D)
    scale = head_dim**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Add relative position bias
    dist_emb = state_dict[p + "distance_embedding.weight"]
    pos_bias = w2v_relative_position_bias(q, dist_emb, seq_len)
    scores = scores + pos_bias

    # Softmax + weighted sum
    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # Merge heads: [B, H, T, D] -> [B, T, dim]
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, dim)

    # Output projection
    attn_output = F.linear(
        attn_output,
        state_dict[p + "linear_out.weight"],
        state_dict[p + "linear_out.bias"],
    )
    return attn_output


def w2v_conv_module_forward(
    x: torch.Tensor,
    state_dict_prefix: str,
    state_dict: Dict[str, torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Conformer convolution module.

    LayerNorm -> pointwise_conv1 (1024->2048, k=1) -> GLU -> causal pad ->
    depthwise_conv (1024, k=31, groups=1024) -> depthwise_layer_norm ->
    SiLU -> pointwise_conv2 (1024->1024, k=1)

    Args:
        x: [B, T, 1024]
        state_dict_prefix: e.g. "encoder.layers.0.conv_module."
        state_dict: conv module weights
    Returns:
        [B, T, 1024]
    """
    p = state_dict_prefix

    # LayerNorm
    h = F.layer_norm(
        x,
        [x.shape[-1]],
        state_dict[p + "layer_norm.weight"],
        state_dict[p + "layer_norm.bias"],
        eps=eps,
    )

    # Transpose to [B, C, T] for conv ops
    h = h.transpose(1, 2)

    # Pointwise conv1: Conv1d(1024, 2048, k=1, bias=False)
    h = F.conv1d(h, state_dict[p + "pointwise_conv1.weight"])

    # GLU along channel dim: split into two halves, sigmoid gate
    h1, h2 = h.chunk(2, dim=1)
    h = h1 * torch.sigmoid(h2)

    # Causal left-pad for depthwise conv (kernel=31, causal: pad 30 left, 0 right)
    h = F.pad(h, (30, 0))

    # Depthwise conv: Conv1d(1024, 1024, k=31, groups=1024, bias=False)
    h = F.conv1d(h, state_dict[p + "depthwise_conv.weight"], groups=1024)

    # Depthwise layer norm: transpose -> LN -> transpose
    h = h.transpose(1, 2)  # [B, T, 1024]
    h = F.layer_norm(
        h,
        [h.shape[-1]],
        state_dict[p + "depthwise_layer_norm.weight"],
        state_dict[p + "depthwise_layer_norm.bias"],
        eps=eps,
    )
    h = h.transpose(1, 2)  # [B, 1024, T]

    # SiLU
    h = F.silu(h)

    # Pointwise conv2: Conv1d(1024, 1024, k=1, bias=False)
    h = F.conv1d(h, state_dict[p + "pointwise_conv2.weight"])

    # Back to [B, T, 1024]
    h = h.transpose(1, 2)

    return h


def w2v_conformer_layer_forward(
    x: torch.Tensor,
    layer_idx: int,
    state_dict: Dict[str, torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Single Conformer encoder layer with Macaron half-step FFNs.

    Sub-block 1: FFN1 (half-step)
    Sub-block 2: Self-Attention with relative position bias
    Sub-block 3: Convolution module
    Sub-block 4: FFN2 (half-step)
    Final: LayerNorm

    Args:
        x: [B, T, 1024]
        layer_idx: layer index (0-15)
        state_dict: full model state dict
    Returns:
        [B, T, 1024]
    """
    prefix = f"encoder.layers.{layer_idx}."
    T = x.shape[1]

    # Sub-block 1: FFN1 (half-step)
    residual = x
    h = F.layer_norm(
        x,
        [x.shape[-1]],
        state_dict[prefix + "ffn1_layer_norm.weight"],
        state_dict[prefix + "ffn1_layer_norm.bias"],
        eps=eps,
    )
    h = w2v_ffn_forward(
        h,
        state_dict[prefix + "ffn1.intermediate_dense.weight"],
        state_dict[prefix + "ffn1.intermediate_dense.bias"],
        state_dict[prefix + "ffn1.output_dense.weight"],
        state_dict[prefix + "ffn1.output_dense.bias"],
    )
    x = h * 0.5 + residual

    # Sub-block 2: Self-Attention
    residual = x
    h = F.layer_norm(
        x,
        [x.shape[-1]],
        state_dict[prefix + "self_attn_layer_norm.weight"],
        state_dict[prefix + "self_attn_layer_norm.bias"],
        eps=eps,
    )
    h = w2v_self_attention_forward(h, prefix + "self_attn.", state_dict, T)
    x = h + residual

    # Sub-block 3: Convolution Module
    residual = x
    h = w2v_conv_module_forward(x, prefix + "conv_module.", state_dict, eps=eps)
    x = h + residual

    # Sub-block 4: FFN2 (half-step)
    residual = x
    h = F.layer_norm(
        x,
        [x.shape[-1]],
        state_dict[prefix + "ffn2_layer_norm.weight"],
        state_dict[prefix + "ffn2_layer_norm.bias"],
        eps=eps,
    )
    h = w2v_ffn_forward(
        h,
        state_dict[prefix + "ffn2.intermediate_dense.weight"],
        state_dict[prefix + "ffn2.intermediate_dense.bias"],
        state_dict[prefix + "ffn2.output_dense.weight"],
        state_dict[prefix + "ffn2.output_dense.bias"],
    )
    x = h * 0.5 + residual

    # Final LayerNorm
    x = F.layer_norm(
        x,
        [x.shape[-1]],
        state_dict[prefix + "final_layer_norm.weight"],
        state_dict[prefix + "final_layer_norm.bias"],
        eps=eps,
    )

    return x


def w2v_encoder_forward(
    input_features: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    num_layers: int = 16,
) -> torch.Tensor:
    """Full Wav2Vec2-BERT encoder forward: feature_projection + N conformer layers.

    Produces hidden_states[num_layers] (the output after the last requested layer).

    Args:
        input_features: [B, T, 160] mel filterbank features from AutoFeatureExtractor
        state_dict: full Wav2Vec2BertModel state dict (with "wav2vec2_bert." prefix stripped)
        num_layers: number of conformer layers to run (default 16 for layers 0-15)
    Returns:
        [B, T, 1024] hidden states after layer num_layers-1
    """
    # Feature projection: LayerNorm(160) -> Linear(160, 1024)
    x = w2v_feature_projection_forward(input_features, state_dict)

    # Run conformer layers
    for i in range(num_layers):
        x = w2v_conformer_layer_forward(x, i, state_dict)

    return x
