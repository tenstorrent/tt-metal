# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone PyTorch reference implementations for rednote-hilab/dots.ocr blocks.

Pure PyTorch (no TTNN imports). Each function mirrors the corresponding HuggingFace
module in the `trust_remote_code` snapshot `modeling_dots_vision.py` exactly, so that
TTNN blocks can be verified against the golden tensors produced here.

Vision tower config (dots_vit): embed_dim=1536, intermediate_size=4224,
num_attention_heads=12 (head_dim=128), patch_size=14, temporal_patch_size=1,
num_channels=3, rms_norm_eps=1e-5, use_bias=False (qkv/proj/fc unbiased),
2D VisionRotaryEmbedding(head_dim//2, theta=1e4), bidirectional block-diagonal attn.
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# vision_rmsnorm
# ---------------------------------------------------------------------------
def vision_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """dots RMSNorm (modeling_dots_vision.RMSNorm).

    Computes the norm in fp32 then casts back to the input dtype, then multiplies
    by the (input-dtype) weight -- matching `output = self._norm(x.float()).type_as(x); return output * self.weight`.
    """
    orig_dtype = x.dtype
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normed = normed.to(orig_dtype)
    return normed * weight


# ---------------------------------------------------------------------------
# vision_patch_embed
# ---------------------------------------------------------------------------
def vision_patch_embed_forward(
    pixel_values: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    num_channels: int = 3,
    temporal_patch_size: int = 1,
    patch_size: int = 14,
    embed_dim: int = 1536,
    eps: float = 1e-5,
) -> torch.Tensor:
    """DotsPatchEmbed.forward: reshape -> Conv2d(patchify) -> flatten -> RMSNorm.

    Args:
        pixel_values: [num_patches, num_channels * temporal_patch_size * patch_size * patch_size]
                      (flattened patches, the Qwen-VL packed format).
        state_dict: keys 'proj.weight' [embed_dim,3,14,14], 'proj.bias' [embed_dim],
                    'norm.weight' [embed_dim].
    Returns:
        [num_patches, embed_dim]
    """
    x = pixel_values.view(-1, num_channels, temporal_patch_size, patch_size, patch_size)[:, :, 0]
    x = F.conv2d(
        x,
        state_dict["proj.weight"],
        state_dict["proj.bias"],
        stride=(patch_size, patch_size),
    ).view(-1, embed_dim)
    return vision_rmsnorm_forward(x, state_dict["norm.weight"], eps=eps)


# ---------------------------------------------------------------------------
# vision RoPE helpers (needed by vision_attention)
# ---------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Matches modeling_dots_vision.apply_rotary_pos_emb_vision (fp32 internally)."""
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (_rotate_half(tensor) * sin)
    return output.to(orig_dtype)


def vision_rotary_freqs(seqlen: int, head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """VisionRotaryEmbedding(head_dim//2).forward(seqlen) -> [seqlen, head_dim//4] freqs.

    Returns the per-position freqs table used to index the 2D rope (rot_pos_emb gathers
    rows by (h,w) pos ids and flattens). For the standalone block test we use freqs of the
    appropriate flattened width directly; see vision_attention_forward.
    """
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    seq = torch.arange(seqlen, dtype=torch.float)
    return torch.outer(seq, inv_freq)


# ---------------------------------------------------------------------------
# vision_attention
# ---------------------------------------------------------------------------
def vision_attention_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    num_heads: int = 12,
    bias: bool = False,
) -> torch.Tensor:
    """VisionAttention (eager) forward: fused QKV -> 2D RoPE on q,k -> block-diagonal
    bidirectional softmax attention (via cu_seqlens mask) -> output proj.

    Args:
        hidden_states: [seq_length, dim]
        state_dict: 'qkv.weight' [3*dim, dim] (+ 'qkv.bias'), 'proj.weight' [dim, dim] (+ 'proj.bias').
        cu_seqlens: 1D int tensor of cumulative seqlens, starting with 0.
        rotary_pos_emb: [seq_length, head_dim//2] freqs (post rot_pos_emb flatten).
    Returns:
        [seq_length, dim]
    """
    seq_length, dim = hidden_states.shape
    head_dim = dim // num_heads

    qkv_b = state_dict.get("qkv.bias") if bias else None
    qkv = F.linear(hidden_states, state_dict["qkv.weight"], qkv_b)
    q, k, v = qkv.reshape(seq_length, 3, num_heads, -1).permute(1, 0, 2, 3).unbind(0)

    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    attention_mask = torch.full([1, seq_length, seq_length], torch.finfo(q.dtype).min, dtype=q.dtype)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)

    proj_b = state_dict.get("proj.bias") if bias else None
    return F.linear(attn_output, state_dict["proj.weight"], proj_b)


# ---------------------------------------------------------------------------
# vision_mlp
# ---------------------------------------------------------------------------
def vision_mlp_forward(x: torch.Tensor, state_dict: Dict[str, torch.Tensor], bias: bool = False) -> torch.Tensor:
    """DotsSwiGLUFFN: fc2(silu(fc1(x)) * fc3(x)). All unbiased (use_bias=False)."""
    fc1_b = state_dict.get("fc1.bias") if bias else None
    fc2_b = state_dict.get("fc2.bias") if bias else None
    fc3_b = state_dict.get("fc3.bias") if bias else None
    h = F.silu(F.linear(x, state_dict["fc1.weight"], fc1_b)) * F.linear(x, state_dict["fc3.weight"], fc3_b)
    return F.linear(h, state_dict["fc2.weight"], fc2_b)
