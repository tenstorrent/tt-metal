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


# ---------------------------------------------------------------------------
# vision_block (one full DotsVisionBlock layer)
# ---------------------------------------------------------------------------
def vision_block_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    num_heads: int = 12,
    eps: float = 1e-5,
    bias: bool = False,
) -> torch.Tensor:
    """DotsVisionBlock.forward (pre-norm, residual):

        h = h + attn(norm1(h))
        h = h + mlp(norm2(h))

    Composes vision_rmsnorm + vision_attention + vision_mlp.

    Args:
        hidden_states: [seq_length, dim]
        state_dict: prefixed weights:
            'norm1.weight', 'attn.qkv.weight' (+ '.bias' if bias),
            'attn.proj.weight' (+ '.bias'), 'norm2.weight',
            'mlp.fc1.weight', 'mlp.fc2.weight', 'mlp.fc3.weight' (+ biases).
        cu_seqlens: cumulative seqlens (block-diagonal attn mask).
        rotary_pos_emb: [seq_length, head_dim//2] freqs.
    Returns:
        [seq_length, dim]
    """
    attn_sd = {
        "qkv.weight": state_dict["attn.qkv.weight"],
        "proj.weight": state_dict["attn.proj.weight"],
    }
    if bias:
        attn_sd["qkv.bias"] = state_dict.get("attn.qkv.bias")
        attn_sd["proj.bias"] = state_dict.get("attn.proj.bias")

    normed1 = vision_rmsnorm_forward(hidden_states, state_dict["norm1.weight"], eps=eps)
    hidden_states = hidden_states + vision_attention_forward(
        normed1, attn_sd, cu_seqlens, rotary_pos_emb, num_heads=num_heads, bias=bias
    )

    mlp_sd = {
        "fc1.weight": state_dict["mlp.fc1.weight"],
        "fc2.weight": state_dict["mlp.fc2.weight"],
        "fc3.weight": state_dict["mlp.fc3.weight"],
    }
    if bias:
        mlp_sd["fc1.bias"] = state_dict.get("mlp.fc1.bias")
        mlp_sd["fc2.bias"] = state_dict.get("mlp.fc2.bias")
        mlp_sd["fc3.bias"] = state_dict.get("mlp.fc3.bias")

    normed2 = vision_rmsnorm_forward(hidden_states, state_dict["norm2.weight"], eps=eps)
    hidden_states = hidden_states + vision_mlp_forward(normed2, mlp_sd, bias=bias)
    return hidden_states


# ---------------------------------------------------------------------------
# vision_patch_merger (PatchMerger: LayerNorm -> view(merge) -> Linear -> GELU -> Linear)
# ---------------------------------------------------------------------------
def vision_patch_merger_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    context_dim: int = 1536,
    spatial_merge_size: int = 2,
    ln_eps: float = 1e-6,
) -> torch.Tensor:
    """PatchMerger.forward (pre_norm='layernorm'):

        x = mlp(ln_q(x).view(-1, context_dim * merge**2))
        mlp = Linear(hidden, hidden) -> GELU -> Linear(hidden, out_dim)

    NOTE: LayerNorm (not RMSNorm), with bias; mlp Linears are biased.

    Args:
        x: [num_patches, context_dim]
        state_dict: 'ln_q.weight','ln_q.bias',
            'mlp.0.weight','mlp.0.bias','mlp.2.weight','mlp.2.bias'.
    Returns:
        [num_patches // merge**2, out_dim]
    """
    hidden_size = context_dim * (spatial_merge_size**2)
    x = F.layer_norm(x, (context_dim,), state_dict["ln_q.weight"], state_dict["ln_q.bias"], eps=ln_eps)
    x = x.view(-1, hidden_size)
    x = F.linear(x, state_dict["mlp.0.weight"], state_dict["mlp.0.bias"])
    x = F.gelu(x)
    x = F.linear(x, state_dict["mlp.2.weight"], state_dict["mlp.2.bias"])
    return x


# ---------------------------------------------------------------------------
# vision_tower (full DotsVisionTransformer, reduced layer count for golden)
# ---------------------------------------------------------------------------
def _vision_rot_pos_emb(grid_thw, spatial_merge_size: int, head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Mirror DotsVisionTransformer.rot_pos_emb: build 2D (h,w) pos ids per grid,
    gather from a VisionRotaryEmbedding(head_dim//2) table, flatten."""
    pos_ids = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = int(grid_thw[:, 1:].max())

    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    seq = torch.arange(max_grid_size, dtype=torch.float)
    rotary_pos_emb_full = torch.outer(seq, inv_freq)  # [max_grid_size, head_dim//4]
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # [seq_length, head_dim//2]
    return rotary_pos_emb


def vision_tower_forward(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    num_layers: int,
    embed_dim: int = 1536,
    num_heads: int = 12,
    num_channels: int = 3,
    temporal_patch_size: int = 1,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    rms_norm_eps: float = 1e-5,
    ln_eps: float = 1e-6,
    post_norm: bool = True,
    bias: bool = False,
    hidden_size: int = 1536,
) -> torch.Tensor:
    """DotsVisionTransformer.forward (fp32 path, bf16=False):

        patch_embed -> N x vision_block -> [post_trunk_norm] -> patch_merger.

    `num_layers` lets the golden run at a REDUCED depth (the real config has 42).
    The full-depth check is deferred to the real_weights phase.

    Args:
        pixel_values: [num_patches, num_channels*temporal_patch_size*patch_size**2]
        grid_thw: [num_images, 3] (t, h, w) grids; sum(t*h*w) == num_patches.
        state_dict: flat keys:
            'patch_embed.proj.weight/bias', 'patch_embed.norm.weight',
            'blocks.{i}.norm1.weight', 'blocks.{i}.attn.qkv.weight',
            'blocks.{i}.attn.proj.weight', 'blocks.{i}.norm2.weight',
            'blocks.{i}.mlp.fc{1,2,3}.weight',
            'post_trunk_norm.weight' (if post_norm),
            'merger.ln_q.weight/bias', 'merger.mlp.0.weight/bias', 'merger.mlp.2.weight/bias'.
    Returns:
        [num_patches // merge**2, hidden_size]
    """
    head_dim = embed_dim // num_heads

    pe_sd = {
        "proj.weight": state_dict["patch_embed.proj.weight"],
        "proj.bias": state_dict["patch_embed.proj.bias"],
        "norm.weight": state_dict["patch_embed.norm.weight"],
    }
    hidden_states = vision_patch_embed_forward(
        pixel_values,
        pe_sd,
        num_channels=num_channels,
        temporal_patch_size=temporal_patch_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        eps=rms_norm_eps,
    )

    rotary_pos_emb = _vision_rot_pos_emb(grid_thw, spatial_merge_size, head_dim)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for i in range(num_layers):
        blk_sd = {
            "norm1.weight": state_dict[f"blocks.{i}.norm1.weight"],
            "attn.qkv.weight": state_dict[f"blocks.{i}.attn.qkv.weight"],
            "attn.proj.weight": state_dict[f"blocks.{i}.attn.proj.weight"],
            "norm2.weight": state_dict[f"blocks.{i}.norm2.weight"],
            "mlp.fc1.weight": state_dict[f"blocks.{i}.mlp.fc1.weight"],
            "mlp.fc2.weight": state_dict[f"blocks.{i}.mlp.fc2.weight"],
            "mlp.fc3.weight": state_dict[f"blocks.{i}.mlp.fc3.weight"],
        }
        hidden_states = vision_block_forward(
            hidden_states, blk_sd, cu_seqlens, rotary_pos_emb, num_heads=num_heads, eps=rms_norm_eps, bias=bias
        )

    if post_norm:
        hidden_states = vision_rmsnorm_forward(hidden_states, state_dict["post_trunk_norm.weight"], eps=rms_norm_eps)

    merger_sd = {
        "ln_q.weight": state_dict["merger.ln_q.weight"],
        "ln_q.bias": state_dict["merger.ln_q.bias"],
        "mlp.0.weight": state_dict["merger.mlp.0.weight"],
        "mlp.0.bias": state_dict["merger.mlp.0.bias"],
        "mlp.2.weight": state_dict["merger.mlp.2.weight"],
        "mlp.2.bias": state_dict["merger.mlp.2.bias"],
    }
    hidden_states = vision_patch_merger_forward(
        hidden_states, merger_sd, context_dim=embed_dim, spatial_merge_size=spatial_merge_size, ln_eps=ln_eps
    )
    return hidden_states


# ---------------------------------------------------------------------------
# embedding (Qwen2 token embedding)
# ---------------------------------------------------------------------------
def embedding_forward(input_ids: torch.Tensor, weight: torch.Tensor, padding_idx=None) -> torch.Tensor:
    """nn.Embedding lookup: weight[input_ids].

    Args:
        input_ids: [batch, seq_len] (or any int shape).
        weight: [vocab_size, hidden_size] embedding table.
    Returns:
        [*input_ids.shape, hidden_size]
    """
    return F.embedding(input_ids, weight, padding_idx=padding_idx)
