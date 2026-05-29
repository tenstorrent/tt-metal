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


# ===========================================================================
# Language-model (Qwen2) blocks
# ===========================================================================
# LM config (dots.ocr Qwen2): hidden_size=1536, intermediate_size=8960,
# num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2 (GQA),
# head_dim=128, vocab_size=151936, rope_theta=1e6, rms_norm_eps=1e-6,
# hidden_act=silu (SwiGLU), attention_bias=true (q/k/v_proj biased, o_proj not),
# tie_word_embeddings=false.


# ---------------------------------------------------------------------------
# rmsnorm (Qwen2RMSNorm)
# ---------------------------------------------------------------------------
def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen2RMSNorm forward.

    Mirrors HF exactly: normalize in fp32, cast back to input dtype, then
    `weight * hidden.to(input_dtype)`. eps defaults to 1e-6 (LM) vs 1e-5 (vision).
    """
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return weight * xf.to(input_dtype)


# ---------------------------------------------------------------------------
# rope (Qwen2RotaryEmbedding: position_ids -> (cos, sin))
# ---------------------------------------------------------------------------
def rope_forward(
    position_ids: torch.Tensor,
    head_dim: int = 128,
    rope_theta: float = 1000000.0,
    dtype: torch.dtype = torch.float32,
    attention_scaling: float = 1.0,
) -> tuple:
    """Qwen2RotaryEmbedding.forward (default rope_type, attention_scaling=1.0).

    Builds the rotary cos/sin tables from inv_freq and position ids. The freqs
    are computed in fp32 then cast to `dtype`.

    Args:
        position_ids: [batch, seq_len] int positions.
        head_dim: per-head dim (rotary applies over the full head_dim).
        rope_theta: base (dots.ocr LM uses 1e6).
    Returns:
        (cos, sin) each [batch, seq_len, head_dim].
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _rotate_half_lm(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_lm(q, k, cos, sin, unsqueeze_dim: int = 1):
    """HF apply_rotary_pos_emb for the LM (1D RoPE)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_lm(q) * sin)
    k_embed = (k * cos) + (_rotate_half_lm(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# attention (Qwen2Attention, GQA + QKV bias + 1D RoPE, causal)
# ---------------------------------------------------------------------------
def attention_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    position_embeddings: tuple,
    attention_mask: torch.Tensor = None,
    num_heads: int = 12,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    bias: bool = True,
) -> torch.Tensor:
    """Qwen2Attention (eager) forward.

    Fused-less projections q/k/v with bias, GQA (num_heads // num_kv_heads),
    1D RoPE on q,k, causal eager attention (repeat_kv), o_proj (no bias).

    Args:
        hidden_states: [batch, seq_len, hidden].
        state_dict: 'q_proj.weight'(+'.bias'), 'k_proj.weight'(+'.bias'),
            'v_proj.weight'(+'.bias'), 'o_proj.weight'.
        position_embeddings: (cos, sin) from rope_forward, each [batch, seq, head_dim].
        attention_mask: additive mask [batch, 1, q_len, kv_len]; if None a causal
            mask is constructed.
    Returns:
        [batch, seq_len, hidden].
    """
    bsz, q_len, _ = hidden_states.shape
    scaling = head_dim**-0.5
    n_rep = num_heads // num_kv_heads

    qb = state_dict.get("q_proj.bias") if bias else None
    kb = state_dict.get("k_proj.bias") if bias else None
    vb = state_dict.get("v_proj.bias") if bias else None

    q = F.linear(hidden_states, state_dict["q_proj.weight"], qb).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = (
        F.linear(hidden_states, state_dict["k_proj.weight"], kb)
        .view(bsz, q_len, num_kv_heads, head_dim)
        .transpose(1, 2)
    )
    v = (
        F.linear(hidden_states, state_dict["v_proj.weight"], vb)
        .view(bsz, q_len, num_kv_heads, head_dim)
        .transpose(1, 2)
    )

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_lm(q, k, cos, sin)

    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling

    if attention_mask is None:
        causal = torch.full((q_len, q_len), torch.finfo(q.dtype).min, dtype=q.dtype)
        causal = torch.triu(causal, diagonal=1)
        attention_mask = causal[None, None, :, :]
    attn_weights = attn_weights + attention_mask[:, :, :, : k.shape[-2]]

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    return F.linear(attn_output, state_dict["o_proj.weight"], None)


# ---------------------------------------------------------------------------
# mlp (Qwen2MLP, SwiGLU SiLU, no bias)
# ---------------------------------------------------------------------------
def mlp_forward(x: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Qwen2MLP: down_proj(silu(gate_proj(x)) * up_proj(x)). All unbiased."""
    gate = F.silu(F.linear(x, state_dict["gate_proj.weight"]))
    up = F.linear(x, state_dict["up_proj.weight"])
    return F.linear(gate * up, state_dict["down_proj.weight"])


# ---------------------------------------------------------------------------
# decoder_layer (one full Qwen2DecoderLayer)
# ---------------------------------------------------------------------------
def decoder_layer_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    position_embeddings: tuple,
    attention_mask: torch.Tensor = None,
    num_heads: int = 12,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    eps: float = 1e-6,
    bias: bool = True,
) -> torch.Tensor:
    """Qwen2DecoderLayer.forward (pre-norm, residual):

        residual = h
        h = input_layernorm(h)
        h = residual + self_attn(h)
        residual = h
        h = post_attention_layernorm(h)
        h = residual + mlp(h)

    Composes rmsnorm + attention + mlp.

    Args:
        hidden_states: [batch, seq_len, hidden].
        state_dict: prefixed weights:
            'input_layernorm.weight',
            'self_attn.q_proj.weight'(+'.bias'), 'self_attn.k_proj.weight'(+'.bias'),
            'self_attn.v_proj.weight'(+'.bias'), 'self_attn.o_proj.weight',
            'post_attention_layernorm.weight',
            'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'.
        position_embeddings: (cos, sin) from rope_forward.
        attention_mask: additive causal mask; if None a causal mask is built.
    Returns:
        [batch, seq_len, hidden].
    """
    attn_sd = {
        "q_proj.weight": state_dict["self_attn.q_proj.weight"],
        "k_proj.weight": state_dict["self_attn.k_proj.weight"],
        "v_proj.weight": state_dict["self_attn.v_proj.weight"],
        "o_proj.weight": state_dict["self_attn.o_proj.weight"],
    }
    if bias:
        attn_sd["q_proj.bias"] = state_dict.get("self_attn.q_proj.bias")
        attn_sd["k_proj.bias"] = state_dict.get("self_attn.k_proj.bias")
        attn_sd["v_proj.bias"] = state_dict.get("self_attn.v_proj.bias")

    residual = hidden_states
    normed = rmsnorm_forward(hidden_states, state_dict["input_layernorm.weight"], eps=eps)
    attn_out = attention_forward(
        normed,
        attn_sd,
        position_embeddings,
        attention_mask=attention_mask,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bias=bias,
    )
    hidden_states = residual + attn_out

    residual = hidden_states
    normed = rmsnorm_forward(hidden_states, state_dict["post_attention_layernorm.weight"], eps=eps)
    mlp_sd = {
        "gate_proj.weight": state_dict["mlp.gate_proj.weight"],
        "up_proj.weight": state_dict["mlp.up_proj.weight"],
        "down_proj.weight": state_dict["mlp.down_proj.weight"],
    }
    hidden_states = residual + mlp_forward(normed, mlp_sd)
    return hidden_states


# ---------------------------------------------------------------------------
# lm_head (untied Linear hidden -> vocab, no bias)
# ---------------------------------------------------------------------------
def lm_head_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LM head: Linear(hidden_size -> vocab_size, bias=False).

    Args:
        x: [batch, seq_len, hidden_size] final hidden states.
        weight: [vocab_size, hidden_size] (untied, tie_word_embeddings=False).
    Returns:
        logits [batch, seq_len, vocab_size].
    """
    return F.linear(x, weight, None)


# ---------------------------------------------------------------------------
# language_model (full Qwen2ForCausalLM: embed -> N decoder layers -> norm -> lm_head)
# ---------------------------------------------------------------------------
def language_model_forward(
    input_ids: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    num_layers: int,
    num_heads: int = 12,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    rope_theta: float = 1000000.0,
    eps: float = 1e-6,
    bias: bool = True,
    position_ids: torch.Tensor = None,
) -> torch.Tensor:
    """Qwen2ForCausalLM.forward (causal, no KV cache):

        embed_tokens -> rotary_emb(cos,sin) -> N x decoder_layer (causal) -> norm -> lm_head.

    `num_layers` lets the golden run at a REDUCED depth (real config has 28 layers).
    The full-depth check is deferred to the real_weights phase.

    Args:
        input_ids: [batch, seq_len] int token ids.
        state_dict: flat keys:
            'embed_tokens.weight',
            'layers.{i}.input_layernorm.weight',
            'layers.{i}.self_attn.{q,k,v}_proj.weight' (+ '.bias'),
            'layers.{i}.self_attn.o_proj.weight',
            'layers.{i}.post_attention_layernorm.weight',
            'layers.{i}.mlp.{gate,up,down}_proj.weight',
            'norm.weight', 'lm_head.weight'.
    Returns:
        logits [batch, seq_len, vocab_size].
    """
    bsz, seq_len = input_ids.shape

    hidden_states = embedding_forward(input_ids, state_dict["embed_tokens.weight"])

    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    cos, sin = rope_forward(position_ids, head_dim=head_dim, rope_theta=rope_theta, dtype=hidden_states.dtype)

    # Shared causal additive mask.
    causal = torch.full((seq_len, seq_len), torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
    causal = torch.triu(causal, diagonal=1)
    attention_mask = causal[None, None, :, :]

    for i in range(num_layers):
        layer_sd = {
            "input_layernorm.weight": state_dict[f"layers.{i}.input_layernorm.weight"],
            "self_attn.q_proj.weight": state_dict[f"layers.{i}.self_attn.q_proj.weight"],
            "self_attn.k_proj.weight": state_dict[f"layers.{i}.self_attn.k_proj.weight"],
            "self_attn.v_proj.weight": state_dict[f"layers.{i}.self_attn.v_proj.weight"],
            "self_attn.o_proj.weight": state_dict[f"layers.{i}.self_attn.o_proj.weight"],
            "post_attention_layernorm.weight": state_dict[f"layers.{i}.post_attention_layernorm.weight"],
            "mlp.gate_proj.weight": state_dict[f"layers.{i}.mlp.gate_proj.weight"],
            "mlp.up_proj.weight": state_dict[f"layers.{i}.mlp.up_proj.weight"],
            "mlp.down_proj.weight": state_dict[f"layers.{i}.mlp.down_proj.weight"],
        }
        if bias:
            layer_sd["self_attn.q_proj.bias"] = state_dict.get(f"layers.{i}.self_attn.q_proj.bias")
            layer_sd["self_attn.k_proj.bias"] = state_dict.get(f"layers.{i}.self_attn.k_proj.bias")
            layer_sd["self_attn.v_proj.bias"] = state_dict.get(f"layers.{i}.self_attn.v_proj.bias")
        hidden_states = decoder_layer_forward(
            hidden_states,
            layer_sd,
            (cos, sin),
            attention_mask=attention_mask,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            bias=bias,
        )

    hidden_states = rmsnorm_forward(hidden_states, state_dict["norm.weight"], eps=eps)
    return lm_head_forward(hidden_states, state_dict["lm_head.weight"])
