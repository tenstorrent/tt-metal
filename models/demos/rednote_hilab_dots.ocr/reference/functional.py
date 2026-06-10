# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch reference implementations for dots.ocr blocks.

Standalone functions, no TTNN imports. Faithful to
modeling_dots_vision.py from rednote-hilab/dots.ocr.
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# vision_rmsnorm
# ---------------------------------------------------------------------------
def vision_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm exactly as dots.ocr RMSNorm: normalize in fp32, scale after cast back."""
    out = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).type_as(x)
    return out * weight


# ---------------------------------------------------------------------------
# vision_patch_embed
# ---------------------------------------------------------------------------
def vision_patch_embed_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    num_channels: int = 3,
    temporal_patch_size: int = 1,
    patch_size: int = 14,
    embed_dim: int = 1536,
    eps: float = 1e-5,
) -> torch.Tensor:
    """DotsPatchEmbed: flattened patches -> Conv2d(stride=patch) -> RMSNorm.

    Args:
        x: [num_patches, C*T*P*P] flattened patches.
        state_dict: proj.weight [E, C, P, P], proj.bias [E], norm.weight [E].
    Returns: [num_patches, embed_dim]
    """
    x = x.view(-1, num_channels, temporal_patch_size, patch_size, patch_size)[:, :, 0]
    x = F.conv2d(x, state_dict["proj.weight"], state_dict["proj.bias"], stride=patch_size)
    x = x.view(-1, embed_dim)
    return vision_rmsnorm_forward(x, state_dict["norm.weight"], eps)


# ---------------------------------------------------------------------------
# vision rotary helpers (host-side tables)
# ---------------------------------------------------------------------------
def vision_rot_pos_emb(
    grid_thw: torch.Tensor, head_dim: int = 128, spatial_merge_size: int = 2, theta: float = 10000.0
) -> torch.Tensor:
    """2D rotary table per patch, [seq, head_dim//4*2] (flattened h/w halves)."""
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    pos_ids = []
    for t, h, w in grid_thw.tolist():
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos = (
            hpos.reshape(h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos = (
            wpos.reshape(h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid = int(grid_thw[:, 1:].max())
    freqs = torch.outer(torch.arange(max_grid, dtype=torch.float), inv_freq)
    return freqs[pos_ids].flatten(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig = t.dtype
    t = t.float()
    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    return ((t * cos) + (_rotate_half(t) * sin)).to(orig)


# ---------------------------------------------------------------------------
# vision_attention
# ---------------------------------------------------------------------------
def vision_attention_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    num_heads: int = 12,
) -> torch.Tensor:
    """Eager MHA with fused QKV (no bias), 2D rope, block-diagonal cu_seqlens mask.

    x: [seq, dim]; state_dict: qkv.weight [3*dim, dim], proj.weight [dim, dim].
    """
    seq, dim = x.shape
    head_dim = dim // num_heads
    qkv = F.linear(x, state_dict["qkv.weight"])
    q, k, v = qkv.reshape(seq, 3, num_heads, head_dim).permute(1, 0, 2, 3).unbind(0)
    q = _apply_rotary_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = _apply_rotary_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    mask = torch.full([1, seq, seq], torch.finfo(q.dtype).min, dtype=q.dtype)
    for i in range(1, len(cu_seqlens)):
        mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim) + mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v).transpose(0, 1).reshape(seq, -1)
    return F.linear(out, state_dict["proj.weight"])


# ---------------------------------------------------------------------------
# vision_mlp
# ---------------------------------------------------------------------------
def vision_mlp_forward(x: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """SwiGLU: fc2(silu(fc1(x)) * fc3(x)), no biases."""
    return F.linear(
        F.silu(F.linear(x, state_dict["fc1.weight"])) * F.linear(x, state_dict["fc3.weight"]), state_dict["fc2.weight"]
    )


# ---------------------------------------------------------------------------
# vision_block
# ---------------------------------------------------------------------------
def vision_block_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    num_heads: int = 12,
    eps: float = 1e-5,
) -> torch.Tensor:
    """DotsVisionBlock: pre-norm residual; x + attn(norm1(x)); x + mlp(norm2(x)).

    state_dict keys: norm1.weight, attn.qkv.weight, attn.proj.weight,
    norm2.weight, mlp.fc1.weight, mlp.fc2.weight, mlp.fc3.weight.
    """
    h = x + vision_attention_forward(
        vision_rmsnorm_forward(x, state_dict["norm1.weight"], eps),
        {"qkv.weight": state_dict["attn.qkv.weight"], "proj.weight": state_dict["attn.proj.weight"]},
        cu_seqlens,
        rotary_pos_emb,
        num_heads=num_heads,
    )
    return h + vision_mlp_forward(
        vision_rmsnorm_forward(h, state_dict["norm2.weight"], eps),
        {
            "fc1.weight": state_dict["mlp.fc1.weight"],
            "fc2.weight": state_dict["mlp.fc2.weight"],
            "fc3.weight": state_dict["mlp.fc3.weight"],
        },
    )


# ---------------------------------------------------------------------------
# patch_merger
# ---------------------------------------------------------------------------
def patch_merger_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    spatial_merge_size: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PatchMerger: LayerNorm(eps=1e-6) -> view(-1, C*m^2) -> Linear -> GELU -> Linear.

    state_dict keys: ln_q.weight, ln_q.bias, mlp.0.weight, mlp.0.bias,
    mlp.2.weight, mlp.2.bias.
    """
    dim = x.shape[-1]
    x = F.layer_norm(x, (dim,), state_dict["ln_q.weight"], state_dict["ln_q.bias"], eps)
    x = x.view(-1, dim * spatial_merge_size**2)
    x = F.gelu(F.linear(x, state_dict["mlp.0.weight"], state_dict["mlp.0.bias"]))
    return F.linear(x, state_dict["mlp.2.weight"], state_dict["mlp.2.bias"])


# ---------------------------------------------------------------------------
# vision_transformer (full DotsVisionTransformer)
# ---------------------------------------------------------------------------
def vision_transformer_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    grid_thw: torch.Tensor,
    num_layers: int = 42,
    num_heads: int = 12,
    eps: float = 1e-5,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Full vision tower: patch_embed -> num_layers x vision_block ->
    post_trunk_norm -> patch_merger.

    state_dict is flat with ``vision_tower.``-stripped HF keys, e.g.
    patch_embed.patchifier.proj.weight, blocks.0.norm1.weight, ...,
    post_trunk_norm.weight, merger.ln_q.weight.
    """
    h = vision_patch_embed_forward(
        x,
        {
            "proj.weight": state_dict["patch_embed.patchifier.proj.weight"],
            "proj.bias": state_dict["patch_embed.patchifier.proj.bias"],
            "norm.weight": state_dict["patch_embed.patchifier.norm.weight"],
        },
        eps=eps,
    )
    head_dim = h.shape[-1] // num_heads
    rotary_pos_emb = vision_rot_pos_emb(grid_thw, head_dim=head_dim, spatial_merge_size=spatial_merge_size)
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    for i in range(num_layers):
        prefix = f"blocks.{i}."
        block_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        h = vision_block_forward(h, block_sd, cu_seqlens, rotary_pos_emb, num_heads=num_heads, eps=eps)
    h = vision_rmsnorm_forward(h, state_dict["post_trunk_norm.weight"], eps)
    merger_sd = {k[len("merger.") :]: v for k, v in state_dict.items() if k.startswith("merger.")}
    return patch_merger_forward(h, merger_sd, spatial_merge_size=spatial_merge_size)


# ---------------------------------------------------------------------------
# embedding (text token embedding)
# ---------------------------------------------------------------------------
def embedding_forward(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Token embedding lookup: weight [vocab, hidden], untied from lm_head."""
    return F.embedding(input_ids, weight)


# ---------------------------------------------------------------------------
# text_rmsnorm (Qwen2RMSNorm, eps=1e-6)
# ---------------------------------------------------------------------------
def text_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen2RMSNorm: fp32 variance + rsqrt, cast back, THEN scale by weight."""
    out = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).type_as(x)
    return weight * out


# ---------------------------------------------------------------------------
# text rotary helpers (host-side tables, Qwen2 default rope theta=1e6)
# ---------------------------------------------------------------------------
def text_rope_cos_sin(position_ids: torch.Tensor, head_dim: int = 128, theta: float = 1e6):
    """cos/sin tables matching Qwen2RotaryEmbedding (default rope, scaling=1).

    position_ids: [batch, seq] int. Returns cos, sin each [batch, seq, head_dim].
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    freqs = position_ids[..., None].float() * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _apply_rotary_text(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q/k: [batch, heads, seq, head_dim]; cos/sin: [batch, seq, head_dim]."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# text_attention (Qwen2 GQA: 12 Q heads, 2 KV heads, QKV bias, causal)
# ---------------------------------------------------------------------------
def text_attention_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int = 12,
    num_kv_heads: int = 2,
) -> torch.Tensor:
    """Eager causal GQA exactly as Qwen2Attention.

    x: [batch, seq, hidden]. state_dict keys: q_proj.weight/.bias,
    k_proj.weight/.bias, v_proj.weight/.bias, o_proj.weight (no bias).
    cos/sin from text_rope_cos_sin.
    """
    bsz, seq, hidden = x.shape
    head_dim = hidden // num_heads
    q = F.linear(x, state_dict["q_proj.weight"], state_dict["q_proj.bias"])
    k = F.linear(x, state_dict["k_proj.weight"], state_dict["k_proj.bias"])
    v = F.linear(x, state_dict["v_proj.weight"], state_dict["v_proj.bias"])
    q = q.view(bsz, seq, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq, num_kv_heads, head_dim).transpose(1, 2)
    q, k = _apply_rotary_text(q, k, cos, sin)

    rep = num_heads // num_kv_heads
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)

    mask = torch.triu(torch.full((seq, seq), torch.finfo(q.dtype).min, dtype=q.dtype), diagonal=1)
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim) + mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq, hidden)
    return F.linear(out, state_dict["o_proj.weight"])


# ---------------------------------------------------------------------------
# text_mlp (Qwen2 SwiGLU 1536 -> 8960 -> 1536, no bias)
# ---------------------------------------------------------------------------
def text_mlp_forward(x: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """down(silu(gate(x)) * up(x)), no biases."""
    return F.linear(
        F.silu(F.linear(x, state_dict["gate_proj.weight"])) * F.linear(x, state_dict["up_proj.weight"]),
        state_dict["down_proj.weight"],
    )


# ---------------------------------------------------------------------------
# decoder_layer (Qwen2DecoderLayer: pre-norm residual)
# ---------------------------------------------------------------------------
def decoder_layer_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int = 12,
    num_kv_heads: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """x + attn(input_layernorm(x)); h + mlp(post_attention_layernorm(h)).

    state_dict keys: input_layernorm.weight, self_attn.{q,k,v}_proj.weight/.bias,
    self_attn.o_proj.weight, post_attention_layernorm.weight,
    mlp.{gate,up,down}_proj.weight.
    """
    attn_sd = {k[len("self_attn.") :]: v for k, v in state_dict.items() if k.startswith("self_attn.")}
    mlp_sd = {k[len("mlp.") :]: v for k, v in state_dict.items() if k.startswith("mlp.")}
    h = x + text_attention_forward(
        text_rmsnorm_forward(x, state_dict["input_layernorm.weight"], eps),
        attn_sd,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    return h + text_mlp_forward(text_rmsnorm_forward(h, state_dict["post_attention_layernorm.weight"], eps), mlp_sd)


# ---------------------------------------------------------------------------
# lm_head (untied vocab projection, no bias)
# ---------------------------------------------------------------------------
def lm_head_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Final vocab projection: Linear(hidden -> vocab), no bias, untied from embed_tokens.

    x: [batch, seq, hidden]; weight: [vocab, hidden]. Returns logits [batch, seq, vocab].
    """
    return F.linear(x, weight)
