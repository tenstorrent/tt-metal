# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for HunyuanImage3SDPAAttention (GQA + 2D RoPE + QK-norm).
# Extracted verbatim from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     repeat_kv                    lines  128-137
#     HunyuanImage3SDPAAttention   lines 1257-1370
#
# Depends on rms_norm.py and rope_2d.py in this same directory.
# Used as the golden reference for TT-Metal numeric validation.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import HunyuanRMSNorm
from .rope_2d import apply_rotary_pos_emb, build_batch_2d_rope


# ---------------------------------------------------------------------------
# GQA helper
# ---------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads for Grouped Query Attention.
    Input:  [batch, num_key_value_heads, seq_len, head_dim]
    Output: [batch, num_attention_heads,  seq_len, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Minimal config container (mirrors the fields used by the attention module)
# ---------------------------------------------------------------------------


class AttentionConfig:
    """
    Minimal stand-in for HunyuanImage3Config — only the fields needed by
    HunyuanImage3SDPAAttention.

    Default values match the HunyuanImage-3.0 7B/14B architecture:
        hidden_size          = 4096
        num_attention_heads  = 32
        attention_head_dim   = 128
        num_key_value_heads  = 8    (GQA: 4 query groups per KV head)
        use_qk_norm          = True
        use_rotary_pos_emb   = True
        rms_norm_eps         = 1e-6
        attention_bias       = False
        attention_dropout    = 0.0
        max_position_embeddings = 2048
        rope_theta           = 10000.0
        rope_scaling         = {"type": "custom"}
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_key_value_heads: int = 8,
        use_qk_norm: bool = True,
        use_rotary_pos_emb: bool = True,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_key_value_heads = num_key_value_heads
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = {"type": "custom"}


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------


class HunyuanImage3SDPAAttention(nn.Module):
    """
    Hunyuan SDPA attention with:
      - Fused QKV projection
      - Grouped Query Attention (GQA) via repeat_kv
      - 2D Rotary Position Embeddings passed in as (cos, sin) pair
      - Optional per-head QK LayerNorm (use_qk_norm=True)
      - torch.nn.functional.scaled_dot_product_attention backend

    Forward args:
        hidden_states:  [B, S, hidden_size]
        attention_mask: [B, 1, S, S] float mask (0=attend, -inf=mask) or None
        custom_pos_emb: (cos, sin) tuple from build_batch_2d_rope
                        cos/sin shape: [B, S, head_dim]

    Returns:
        attn_output:    [B, S, hidden_size]
        None            (no attention weights — SDPA doesn't expose them)
        None            (no KV-cache update in this reference)

    Notes for TT-Metal port:
    - QKV split: use ttnn.experimental.nlp_create_qkv_heads() — handles GQA
      expansion internally; pass num_heads and num_kv_heads.
    - RoPE:      use ttnn.experimental.rotary_embedding_llama() with
      transformation matrix from get_rot_transformation_mat().
    - QK-norm:   apply ttnn.rms_norm per-head after RoPE; weight shape must
      be [1, 1, head_dim//32, 32] for tile alignment.
    - SDPA:      ttnn.transformer.scaled_dot_product_attention() with
      is_causal=True for causal text; is_causal=False + explicit mask for
      the mixed causal/bidirectional Hunyuan mask.
    - Concat:    ttnn.experimental.nlp_concat_heads()
    - o_proj:    ttnn.linear()
    """

    def __init__(self, config: AttentionConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads

        self.use_qk_norm = config.use_qk_norm
        self.use_rotary_pos_emb = config.use_rotary_pos_emb

        # Fused QKV projection: [hidden] -> [Q_dim + 2*KV_dim]
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size_q + 2 * self.hidden_size_kv,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(self.hidden_size_q, self.hidden_size, bias=config.attention_bias)

        # Per-head QK normalisation (applied after RoPE)
        if self.use_qk_norm:
            self.query_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        custom_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None]:
        bsz, q_len, _ = hidden_states.size()

        # ---- 1. Fused QKV projection ----------------------------------------
        qkv_states = self.qkv_proj(hidden_states)

        # Reshape to expose head structure, then split Q / K / V
        # [B, S, kv_heads, (q_groups + 1 + 1), head_dim]
        qkv_states = qkv_states.reshape(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.num_key_value_groups + 2,
            self.head_dim,
        )
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)
        # -> [B, num_heads,    S, head_dim]
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # -> [B, num_kv_heads, S, head_dim]
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ---- 2. 2D RoPE -------------------------------------------------------
        if self.use_rotary_pos_emb and custom_pos_emb is not None:
            cos, sin = custom_pos_emb  # [B, S, head_dim]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ---- 3. Per-head QK normalisation (applied after RoPE) ---------------
        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        # ---- 4. GQA: expand KV heads to match Q heads -----------------------
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ---- 5. Scaled dot-product attention ---------------------------------
        # CUDA contiguity required for custom attn_mask with SDPA backend
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask if not kwargs.get("is_causal") else None,
            dropout_p=0.0,
            is_causal=bool(kwargs.get("is_causal")),
        )

        # ---- 6. Merge heads and output projection ----------------------------
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None


# ---------------------------------------------------------------------------
# Causal / mixed mask builder
# ---------------------------------------------------------------------------


def build_causal_mask(seq_len: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Standard lower-triangular causal mask.
    Returns [1, 1, seq_len, seq_len] with 0 for attend, -inf for mask.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).to(dtype)


def build_hunyuan_mixed_mask(
    seq_len: int,
    text_len: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Hunyuan mixed attention mask:
      - Text tokens (0 .. text_len-1): causal (attend only to earlier tokens)
      - Image tokens (text_len .. seq_len-1): bidirectional among themselves,
        but can also attend to all text tokens.

    Layout (0 = attend, -inf = block):
        [  causal_text  |  text→image=-inf  ]
        [  img→text=0   |  img_bidirectional ]

    Returns [1, 1, seq_len, seq_len].
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    # Text block: causal
    mask[:text_len, :text_len] = torch.triu(torch.full((text_len, text_len), float("-inf"), dtype=dtype), diagonal=1)
    # Image→text: attend to all text (already 0 from above if we zero it)
    mask[text_len:, :text_len] = 0.0
    # Image→image: bidirectional (all zeros)
    mask[text_len:, text_len:] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Quick numeric smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(7)
    B, S = 1, 256
    cfg = AttentionConfig(
        hidden_size=4096,
        num_attention_heads=32,
        attention_head_dim=128,
        num_key_value_heads=8,
        use_qk_norm=True,
        use_rotary_pos_emb=True,
    )
    model = HunyuanImage3SDPAAttention(cfg).eval()

    x = torch.randn(B, S, cfg.hidden_size, dtype=torch.bfloat16)

    # Build 2D RoPE (text-only for this smoke test)
    cos, sin = build_batch_2d_rope(seq_len=S, n_elem=cfg.attention_head_dim)
    # cos/sin: [B, S, head_dim] — pass full batched tensors; apply_rotary_pos_emb
    # unsqueezes dim-1 to get [B, 1, S, head_dim] for broadcasting over heads.
    custom_pos_emb = (cos, sin)

    # Causal mask
    mask = build_causal_mask(S, dtype=torch.float32)

    model = model.to(torch.bfloat16)
    with torch.no_grad():
        out, _, _ = model(x, attention_mask=mask, custom_pos_emb=custom_pos_emb)

    print(f"input  shape : {x.shape}")
    print(f"output shape : {out.shape}")
    print(f"output mean  : {out.float().mean():.6f}  std: {out.float().std():.6f}")
    print("smoke-test passed")
