# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone PyTorch reference implementations for MiniMax-M2.5 blocks.

These functions exactly mirror the HuggingFace modeling_minimax_m2.py forward pass
and are used to generate golden outputs for TTNN verification.

Architecture reference: models/demos/minimax_m2/ARCHITECTURE.md
HuggingFace: https://huggingface.co/MiniMaxAI/MiniMax-M2.5
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config dataclass (mirrors config.json — no HF dependency)
# ---------------------------------------------------------------------------


@dataclass
class MiniMaxM2Config:
    hidden_size: int = 3072
    head_dim: int = 128
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    num_hidden_layers: int = 62
    intermediate_size: int = 1536  # per-expert FFN hidden dim
    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    rotary_dim: int = 64  # partial RoPE: only first 64 head dims
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 200_064
    max_position_embeddings: int = 196_608
    use_qk_norm: bool = True
    use_routing_bias: bool = True
    attention_dropout: float = 0.0
    router_jitter_noise: float = 0.0

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


# ---------------------------------------------------------------------------
# 1. RMSNorm
# ---------------------------------------------------------------------------


def rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm — identical to MiniMaxM2RMSNorm.

    Args:
        x:      [..., hidden_size]
        weight: [hidden_size]
    """
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_norm.to(x.dtype)


# ---------------------------------------------------------------------------
# 2. RoPE (partial — rotary_dim=64 of head_dim=128)
# ---------------------------------------------------------------------------


def build_rope_cache(
    seq_len: int,
    rotary_dim: int,
    rope_theta: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin cache for partial RoPE.

    Returns:
        cos, sin: [seq_len, rotary_dim]  (NOT full head_dim)
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [seq_len, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, rotary_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to only the first rotary_dim dims; pass the rest through.

    Args:
        q:   [batch, num_q_heads,  seq_len, head_dim]
        k:   [batch, num_kv_heads, seq_len, head_dim]
        cos: [seq_len, rotary_dim]   (rotary_dim = 64)
        sin: [seq_len, rotary_dim]
    """
    rotary_dim = cos.shape[-1]

    # Unsqueeze for broadcast: [1, 1, seq_len, rotary_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split: rotate first rotary_dim, passthrough the rest
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin

    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


# ---------------------------------------------------------------------------
# 3. GQA Attention with QK-norm + partial RoPE
# ---------------------------------------------------------------------------


def attention_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: MiniMaxM2Config,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MiniMaxM2Attention forward pass.

    Args:
        x:          [batch, seq_len, hidden_size]
        state_dict: keys: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm (all .weight)
        cos, sin:   [seq_len, rotary_dim]  from build_rope_cache()
        attention_mask: [batch, 1, seq_len, seq_len] or None

    Returns:
        [batch, seq_len, hidden_size]
    """
    B, S, H = x.shape
    num_q = config.num_attention_heads
    num_kv = config.num_key_value_heads
    head_dim = config.head_dim
    groups = config.num_key_value_groups

    # QKV projections
    q = F.linear(x, state_dict["q_proj.weight"])  # [B, S, num_q * head_dim]
    k = F.linear(x, state_dict["k_proj.weight"])  # [B, S, num_kv * head_dim]
    v = F.linear(x, state_dict["v_proj.weight"])  # [B, S, num_kv * head_dim]

    # QK-norm (per_layer): applied to flattened Q/K BEFORE reshape
    # — mirrors HF: q_norm is RMSNorm(num_q * head_dim), k_norm is RMSNorm(num_kv * head_dim)
    if config.use_qk_norm:
        q = rmsnorm_forward(q, state_dict["q_norm.weight"], config.rms_norm_eps)
        k = rmsnorm_forward(k, state_dict["k_norm.weight"], config.rms_norm_eps)

    # Reshape to [B, heads, S, head_dim]
    q = q.view(B, S, num_q, head_dim).transpose(1, 2)
    k = k.view(B, S, num_kv, head_dim).transpose(1, 2)
    v = v.view(B, S, num_kv, head_dim).transpose(1, 2)

    # Partial RoPE (rotary_dim=64)
    q, k = apply_partial_rope(q, k, cos, sin)

    # GQA: repeat K, V to match Q head count
    if groups > 1:
        k = k.unsqueeze(2).expand(B, num_kv, groups, S, head_dim).reshape(B, num_q, S, head_dim)
        v = v.unsqueeze(2).expand(B, num_kv, groups, S, head_dim).reshape(B, num_q, S, head_dim)

    # Scaled dot-product attention
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_q, S, S]
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_out = torch.matmul(attn_weights, v)  # [B, num_q, S, head_dim]
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, num_q * head_dim)

    return F.linear(attn_out, state_dict["o_proj.weight"])


# ---------------------------------------------------------------------------
# 4. MoE: Router + Experts (SwiGLU)
# ---------------------------------------------------------------------------


def expert_mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    """Single expert SwiGLU MLP.

    Args:
        x:  [..., hidden_size]
        w1: [intermediate_size, hidden_size]  gate proj
        w2: [hidden_size, intermediate_size]  down proj
        w3: [intermediate_size, hidden_size]  up proj
    """
    gate = F.silu(F.linear(x, w1))
    up = F.linear(x, w3)
    return F.linear(gate * up, w2)


def moe_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    config: MiniMaxM2Config,
) -> torch.Tensor:
    """MiniMaxM2SparseMoeBlock forward pass.

    Args:
        x:          [batch, seq_len, hidden_size]
        state_dict: keys:
            gate.weight            [num_experts, hidden_size]
            e_score_correction_bias [num_experts]
            experts.{j}.w1.weight  [intermediate_size, hidden_size]  for j in 0..num_experts-1
            experts.{j}.w2.weight  [hidden_size, intermediate_size]
            experts.{j}.w3.weight  [intermediate_size, hidden_size]

    Returns:
        [batch, seq_len, hidden_size]
    """
    B, S, H = x.shape
    T = B * S
    top_k = config.num_experts_per_tok

    hidden = x.view(T, H)

    # Router: sigmoid + routing_bias + top-k + normalize
    router_logits = F.linear(hidden, state_dict["gate.weight"])  # [T, num_experts]
    routing_weights = torch.sigmoid(router_logits.float())

    if config.use_routing_bias:
        scores = routing_weights + state_dict["e_score_correction_bias"].float()
    else:
        scores = routing_weights

    _, top_k_index = torch.topk(scores, top_k, dim=-1, sorted=False)  # [T, top_k]
    top_k_weights = routing_weights.gather(1, top_k_index)  # [T, top_k]
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights.to(hidden.dtype)

    # Dispatch tokens to selected experts
    final_out = torch.zeros_like(hidden)
    expert_mask = F.one_hot(top_k_index, num_classes=config.num_local_experts).permute(2, 1, 0)
    # expert_mask: [num_experts, top_k, T]

    expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=False).squeeze(-1)
    for expert_idx in expert_hit:
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        # idx: which of the top_k slots, top_x: which tokens
        tokens_for_expert = hidden[top_x]  # [n, H]
        w1 = state_dict[f"experts.{expert_idx.item()}.w1.weight"]
        w2 = state_dict[f"experts.{expert_idx.item()}.w2.weight"]
        w3 = state_dict[f"experts.{expert_idx.item()}.w3.weight"]
        expert_out = expert_mlp_forward(tokens_for_expert, w1, w2, w3)  # [n, H]
        expert_out = expert_out * top_k_weights[top_x, idx, None]
        final_out.index_add_(0, top_x, expert_out.to(final_out.dtype))

    return final_out.view(B, S, H)


# ---------------------------------------------------------------------------
# 5. Decoder Layer
# ---------------------------------------------------------------------------


def decoder_layer_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: MiniMaxM2Config,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MiniMaxM2DecoderLayer — pre-norm, attention, residual, pre-norm, MoE, residual.

    Args:
        x:          [batch, seq_len, hidden_size]
        state_dict: all weights for this layer, keys prefixed with the sub-module name
            input_layernorm.weight
            self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight
            post_attention_layernorm.weight
            block_sparse_moe.{gate.weight, e_score_correction_bias, experts.*}
    """
    # --- Attention sub-layer ---
    residual = x
    x = rmsnorm_forward(x, state_dict["input_layernorm.weight"], config.rms_norm_eps)

    attn_sd = {k.removeprefix("self_attn."): v for k, v in state_dict.items() if k.startswith("self_attn.")}
    x = attention_forward(x, attn_sd, cos, sin, config, attention_mask)
    x = residual + x

    # --- MoE sub-layer ---
    residual = x
    x = rmsnorm_forward(x, state_dict["post_attention_layernorm.weight"], config.rms_norm_eps)

    moe_sd = {
        k.removeprefix("block_sparse_moe."): v for k, v in state_dict.items() if k.startswith("block_sparse_moe.")
    }
    x, _ = moe_forward(x, moe_sd, config), None
    x = residual + x

    return x


def decoder_layer_forward_with_moe_logits(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: MiniMaxM2Config,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as decoder_layer_forward but also returns router logits for debugging."""
    residual = x
    x = rmsnorm_forward(x, state_dict["input_layernorm.weight"], config.rms_norm_eps)
    attn_sd = {k.removeprefix("self_attn."): v for k, v in state_dict.items() if k.startswith("self_attn.")}
    x = attention_forward(x, attn_sd, cos, sin, config, attention_mask)
    x = residual + x

    residual = x
    x = rmsnorm_forward(x, state_dict["post_attention_layernorm.weight"], config.rms_norm_eps)
    moe_sd = {
        k.removeprefix("block_sparse_moe."): v for k, v in state_dict.items() if k.startswith("block_sparse_moe.")
    }

    B, S, H = x.shape
    T = B * S
    hidden = x.view(T, H)
    router_logits = F.linear(hidden, moe_sd["gate.weight"])  # capture router logits

    x = moe_forward(x, moe_sd, config)
    x = residual + x
    return x, router_logits


# ---------------------------------------------------------------------------
# 6. Full Model Forward
# ---------------------------------------------------------------------------


def build_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask: [1, 1, seq_len, seq_len]."""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def model_forward(
    input_ids: torch.Tensor,
    model_state_dict: Dict[str, torch.Tensor],
    config: MiniMaxM2Config,
) -> torch.Tensor:
    """Full MiniMaxM2Model forward pass (no KV-cache).

    Args:
        input_ids:        [batch, seq_len]
        model_state_dict: full state_dict with keys like
            model.embed_tokens.weight
            model.layers.{i}.{...}
            model.norm.weight
            lm_head.weight

    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    B, S = input_ids.shape
    device = input_ids.device

    # Embedding
    x = F.embedding(input_ids, model_state_dict["model.embed_tokens.weight"])

    # Build RoPE cache
    cos, sin = build_rope_cache(S, config.rotary_dim, config.rope_theta, dtype=x.dtype, device=device)

    # Causal mask
    causal_mask = build_causal_mask(S, x.dtype, device)

    # Decoder layers
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        layer_sd = {k.removeprefix(prefix): v for k, v in model_state_dict.items() if k.startswith(prefix)}
        x = decoder_layer_forward(x, layer_sd, cos, sin, config, causal_mask)

    # Final norm + LM head
    x = rmsnorm_forward(x, model_state_dict["model.norm.weight"], config.rms_norm_eps)
    logits = F.linear(x, model_state_dict["lm_head.weight"])

    return logits


# ---------------------------------------------------------------------------
# 7. Random-weight model factory (for testing without real checkpoint)
# ---------------------------------------------------------------------------


def make_random_state_dict(
    config: MiniMaxM2Config,
    num_layers: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Create a random state_dict matching MiniMax-M2.5 weight shapes.

    Args:
        config:     MiniMaxM2Config instance
        num_layers: Override num_hidden_layers (useful for fast tests)
        dtype:      Weight dtype
        seed:       Random seed for reproducibility

    Returns:
        state_dict with ALL weight tensors for `model_forward()`
    """
    torch.manual_seed(seed)
    n_layers = num_layers if num_layers is not None else config.num_hidden_layers
    H = config.hidden_size
    D = config.head_dim
    NQ = config.num_attention_heads
    NK = config.num_key_value_heads
    E = config.num_local_experts
    FF = config.intermediate_size
    V = config.vocab_size

    sd = {}

    def _w(*shape):
        return torch.randn(*shape, dtype=dtype) * 0.02

    # Embedding + LM head
    sd["model.embed_tokens.weight"] = _w(V, H)
    sd["lm_head.weight"] = _w(V, H)
    sd["model.norm.weight"] = torch.ones(H, dtype=dtype)

    for i in range(n_layers):
        p = f"model.layers.{i}."

        # Norms
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=dtype)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=dtype)

        # Attention
        sd[p + "self_attn.q_proj.weight"] = _w(NQ * D, H)
        sd[p + "self_attn.k_proj.weight"] = _w(NK * D, H)
        sd[p + "self_attn.v_proj.weight"] = _w(NK * D, H)
        sd[p + "self_attn.o_proj.weight"] = _w(H, NQ * D)
        sd[p + "self_attn.q_norm.weight"] = torch.ones(NQ * D, dtype=dtype)
        sd[p + "self_attn.k_norm.weight"] = torch.ones(NK * D, dtype=dtype)

        # MoE router
        sd[p + "block_sparse_moe.gate.weight"] = _w(E, H)
        sd[p + "block_sparse_moe.e_score_correction_bias"] = torch.zeros(E, dtype=dtype)

        # Experts
        for j in range(E):
            ep = p + f"block_sparse_moe.experts.{j}."
            sd[ep + "w1.weight"] = _w(FF, H)
            sd[ep + "w2.weight"] = _w(H, FF)
            sd[ep + "w3.weight"] = _w(FF, H)

    return sd
