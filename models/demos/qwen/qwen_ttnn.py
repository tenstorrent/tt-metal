"""
Qwen2.5-Coder Bring-Up for Tenstorrent TTNN
Full decoder-only transformer implementation with Grouped Query Attention (GQA),
RoPE, SwiGLU MLP, and RMSNorm mapped to TTNN operations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class Qwen2_5CoderConfig:
    vocab_size: int = 151936
    hidden_size: int = 896          # Qwen2.5-0.5B default
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2     # GQA: 14 query heads, 2 KV heads (group size = 7)
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

class TTNN_RMSNorm(nn.Module):
    """
    TTNN-compatible Root Mean Square Normalization.
    Equivalent to ttnn.rms_norm / (x * rsqrt(mean(x^2) + eps)) * weight.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class TTNN_RotaryEmbedding(nn.Module):
    """
    TTNN Rotary Position Embedding (RoPE) for Qwen2.5-Coder.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape cos, sin for broadcasting: [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class TTNN_Qwen2_5MLP(nn.Module):
    """
    TTNN SwiGLU Feed-Forward Network:
    gate = silu(linear(x, gate_proj))
    up = linear(x, up_proj)
    out = linear(gate * up, down_proj)
    """
    def __init__(self, config: Qwen2_5CoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class TTNN_Qwen2_5Attention(nn.Module):
    """
    TTNN Grouped Query Attention (GQA) for Qwen2.5-Coder.
    Supports key-value head broadcasting and causal masking.
    """
    def __init__(self, config: Qwen2_5CoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = TTNN_RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV heads for GQA (Grouped Query Attention)
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

class TTNN_Qwen2_5DecoderLayer(nn.Module):
    """
    Single Decoder Transformer Block:
    x = x + Attention(RMSNorm(x))
    x = x + MLP(RMSNorm(x))
    """
    def __init__(self, config: Qwen2_5CoderConfig):
        super().__init__()
        self.input_layernorm = TTNN_RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TTNN_Qwen2_5Attention(config)
        self.post_attention_layernorm = TTNN_RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TTNN_Qwen2_5MLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out

        # Pre-LN MLP
        normed_mlp = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_mlp)
        hidden_states = hidden_states + mlp_out

        return hidden_states

class TTNN_Qwen2_5Model(nn.Module):
    """
    TTNN Qwen2.5-Coder Base Backbone.
    """
    def __init__(self, config: Qwen2_5CoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TTNN_Qwen2_5DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = TTNN_RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape

        if attention_mask is None and seq_len > 1:
            # Create causal mask: lower triangular matrix
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1)
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class TTNN_Qwen2_5ForCausalLM(nn.Module):
    """
    TTNN Qwen2.5-Coder for Causal Language Modeling / Code Generation.
    """
    def __init__(self, config: Qwen2_5CoderConfig):
        super().__init__()
        self.config = config
        self.model = TTNN_Qwen2_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
