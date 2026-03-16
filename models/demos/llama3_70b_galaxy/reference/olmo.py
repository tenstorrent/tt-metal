# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Reference Implementation.

Module-based implementation for verification against HuggingFace and TTNN.
Key differences from Qwen3-32B:
- 40 Q heads (not 64)
- QK-norm (per-head RMSNorm applied after QKV projection, before RoPE)
- YaRN RoPE (not linear scaling)
- Hybrid sliding window attention (3 sliding + 1 full)
- Post-sublayer norm (norm applied after each sublayer, before residual add)
- intermediate_size=27648 (not 25600)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class OlmoModelArgs:
    """OLMo-3.1-32B model arguments."""

    dim: int = 5120
    n_layers: int = 64
    n_heads: int = 40  # Query heads
    n_kv_heads: int = 8  # KV heads (GQA ratio 5:1)
    vocab_size: int = 100278
    intermediate_size: int = 27648
    norm_eps: float = 1e-6
    head_dim: int = 128

    # RoPE config (YaRN)
    rope_theta: float = 500000.0
    rope_type: str = "yarn"
    rope_scaling_factor: float = 8.0
    original_max_position_embeddings: int = 8192
    attention_factor: float = 1.2079441541679836
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    # Sliding window
    sliding_window: int = 4096

    # Runtime
    max_batch_size: int = 32
    max_seq_len: int = 65536

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Compute derived values
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads  # 40 / 8 = 5

    def get_layer_type(self, layer_id: int) -> str:
        """Get attention type for a layer (sliding or full)."""
        # Pattern: 3 sliding + 1 full, repeated
        if (layer_id + 1) % 4 == 0:
            return "full_attention"
        return "sliding_attention"

    def get_sliding_window(self, layer_id: int) -> Optional[int]:
        """Get sliding window size for a layer (None for full attention)."""
        if self.get_layer_type(layer_id) == "sliding_attention":
            return self.sliding_window
        return None


class RMSNorm(nn.Module):
    """RMSNorm layer."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def yarn_find_correction_dim(num_rotations, dim, base=10000.0, max_pos=2048):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000.0, max_pos=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def precompute_yarn_freqs_cis(args: OlmoModelArgs) -> Tuple[torch.Tensor, float]:
    """Compute YaRN RoPE frequencies for OLMo."""
    dim = args.head_dim
    end = args.max_seq_len * 2
    base = args.rope_theta

    # Base frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN interpolation
    low, high = yarn_find_correction_range(
        args.beta_fast, args.beta_slow, dim, base, args.original_max_position_embeddings
    )
    inv_freq_mask = 1 - yarn_linear_ramp_mask(low, high, dim // 2)

    # Apply scaling: blend unscaled (high-freq) with scaled (low-freq) per YaRN paper
    inv_freq_scaled = inv_freq / args.rope_scaling_factor
    inv_freq = inv_freq * inv_freq_mask + inv_freq_scaled * (1 - inv_freq_mask)

    # Compute frequencies
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis, args.attention_factor


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500000.0,
    use_yarn: bool = True,
    args: Optional[OlmoModelArgs] = None,
) -> torch.Tensor:
    """Precompute RoPE frequencies."""
    if use_yarn and args is not None:
        freqs_cis, _ = precompute_yarn_freqs_cis(args)
        return freqs_cis

    # Standard RoPE fallback
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rotate_half(x):
    """Rotates half the hidden dims of the input (HF/neox-style).

    OLMo was trained with HF's rotate_half where element pairs (i, i+dim//2)
    form rotation groups, NOT adjacent pairs (2i, 2i+1) like GPT-J/Meta style.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings using rotate_half (HF/neox-style).

    freqs_cis: complex tensor [seq_len, head_dim//2] from precompute_yarn_freqs_cis.
    xq: [batch, seq, n_heads, head_dim], xk: [batch, seq, n_kv_heads, head_dim].
    """
    cos_half = freqs_cis.real.float()  # [seq, dim//2]
    sin_half = freqs_cis.imag.float()  # [seq, dim//2]

    cos = torch.cat([cos_half, cos_half], dim=-1)  # [seq, dim]
    sin = torch.cat([sin_half, sin_half], dim=-1)  # [seq, dim]

    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)

    xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
    xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    return xq_out, xk_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA."""
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """OLMo attention with GQA and optional sliding window."""

    def __init__(self, args: OlmoModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_dim = args.head_dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = args.n_heads // args.n_kv_heads

        # Get sliding window for this layer
        self.sliding_window = args.get_sliding_window(layer_id)

        # Projections (no bias in OLMo)
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        # QK-norm: global RMSNorm over the full Q/K vectors (matching HF Olmo3Attention)
        # q_norm_weight: [n_heads * head_dim] = [5120]
        # k_norm_weight: [n_kv_heads * head_dim] = [1024]
        self.q_norm_weight = nn.Parameter(torch.ones(args.n_heads * args.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(args.n_kv_heads * args.head_dim))

        # KV cache
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.head_dim))

        # YaRN mscale
        self.mscale = args.attention_factor

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape

        # QKV projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # xq: [bsz, seqlen, n_heads*head_dim], xk: [bsz, seqlen, n_kv_heads*head_dim]

        # QK-norm: global RMSNorm over the full Q/K flat vectors (matching HF Olmo3Attention)
        # Applied BEFORE head reshape, normalizing all heads together
        def _global_rms_norm(t, weight):
            t_f = t.float()
            rms = torch.rsqrt(t_f.pow(2).mean(-1, keepdim=True) + 1e-6)
            return (t_f * rms * weight.to(t_f.device)).type_as(t)

        xq = _global_rms_norm(xq, self.q_norm_weight)  # [bsz, seqlen, n_heads*head_dim]
        xk = _global_rms_norm(xk, self.k_norm_weight)  # [bsz, seqlen, n_kv_heads*head_dim]

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Update cache
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # Repeat KV for GQA
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for attention
        xq = xq.transpose(1, 2)  # [bsz, n_heads, seqlen, head_dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention with YaRN mscale
        scale = (self.head_dim**-0.5) * self.mscale
        scores = torch.matmul(xq, keys.transpose(2, 3)) * scale

        # Apply causal mask
        if mask is not None:
            scores = scores + mask

        # Apply sliding window mask if needed
        if self.sliding_window is not None and seqlen > 1:
            cache_len = start_pos
            total_len = cache_len + seqlen
            # Create sliding window mask for cached positions
            positions = torch.arange(total_len, device=scores.device)
            query_positions = torch.arange(cache_len, total_len, device=scores.device)
            distance = query_positions[:, None] - positions[None, :]
            sliding_mask = torch.where(
                distance > self.sliding_window,
                torch.full_like(distance, float("-inf"), dtype=scores.dtype),
                torch.zeros_like(distance, dtype=scores.dtype),
            )
            scores = scores + sliding_mask.unsqueeze(0).unsqueeze(0)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU MLP for OLMo."""

    def __init__(self, args: OlmoModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.intermediate_size, bias=False)  # gate_proj
        self.w2 = nn.Linear(args.intermediate_size, args.dim, bias=False)  # down_proj
        self.w3 = nn.Linear(args.dim, args.intermediate_size, bias=False)  # up_proj

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """OLMo decoder block."""

    def __init__(self, layer_id: int, args: OlmoModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args, layer_id)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # Post-sublayer norm (OLMo3 actual architecture):
        # norm applied AFTER each sublayer, BEFORE the residual add.
        # attention_norm = post_attention_layernorm (HF key)
        # ffn_norm       = post_feedforward_layernorm (HF key)
        h = x + self.attention_norm(self.attention(x, start_pos, freqs_cis, mask))
        out = h + self.ffn_norm(self.feed_forward(h))
        return out


class Transformer(nn.Module):
    """OLMo-3.1-32B Transformer model."""

    def __init__(self, params: OlmoModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Precompute YaRN RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            params.head_dim,
            params.max_seq_len * 2,
            params.rope_theta,
            use_yarn=True,
            args=params,
        )

    @torch.inference_mode()
    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mode: str = "decode",
    ):
        """Forward pass.

        Args:
            embeddings: Token embeddings [batch, seq_len, dim]
            start_pos: Starting position for KV cache
            mode: "decode" or "prefill"

        Returns:
            Logits or hidden states depending on mode
        """
        _bsz, seqlen, _dim = embeddings.shape
        h = embeddings

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # Causal mask
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=h.device), mask]).type_as(h)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        if mode == "decode":
            h = self.norm(h)
            h = self.output(h)
        else:
            assert mode == "prefill"

        return h.float()

    @classmethod
    def from_hf(cls, model_name: str = "allenai/OLMo-3.1-32B-Think"):
        """Load from HuggingFace checkpoint."""
        from transformers import AutoModelForCausalLM, AutoConfig

        # Load HF config
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Create our args
        args = OlmoModelArgs(
            dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=hf_config.num_key_value_heads,
            vocab_size=hf_config.vocab_size,
            intermediate_size=hf_config.intermediate_size,
            norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            sliding_window=hf_config.sliding_window or 4096,
        )

        # Handle YaRN config if present
        if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling:
            rs = hf_config.rope_scaling
            args.rope_scaling_factor = rs.get("factor", 8.0)
            args.original_max_position_embeddings = rs.get("original_max_position_embeddings", 8192)
            args.attention_factor = rs.get("attention_factor", 1.2079)
            args.beta_fast = rs.get("beta_fast", 32.0)
            args.beta_slow = rs.get("beta_slow", 1.0)

        # Create model
        model = cls(args)

        # Load HF weights
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Map weights
        model.tok_embeddings.weight.data = hf_model.model.embed_tokens.weight.data.clone()

        for i, layer in enumerate(model.layers):
            hf_layer = hf_model.model.layers[i]
            # Attention
            layer.attention.wq.weight.data = hf_layer.self_attn.q_proj.weight.data.clone()
            layer.attention.wk.weight.data = hf_layer.self_attn.k_proj.weight.data.clone()
            layer.attention.wv.weight.data = hf_layer.self_attn.v_proj.weight.data.clone()
            layer.attention.wo.weight.data = hf_layer.self_attn.o_proj.weight.data.clone()
            # MLP
            layer.feed_forward.w1.weight.data = hf_layer.mlp.gate_proj.weight.data.clone()
            layer.feed_forward.w2.weight.data = hf_layer.mlp.down_proj.weight.data.clone()
            layer.feed_forward.w3.weight.data = hf_layer.mlp.up_proj.weight.data.clone()
            # Norms — OLMo3 post-sublayer-norm:
            # attention_norm applied AFTER attention = post_attention_layernorm
            # ffn_norm       applied AFTER FFN       = post_feedforward_layernorm
            layer.attention_norm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()
            layer.ffn_norm.weight.data = hf_layer.post_feedforward_layernorm.weight.data.clone()
            # QK-norm: global [n_heads*head_dim] and [n_kv_heads*head_dim] weights
            layer.attention.q_norm_weight.data = hf_layer.self_attn.q_norm.weight.data.clone()
            layer.attention.k_norm_weight.data = hf_layer.self_attn.k_norm.weight.data.clone()

        model.norm.weight.data = hf_model.model.norm.weight.data.clone()
        model.output.weight.data = hf_model.lm_head.weight.data.clone()

        del hf_model
        return model


def print_layer_types(args: OlmoModelArgs):
    """Print the layer type pattern for OLMo."""
    print("OLMo-3.1-32B Layer Types:")
    print("=" * 50)
    sliding_count = 0
    full_count = 0
    for i in range(args.n_layers):
        layer_type = args.get_layer_type(i)
        window = args.get_sliding_window(i)
        if layer_type == "sliding_attention":
            sliding_count += 1
        else:
            full_count += 1
        if i < 8 or i >= args.n_layers - 4:  # Print first 8 and last 4
            print(f"Layer {i:2d}: {layer_type:18s} (window={window})")
        elif i == 8:
            print("...")
    print("=" * 50)
    print(f"Total: {sliding_count} sliding + {full_count} full = {args.n_layers} layers")


if __name__ == "__main__":
    # Print layer types for verification
    args = OlmoModelArgs()
    print_layer_types(args)
