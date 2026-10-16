# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pure PyTorch reference implementation of the Llama 3.1 8B decoder.

Used for golden comparison against the Tenstorrent device implementation.
All computation is done in torch.bfloat16.

RoPE uses the standard HuggingFace half-split convention:
    x1, x2 = x[..., :D//2], x[..., D//2:]
    rotate_half(x) = [-x2, x1]
    x_rotated = x * cos + rotate_half(x) * sin

This matches the golden trace stored at
/mnt/models/meta-llama/Llama-3.1-8B-Instruct/golden/synthetic_10240/kv_cache/.
The device (ttnn) uses the Meta interleaved convention; the PCC check code
permutes golden K from HF to Meta for comparison.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.llama_3_1_8b.config import Llama31Config

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class LlamaRMSNorm(nn.Module):
    """Standard RMSNorm (no Gemma +1 fold)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (Llama3-style)
# ---------------------------------------------------------------------------


class LlamaRotaryEmbedding(nn.Module):
    """Llama 3.1 RoPE with llama3-style frequency scaling.

    Follows the HuggingFace transformers implementation exactly
    (transformers.modeling_rope_utils._compute_llama3_parameters).

    The llama3 scaling adjusts the base inv_freq as follows:
        - Compute wavelengths: wavelen = 2*pi / inv_freq
        - wavelen < high_freq_wavelen  => keep inv_freq unchanged
        - wavelen > low_freq_wavelen   => divide inv_freq by factor (interpolate)
        - in between                   => smooth transition

    where:
        low_freq_wavelen  = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
    """

    inv_freq: torch.Tensor

    def __init__(self, config: Llama31Config, device: torch.device | None = None):
        super().__init__()
        self.config = config
        dim = config.head_dim

        base = config.rope_theta
        factor = config.rope_scaling.factor
        low_freq_factor = config.rope_scaling.low_freq_factor
        high_freq_factor = config.rope_scaling.high_freq_factor
        old_context_len = config.rope_scaling.original_max_position_embeddings

        # Base inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )

        # Llama3 scaling
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: keep unchanged
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # Smooth interpolation for medium frequencies
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        self.register_buffer("inv_freq", inv_freq_llama, persistent=False)
        # attention_scaling is 1.0 for llama3 type (unused but kept for completeness)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for the given positions.

        Args:
            x: Input tensor, used only for dtype/device.
               Shape: [batch_size, seq_len, hidden_size]
            position_ids: Position indices.
               Shape: [batch_size, seq_len]

        Returns:
            (cos, sin) each of shape [batch_size, seq_len, head_dim]
        """
        # inv_freq: [dim//2]
        # Expand for batch matmul: [batch, dim//2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Compute frequencies: [batch, dim//2, seq_len] -> transpose -> [batch, seq_len, dim//2]
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # Double to full head_dim: [batch, seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# RoPE application (HF half-split convention)
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (HF convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K tensors.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor   [batch, n_kv_heads, seq_len, head_dim]
        cos: Cosine of position embedding [batch, seq_len, head_dim]
        sin: Sine of position embedding   [batch, seq_len, head_dim]
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting (default 1 for heads dim)

    Returns:
        (q_rotated, k_rotated)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# GQA repeat helper
# ---------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    Input:  [batch, num_kv_heads, seq_len, head_dim]
    Output: [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class LlamaAttention(nn.Module):
    """Multi-headed GQA attention with RoPE on Q and K.

    Uses torch.nn.functional.scaled_dot_product_attention with is_causal=True.
    No bias on any projections.
    """

    def __init__(self, config: Llama31Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) each [batch, seq_len, head_dim]

        Returns:
            (output, k_post_rope, v) where:
                output: [batch, seq_len, hidden_size]
                k_post_rope: [batch, num_kv_heads, seq_len, head_dim] (post-RoPE keys)
                v: [batch, num_kv_heads, seq_len, head_dim]
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Save post-RoPE K for golden comparison before GQA expansion
        k_post_rope = key_states.clone()

        # GQA: expand K, V to match Q head count
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states_expanded = repeat_kv(value_states, self.num_kv_groups)

        # Scaled dot-product attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states_expanded,
            is_causal=True,
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, k_post_rope, value_states


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class LlamaMLP(nn.Module):
    """Llama MLP: gate_proj with SiLU, up_proj, down_proj. No bias."""

    def __init__(self, config: Llama31Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer: norm -> attention -> residual -> norm -> MLP -> residual."""

    def __init__(self, config: Llama31Config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (hidden_states, k_post_rope, v)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, k_post_rope, v = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, k_post_rope, v


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class LlamaModel(nn.Module):
    """Llama 3.1 model: embed_tokens -> decoder_layers -> final norm."""

    def __init__(self, config: Llama31Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        num_layers: int | None = None,
    ) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            num_layers: If set, only run the first N decoder layers.

        Returns:
            (hidden_states, kv_cache) where:
                hidden_states: [batch, seq_len, hidden_size]
                kv_cache: {layer_idx: {"k": k_tensor, "v": v_tensor}}
                    k: [batch, num_kv_heads, seq_len, head_dim] (post-RoPE)
                    v: [batch, num_kv_heads, seq_len, head_dim]
        """
        hidden_states = self.embed_tokens(input_ids)

        # Compute position embeddings once
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        n_layers = num_layers if num_layers is not None else self.config.num_hidden_layers
        kv_cache: dict[int, dict[str, torch.Tensor]] = {}

        for layer_idx in range(n_layers):
            hidden_states, k_post_rope, v = self.layers[layer_idx](hidden_states, position_embeddings)
            kv_cache[layer_idx] = {"k": k_post_rope, "v": v}

        hidden_states = self.norm(hidden_states)
        return hidden_states, kv_cache


class LlamaForCausalLM(nn.Module):
    """Llama 3.1 causal language model: model + lm_head."""

    def __init__(self, config: Llama31Config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        num_layers: int | None = None,
    ) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            num_layers: If set, only run the first N decoder layers.

        Returns:
            (logits, kv_cache) where:
                logits: [batch, seq_len, vocab_size]
                kv_cache: {layer_idx: {"k": k_tensor, "v": v_tensor}}
        """
        hidden_states, kv_cache = self.model(input_ids, num_layers=num_layers)
        logits = self.lm_head(hidden_states)
        return logits, kv_cache


# ---------------------------------------------------------------------------
# Weight loading from HF checkpoint
# ---------------------------------------------------------------------------


def _map_hf_state_dict(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HuggingFace state_dict keys to our reference model keys.

    HF uses:
        model.embed_tokens.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight

    Our model uses the same key structure so the mapping is identity for
    LlamaForCausalLM, but we map the inner LlamaModel keys too.
    """
    mapped = {}
    for key, value in hf_state_dict.items():
        mapped[key] = value
    return mapped


def load_reference_from_hf(model_path: str | Path, num_layers: int | None = None) -> LlamaForCausalLM:
    """Load the HF checkpoint into the reference model, converting weights to bf16.

    Args:
        model_path: Path to the HuggingFace model directory
            (e.g. /mnt/models/meta-llama/Llama-3.1-8B-Instruct).
        num_layers: If set, only load this many decoder layers (for faster testing).

    Returns:
        LlamaForCausalLM with loaded weights in bfloat16.
    """
    import json

    model_path = Path(model_path)

    # Load HF config
    with open(model_path / "config.json") as f:
        hf_config_dict = json.load(f)

    config = Llama31Config.from_hf_config(hf_config_dict)
    if num_layers is not None:
        config.num_hidden_layers = num_layers

    # Create model
    model = LlamaForCausalLM(config)

    # Load state dict from safetensors or pytorch bin
    from safetensors import safe_open

    state_dict = {}
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if safetensor_files:
        for sf_path in safetensor_files:
            with safe_open(str(sf_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    else:
        # Fall back to pytorch_model.bin / pytorch_model-*.bin
        bin_files = sorted(model_path.glob("pytorch_model*.bin"))
        for bin_path in bin_files:
            shard = torch.load(bin_path, map_location="cpu", weights_only=True)
            state_dict.update(shard)

    # Map and load
    mapped = _map_hf_state_dict(state_dict)

    # Filter to only the layers we need (if num_layers is set)
    if num_layers is not None:
        filtered = {}
        for key, value in mapped.items():
            # Check if this is a layer key
            if "model.layers." in key:
                # Extract layer index
                parts = key.split(".")
                layer_idx_pos = parts.index("layers") + 1
                layer_idx = int(parts[layer_idx_pos])
                if layer_idx >= num_layers:
                    continue
            filtered[key] = value
        mapped = filtered

    # Load state dict (strict=False to handle potential lm_head tying)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        # If lm_head is missing and tie_word_embeddings is True, tie them
        if config.tie_word_embeddings and all("lm_head" in k for k in missing):
            model.lm_head.weight = model.model.embed_tokens.weight
        else:
            print(f"Warning: missing keys when loading state dict: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading state dict: {unexpected}")

    # Convert to bfloat16
    model = model.to(torch.bfloat16)
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Forward pass with KV extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_reference_forward(
    model: LlamaForCausalLM,
    token_ids: torch.Tensor,
    num_layers: int | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Run the reference forward pass and return per-layer K/V tensors.

    Args:
        model: LlamaForCausalLM reference model (should be in bf16).
        token_ids: Token ID tensor of shape [batch, seq_len] or [seq_len].
            If 1D, a batch dimension is added automatically.
        num_layers: If set, only run the first N layers. Must be <= model's
            configured num_hidden_layers.

    Returns:
        Dict mapping layer_idx to {"k": k_tensor, "v": v_tensor} where:
            k: [1, num_kv_heads, seq_len, head_dim] (post-RoPE, HF convention)
            v: [1, num_kv_heads, seq_len, head_dim]
    """
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)

    token_ids = token_ids.to(next(model.parameters()).device)

    _, kv_cache = model(token_ids, num_layers=num_layers)

    return kv_cache
