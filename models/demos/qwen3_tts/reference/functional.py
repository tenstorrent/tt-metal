# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Functional implementations for Qwen3-TTS modules.
Each function is standalone and takes (x, state_dict/weights, config) as arguments.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

torch.manual_seed(0)


# =============================================================================
# Configuration dataclass
# =============================================================================
class Qwen3TTSConfig:
    """Configuration for Qwen3-TTS Talker model."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 6144,
        text_vocab_size: int = 151936,
        audio_vocab_size: int = 3072,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        num_code_groups: int = 16,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        mrope_section: Tuple[int, int, int] = (24, 20, 20),
        mrope_interleaved: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.num_code_groups = num_code_groups
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved


class Qwen3TTSCodePredictorConfig:
    """Configuration for Qwen3-TTS Code Predictor model."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 3072,
        vocab_size: int = 2048,
        max_position_embeddings: int = 65536,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        num_code_groups: int = 16,
        attention_bias: bool = False,
        hidden_act: str = "silu",
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.num_code_groups = num_code_groups
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act


def get_default_talker_config() -> Qwen3TTSConfig:
    """Returns the default Talker configuration for Qwen3-TTS-12Hz-1.7B."""
    return Qwen3TTSConfig()


def get_default_code_predictor_config() -> Qwen3TTSCodePredictorConfig:
    """Returns the default Code Predictor configuration for Qwen3-TTS-12Hz-1.7B."""
    return Qwen3TTSCodePredictorConfig()


# =============================================================================
# RMSNorm
# =============================================================================
def rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Applies RMSNorm to the input tensor.

    Args:
        hidden_states: Input tensor of shape [..., hidden_size]
        weight: Normalization weight of shape [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor of same shape as input
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================
def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes RoPE frequencies (cos and sin) for standard 1D positions.

    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors of shape [1, max_seq_len, head_dim]
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(position_ids, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0)  # [1, seq_len, head_dim]
    sin = emb.sin().unsqueeze(0)
    return cos, sin


def compute_mrope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes MROPE frequencies for 3D positions (temporal, height, width).

    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors of shape [3, 1, max_seq_len, head_dim]
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # For MROPE, we compute freqs for each dimension
    freqs = torch.outer(position_ids, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    # Expand for 3 dimensions (temporal, height, width)
    cos = emb.cos().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)  # [3, 1, seq_len, head_dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies standard 1D RoPE to query and key tensors.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine frequencies of shape [1, seq_len, head_dim] or [batch, 1, seq_len, head_dim]
        sin: Sine frequencies of shape [1, seq_len, head_dim] or [batch, 1, seq_len, head_dim]

    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Preserve input dtype
    orig_dtype = q.dtype
    # Add head dimension if needed
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)
    # Cast cos/sin to input dtype
    cos = cos.to(orig_dtype)
    sin = sin.to(orig_dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: Tuple[int, int, int] = (24, 20, 20),
    mrope_interleaved: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Multimodal RoPE to query and key tensors.

    The MROPE splits the head dimension into 3 sections for temporal, height, and width.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine frequencies of shape [3, batch, seq_len, head_dim]
        sin: Sine frequencies of shape [3, batch, seq_len, head_dim]
        mrope_section: Tuple of (temporal, height, width) dimensions
        mrope_interleaved: Whether to use interleaved MROPE

    Returns:
        Tuple of rotated (q, k) tensors
    """
    if mrope_interleaved:
        # Interleaved MROPE applies different frequencies to interleaved dimensions
        def apply_interleaved_rope(x, modality_num):
            x_t = x[0].clone()
            index_ranges = []
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                index_ranges.append((beg_idx, end_idx))
            for beg_idx, end_idx in index_ranges:
                x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(1)
        sin = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(1)
    else:
        # Non-interleaved MROPE concatenates sections
        mrope_section_doubled = list(mrope_section) * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section_doubled, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section_doubled, dim=-1))], dim=-1).unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# MLP (SwiGLU)
# =============================================================================
def swiglu_mlp(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Applies SwiGLU MLP block.

    Architecture:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
        gate_proj_weight: Gate projection weight [intermediate_size, hidden_size]
        up_proj_weight: Up projection weight [intermediate_size, hidden_size]
        down_proj_weight: Down projection weight [hidden_size, intermediate_size]

    Returns:
        Output tensor of shape [batch, seq_len, hidden_size]
    """
    gate = F.linear(hidden_states, gate_proj_weight)
    up = F.linear(hidden_states, up_proj_weight)
    hidden_states = F.silu(gate) * up
    hidden_states = F.linear(hidden_states, down_proj_weight)
    return hidden_states


# =============================================================================
# Attention
# =============================================================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands key/value heads to match query heads for GQA.

    Args:
        hidden_states: KV tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each KV head

    Returns:
        Expanded tensor of shape [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def attention(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    rms_norm_eps: float = 1e-6,
    attention_mask: Optional[torch.Tensor] = None,
    use_mrope: bool = False,
    mrope_section: Tuple[int, int, int] = (24, 20, 20),
    mrope_interleaved: bool = True,
) -> torch.Tensor:
    """
    Applies multi-head attention with QK-norm and RoPE.

    Args:
        hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
        q_proj_weight: Query projection weight
        k_proj_weight: Key projection weight
        v_proj_weight: Value projection weight
        o_proj_weight: Output projection weight
        q_norm_weight: Query normalization weight
        k_norm_weight: Key normalization weight
        cos: Cosine frequencies for RoPE
        sin: Sine frequencies for RoPE
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension of each attention head
        rms_norm_eps: Epsilon for RMSNorm
        attention_mask: Optional attention mask
        use_mrope: Whether to use multimodal RoPE
        mrope_section: MROPE section sizes
        mrope_interleaved: Whether MROPE is interleaved

    Returns:
        Output tensor of shape [batch, seq_len, hidden_size]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    # Project Q, K, V
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim)

    # Apply QK-norm (on head_dim dimension)
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)

    # Transpose to [batch, heads, seq_len, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Apply RoPE
    if use_mrope:
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, mrope_section, mrope_interleaved
        )
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Expand KV heads for GQA
    key_states = repeat_kv(key_states, num_kv_groups)
    value_states = repeat_kv(value_states, num_kv_groups)

    # Compute attention scores
    scaling = head_dim**-0.5
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling

    # Apply causal mask (always apply for decoder-only models)
    causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    attn_weights = attn_weights + causal_mask

    # Apply additional attention mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax and dropout
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, value_states)

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, -1)
    attn_output = F.linear(attn_output, o_proj_weight)

    return attn_output


# =============================================================================
# Decoder Layer
# =============================================================================
def decoder_layer(
    hidden_states: torch.Tensor,
    layer_weights: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: Qwen3TTSConfig,
    attention_mask: Optional[torch.Tensor] = None,
    use_mrope: bool = True,
) -> torch.Tensor:
    """
    Applies a single Qwen3-TTS decoder layer.

    Architecture:
        x = x + attention(norm(x))
        x = x + mlp(norm(x))

    Args:
        hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
        layer_weights: Dictionary containing layer weights:
            - input_layernorm.weight
            - self_attn.q_proj.weight
            - self_attn.k_proj.weight
            - self_attn.v_proj.weight
            - self_attn.o_proj.weight
            - self_attn.q_norm.weight
            - self_attn.k_norm.weight
            - post_attention_layernorm.weight
            - mlp.gate_proj.weight
            - mlp.up_proj.weight
            - mlp.down_proj.weight
        cos: Cosine frequencies for RoPE
        sin: Sine frequencies for RoPE
        config: Model configuration
        attention_mask: Optional attention mask
        use_mrope: Whether to use multimodal RoPE

    Returns:
        Output tensor of shape [batch, seq_len, hidden_size]
    """
    residual = hidden_states

    # Pre-norm for attention
    hidden_states = rms_norm(hidden_states, layer_weights["input_layernorm.weight"], config.rms_norm_eps)

    # Self-attention
    hidden_states = attention(
        hidden_states,
        q_proj_weight=layer_weights["self_attn.q_proj.weight"],
        k_proj_weight=layer_weights["self_attn.k_proj.weight"],
        v_proj_weight=layer_weights["self_attn.v_proj.weight"],
        o_proj_weight=layer_weights["self_attn.o_proj.weight"],
        q_norm_weight=layer_weights["self_attn.q_norm.weight"],
        k_norm_weight=layer_weights["self_attn.k_norm.weight"],
        cos=cos,
        sin=sin,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        attention_mask=attention_mask,
        use_mrope=use_mrope,
        mrope_section=config.mrope_section,
        mrope_interleaved=config.mrope_interleaved,
    )

    hidden_states = residual + hidden_states

    # Pre-norm for MLP
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights["post_attention_layernorm.weight"], config.rms_norm_eps)

    # MLP
    hidden_states = swiglu_mlp(
        hidden_states,
        gate_proj_weight=layer_weights["mlp.gate_proj.weight"],
        up_proj_weight=layer_weights["mlp.up_proj.weight"],
        down_proj_weight=layer_weights["mlp.down_proj.weight"],
    )

    hidden_states = residual + hidden_states

    return hidden_states


# =============================================================================
# Full Talker Model
# =============================================================================
def talker_forward(
    input_ids: torch.Tensor,
    weights: dict,
    config: Qwen3TTSConfig,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass through the Qwen3-TTS Talker model.

    Args:
        input_ids: Input token IDs of shape [batch, seq_len]
        weights: Dictionary containing all model weights (after extract_talker_weights)
                 Expected keys:
                 - codec_embedding.weight
                 - text_embedding.weight
                 - layers.{i}.input_layernorm.weight
                 - layers.{i}.self_attn.q_proj.weight, etc.
                 - norm.weight
        config: Model configuration
        position_ids: Optional position IDs of shape [3, batch, seq_len] for MROPE
        attention_mask: Optional attention mask

    Returns:
        Hidden states of shape [batch, seq_len, hidden_size]
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Token embeddings (for audio codec tokens)
    hidden_states = F.embedding(input_ids, weights["codec_embedding.weight"])

    # Compute RoPE frequencies for MROPE
    cos, sin = compute_mrope_frequencies(config.head_dim, seq_len, config.rope_theta, device)
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)

    # Create causal attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
            diagonal=1,
        )
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    # Process through decoder layers
    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"layers.{layer_idx}."
        layer_weights = {k.replace(layer_prefix, ""): v for k, v in weights.items() if k.startswith(layer_prefix)}
        hidden_states = decoder_layer(
            hidden_states,
            layer_weights,
            cos,
            sin,
            config,
            attention_mask=attention_mask,
            use_mrope=True,
        )

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["norm.weight"], config.rms_norm_eps)

    return hidden_states


# =============================================================================
# Code Predictor Model
# =============================================================================
def code_predictor_forward(
    inputs_embeds: torch.Tensor,
    weights: dict,
    config: Qwen3TTSCodePredictorConfig,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass through the Qwen3-TTS Code Predictor model.

    Args:
        inputs_embeds: Input embeddings of shape [batch, seq_len, hidden_size]
        weights: Dictionary containing all model weights
        config: Code predictor configuration
        position_ids: Optional position IDs
        attention_mask: Optional attention mask

    Returns:
        Hidden states of shape [batch, seq_len, hidden_size]
    """
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device

    hidden_states = inputs_embeds

    # Compute standard RoPE frequencies
    cos, sin = compute_rope_frequencies(config.head_dim, seq_len, config.rope_theta, device)
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)

    # Create causal attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
            diagonal=1,
        )
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    # Process through decoder layers (using standard RoPE, not MROPE)
    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"layers.{layer_idx}."
        layer_weights = {k.replace(layer_prefix, ""): v for k, v in weights.items() if k.startswith(layer_prefix)}

        # Code predictor uses standard attention (not MROPE)
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer_weights["input_layernorm.weight"], config.rms_norm_eps)

        hidden_states = attention(
            hidden_states,
            q_proj_weight=layer_weights["self_attn.q_proj.weight"],
            k_proj_weight=layer_weights["self_attn.k_proj.weight"],
            v_proj_weight=layer_weights["self_attn.v_proj.weight"],
            o_proj_weight=layer_weights["self_attn.o_proj.weight"],
            q_norm_weight=layer_weights["self_attn.q_norm.weight"],
            k_norm_weight=layer_weights["self_attn.k_norm.weight"],
            cos=cos,
            sin=sin,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_mask=attention_mask,
            use_mrope=False,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer_weights["post_attention_layernorm.weight"], config.rms_norm_eps)

        hidden_states = swiglu_mlp(
            hidden_states,
            gate_proj_weight=layer_weights["mlp.gate_proj.weight"],
            up_proj_weight=layer_weights["mlp.up_proj.weight"],
            down_proj_weight=layer_weights["mlp.down_proj.weight"],
        )

        hidden_states = residual + hidden_states

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["norm.weight"], config.rms_norm_eps)

    return hidden_states


# =============================================================================
# Utility functions for loading weights
# =============================================================================
def extract_layer_weights(state_dict: dict, layer_idx: int, prefix: str = "talker.model.") -> dict:
    """
    Extracts weights for a specific decoder layer.

    Args:
        state_dict: Full model state dict
        layer_idx: Index of the layer
        prefix: Model prefix (e.g., "talker.model." or "talker.code_predictor.model.")

    Returns:
        Dictionary with layer weights
    """
    layer_prefix = f"{prefix}layers.{layer_idx}."
    return {k.replace(layer_prefix, ""): v for k, v in state_dict.items() if k.startswith(layer_prefix)}


def extract_talker_weights(state_dict: dict) -> dict:
    """
    Extracts Talker model weights from full state dict.

    The HuggingFace Qwen3-TTS model uses the following weight key format:
    - talker.model.layers.{i}.* for decoder layers
    - talker.model.norm.weight for final norm
    - talker.model.codec_embedding.weight for codec embeddings
    - talker.model.text_embedding.weight for text embeddings

    Args:
        state_dict: Full model state dict

    Returns:
        Dictionary with Talker weights (with prefix removed)
    """
    prefix = "talker.model."
    talker_weights = {}
    for k, v in state_dict.items():
        if k.startswith(prefix) and not k.startswith("talker.code_predictor"):
            talker_weights[k.replace(prefix, "")] = v
    return talker_weights


def extract_code_predictor_weights(state_dict: dict) -> dict:
    """
    Extracts Code Predictor model weights from full state dict.

    The HuggingFace Qwen3-TTS model uses the following weight key format:
    - talker.code_predictor.model.layers.{i}.* for decoder layers
    - talker.code_predictor.model.norm.weight for final norm
    - talker.code_predictor.model.codec_embedding.{g}.weight for codec embeddings
    - talker.code_predictor.lm_head.{g}.weight for language model heads

    Args:
        state_dict: Full model state dict

    Returns:
        Dictionary with Code Predictor weights (with prefix removed)
    """
    prefix = "talker.code_predictor."
    cp_weights = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            cp_weights[k.replace(prefix, "")] = v
    return cp_weights


# =============================================================================
# Speech Tokenizer Decoder Configuration
# =============================================================================
class SpeechTokenizerDecoderConfig:
    """Configuration for Speech Tokenizer Decoder."""

    def __init__(
        self,
        # Quantizer config
        num_quantizers: int = 16,
        codebook_size: int = 2048,
        codebook_dim: int = 256,
        latent_dim: int = 1024,
        # Pre-transformer config
        pre_transformer_hidden_size: int = 512,
        pre_transformer_intermediate_size: int = 1024,
        pre_transformer_num_layers: int = 8,
        pre_transformer_num_heads: int = 16,
        pre_transformer_head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        # Decoder config
        decoder_dim: int = 1536,
        upsample_rates: Tuple[int, ...] = (8, 5, 4, 3),
        upsampling_ratios: Tuple[int, ...] = (2, 2),
        # Audio config
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
    ):
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.latent_dim = latent_dim
        self.pre_transformer_hidden_size = pre_transformer_hidden_size
        self.pre_transformer_intermediate_size = pre_transformer_intermediate_size
        self.pre_transformer_num_layers = pre_transformer_num_layers
        self.pre_transformer_num_heads = pre_transformer_num_heads
        self.pre_transformer_head_dim = pre_transformer_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.decoder_dim = decoder_dim
        self.upsample_rates = upsample_rates
        self.upsampling_ratios = upsampling_ratios
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate


def get_default_speech_tokenizer_config() -> SpeechTokenizerDecoderConfig:
    """Returns the default Speech Tokenizer Decoder configuration."""
    return SpeechTokenizerDecoderConfig()


# =============================================================================
# Speech Tokenizer Decoder - Codebook Lookup
# =============================================================================
def codebook_lookup(
    token_ids: torch.Tensor,
    codebooks: list,
    input_proj_weight: torch.Tensor,
    output_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Looks up embeddings from RVQ codebooks and sums them.

    NOTE: This is a legacy function. Use codebook_lookup_rvq instead for proper RVQ processing.

    Args:
        token_ids: Token IDs of shape [batch, num_quantizers, seq_len]
        codebooks: List of codebook embeddings, each [codebook_size, codebook_dim]
        input_proj_weight: Input projection [codebook_dim, latent_dim, 1] (Conv1d)
        output_proj_weight: Output projection [latent_dim, codebook_dim, 1] (Conv1d)

    Returns:
        Embeddings of shape [batch, seq_len, latent_dim]
    """
    batch_size, num_quantizers, seq_len = token_ids.shape

    # Sum embeddings from all codebooks
    embeddings = None
    for i, codebook in enumerate(codebooks):
        ids = token_ids[:, i, :]  # [batch, seq_len]
        emb = F.embedding(ids, codebook)  # [batch, seq_len, codebook_dim]

        if embeddings is None:
            embeddings = emb
        else:
            embeddings = embeddings + emb

    # Project to latent dim: Conv1d with kernel_size=1
    # Reshape for conv1d: [batch, channels, seq_len]
    embeddings = embeddings.transpose(1, 2)  # [batch, codebook_dim, seq_len]

    # Apply input projection (effectively a linear layer)
    # Conv1d weight shape: [out_channels, in_channels, kernel_size]
    embeddings = F.conv1d(embeddings, input_proj_weight)  # [batch, latent_dim, seq_len]

    # Apply output projection
    embeddings = F.conv1d(embeddings, output_proj_weight)  # [batch, latent_dim, seq_len]

    # Transpose back: [batch, seq_len, latent_dim]
    embeddings = embeddings.transpose(1, 2)

    return embeddings


def codebook_lookup_rvq(
    token_ids: torch.Tensor,
    rvq_first_codebook: torch.Tensor,
    rvq_rest_codebooks: list,
    rvq_first_output_proj: Optional[torch.Tensor],
    rvq_rest_output_proj: Optional[torch.Tensor],
    rvq_first_cluster_usage: Optional[torch.Tensor] = None,
    rvq_rest_cluster_usages: Optional[list] = None,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Proper RVQ codebook lookup with separate rvq_first and rvq_rest processing.

    CRITICAL: Codebook embeddings must be normalized by cluster_usage:
        embedding = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]

    Architecture (matching official qwen_tts):
    - rvq_first (semantic): 1 codebook -> sum -> output_proj (256 -> 256)
    - rvq_rest (acoustic): 15 codebooks -> sum -> output_proj (256 -> 256)
    - ADD the results (not concatenate!) -> [batch, codebook_dim, seq_len]

    Args:
        token_ids: Token IDs of shape [batch, num_quantizers, seq_len]
        rvq_first_codebook: First codebook embedding_sum [codebook_size, codebook_dim]
        rvq_rest_codebooks: List of remaining codebook embedding_sums
        rvq_first_output_proj: Output projection for rvq_first [256, 256, 1]
        rvq_rest_output_proj: Output projection for rvq_rest [256, 256, 1]
        rvq_first_cluster_usage: Cluster usage for first codebook [codebook_size]
        rvq_rest_cluster_usages: List of cluster usages for remaining codebooks
        epsilon: Small value to avoid division by zero

    Returns:
        Embeddings of shape [batch, codebook_dim, seq_len] (channels-first format!)
    """
    batch_size, num_quantizers, seq_len = token_ids.shape
    device = token_ids.device

    # Helper function to normalize codebook
    def normalize_codebook(embedding_sum, cluster_usage):
        if cluster_usage is not None:
            return embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]
        return embedding_sum

    # Process RVQ First (semantic codebook - index 0)
    rvq_first_emb = None
    if rvq_first_codebook is not None and num_quantizers > 0:
        # Normalize codebook by cluster usage
        codebook = normalize_codebook(rvq_first_codebook, rvq_first_cluster_usage)
        ids = token_ids[:, 0, :]  # [batch, seq_len]
        rvq_first_emb = F.embedding(ids, codebook)  # [batch, seq_len, 256]

        # Transpose to channels-first: [batch, 256, seq_len]
        rvq_first_emb = rvq_first_emb.transpose(1, 2)

        # Apply output projection if available
        if rvq_first_output_proj is not None:
            rvq_first_emb = F.conv1d(rvq_first_emb, rvq_first_output_proj)

    # Process RVQ Rest (acoustic codebooks - indices 1-15)
    rvq_rest_emb = None
    if len(rvq_rest_codebooks) > 0 and num_quantizers > 1:
        for i, codebook_sum in enumerate(rvq_rest_codebooks):
            if i + 1 >= num_quantizers:
                break

            # Get cluster usage for this codebook
            cluster_usage = None
            if rvq_rest_cluster_usages is not None and i < len(rvq_rest_cluster_usages):
                cluster_usage = rvq_rest_cluster_usages[i]

            # Normalize codebook
            codebook = normalize_codebook(codebook_sum, cluster_usage)

            ids = token_ids[:, i + 1, :]  # [batch, seq_len]
            emb = F.embedding(ids, codebook)  # [batch, seq_len, 256]
            emb = emb.transpose(1, 2)  # [batch, 256, seq_len]

            if rvq_rest_emb is None:
                rvq_rest_emb = emb
            else:
                rvq_rest_emb = rvq_rest_emb + emb

        # Apply output projection if available
        if rvq_rest_emb is not None and rvq_rest_output_proj is not None:
            rvq_rest_emb = F.conv1d(rvq_rest_emb, rvq_rest_output_proj)

    # ADD rvq_first and rvq_rest (not concatenate!)
    # Official qwen_tts: quantized = rvq_first.decode(...) + rvq_rest.decode(...)
    if rvq_first_emb is not None and rvq_rest_emb is not None:
        embeddings = rvq_first_emb + rvq_rest_emb  # [batch, 256, seq_len]
    elif rvq_first_emb is not None:
        embeddings = rvq_first_emb
    elif rvq_rest_emb is not None:
        embeddings = rvq_rest_emb
    else:
        raise ValueError("No RVQ codebooks available")

    return embeddings  # [batch, codebook_dim, seq_len]


# =============================================================================
# Speech Tokenizer Decoder - Pre-Transformer Attention (simplified, no QK-norm)
# =============================================================================
def pre_transformer_attention(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_proj_bias: Optional[torch.Tensor] = None,
    k_proj_bias: Optional[torch.Tensor] = None,
    v_proj_bias: Optional[torch.Tensor] = None,
    o_proj_bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    num_heads: int = 16,
    head_dim: int = 64,
    sliding_window: int = 72,
) -> torch.Tensor:
    """
    Pre-transformer attention (simpler than main model, with sliding window).

    Args:
        hidden_states: Input of shape [batch, seq_len, hidden_size]
        q/k/v/o_proj_weight: Projection weights
        q/k/v/o_proj_bias: Optional projection biases
        cos, sin: RoPE frequencies
        num_heads: Number of attention heads
        head_dim: Head dimension
        sliding_window: Sliding window size for attention

    Returns:
        Output of shape [batch, seq_len, hidden_size]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Project Q, K, V
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Apply RoPE if provided
    if cos is not None and sin is not None:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Compute attention scores
    scaling = head_dim**-0.5
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling

    # Apply causal mask with sliding window
    causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    # Apply sliding window mask
    if sliding_window is not None and sliding_window < seq_len:
        sliding_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype),
            diagonal=-sliding_window,
        )
        sliding_mask = torch.tril(sliding_mask, diagonal=-1)
        causal_mask = causal_mask + sliding_mask

    attn_weights = attn_weights + causal_mask

    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, value_states)

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, -1)
    attn_output = F.linear(attn_output, o_proj_weight, o_proj_bias)

    return attn_output


# =============================================================================
# Speech Tokenizer Decoder - Pre-Transformer Layer
# =============================================================================
def pre_transformer_layer(
    hidden_states: torch.Tensor,
    layer_weights: dict,
    config: SpeechTokenizerDecoderConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Single pre-transformer layer with attention and MLP.

    Args:
        hidden_states: Input of shape [batch, seq_len, hidden_size]
        layer_weights: Dictionary with layer weights
        config: Decoder configuration
        cos, sin: RoPE frequencies

    Returns:
        Output of shape [batch, seq_len, hidden_size]
    """
    residual = hidden_states

    # Pre-norm for attention
    hidden_states = rms_norm(hidden_states, layer_weights["input_layernorm.weight"], config.rms_norm_eps)

    # Self-attention
    attn_output = pre_transformer_attention(
        hidden_states,
        q_proj_weight=layer_weights["self_attn.q_proj.weight"],
        k_proj_weight=layer_weights["self_attn.k_proj.weight"],
        v_proj_weight=layer_weights["self_attn.v_proj.weight"],
        o_proj_weight=layer_weights["self_attn.o_proj.weight"],
        q_proj_bias=layer_weights.get("self_attn.q_proj.bias"),
        k_proj_bias=layer_weights.get("self_attn.k_proj.bias"),
        v_proj_bias=layer_weights.get("self_attn.v_proj.bias"),
        o_proj_bias=layer_weights.get("self_attn.o_proj.bias"),
        cos=cos,
        sin=sin,
        num_heads=config.pre_transformer_num_heads,
        head_dim=config.pre_transformer_head_dim,
    )

    # Apply layer scale if present (key can be self_attn_layer_scale or self_attn_layer_scale.scale)
    layer_scale_key = None
    if "self_attn_layer_scale.scale" in layer_weights:
        layer_scale_key = "self_attn_layer_scale.scale"
    elif "self_attn_layer_scale" in layer_weights:
        layer_scale_key = "self_attn_layer_scale"
    if layer_scale_key:
        attn_output = attn_output * layer_weights[layer_scale_key]

    hidden_states = residual + attn_output

    # Pre-norm for MLP
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights["post_attention_layernorm.weight"], config.rms_norm_eps)

    # MLP (SwiGLU)
    mlp_output = swiglu_mlp(
        hidden_states,
        gate_proj_weight=layer_weights["mlp.gate_proj.weight"],
        up_proj_weight=layer_weights["mlp.up_proj.weight"],
        down_proj_weight=layer_weights["mlp.down_proj.weight"],
    )

    # Apply layer scale if present (key can be mlp_layer_scale or mlp_layer_scale.scale)
    mlp_layer_scale_key = None
    if "mlp_layer_scale.scale" in layer_weights:
        mlp_layer_scale_key = "mlp_layer_scale.scale"
    elif "mlp_layer_scale" in layer_weights:
        mlp_layer_scale_key = "mlp_layer_scale"
    if mlp_layer_scale_key:
        mlp_output = mlp_output * layer_weights[mlp_layer_scale_key]

    hidden_states = residual + mlp_output

    return hidden_states


# =============================================================================
# Speech Tokenizer Decoder - Pre-Transformer Forward
# =============================================================================
def pre_transformer_forward(
    embeddings: torch.Tensor,
    weights: dict,
    config: SpeechTokenizerDecoderConfig,
) -> torch.Tensor:
    """
    Forward pass through the pre-transformer.

    Matches official qwen_tts implementation:
    1. input_proj: latent_dim -> hidden_size
    2. transformer layers
    3. final norm
    4. output_proj: hidden_size -> latent_dim

    Args:
        embeddings: Input embeddings of shape [batch, seq_len, latent_dim]
        weights: Dictionary with all pre-transformer weights
        config: Decoder configuration

    Returns:
        Output of shape [batch, seq_len, latent_dim]
    """
    batch_size, seq_len, _ = embeddings.shape
    device = embeddings.device

    # Input projection: latent_dim (1024) -> hidden_size (512)
    hidden_states = F.linear(embeddings, weights["input_proj.weight"], weights.get("input_proj.bias"))

    # Compute RoPE frequencies
    cos, sin = compute_rope_frequencies(config.pre_transformer_head_dim, seq_len, config.rope_theta, device)
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)

    # Process through transformer layers
    for layer_idx in range(config.pre_transformer_num_layers):
        layer_prefix = f"layers.{layer_idx}."
        layer_weights = {k.replace(layer_prefix, ""): v for k, v in weights.items() if k.startswith(layer_prefix)}
        hidden_states = pre_transformer_layer(hidden_states, layer_weights, config, cos, sin)

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["norm.weight"], config.rms_norm_eps)

    # Output projection (always apply, matching official qwen_tts)
    # This projects from hidden_size (512) back to latent_dim (512)
    if "output_proj.weight" in weights:
        hidden_states = F.linear(hidden_states, weights["output_proj.weight"], weights.get("output_proj.bias"))

    return hidden_states


# =============================================================================
# Speech Tokenizer Decoder - ConvNeXt Upsample Block
# =============================================================================
def convnext_block(
    x: torch.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: Optional[torch.Tensor],
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: Optional[torch.Tensor],
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ConvNeXt block for upsampling.

    Args:
        x: Input of shape [batch, channels, seq_len]
        dwconv_weight: Depthwise conv weight [channels, 1, kernel_size]
        pwconv1/2_weight: Pointwise conv weights
        norm_weight/bias: LayerNorm weights
        gamma: Optional layer scale

    Returns:
        Output of shape [batch, channels, seq_len]
    """
    residual = x

    # Depthwise conv (groups=channels) with CAUSAL padding (left-only)
    # Official uses Qwen3TTSTokenizerV2CausalConvNet with padding = kernel_size - 1 on left
    kernel_size = dwconv_weight.shape[-1]
    left_pad = kernel_size - 1  # All padding on left for causal
    x = F.pad(x, (left_pad, 0), mode="constant", value=0)
    x = F.conv1d(x, dwconv_weight, dwconv_bias, groups=x.shape[1])

    # LayerNorm (transpose to [batch, seq_len, channels] for LN)
    x = x.transpose(1, 2)
    x = F.layer_norm(x, [x.shape[-1]], norm_weight, norm_bias)

    # Pointwise convs (implemented as linear layers)
    x = F.linear(x, pwconv1_weight, pwconv1_bias)
    x = F.gelu(x)
    x = F.linear(x, pwconv2_weight, pwconv2_bias)

    # Transpose back
    x = x.transpose(1, 2)

    # Layer scale
    if gamma is not None:
        x = x * gamma.view(1, -1, 1)

    return residual + x


def upsample_block(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    convnext_weights: dict,
    upsample_rate: int,
) -> torch.Tensor:
    """
    Upsample block: ConvTranspose1d + ConvNeXt.

    Args:
        x: Input of shape [batch, channels, seq_len]
        conv_weight: ConvTranspose1d weight
        conv_bias: ConvTranspose1d bias
        convnext_weights: Weights for ConvNeXt block
        upsample_rate: Upsampling factor

    Returns:
        Output of shape [batch, channels, seq_len * upsample_rate]
    """
    # Transposed convolution for upsampling
    x = F.conv_transpose1d(x, conv_weight, conv_bias, stride=upsample_rate)

    # ConvNeXt block
    x = convnext_block(
        x,
        dwconv_weight=convnext_weights["dwconv.conv.weight"],
        dwconv_bias=convnext_weights.get("dwconv.conv.bias"),
        pwconv1_weight=convnext_weights["pwconv1.weight"],
        pwconv1_bias=convnext_weights.get("pwconv1.bias"),
        pwconv2_weight=convnext_weights["pwconv2.weight"],
        pwconv2_bias=convnext_weights.get("pwconv2.bias"),
        norm_weight=convnext_weights["norm.weight"],
        norm_bias=convnext_weights["norm.bias"],
        gamma=convnext_weights.get("gamma"),
    )

    return x


# =============================================================================
# Speech Tokenizer Decoder - Conv Decoder Block
# =============================================================================
def snake_activation(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    """Snake activation: x + (1/exp(beta)) * sin^2(exp(alpha) * x)

    IMPORTANT: alpha and beta are stored as log values, so we must exp() them!
    This matches the official qwen_tts SnakeBeta implementation.

    Args:
        x: Input tensor of shape [batch, channels, seq_len]
        alpha: Per-channel log-alpha parameter [channels]
        beta: Per-channel log-beta parameter [channels]
        epsilon: Small value to avoid division by zero

    Returns:
        Output tensor of shape [batch, channels, seq_len]
    """
    # Reshape alpha and beta for broadcasting: [1, channels, 1]
    alpha = alpha.view(1, -1, 1)
    beta = beta.view(1, -1, 1)
    # CRITICAL: Apply exp() to convert from log-space (matching official qwen_tts)
    alpha = torch.exp(alpha)
    beta = torch.exp(beta)
    return x + (1.0 / (beta + epsilon)) * torch.sin(alpha * x).pow(2)


def conv_decoder_block(
    x: torch.Tensor,
    block_weights: dict,
    upsample_rate: int,
    num_residual_layers: int = 3,
    dilations: list = None,
) -> torch.Tensor:
    """
    Conv decoder block with upsampling and residual layers (causal padding).

    Args:
        x: Input of shape [batch, in_channels, seq_len]
        block_weights: Dictionary with block weights
        upsample_rate: Upsampling factor
        num_residual_layers: Number of residual layers
        dilations: Dilation rates for residual layers (default: [1, 3, 9])

    Returns:
        Output of shape [batch, out_channels, seq_len * upsample_rate]
    """
    if dilations is None:
        dilations = [1, 3, 9]

    # Snake activation before upsampling
    # Keys can be "alpha"/"beta" or "block.0.alpha"/"block.0.beta"
    alpha_key = "block.0.alpha" if "block.0.alpha" in block_weights else "alpha"
    beta_key = "block.0.beta" if "block.0.beta" in block_weights else "beta"
    if alpha_key in block_weights and beta_key in block_weights:
        x = snake_activation(x, block_weights[alpha_key], block_weights[beta_key])

    # Transposed conv for upsampling (in block.1)
    # Official qwen_tts uses NO padding in conv_transpose1d, then trims right side
    if "block.1.conv.weight" in block_weights:
        kernel_size = block_weights["block.1.conv.weight"].shape[-1]
        x = F.conv_transpose1d(
            x,
            block_weights["block.1.conv.weight"],
            block_weights.get("block.1.conv.bias"),
            stride=upsample_rate,
        )
        # Trim right side: pad = kernel_size - stride
        right_pad = kernel_size - upsample_rate
        if right_pad > 0:
            x = x[..., :-right_pad]

    # Residual layers (block.2, block.3, block.4) with dilations
    for i, dilation in zip(range(2, 2 + num_residual_layers), dilations):
        residual = x

        # First activation + conv (dilated, causal)
        act1_key = f"block.{i}.act1"
        if f"{act1_key}.alpha" in block_weights:
            x = snake_activation(x, block_weights[f"{act1_key}.alpha"], block_weights[f"{act1_key}.beta"])

        conv1_weight = block_weights.get(f"block.{i}.conv1.conv.weight")
        conv1_bias = block_weights.get(f"block.{i}.conv1.conv.bias")
        if conv1_weight is not None:
            kernel_size = conv1_weight.shape[-1]
            # Causal padding for dilated conv: effective_kernel = (kernel-1)*dilation + 1
            effective_kernel = (kernel_size - 1) * dilation + 1
            x = F.pad(x, (effective_kernel - 1, 0), mode="constant", value=0)
            x = F.conv1d(x, conv1_weight, conv1_bias, dilation=dilation)

        # Second activation + conv (1x1, no dilation needed)
        act2_key = f"block.{i}.act2"
        if f"{act2_key}.alpha" in block_weights:
            x = snake_activation(x, block_weights[f"{act2_key}.alpha"], block_weights[f"{act2_key}.beta"])

        conv2_weight = block_weights.get(f"block.{i}.conv2.conv.weight")
        conv2_bias = block_weights.get(f"block.{i}.conv2.conv.bias")
        if conv2_weight is not None:
            x = F.conv1d(x, conv2_weight, conv2_bias)

        x = residual + x

    return x


# =============================================================================
# Speech Tokenizer Decoder - Full Forward Pass
# =============================================================================
def speech_tokenizer_decoder_forward(
    token_ids: torch.Tensor,
    weights: dict,
    config: SpeechTokenizerDecoderConfig,
) -> torch.Tensor:
    """
    Full forward pass of the Speech Tokenizer Decoder.

    Converts codec tokens to audio waveform.

    Args:
        token_ids: Token IDs of shape [batch, num_quantizers, seq_len]
        weights: Dictionary with all decoder weights
        config: Decoder configuration

    Returns:
        Audio waveform of shape [batch, 1, num_samples]
    """
    batch_size, num_quantizers, seq_len = token_ids.shape
    device = token_ids.device

    # 1. Codebook lookup with proper RVQ processing
    # Get RVQ First (semantic codebook) - MUST normalize by cluster_usage!
    rvq_first_codebook = weights.get("quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
    rvq_first_cluster_usage = weights.get("quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")
    rvq_first_output_proj = weights.get("quantizer.rvq_first.output_proj.weight")

    # Get RVQ Rest (acoustic codebooks)
    rvq_rest_codebooks = []
    rvq_rest_cluster_usages = []
    for i in range(num_quantizers - 1):
        key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        usage_key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"
        if key in weights:
            rvq_rest_codebooks.append(weights[key])
            rvq_rest_cluster_usages.append(weights.get(usage_key))
    rvq_rest_output_proj = weights.get("quantizer.rvq_rest.output_proj.weight")

    # Use proper RVQ lookup with cluster_usage normalization
    embeddings = codebook_lookup_rvq(
        token_ids,
        rvq_first_codebook,
        rvq_rest_codebooks,
        rvq_first_output_proj,
        rvq_rest_output_proj,
        rvq_first_cluster_usage,
        rvq_rest_cluster_usages,
    )
    # embeddings is now [batch, codebook_dim, seq_len] (channels-first)

    # 2. Pre-transformer
    pre_transformer_weights = {
        k.replace("pre_transformer.", ""): v for k, v in weights.items() if k.startswith("pre_transformer.")
    }

    # 3. Pre-conv (causal padding for streaming)
    # embeddings is [batch, codebook_dim, seq_len]
    if "pre_conv.conv.weight" in weights:
        conv_weight = weights["pre_conv.conv.weight"]
        kernel_size = conv_weight.shape[-1]
        # Causal padding: pad left side only
        hidden_states = F.pad(embeddings, (kernel_size - 1, 0), mode="constant", value=0)
        hidden_states = F.conv1d(hidden_states, conv_weight, weights.get("pre_conv.conv.bias"))
    else:
        hidden_states = embeddings

    # 4. Pre-transformer
    # Transpose to [batch, seq_len, hidden] for transformer
    hidden_states = hidden_states.transpose(1, 2)  # [batch, seq_len, latent_dim]

    if pre_transformer_weights:
        hidden_states = pre_transformer_forward(hidden_states, pre_transformer_weights, config)
    # hidden_states is [batch, seq_len, latent_dim]

    # Transpose back to channels-first for conv decoder
    hidden_states = hidden_states.transpose(1, 2)  # [batch, latent_dim, seq_len]

    # 5. Upsampler (ConvNeXt blocks)
    for i, ratio in enumerate(config.upsampling_ratios):
        upsample_prefix = f"upsample.{i}."
        conv_weight = weights.get(f"{upsample_prefix}0.conv.weight")
        conv_bias = weights.get(f"{upsample_prefix}0.conv.bias")

        if conv_weight is not None:
            # Upsample with ConvTranspose1d
            hidden_states = F.conv_transpose1d(hidden_states, conv_weight, conv_bias, stride=ratio)
            # Trim padding from conv_transpose
            pad_to_trim = conv_weight.shape[-1] - ratio
            if pad_to_trim > 0:
                hidden_states = hidden_states[..., :-pad_to_trim]

            # ConvNeXt block
            convnext_weights = {
                k.replace(f"{upsample_prefix}1.", ""): v
                for k, v in weights.items()
                if k.startswith(f"{upsample_prefix}1.")
            }
            if convnext_weights:
                hidden_states = convnext_block(
                    hidden_states,
                    dwconv_weight=convnext_weights.get("dwconv.conv.weight"),
                    dwconv_bias=convnext_weights.get("dwconv.conv.bias"),
                    pwconv1_weight=convnext_weights.get("pwconv1.weight"),
                    pwconv1_bias=convnext_weights.get("pwconv1.bias"),
                    pwconv2_weight=convnext_weights.get("pwconv2.weight"),
                    pwconv2_bias=convnext_weights.get("pwconv2.bias"),
                    norm_weight=convnext_weights.get("norm.weight"),
                    norm_bias=convnext_weights.get("norm.bias"),
                    gamma=convnext_weights.get("gamma"),
                )

    # 6. Conv decoder (main upsampling)
    # Initial conv (causal padding)
    if "decoder.0.conv.weight" in weights:
        conv_weight = weights["decoder.0.conv.weight"]
        kernel_size = conv_weight.shape[-1]
        hidden_states = F.pad(hidden_states, (kernel_size - 1, 0), mode="constant", value=0)
        hidden_states = F.conv1d(hidden_states, conv_weight, weights.get("decoder.0.conv.bias"))

    # Decoder blocks with upsampling
    for i, rate in enumerate(config.upsample_rates):
        block_prefix = f"decoder.{i + 1}."
        block_weights = {k.replace(block_prefix, ""): v for k, v in weights.items() if k.startswith(block_prefix)}
        if block_weights:
            hidden_states = conv_decoder_block(hidden_states, block_weights, rate)

    # Final activation + conv (causal padding)
    if "decoder.5.alpha" in weights:
        hidden_states = snake_activation(hidden_states, weights["decoder.5.alpha"], weights["decoder.5.beta"])

    if "decoder.6.conv.weight" in weights:
        conv_weight = weights["decoder.6.conv.weight"]
        kernel_size = conv_weight.shape[-1]
        hidden_states = F.pad(hidden_states, (kernel_size - 1, 0), mode="constant", value=0)
        hidden_states = F.conv1d(hidden_states, conv_weight, weights.get("decoder.6.conv.bias"))

    # Clamp to audio range [-1, 1] (matching official qwen_tts)
    audio = hidden_states.clamp(min=-1, max=1)

    return audio


def extract_speech_tokenizer_decoder_weights(state_dict: dict) -> dict:
    """
    Extracts Speech Tokenizer Decoder weights from state dict.

    The weights are in speech_tokenizer/model.safetensors with prefix "decoder."

    Args:
        state_dict: Speech tokenizer state dict

    Returns:
        Dictionary with decoder weights (prefix removed)
    """
    prefix = "decoder."
    decoder_weights = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            decoder_weights[k.replace(prefix, "")] = v
    return decoder_weights


# =============================================================================
# Speech Tokenizer Encoder - Configuration
# =============================================================================
class SpeechTokenizerEncoderConfig:
    """Configuration for Speech Tokenizer Encoder."""

    def __init__(
        self,
        # Audio config
        sample_rate: int = 24000,
        audio_channels: int = 1,
        # Conv encoder config
        encoder_dim: int = 512,
        encoder_rates: list = None,  # Downsampling rates
        # Transformer config
        num_transformer_layers: int = 8,
        transformer_hidden_size: int = 512,
        transformer_intermediate_size: int = 2048,
        transformer_num_heads: int = 8,
        transformer_head_dim: int = 64,
        norm_eps: float = 1e-5,
        # Quantizer config
        num_semantic_quantizers: int = 1,
        num_acoustic_quantizers: int = 15,
        codebook_size: int = 2048,
        codebook_dim: int = 256,
    ):
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates or [8, 6, 5, 4]  # Total: 960x downsample
        self.num_transformer_layers = num_transformer_layers
        self.transformer_hidden_size = transformer_hidden_size
        self.transformer_intermediate_size = transformer_intermediate_size
        self.transformer_num_heads = transformer_num_heads
        self.transformer_head_dim = transformer_head_dim
        self.norm_eps = norm_eps
        self.num_semantic_quantizers = num_semantic_quantizers
        self.num_acoustic_quantizers = num_acoustic_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim


# =============================================================================
# Speech Tokenizer Encoder - Components
# =============================================================================
def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standard Layer Normalization."""
    return F.layer_norm(x, weight.shape, weight, bias, eps)


def encoder_residual_block(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: Optional[torch.Tensor],
    conv2_weight: torch.Tensor,
    conv2_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Encoder residual block with two convolutions.

    Args:
        x: Input [batch, channels, seq_len]
        conv1_weight: First conv weight
        conv1_bias: First conv bias
        conv2_weight: Second conv weight (1x1)
        conv2_bias: Second conv bias

    Returns:
        Output [batch, channels, seq_len]
    """
    residual = x

    # Conv1 with causal padding
    kernel_size = conv1_weight.shape[-1]
    x = F.pad(x, (kernel_size - 1, 0), mode="constant", value=0)
    x = F.conv1d(x, conv1_weight, conv1_bias)
    x = F.elu(x)

    # Conv2 (1x1, no padding needed)
    x = F.conv1d(x, conv2_weight, conv2_bias)
    x = F.elu(x)

    return residual + x


def encoder_conv_block(
    x: torch.Tensor,
    weights: dict,
    layer_idx: int,
    downsample: bool = False,
) -> torch.Tensor:
    """
    Conv encoder block - either simple conv or residual block.

    Args:
        x: Input [batch, in_channels, seq_len]
        weights: Dictionary with layer weights
        layer_idx: Layer index
        downsample: Whether this is a downsampling layer

    Returns:
        Output [batch, out_channels, seq_len'] where seq_len' may be downsampled
    """
    prefix = f"encoder.layers.{layer_idx}."

    if downsample:
        # Downsampling conv
        conv_weight = weights.get(f"{prefix}conv.weight")
        conv_bias = weights.get(f"{prefix}conv.bias")
        if conv_weight is not None:
            kernel_size = conv_weight.shape[-1]
            # Causal padding
            x = F.pad(x, (kernel_size - 1, 0), mode="constant", value=0)
            stride = kernel_size // 2 if kernel_size > 1 else 1  # Approximate stride
            x = F.conv1d(x, conv_weight, conv_bias, stride=stride)
            x = F.elu(x)
    else:
        # Residual block
        conv1_weight = weights.get(f"{prefix}block.1.conv.weight")
        conv1_bias = weights.get(f"{prefix}block.1.conv.bias")
        conv2_weight = weights.get(f"{prefix}block.3.conv.weight")
        conv2_bias = weights.get(f"{prefix}block.3.conv.bias")

        if conv1_weight is not None:
            x = encoder_residual_block(x, conv1_weight, conv1_bias, conv2_weight, conv2_bias)

    return x


def encoder_transformer_layer(
    hidden_states: torch.Tensor,
    weights: dict,
    layer_idx: int,
    config: SpeechTokenizerEncoderConfig,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Encoder transformer layer with LayerNorm, attention, and MLP.

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        weights: Layer weights
        layer_idx: Layer index
        config: Encoder config
        attention_mask: Optional attention mask

    Returns:
        Output [batch, seq_len, hidden_size]
    """
    prefix = f"encoder_transformer.layers.{layer_idx}."

    # Pre-attention LayerNorm
    ln1_weight = weights.get(f"{prefix}input_layernorm.weight")
    ln1_bias = weights.get(f"{prefix}input_layernorm.bias")
    normed = layer_norm(hidden_states, ln1_weight, ln1_bias, config.norm_eps)

    # Self-attention
    q_weight = weights.get(f"{prefix}self_attn.q_proj.weight")
    k_weight = weights.get(f"{prefix}self_attn.k_proj.weight")
    v_weight = weights.get(f"{prefix}self_attn.v_proj.weight")
    o_weight = weights.get(f"{prefix}self_attn.o_proj.weight")

    batch_size, seq_len, hidden_size = normed.shape
    num_heads = config.transformer_num_heads
    head_dim = hidden_size // num_heads

    # QKV projections
    q = F.linear(normed, q_weight)
    k = F.linear(normed, k_weight)
    v = F.linear(normed, v_weight)

    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention (bidirectional for encoder)
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # Reshape and project
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    attn_output = F.linear(attn_output, o_weight)

    # Layer scale for attention
    attn_scale = weights.get(f"{prefix}self_attn_layer_scale.scale")
    if attn_scale is not None:
        attn_output = attn_output * attn_scale

    # Residual connection
    hidden_states = hidden_states + attn_output

    # Post-attention LayerNorm
    ln2_weight = weights.get(f"{prefix}post_attention_layernorm.weight")
    ln2_bias = weights.get(f"{prefix}post_attention_layernorm.bias")
    normed = layer_norm(hidden_states, ln2_weight, ln2_bias, config.norm_eps)

    # MLP (GELU activation)
    fc1_weight = weights.get(f"{prefix}mlp.fc1.weight")
    fc2_weight = weights.get(f"{prefix}mlp.fc2.weight")

    mlp_output = F.linear(normed, fc1_weight)
    mlp_output = F.gelu(mlp_output)
    mlp_output = F.linear(mlp_output, fc2_weight)

    # Layer scale for MLP
    mlp_scale = weights.get(f"{prefix}mlp_layer_scale.scale")
    if mlp_scale is not None:
        mlp_output = mlp_output * mlp_scale

    # Residual connection
    hidden_states = hidden_states + mlp_output

    return hidden_states


def rvq_encode(
    z: torch.Tensor,
    codebooks: list,
    cluster_usages: list,
    input_proj: Optional[torch.Tensor] = None,
    output_proj: Optional[torch.Tensor] = None,
    epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Residual Vector Quantization encoding.

    Args:
        z: Input features [batch, seq_len, dim] or [batch, dim, seq_len]
        codebooks: List of codebook embedding_sum tensors
        cluster_usages: List of cluster_usage tensors for normalization
        input_proj: Optional input projection weight
        output_proj: Optional output projection weight
        epsilon: Small value for numerical stability

    Returns:
        Tuple of (codes, quantized) where:
            codes: Token IDs [batch, num_codebooks, seq_len]
            quantized: Quantized features [batch, seq_len, dim]
    """
    # Ensure [batch, seq_len, dim] format
    if z.dim() == 3 and z.shape[1] > z.shape[2]:
        # Likely [batch, dim, seq_len], transpose
        z = z.transpose(1, 2)

    batch_size, seq_len, dim = z.shape

    # Input projection if provided
    if input_proj is not None:
        # input_proj is Conv1d weight [out, in, 1]
        z_proj = z.transpose(1, 2)  # [batch, dim, seq_len]
        z_proj = F.conv1d(z_proj, input_proj)
        z_proj = z_proj.transpose(1, 2)  # [batch, seq_len, proj_dim]
    else:
        z_proj = z

    codes_list = []
    residual = z_proj.clone()
    quantized_sum = torch.zeros_like(z_proj)

    for i, (codebook_sum, cluster_usage) in enumerate(zip(codebooks, cluster_usages)):
        # Normalize codebook
        if cluster_usage is not None:
            codebook = codebook_sum / cluster_usage.clamp(min=epsilon)[:, None]
        else:
            codebook = codebook_sum

        # Find nearest codebook entry
        # residual: [batch, seq_len, dim], codebook: [num_codes, dim]
        residual_flat = residual.reshape(-1, residual.shape[-1])  # [batch*seq_len, dim]

        # Compute distances
        distances = torch.cdist(residual_flat, codebook)  # [batch*seq_len, num_codes]
        codes = distances.argmin(dim=-1)  # [batch*seq_len]
        codes = codes.view(batch_size, seq_len)  # [batch, seq_len]
        codes_list.append(codes)

        # Get quantized values
        quantized = F.embedding(codes, codebook)  # [batch, seq_len, dim]
        quantized_sum = quantized_sum + quantized

        # Update residual
        residual = residual - quantized

    # Stack codes: [batch, num_codebooks, seq_len]
    codes = torch.stack(codes_list, dim=1)

    # Output projection if provided
    if output_proj is not None:
        quantized_sum = quantized_sum.transpose(1, 2)  # [batch, dim, seq_len]
        quantized_sum = F.conv1d(quantized_sum, output_proj)
        quantized_sum = quantized_sum.transpose(1, 2)  # [batch, seq_len, out_dim]

    return codes, quantized_sum


def speech_tokenizer_encoder_forward_mimi(
    audio: torch.Tensor,
    model_path: Optional[str] = None,
    num_quantizers: int = 16,
) -> torch.Tensor:
    """
    Speech Tokenizer Encoder using MimiModel from transformers with key remapping.

    The Qwen3-TTS speech tokenizer checkpoint has keys prefixed with 'encoder.'
    that don't match MimiModel's expected format. This function remaps the keys.

    Args:
        audio: Audio waveform [batch, num_samples] or [batch, 1, num_samples] @ 24kHz
        model_path: Path to speech tokenizer model (default: Qwen HuggingFace)
        num_quantizers: Number of quantizers to use (default: 16)

    Returns:
        RVQ codes [batch, num_quantizers, seq_len] @ ~12.5Hz
    """
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import MimiConfig, MimiModel

    # Get model path
    if model_path is None:
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        model_path = snapshot_download(model_id, allow_patterns=["speech_tokenizer/*"])
        model_path = Path(model_path) / "speech_tokenizer"
    else:
        model_path = Path(model_path)

    # Load config
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # The encoder config is nested under 'encoder_config'
    encoder_config_dict = config_dict.get("encoder_config", config_dict)

    # Create MimiConfig (mapping qwen3_tts_tokenizer_12hz config to mimi)
    # Note: num_codebooks was renamed to num_quantizers in newer transformers
    mimi_config = MimiConfig(
        sampling_rate=encoder_config_dict.get("sampling_rate", 24000),
        audio_channels=encoder_config_dict.get("audio_channels", 1),
        hidden_size=encoder_config_dict.get("hidden_size", 512),
        num_hidden_layers=encoder_config_dict.get("num_hidden_layers", 8),
        num_attention_heads=encoder_config_dict.get("num_attention_heads", 8),
        num_key_value_heads=encoder_config_dict.get("num_key_value_heads", 8),
        head_dim=encoder_config_dict.get("head_dim", 64),
        intermediate_size=encoder_config_dict.get("intermediate_size", 2048),
        num_quantizers=encoder_config_dict.get("num_codebooks", encoder_config_dict.get("num_quantizers", 32)),
        codebook_size=encoder_config_dict.get("codebook_size", 2048),
        codebook_dim=encoder_config_dict.get("codebook_dim", 256),
        upsample_groups=encoder_config_dict.get("upsample_groups", 512),
        num_filters=encoder_config_dict.get("num_filters", 64),
        residual_kernel_size=encoder_config_dict.get("residual_kernel_size", 3),
        use_causal_conv=encoder_config_dict.get("use_causal_conv", True),
        compress=encoder_config_dict.get("compress", 2),
        trim_right_ratio=encoder_config_dict.get("trim_right_ratio", 1.0),
    )

    # Create model with config
    model = MimiModel(mimi_config)

    # Load weights and remap keys
    weights_path = model_path / "model.safetensors"
    raw_weights = load_file(weights_path)

    # Remap keys: remove 'encoder.' prefix where it creates double prefix
    remapped_weights = {}
    for key, value in raw_weights.items():
        new_key = key
        # Handle encoder weights (encoder.encoder.* -> encoder.*)
        if key.startswith("encoder.encoder."):
            new_key = "encoder." + key[16:]  # Remove "encoder.encoder." -> "encoder."
        elif key.startswith("encoder.encoder_transformer."):
            new_key = (
                "encoder_transformer." + key[28:]
            )  # Remove "encoder.encoder_transformer." -> "encoder_transformer."
        elif key.startswith("encoder.quantizer."):
            new_key = "quantizer." + key[18:]  # Remove "encoder.quantizer." -> "quantizer."
        elif key.startswith("encoder.downsample."):
            new_key = "downsample." + key[19:]  # Remove "encoder.downsample." -> "downsample."
        # Skip decoder weights (we only need encoder)
        elif key.startswith("decoder."):
            continue
        remapped_weights[new_key] = value

    # Load remapped weights (ignore missing decoder weights)
    model.load_state_dict(remapped_weights, strict=False)
    model.eval()

    # Ensure correct input format [batch, 1, num_samples]
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)

    # Move model to same device as audio
    device = audio.device
    model = model.to(device)

    # Encode using MimiModel
    with torch.no_grad():
        encoder_outputs = model.encode(audio, num_quantizers=num_quantizers)
        codes = encoder_outputs.audio_codes  # [batch, num_quantizers, seq_len]

    return codes


def speech_tokenizer_encoder_forward(
    audio: torch.Tensor,
    weights: dict = None,
    config: Optional[SpeechTokenizerEncoderConfig] = None,
    use_mimi: bool = True,
) -> torch.Tensor:
    """
    Speech Tokenizer Encoder forward pass.

    By default, uses MimiModel from transformers for accurate results.
    Set use_mimi=False to use the custom implementation (may have accuracy issues).

    Args:
        audio: Audio waveform [batch, 1, num_samples] or [batch, num_samples]
        weights: Dictionary with encoder weights (only used if use_mimi=False)
        config: Encoder configuration (only used if use_mimi=False)
        use_mimi: If True, use MimiModel from transformers (recommended)

    Returns:
        RVQ codes [batch, num_quantizers, seq_len]
    """
    # Use MimiModel for accurate encoding
    if use_mimi:
        return speech_tokenizer_encoder_forward_mimi(audio)

    # Legacy custom implementation (kept for reference, may have accuracy issues)
    if config is None:
        config = SpeechTokenizerEncoderConfig()

    if weights is None:
        raise ValueError("weights must be provided when use_mimi=False")

    # Ensure [batch, 1, num_samples] format
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)

    batch_size = audio.shape[0]
    hidden = audio

    # Conv encoder layers
    # Layer structure: 0=conv, 1=res, 3=conv, 4=res, 6=conv, 7=res, 9=conv, 10=res, 12=conv, 14=conv
    layer_indices = [0, 1, 3, 4, 6, 7, 9, 10, 12, 14]
    downsample_layers = {0, 3, 6, 9, 12, 14}

    for idx in layer_indices:
        is_downsample = idx in downsample_layers
        # Check if weights exist for this layer
        if f"encoder.layers.{idx}.conv.weight" in weights or f"encoder.layers.{idx}.block.1.conv.weight" in weights:
            hidden = encoder_conv_block(hidden, weights, idx, downsample=is_downsample)

    # Downsample conv (final temporal downsampling)
    if "downsample.conv.weight" in weights:
        conv_weight = weights["downsample.conv.weight"]
        kernel_size = conv_weight.shape[-1]
        hidden = F.pad(hidden, (kernel_size - 1, 0), mode="constant", value=0)
        hidden = F.conv1d(hidden, conv_weight, stride=kernel_size // 2)

    # Transpose for transformer: [batch, channels, seq_len] -> [batch, seq_len, channels]
    hidden = hidden.transpose(1, 2)

    # Encoder transformer layers
    for layer_idx in range(config.num_transformer_layers):
        hidden = encoder_transformer_layer(hidden, weights, layer_idx, config)

    # RVQ quantization
    # Semantic quantizer (1 codebook)
    semantic_codes = None
    semantic_prefix = "quantizer.semantic_residual_vector_quantizer."
    semantic_codebooks = []
    semantic_usages = []
    for i in range(config.num_semantic_quantizers):
        cb_key = f"{semantic_prefix}layers.{i}.codebook.embed_sum"
        usage_key = f"{semantic_prefix}layers.{i}.codebook.cluster_usage"
        if cb_key in weights:
            semantic_codebooks.append(weights[cb_key])
            semantic_usages.append(weights.get(usage_key))

    if semantic_codebooks:
        semantic_input_proj = weights.get(f"{semantic_prefix}input_proj.weight")
        semantic_output_proj = weights.get(f"{semantic_prefix}output_proj.weight")
        semantic_codes, _ = rvq_encode(
            hidden, semantic_codebooks, semantic_usages, semantic_input_proj, semantic_output_proj
        )

    # Acoustic quantizer (15 codebooks)
    acoustic_codes = None
    acoustic_prefix = "quantizer.acoustic_residual_vector_quantizer."
    acoustic_codebooks = []
    acoustic_usages = []
    for i in range(config.num_acoustic_quantizers):
        cb_key = f"{acoustic_prefix}layers.{i}.codebook.embed_sum"
        usage_key = f"{acoustic_prefix}layers.{i}.codebook.cluster_usage"
        if cb_key in weights:
            acoustic_codebooks.append(weights[cb_key])
            acoustic_usages.append(weights.get(usage_key))

    if acoustic_codebooks:
        acoustic_input_proj = weights.get(f"{acoustic_prefix}input_proj.weight")
        acoustic_output_proj = weights.get(f"{acoustic_prefix}output_proj.weight")
        acoustic_codes, _ = rvq_encode(
            hidden, acoustic_codebooks, acoustic_usages, acoustic_input_proj, acoustic_output_proj
        )

    # Combine semantic and acoustic codes
    if semantic_codes is not None and acoustic_codes is not None:
        codes = torch.cat([semantic_codes, acoustic_codes], dim=1)
    elif semantic_codes is not None:
        codes = semantic_codes
    elif acoustic_codes is not None:
        codes = acoustic_codes
    else:
        raise ValueError("No quantizer codebooks found in weights")

    return codes


def extract_speech_tokenizer_encoder_weights(state_dict: dict) -> dict:
    """
    Extracts Speech Tokenizer Encoder weights from state dict.

    Args:
        state_dict: Speech tokenizer state dict

    Returns:
        Dictionary with encoder weights (prefix "encoder." removed)
    """
    prefix = "encoder."
    encoder_weights = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            encoder_weights[k[len(prefix) :]] = v
    return encoder_weights


# =============================================================================
# Speaker Encoder - Configuration
# =============================================================================
class SpeakerEncoderConfig:
    """Configuration for Speaker Encoder."""

    def __init__(
        self,
        # Input config
        sample_rate: int = 24000,
        n_mels: int = 128,  # Model expects 128 channels (blocks.0.conv: [512, 128, 5])
        # Model config
        channels: int = 512,
        output_dim: int = 2048,
        num_blocks: int = 6,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.channels = channels
        self.output_dim = output_dim
        self.num_blocks = num_blocks


# =============================================================================
# Mel Spectrogram - matching official qwen_tts exactly
# =============================================================================


def compute_mel_spectrogram_qwen(
    audio: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 128,
    sampling_rate: int = 24000,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: int = 0,
    fmax: int = 12000,
    center: bool = False,
) -> torch.Tensor:
    """
    Compute mel spectrogram matching official qwen_tts exactly.

    Uses librosa mel filterbank with slaney norm and dynamic range compression.

    Args:
        audio: Audio waveform [batch, num_samples] or [num_samples]
        n_fft: FFT size (default: 1024)
        num_mels: Number of mel bins (default: 128)
        sampling_rate: Sample rate (default: 24000)
        hop_size: Hop size (default: 256)
        win_size: Window size (default: 1024)
        fmin: Minimum frequency (default: 0)
        fmax: Maximum frequency (default: 12000)
        center: Whether to center frames (default: False)

    Returns:
        Mel spectrogram [batch, num_mels, time] for speaker encoder
        or [batch, time, num_mels] for other uses (transpose as needed)
    """
    from librosa.filters import mel as librosa_mel_fn

    # Ensure batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    device = audio.device

    # Validate input range
    if torch.min(audio) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(audio)}")
    if torch.max(audio) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(audio)}")

    # Create mel filterbank using librosa (slaney norm)
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(device)

    # Hann window
    hann_window = torch.hann_window(win_size).to(device)

    # Reflect padding
    padding = (n_fft - hop_size) // 2
    y = F.pad(audio.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    # STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    # Apply mel filterbank
    mel_spec = torch.matmul(mel_basis, spec)

    # Dynamic range compression: log(clamp(x, 1e-5))
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec  # [batch, num_mels, time]


# =============================================================================
# Speaker Encoder - ECAPA-TDNN Components (matching official qwen_tts exactly)
# =============================================================================


def _conv1d_same_padding(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    """
    Conv1d with 'same' padding using reflect mode (matching official ECAPA-TDNN).

    Args:
        x: Input [batch, in_channels, seq_len]
        weight: Conv weight [out_channels, in_channels, kernel_size]
        bias: Conv bias [out_channels]
        dilation: Dilation factor

    Returns:
        Output [batch, out_channels, seq_len]
    """
    kernel_size = weight.shape[-1]
    # Compute padding for 'same' output size with dilation
    effective_kernel = dilation * (kernel_size - 1) + 1
    pad_total = effective_kernel - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Apply reflect padding
    x_padded = F.pad(x, (pad_left, pad_right), mode="reflect")

    return F.conv1d(x_padded, weight, bias, dilation=dilation)


def time_delay_net_block(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dilation: int = 1,
) -> torch.Tensor:
    """
    Time Delay Network Block: Conv1d(same, reflect) + ReLU

    Args:
        x: Input [batch, in_channels, seq_len]
        conv_weight: Conv weight [out_channels, in_channels, kernel_size]
        conv_bias: Conv bias [out_channels]
        dilation: Dilation factor

    Returns:
        Output [batch, out_channels, seq_len]
    """
    out = _conv1d_same_padding(x, conv_weight, conv_bias, dilation)
    return F.relu(out)


def res2net_block(
    x: torch.Tensor,
    weights: dict,
    prefix: str,
    scale: int = 8,
) -> torch.Tensor:
    """
    Res2Net block with multi-scale feature extraction (matching official exactly).

    Official pattern:
    - Split input into `scale` parts (e.g., 8)
    - Part 0: passed through unchanged
    - Part 1: goes through blocks[0]
    - Part i (i>1): hidden_part + previous_output, then through blocks[i-1]

    Args:
        x: Input [batch, channels, seq_len]
        weights: Weight dict
        prefix: Weight prefix like "blocks.1.res2net_block."
        scale: Number of scales (default 8)

    Returns:
        Output [batch, channels, seq_len]
    """
    batch, channels, seq_len = x.shape
    width = channels // scale

    # Split into scales using chunk (official uses torch.chunk)
    parts = list(torch.chunk(x, scale, dim=1))
    outputs = []

    output_part = None  # Track previous output for cumulative addition
    for i, hidden_part in enumerate(parts):
        if i == 0:
            # First part passes through unchanged
            output_part = hidden_part
        elif i == 1:
            # Second part goes through first TDNN block
            conv_weight = weights.get(f"{prefix}blocks.{i-1}.conv.weight")
            conv_bias = weights.get(f"{prefix}blocks.{i-1}.conv.bias")
            if conv_weight is not None:
                dilation = conv_weight.shape[-1] // 2 if conv_weight.shape[-1] > 1 else 1
                # Get dilation from the kernel and match official behavior
                output_part = time_delay_net_block(hidden_part, conv_weight, conv_bias)
            else:
                output_part = hidden_part
        else:
            # Parts 2+ add previous output before TDNN
            conv_weight = weights.get(f"{prefix}blocks.{i-1}.conv.weight")
            conv_bias = weights.get(f"{prefix}blocks.{i-1}.conv.bias")
            if conv_weight is not None:
                output_part = time_delay_net_block(hidden_part + output_part, conv_weight, conv_bias)
            else:
                output_part = hidden_part + output_part

        outputs.append(output_part)

    # Concatenate all outputs
    return torch.cat(outputs, dim=1)


def squeeze_excitation_block(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Squeeze-and-Excitation block (matching official Conv1d-based implementation).

    Args:
        x: Input [batch, channels, seq_len]
        conv1_weight: First Conv1d weight [se_channels, channels, 1]
        conv1_bias: First Conv1d bias
        conv2_weight: Second Conv1d weight [channels, se_channels, 1]
        conv2_bias: Second Conv1d bias

    Returns:
        Output [batch, channels, seq_len] (channel-wise scaled)
    """
    # Global average pooling
    y = x.mean(dim=2, keepdim=True)  # [batch, channels, 1]

    # Conv1d layers (official uses padding="same", padding_mode="reflect", but kernel=1 means no padding needed)
    y = F.relu(F.conv1d(y, conv1_weight, conv1_bias))
    y = torch.sigmoid(F.conv1d(y, conv2_weight, conv2_bias))

    return x * y


def se_res2net_block(
    x: torch.Tensor,
    weights: dict,
    block_idx: int,
    scale: int = 8,
) -> torch.Tensor:
    """
    SqueezeExcitationRes2NetBlock: TDNN1 → Res2Net → TDNN2 → SE → residual add

    Args:
        x: Input [batch, channels, seq_len]
        weights: Weight dict
        block_idx: Block index (1, 2, or 3)
        scale: Res2Net scale

    Returns:
        Output [batch, channels, seq_len]
    """
    prefix = f"blocks.{block_idx}."
    residual = x

    # TDNN1 (kernel=1, dilation=1)
    tdnn1_weight = weights.get(f"{prefix}tdnn1.conv.weight")
    tdnn1_bias = weights.get(f"{prefix}tdnn1.conv.bias")
    if tdnn1_weight is not None:
        x = time_delay_net_block(x, tdnn1_weight, tdnn1_bias, dilation=1)

    # Res2Net block
    x = res2net_block(x, weights, f"{prefix}res2net_block.", scale)

    # TDNN2 (kernel=1, dilation=1)
    tdnn2_weight = weights.get(f"{prefix}tdnn2.conv.weight")
    tdnn2_bias = weights.get(f"{prefix}tdnn2.conv.bias")
    if tdnn2_weight is not None:
        x = time_delay_net_block(x, tdnn2_weight, tdnn2_bias, dilation=1)

    # SE block
    se_conv1_weight = weights.get(f"{prefix}se_block.conv1.weight")
    se_conv1_bias = weights.get(f"{prefix}se_block.conv1.bias")
    se_conv2_weight = weights.get(f"{prefix}se_block.conv2.weight")
    se_conv2_bias = weights.get(f"{prefix}se_block.conv2.bias")
    if se_conv1_weight is not None:
        x = squeeze_excitation_block(x, se_conv1_weight, se_conv1_bias, se_conv2_weight, se_conv2_bias)

    # Residual connection
    return x + residual


def attentive_statistics_pooling(
    x: torch.Tensor,
    tdnn_weight: torch.Tensor,
    tdnn_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Attentive Statistics Pooling (matching official ECAPA-TDNN implementation).

    Args:
        x: Input [batch, channels, seq_len]
        tdnn_weight: TDNN conv weight [attention_channels, channels*3, kernel]
        tdnn_bias: TDNN conv bias
        conv_weight: Final conv weight [channels, attention_channels, kernel]
        conv_bias: Final conv bias
        eps: Epsilon for numerical stability

    Returns:
        Output [batch, channels * 2, 1] (mean + std concatenated, with time dim preserved)
    """
    batch, channels, seq_len = x.shape

    # Create mask (all ones for full sequence)
    mask = torch.ones(batch, 1, seq_len, device=x.device, dtype=x.dtype)
    total = mask.sum(dim=2, keepdim=True)

    # Compute global statistics (mean, std over time)
    mean = (mask * x).sum(dim=2)  # [batch, channels]
    std = torch.sqrt(((mask * (x - mean.unsqueeze(2)).pow(2)).sum(dim=2)).clamp(eps))  # [batch, channels]

    # Expand back to seq_len for concatenation
    mean_expanded = mean.unsqueeze(2).repeat(1, 1, seq_len)  # [batch, channels, seq_len]
    std_expanded = std.unsqueeze(2).repeat(1, 1, seq_len)  # [batch, channels, seq_len]

    # Concatenate [x, mean, std] -> [batch, channels*3, seq_len]
    attention_input = torch.cat([x, mean_expanded, std_expanded], dim=1)

    # Apply TDNN -> tanh -> conv -> softmax
    attention = _conv1d_same_padding(attention_input, tdnn_weight, tdnn_bias)
    attention = torch.tanh(attention)
    attention = _conv1d_same_padding(attention, conv_weight, conv_bias)

    # Mask out padded positions (not needed for full sequences)
    attention = attention.masked_fill(mask == 0, float("-inf"))

    # Softmax over time dimension
    attention = F.softmax(attention, dim=2)

    # Compute weighted statistics
    weighted_mean = (attention * x).sum(dim=2)  # [batch, channels]
    weighted_std = torch.sqrt(((attention * (x - weighted_mean.unsqueeze(2)).pow(2)).sum(dim=2)).clamp(eps))

    # Concatenate mean and std, add time dimension
    pooled_stats = torch.cat([weighted_mean, weighted_std], dim=1)  # [batch, channels*2]
    pooled_stats = pooled_stats.unsqueeze(2)  # [batch, channels*2, 1]

    return pooled_stats


def speaker_encoder_forward(
    mel_spectrogram: torch.Tensor,
    weights: dict,
    config: Optional[SpeakerEncoderConfig] = None,
) -> torch.Tensor:
    """
    Full Speaker Encoder forward pass (ECAPA-TDNN matching official qwen_tts).

    Architecture:
        Input: [batch, time, 128] mel spectrogram
        -> transpose to [batch, 128, time]
        -> TimeDelayNetBlock(128 → 512)           # blocks[0]
        -> SERes2NetBlock(512 → 512, dilation=2)  # blocks[1]
        -> SERes2NetBlock(512 → 512, dilation=3)  # blocks[2]
        -> SERes2NetBlock(512 → 512, dilation=4)  # blocks[3]
        -> MFA: cat(blocks[1:4]) -> TimeDelayNetBlock(1536 → 1536)
        -> AttentiveStatisticsPooling(1536) -> [batch, 3072, 1]
        -> Conv1d(3072 → 2048)
        -> squeeze to [batch, 2048]

    Args:
        mel_spectrogram: Mel-spectrogram [batch, time, n_mels] or [batch, n_mels, time]
        weights: Dictionary with speaker encoder weights (prefix "speaker_encoder." removed)
        config: Speaker encoder configuration

    Returns:
        Speaker embedding [batch, output_dim]
    """
    if config is None:
        config = SpeakerEncoderConfig()

    # Convert to float32 for computation
    hidden = mel_spectrogram.float()

    # Transpose if input is [batch, time, n_mels] (official expects this)
    # Detect based on dimensions: mel_dim is typically 128, time is usually larger
    if hidden.shape[1] > hidden.shape[2]:
        # Input is [batch, time, n_mels], transpose to [batch, n_mels, time]
        hidden = hidden.transpose(1, 2)

    # Helper to get weight as float32
    def get_weight(key):
        w = weights.get(key)
        return w.float() if w is not None else None

    # Track outputs from each block for MFA
    hidden_states_list = []

    # blocks[0]: Initial TimeDelayNetBlock(128 → 512, kernel=5, dilation=1)
    conv_weight = get_weight("blocks.0.conv.weight")
    conv_bias = get_weight("blocks.0.conv.bias")
    if conv_weight is not None:
        hidden = time_delay_net_block(hidden, conv_weight, conv_bias, dilation=1)
    hidden_states_list.append(hidden)

    # Convert weights to float32
    weights_f32 = {k: v.float() if v is not None else None for k, v in weights.items()}

    # blocks[1-3]: SERes2NetBlocks
    for block_idx in range(1, 4):  # blocks 1, 2, 3
        if f"blocks.{block_idx}.tdnn1.conv.weight" in weights:
            hidden = se_res2net_block(hidden, weights_f32, block_idx, scale=8)
        hidden_states_list.append(hidden)

    # Multi-scale Feature Aggregation (MFA)
    # Concatenate outputs from blocks[1], blocks[2], blocks[3] (skip blocks[0])
    hidden = torch.cat(hidden_states_list[1:], dim=1)  # [batch, 512*3=1536, time]

    # MFA TimeDelayNetBlock (kernel=1, dilation=1)
    mfa_weight = get_weight("mfa.conv.weight")
    mfa_bias = get_weight("mfa.conv.bias")
    if mfa_weight is not None:
        hidden = time_delay_net_block(hidden, mfa_weight, mfa_bias, dilation=1)

    # Attentive Statistics Pooling
    asp_tdnn_weight = get_weight("asp.tdnn.conv.weight")
    asp_tdnn_bias = get_weight("asp.tdnn.conv.bias")
    asp_conv_weight = get_weight("asp.conv.weight")
    asp_conv_bias = get_weight("asp.conv.bias")

    if asp_tdnn_weight is not None:
        hidden = attentive_statistics_pooling(
            hidden, asp_tdnn_weight, asp_tdnn_bias, asp_conv_weight, asp_conv_bias
        )  # [batch, 3072, 1]
    else:
        # Fallback
        mean = hidden.mean(dim=2)
        std = hidden.std(dim=2)
        hidden = torch.cat([mean, std], dim=1).unsqueeze(2)

    # Final FC: Conv1d(3072 → 2048, kernel=1)
    fc_weight = get_weight("fc.weight")
    fc_bias = get_weight("fc.bias")
    if fc_weight is not None:
        hidden = F.conv1d(hidden, fc_weight, fc_bias)

    # Squeeze time dimension
    hidden = hidden.squeeze(-1)  # [batch, 2048]

    return hidden


def extract_speaker_encoder_weights(state_dict: dict) -> dict:
    """
    Extracts Speaker Encoder weights from state dict.

    Args:
        state_dict: Main model state dict

    Returns:
        Dictionary with speaker encoder weights (prefix "speaker_encoder." removed)
    """
    prefix = "speaker_encoder."
    speaker_weights = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            speaker_weights[k[len(prefix) :]] = v
    return speaker_weights
