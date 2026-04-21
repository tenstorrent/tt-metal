# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Training FLOPs calculation for Llama architecture.

Uses the standard LLM FLOPs formula: 6*N + 12*L*H*Q*T
where N is calculated from the actual model config parameters.

Note: TinyLlama uses intermediate_size=5632 instead of the standard 4*hidden_size.
"""

from __future__ import annotations


def calculate_flops_per_token(config, seq_len: int) -> int:
    """Calculate training FLOPs per token for Llama architecture.

    Uses the standard LLM FLOPs formula from PaLM paper:
      flops = 6 * N + 12 * L * H * Q * T

    Args:
        config: LlamaConfig instance with actual model parameters
        seq_len: Sequence length used for training

    Returns:
        FLOPs per token (int)
    """
    # Get the actual vocab size from the model config
    vocab_size = config.vocab_size

    # Parameter calculation using actual Llama architecture
    # Embedding: vocab_size * hidden_size
    embed_params = vocab_size * config.hidden_size

    # Get intermediate size using the same logic as LlamaMLP
    intermediate_size = getattr(config, "intermediate_size", None)
    if intermediate_size is None:
        # Same calculation as in LlamaMLP: (4 * embedding_size * 2) // 3, rounded to multiple of 256
        multiple_of = 256
        unrounded_size = (4 * config.hidden_size * 2) // 3
        intermediate_size = ((unrounded_size + multiple_of - 1) // multiple_of) * multiple_of

    # Transformer layers - actual Llama structure per layer:
    # Attention:
    #   - q_proj: hidden_size × hidden_size
    #   - k_proj, v_proj: hidden_size × (num_key_value_heads * head_dim)
    #   - o_proj: hidden_size × hidden_size
    #   - attention_norm: hidden_size
    # MLP (SwiGLU):
    #   - gate_proj: hidden_size × intermediate_size
    #   - up_proj: hidden_size × intermediate_size
    #   - down_proj: intermediate_size × hidden_size
    #   - mlp_norm: hidden_size

    head_dim = config.hidden_size // config.num_attention_heads
    kv_size = config.num_key_value_heads * head_dim

    attention_params = (
        config.hidden_size * config.hidden_size
        + config.hidden_size * kv_size  # q_proj
        + config.hidden_size * kv_size  # k_proj
        + config.hidden_size * config.hidden_size  # v_proj
        + config.hidden_size  # o_proj  # attention_norm
    )

    mlp_params = (
        config.hidden_size * intermediate_size
        + config.hidden_size * intermediate_size  # gate_proj
        + intermediate_size * config.hidden_size  # up_proj
        + config.hidden_size  # down_proj  # mlp_norm
    )

    params_per_layer = attention_params + mlp_params
    transformer_params = config.num_hidden_layers * params_per_layer

    # Output head (if not weight-tied)
    head_params = vocab_size * config.hidden_size
    if hasattr(config, "weight_tying") and config.weight_tying.name == "Enabled":
        head_params = 0

    # Final layer norm (RMSNorm - no bias)
    final_norm_params = config.hidden_size

    # Total parameters
    N = embed_params + transformer_params + head_params + final_norm_params

    # Standard LLM FLOPs formula
    L = config.num_hidden_layers
    H = config.num_attention_heads
    Q = config.hidden_size // config.num_attention_heads
    T = seq_len

    flops = 6 * N + 12 * L * H * Q * T

    return int(flops)
