# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Training FLOPs calculation for Qwen3 architecture.

Uses the standard LLM FLOPs formula: 6*N + 12*L*H*Q*T
where N accounts for Qwen3-specific parameter layout:
  - Separate Q, K, V projections with explicit head_dim
  - QK-Norm weights (per-head RMSNorm on Q and K)
  - Optional attention bias
  - SwiGLU MLP (same as Llama)
"""

from __future__ import annotations


def calculate_flops_per_token(config, seq_len: int) -> int:
    """Calculate training FLOPs per token for Qwen3 architecture.

    Uses the standard LLM FLOPs formula from PaLM paper:
      flops = 6 * N + 12 * L * H * Q * T

    Args:
        config: Qwen3Config instance with actual model parameters
        seq_len: Sequence length used for training

    Returns:
        FLOPs per token (int)
    """
    vocab_size = config.vocab_size
    head_dim = config.head_dim
    q_out = config.num_attention_heads * head_dim
    kv_out = config.num_key_value_heads * head_dim

    # Embedding: vocab_size * hidden_size
    embed_params = vocab_size * config.hidden_size

    # Attention per layer:
    #   q_proj:  hidden_size x q_out
    #   k_proj:  hidden_size x kv_out
    #   v_proj:  hidden_size x kv_out
    #   o_proj:  q_out x hidden_size
    #   q_norm:  head_dim  (RMSNorm weight)
    #   k_norm:  head_dim  (RMSNorm weight)
    #   input_layernorm: hidden_size
    attention_params = (
        config.hidden_size * q_out
        + config.hidden_size * kv_out
        + config.hidden_size * kv_out
        + q_out * config.hidden_size
        + head_dim
        + head_dim
        + config.hidden_size
    )

    if config.attention_bias:
        attention_params += q_out + kv_out + kv_out + config.hidden_size

    # MLP per layer (SwiGLU):
    #   gate_proj: hidden_size x intermediate_size
    #   up_proj:   hidden_size x intermediate_size
    #   down_proj:  intermediate_size x hidden_size
    #   post_attention_layernorm: hidden_size
    mlp_params = (
        config.hidden_size * config.intermediate_size
        + config.hidden_size * config.intermediate_size
        + config.intermediate_size * config.hidden_size
        + config.hidden_size
    )

    params_per_layer = attention_params + mlp_params
    transformer_params = config.num_hidden_layers * params_per_layer

    # Output head (if not weight-tied)
    head_params = vocab_size * config.hidden_size
    if hasattr(config, "weight_tying") and config.weight_tying.name == "Enabled":
        head_params = 0

    # Final layer norm (RMSNorm)
    final_norm_params = config.hidden_size

    N = embed_params + transformer_params + head_params + final_norm_params

    L = config.num_hidden_layers
    H = config.num_attention_heads
    Q = head_dim
    T = seq_len

    flops = 6 * N + 12 * L * H * Q * T

    return int(flops)
