# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Training FLOPs calculation for NanoGPT (GPT2) architecture.

Uses the standard LLM FLOPs formula: 6*N + 12*L*H*Q*T
where N is calculated by enumerating actual model layers.
"""

from __future__ import annotations


def calculate_flops_per_token(config, seq_len: int) -> int:
    """Calculate training FLOPs per token for NanoGPT (GPT2) architecture.

    Uses the standard LLM FLOPs formula from PaLM paper:
      flops = 6 * N + 12 * L * H * Q * T

    N is computed by enumerating the exact layers in the model:
      - MultiHeadAttention: qkv_linear (3*E², bias 3*E) + out_linear (E², bias E)
      - GPTMLP: fc1 (4*E², bias 4*E) + fc2 (4*E², bias E)
      - LayerNorm x2: gamma (E each), beta (E each, only if config.bias)
    All linear layers in attention and MLP hardcode bias=True.

    Args:
        config: NanoGPTConfig instance with actual model parameters
        seq_len: Sequence length used for training

    Returns:
        FLOPs per token (int)
    """
    E = config.n_embd
    vocab_size = config.vocab_size

    embed_params = vocab_size * E

    pos_embed_params = 0
    if hasattr(config, "positional_embedding_type") and config.positional_embedding_type == "trainable":
        pos_embed_params = config.block_size * E

    # Per-layer parameter count (from actual layer constructors):
    # Attention weights: qkv_linear(E, 3E) + out_linear(E, E) = 4E²
    # Attention biases (always on): 3E + E = 4E
    # MLP weights: fc1(E, 4E) + fc2(4E, E) = 8E²
    # MLP biases (always on): 4E + E = 5E
    # LayerNorm gamma (2 per block): 2E
    # LayerNorm beta (2 per block, only if config.bias): 2E
    params_per_layer = 12 * E * E + 11 * E
    if config.bias:
        params_per_layer += 2 * E

    transformer_params = config.n_layer * params_per_layer

    head_params = vocab_size * E
    if hasattr(config, "weight_tying") and config.weight_tying.name == "Enabled":
        head_params = 0

    # Final LayerNorm: gamma always, beta only if config.bias
    final_norm_params = E * (2 if config.bias else 1)

    N = embed_params + pos_embed_params + transformer_params + head_params + final_norm_params

    L = config.n_layer
    H = config.n_head
    Q = E // H
    T = seq_len

    flops = 6 * N + 12 * L * H * Q * T

    return int(flops)
