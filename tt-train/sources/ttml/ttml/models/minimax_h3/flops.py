# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Training FLOPs calculation for the MiniMax-H3 transformer.

Uses the standard LLM FLOPs formula (PaLM): 6*N + 12*L*H*Q*T per token for full fine-tuning, where N is the
per-token parameter count of the block stack and heads. Two MiniMax-H3 specifics:

  - The AdaLN projections (`adaln_proj`, 40% of the parameters) run once per sample on the distinct timesteps,
    not per token, so they are a per-sample term in `calculate_sequence_flops`, not part of N.
  - With a LoRA (`rank > 0`) the base weights are frozen: forward (2N) and input gradients (2N) still run through
    every matmul but the weight gradients (2N) do not, so the per-token cost is 4*N + 6*N_lora + 12*L*H*Q*T.
    The token refiner, patch projections and heads are then forward-only (nothing upstream of the blocks trains).
"""

from __future__ import annotations

ADALN_ROWS = 32  # distinct timesteps padded to one tile
LORA_TARGETS_PER_BLOCK = 6  # to_q, to_k, to_v, to_out, ff.net.0.proj, ff.net.2


def _block_params(config) -> int:
    """Per-token parameters of one transformer block, AdaLN excluded."""
    D, I, F = config.hidden_size, config.num_attention_heads * config.attention_head_dim, config.ffn_dim
    attention = 3 * D * I + I * D + 2 * config.attention_head_dim  # q, k, v, out, qk norms
    mlp = D * 2 * F + F * D  # fused SwiGLU proj, out
    return attention + mlp + 2 * D  # norm1, norm2


def _adaln_params(config) -> int:
    return config.time_embed_dim * 6 * 3 * config.hidden_size + 6 * 3 * config.hidden_size


def _lora_params_per_block(config, rank: int) -> int:
    D, I, F = config.hidden_size, config.num_attention_heads * config.attention_head_dim, config.ffn_dim
    linears = [(D, I), (D, I), (D, I), (I, D), (D, 2 * F), (F, D)]
    return sum(rank * (a + b) for a, b in linears)


def _head_params(config) -> int:
    """Per-token parameters outside the block stack on the packed rows: norm_out and the video head."""
    D = config.hidden_size
    return D + D * config.video_in_channels + config.video_in_channels


def calculate_flops_per_token(config, seq_len: int, rank: int = 0) -> int:
    """Training FLOPs per packed-sequence row through the 50 blocks and the video head.

    Args:
        config: H3Config.
        seq_len: packed (padded) sequence length T, for the attention term.
        rank: LoRA rank; 0 means full fine-tuning (6N convention).
    """
    L, H, Q, T = config.num_layers, config.num_attention_heads, config.attention_head_dim, seq_len
    N = L * _block_params(config) + _head_params(config)
    attention = 12 * L * H * Q * T
    if rank <= 0:
        return int(6 * N + attention)
    N_lora = L * _lora_params_per_block(config, rank)
    return int(4 * N + 6 * N_lora + attention)


def calculate_sequence_flops(config, seq_len: int, text_len: int, rank: int = 0) -> int:
    """Training FLOPs of one packed sample: per-token cost on `seq_len` rows, the token refiner on `text_len`
    rows, the context embedder, and the per-sample AdaLN and norm_out projections."""
    D, I, F, Q = (
        config.hidden_size,
        config.num_attention_heads * config.attention_head_dim,
        config.ffn_dim,
        config.attention_head_dim,
    )
    H, R = config.num_attention_heads, config.num_refiner_layers
    flops = calculate_flops_per_token(config, seq_len, rank) * seq_len
    refiner_params = R * (3 * D * I + I * D + 2 * Q + D * 2 * F + F * D + 2 * D) + D  # blocks + final norm
    embed_params = config.text_dim * D + D
    refiner_attention = 4 * R * H * Q * text_len * text_len
    per_sample = ADALN_ROWS * config.num_layers * _adaln_params(config) + ADALN_ROWS * (config.time_embed_dim * 2 * D)
    if rank <= 0:
        flops += 6 * (refiner_params + embed_params) * text_len + 3 * refiner_attention + 6 * per_sample
    else:
        flops += 2 * (refiner_params + embed_params) * text_len + refiner_attention + 2 * per_sample
    return int(flops)
