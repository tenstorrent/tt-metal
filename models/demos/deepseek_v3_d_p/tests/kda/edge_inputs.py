# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic KDA recurrence inputs aimed at numerical edges (accuracy tier C).

Real inputs rarely isolate the cases that break chunked low-precision kernels, so each case here targets one:
the decay ceiling, long memory, both at once in one head, repeated keys, zero values, and sequence lengths
around 16- and 32-token chunk boundaries. Every gate stays inside what Kimi-K3's bounded gate
``-5·sigmoid(x)`` can emit and is fractional (never format-exact), so a case exercises the same argument
precision a trained layer does. Tensors use the reference's ``[B, T, H, D]`` layout and FP32; cast them to the
device format before use, and score against the FP64 golden of the cast values.
"""

from __future__ import annotations

import torch

# Tails of both the 16-token (FlashKDA) and 32-token (TT) chunk sizes, and a partial third 32-token chunk.
BOUNDARY_LENGTHS = (15, 16, 17, 31, 32, 33, 63, 64, 65)
GATE_LOWER_BOUND = -5.0
EDGE_CASES = ("bounded", "ceiling", "ceiling_with_long_memory", "long_memory", "repeated_keys", "zero_values")


def _bounded_gate(logits: torch.Tensor) -> torch.Tensor:
    return GATE_LOWER_BOUND * torch.sigmoid(logits)


def recurrence_edge_case(
    name: str, sequence: int, *, heads: int, key_dim: int, value_dim: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """``q, k, v, gate, beta`` (and a zero ``initial_state``) for one named edge case."""
    if name not in EDGE_CASES:
        raise ValueError(f"unknown edge case {name!r}; expected one of {EDGE_CASES}")
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    q = randn(1, sequence, heads, key_dim)
    k = randn(1, sequence, heads, key_dim)
    v = randn(1, sequence, heads, value_dim)
    beta = torch.sigmoid(randn(1, sequence, heads))
    noise = randn(1, sequence, heads, key_dim)

    if name == "bounded":
        # The whole bounded range, as random hidden states give.
        gate = _bounded_gate(2.0 * noise)
    elif name == "ceiling":
        # Every channel a few 1e-3 above -5 per token: |G_last| ≈ 159.9 per 32 tokens, never exactly 160.
        gate = _bounded_gate(8.0 + 0.5 * noise)
    elif name == "long_memory":
        # Decay ~1e-5 per token: exp(G) rounds to 1 in BF16 (spacing 2^-8 below 1) for whole chunks.
        gate = _bounded_gate(-13.0 + 0.5 * noise)
    elif name == "ceiling_with_long_memory":
        # Most channels at the ceiling next to a few long-memory channels in the same head, with beta near 1:
        # the in-domain K3 shape that still fails on device (g1b.4.16 k3-strong-saturated).
        gate = _bounded_gate(8.0 + 0.5 * noise)
        long_memory = max(1, key_dim // 16)
        gate[..., :long_memory] = _bounded_gate(-13.0 + 0.5 * noise[..., :long_memory])
        beta = 0.97 + 0.02 * torch.sigmoid(randn(1, sequence, heads))
    elif name == "repeated_keys":
        # One key repeated for every token of a head: the in-chunk triangular system is as ill-conditioned
        # as KDA allows, and beta near 1 keeps every rank-1 update large.
        gate = _bounded_gate(2.0 * noise)
        k = k[:, :1].expand_as(k).clone()
        beta = 0.9 + 0.1 * torch.sigmoid(randn(1, sequence, heads))
    else:  # zero_values
        # Zero values with a zero initial state: the exact output and state are zero, so only an absolute
        # check is meaningful.
        gate = _bounded_gate(2.0 * noise)
        v = torch.zeros_like(v)

    return {
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "initial_state": torch.zeros(1, heads, key_dim, value_dim),
    }
