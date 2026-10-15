# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic KDA recurrence inputs aimed at numerical edges (accuracy tier C).

Real inputs rarely isolate the cases that break chunked low-precision kernels, so each case here targets one:
short memory (the strongest decay), long memory, both at once in one head, repeated keys, and zero values. Every
gate stays inside what Kimi-K3's bounded gate ``-5·sigmoid(x)`` can emit and is fractional (never format-exact),
so a case exercises the same argument precision a trained layer does. Each generated case asserts that it reaches
its edge, so a test cannot pass on inputs that drifted away from it. Tensors use the reference's ``[B, T, H, D]``
layout and FP32; cast them to the device format before use, and score against the FP64 golden of the cast values.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_CHUNK_SIZE

GATE_LOWER_BOUND = KimiK3Config.KDA_GATE_LOWER_BOUND
# Largest |G| one chunk can accumulate: every token's log decay is above the gate bound.
CHUNK_DECAY_LIMIT = -GATE_LOWER_BOUND * KDA_CHUNK_SIZE
EDGE_CASES = ("baseline", "short_memory", "long_memory", "short_and_long_memory", "repeated_keys", "zero_values")


def _bounded_gate(logits: torch.Tensor) -> torch.Tensor:
    return GATE_LOWER_BOUND * torch.sigmoid(logits)


def recurrence_edge_case(
    name: str, sequence: int, *, heads: int, key_dim: int, value_dim: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """``q, k, v, gate, beta_logits`` (and a zero ``initial_state``) for one named edge case.

    ``beta_logits`` is what the device recurrence takes; the recurrence uses ``beta = sigmoid(beta_logits)``.
    """
    assert name in EDGE_CASES, f"unknown edge case {name!r}; expected one of {EDGE_CASES}"
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    q = randn(1, sequence, heads, key_dim)
    k = randn(1, sequence, heads, key_dim)
    v = randn(1, sequence, heads, value_dim)
    beta_logits = randn(1, sequence, heads)
    noise = randn(1, sequence, heads, key_dim)
    long_memory = max(1, key_dim // 16)

    if name == "baseline":
        # The whole bounded range, as random hidden states give.
        gate = _bounded_gate(2.0 * noise)
    elif name == "short_memory":
        # Every channel a few 1e-3 above -5 per token: |G_last| ≈ 159.9 per 32 tokens, never exactly 160.
        gate = _bounded_gate(8.0 + 0.5 * noise)
    elif name == "long_memory":
        # Decay ~1e-5 per token: exp(gate) rounds to 1 in BF16 (spacing 2^-8 below 1).
        gate = _bounded_gate(-13.0 + 0.5 * noise)
    elif name == "short_and_long_memory":
        # Most channels at short memory next to a few long-memory channels in the same head, with beta near 1:
        # the in-domain K3 shape that still fails on device (g1b.4.16 k3-strong-saturated).
        gate = _bounded_gate(8.0 + 0.5 * noise)
        gate[..., :long_memory] = _bounded_gate(-13.0 + 0.5 * noise[..., :long_memory])
        beta_logits = 3.5 + 1.1 * torch.sigmoid(randn(1, sequence, heads))  # beta = sigmoid(...) in [0.97, 0.99]
    elif name == "repeated_keys":
        # One key repeated for every token of a head: the in-chunk triangular system is as ill-conditioned
        # as KDA allows, and beta near 1 keeps every rank-1 update large.
        gate = _bounded_gate(2.0 * noise)
        k = k[:, :1].expand_as(k).clone()
        beta_logits = 2.2 + 3.0 * torch.sigmoid(randn(1, sequence, heads))  # beta = sigmoid(...) in [0.90, 0.995]
    else:  # zero_values
        # Zero values with a zero initial state: the exact output and state are zero, so only an absolute
        # check is meaningful.
        gate = _bounded_gate(2.0 * noise)
        v = torch.zeros_like(v)

    case = {
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta_logits": beta_logits,
        "initial_state": torch.zeros(1, heads, key_dim, value_dim),
    }
    _assert_reaches_edge(name, case, long_memory)
    return case


def _chunk_decay(gate: torch.Tensor) -> torch.Tensor:
    """|G| of each full chunk, per head and channel; a partial tail chunk is not checked."""
    full = gate.shape[1] // KDA_CHUNK_SIZE * KDA_CHUNK_SIZE
    return -gate[:, :full].unflatten(1, (-1, KDA_CHUNK_SIZE)).sum(dim=2)


def _assert_reaches_edge(name: str, case: dict[str, torch.Tensor], long_memory: int) -> None:
    gate, beta = case["gate"], torch.sigmoid(case["beta_logits"])
    assert bool(((gate > GATE_LOWER_BOUND) & (gate < 0.0)).all()), f"{name}: gate outside ({GATE_LOWER_BOUND}, 0)"

    if name in ("short_memory", "short_and_long_memory"):
        decay = _chunk_decay(gate if name == "short_memory" else gate[..., long_memory:])
        assert bool(((decay > CHUNK_DECAY_LIMIT - 1.0) & (decay < CHUNK_DECAY_LIMIT)).all()), (
            f"{name}: chunk |G| in [{decay.min():.3f}, {decay.max():.3f}], "
            f"expected just under {CHUNK_DECAY_LIMIT:g}"
        )
    if name in ("long_memory", "short_and_long_memory"):
        slow = gate if name == "long_memory" else gate[..., :long_memory]
        assert bool((torch.exp(slow).bfloat16() == 1.0).all()), f"{name}: a long-memory decay is not 1 in BF16"
    if name == "short_and_long_memory":
        assert bool(((beta > 0.97) & (beta < 0.99)).all()), f"{name}: beta outside (0.97, 0.99)"
    if name == "repeated_keys":
        assert torch.equal(case["k"], case["k"][:, :1].expand_as(case["k"])), f"{name}: keys differ across tokens"
        assert bool((beta >= 0.9).all()), f"{name}: beta below 0.9"
    if name == "zero_values":
        assert not case["v"].any(), f"{name}: values are not zero"
