# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU feasibility tests for an associative KDA chunk-prefix formulation."""

from __future__ import annotations

import torch

AffineTransform = tuple[torch.Tensor, torch.Tensor]


def _identity(key_dim: int, value_dim: int, *, dtype: torch.dtype) -> AffineTransform:
    return torch.eye(key_dim, dtype=dtype), torch.zeros(key_dim, value_dim, dtype=dtype)


def _compose(after: AffineTransform, before: AffineTransform) -> AffineTransform:
    """Compose state transforms as ``after(before(state))``."""
    after_a, after_b = after
    before_a, before_b = before
    return after_a @ before_a, after_a @ before_b + after_b


def _chunk_transform(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> AffineTransform:
    """Build ``state_out = A @ state_in + B`` for one token chunk."""
    key_dim = k.shape[-1]
    value_dim = v.shape[-1]
    transform_a, transform_b = _identity(key_dim, value_dim, dtype=k.dtype)
    for token in range(k.shape[0]):
        decay = gate[token].exp()
        decayed_a = decay[:, None] * transform_a
        decayed_b = decay[:, None] * transform_b
        transform_a = decayed_a - beta[token] * torch.outer(k[token], k[token] @ decayed_a)
        residual_b = v[token] - k[token] @ decayed_b
        transform_b = decayed_b + beta[token] * torch.outer(k[token], residual_b)
    return transform_a, transform_b


def _exclusive_prefix(transforms: list[AffineTransform]) -> list[AffineTransform]:
    """Work-efficient Blelloch scan over associative affine transforms."""
    if not transforms:
        return []
    key_dim, value_dim = transforms[0][1].shape
    size = 1 << (len(transforms) - 1).bit_length()
    work = list(transforms)
    work.extend(_identity(key_dim, value_dim, dtype=transforms[0][0].dtype) for _ in range(size - len(work)))

    stride = 1
    while stride < size:
        for start in range(0, size, 2 * stride):
            left = start + stride - 1
            right = start + 2 * stride - 1
            work[right] = _compose(work[right], work[left])
        stride *= 2

    work[-1] = _identity(key_dim, value_dim, dtype=transforms[0][0].dtype)
    stride = size // 2
    while stride:
        for start in range(0, size, 2 * stride):
            left = start + stride - 1
            right = start + 2 * stride - 1
            left_total = work[left]
            prefix = work[right]
            work[left] = prefix
            work[right] = _compose(left_total, prefix)
        stride //= 2
    return work[: len(transforms)]


def _run_tokens(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.clone()
    outputs = []
    for token in range(k.shape[0]):
        state = gate[token].exp()[:, None] * state
        residual = v[token] - k[token] @ state
        state = state + beta[token] * torch.outer(k[token], residual)
        outputs.append(q[token] @ state)
    return torch.stack(outputs), state


def _inputs(sequence: int, key_dim: int, value_dim: int, *, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(941)
    q = torch.randn(sequence, key_dim, generator=generator, dtype=dtype) * key_dim**-0.5
    k = torch.randn(sequence, key_dim, generator=generator, dtype=dtype)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    v = torch.randn(sequence, value_dim, generator=generator, dtype=dtype)
    gate = -0.05 * torch.rand(sequence, key_dim, generator=generator, dtype=dtype)
    beta = torch.sigmoid(torch.randn(sequence, generator=generator, dtype=dtype))
    state = 0.02 * torch.randn(key_dim, value_dim, generator=generator, dtype=dtype)
    return q, k, v, gate, beta, state


def test_affine_composition_is_associative() -> None:
    q, k, v, gate, beta, _ = _inputs(12, 8, 6, dtype=torch.float64)
    del q
    transforms = [
        _chunk_transform(k[i : i + 4], v[i : i + 4], gate[i : i + 4], beta[i : i + 4]) for i in range(0, 12, 4)
    ]

    left = _compose(_compose(transforms[2], transforms[1]), transforms[0])
    right = _compose(transforms[2], _compose(transforms[1], transforms[0]))

    torch.testing.assert_close(left[0], right[0], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(left[1], right[1], rtol=1e-13, atol=1e-13)


def test_affine_prefix_reproduces_chunk_entry_states_and_outputs() -> None:
    chunk_size = 4
    q, k, v, gate, beta, initial_state = _inputs(32, 16, 12, dtype=torch.float64)
    expected_output, expected_state = _run_tokens(q, k, v, gate, beta, initial_state)
    chunks = [
        _chunk_transform(
            k[start : start + chunk_size],
            v[start : start + chunk_size],
            gate[start : start + chunk_size],
            beta[start : start + chunk_size],
        )
        for start in range(0, k.shape[0], chunk_size)
    ]
    prefixes = _exclusive_prefix(chunks)

    actual_outputs = []
    actual_state = initial_state
    for chunk, prefix in enumerate(prefixes):
        entry_state = prefix[0] @ initial_state + prefix[1]
        start = chunk * chunk_size
        output, actual_state = _run_tokens(
            q[start : start + chunk_size],
            k[start : start + chunk_size],
            v[start : start + chunk_size],
            gate[start : start + chunk_size],
            beta[start : start + chunk_size],
            entry_state,
        )
        actual_outputs.append(output)

    torch.testing.assert_close(torch.cat(actual_outputs), expected_output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-12, atol=1e-12)


def test_float32_balanced_prefix_stays_close_to_serial_recurrence() -> None:
    chunk_size = 8
    q, k, v, gate, beta, initial_state = _inputs(160, 32, 32, dtype=torch.float32)
    _, expected_state = _run_tokens(q, k, v, gate, beta, initial_state)
    chunks = [
        _chunk_transform(
            k[start : start + chunk_size],
            v[start : start + chunk_size],
            gate[start : start + chunk_size],
            beta[start : start + chunk_size],
        )
        for start in range(0, k.shape[0], chunk_size)
    ]
    final_prefix = _compose(chunks[-1], _exclusive_prefix(chunks)[-1])
    actual_state = final_prefix[0] @ initial_state + final_prefix[1]

    torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-5)
