# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light numerical assertions shared by KDA tests."""

from __future__ import annotations

import torch


def _finiteness(name: str, tensor: torch.Tensor) -> tuple[list[str], str]:
    element_count = tensor.numel()
    nan_count = int(torch.isnan(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    positive_inf_count = int(torch.isposinf(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    negative_inf_count = int(torch.isneginf(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    non_finite_count = nan_count + positive_inf_count + negative_inf_count
    summary = (
        f"{name} finiteness: non_finite={non_finite_count}/{element_count}, "
        f"nan={nan_count}/{element_count}, +inf={positive_inf_count}/{element_count}, "
        f"-inf={negative_inf_count}/{element_count}"
    )
    failures = [] if non_finite_count == 0 else [summary]
    return failures, summary


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected_flat = expected.detach().float().reshape(-1)
    actual_flat = actual.detach().float().reshape(-1)
    if torch.equal(expected_flat, actual_flat):
        return 1.0
    expected_centered = expected_flat.double() - expected_flat.double().mean()
    actual_centered = actual_flat.double() - actual_flat.double().mean()
    denominator = torch.linalg.vector_norm(expected_centered) * torch.linalg.vector_norm(actual_centered)
    if denominator == 0:
        return float(torch.allclose(expected_flat, actual_flat, rtol=1e-5, atol=1e-4))
    return float(torch.dot(expected_centered, actual_centered) / denominator)


def assert_accurate(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    name: str = "accuracy",
    pcc_threshold: float = 0.999,
) -> float:
    """Require matching metadata, finite tensors, and PCC at or above a threshold."""
    failures = []
    if expected.shape != actual.shape:
        failures.append(f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if expected.dtype != actual.dtype:
        failures.append(f"{name} dtype {actual.dtype} != {expected.dtype}")
    expected_failures, expected_summary = _finiteness(f"{name} expected", expected)
    actual_failures, actual_summary = _finiteness(f"{name} actual", actual)
    failures.extend(expected_failures)
    failures.extend(actual_failures)
    if failures:
        raise AssertionError("\n".join(failures))

    pcc = _pcc(expected, actual)
    max_abs = float((expected.float() - actual.float()).abs().max()) if expected.numel() else 0.0
    print(expected_summary)
    print(actual_summary)
    print(f"{name}: PCC={pcc:.6f}, max_abs={max_abs:.6e}")
    if pcc < pcc_threshold:
        raise AssertionError(f"{name} PCC {pcc:.6f} < {pcc_threshold}")
    return pcc


def assert_equal(expected: torch.Tensor, actual: torch.Tensor, *, name: str = "equality") -> None:
    """Require finite tensors with identical metadata and values."""
    failures = []
    if expected.shape != actual.shape:
        failures.append(f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if expected.dtype != actual.dtype:
        failures.append(f"{name} dtype {actual.dtype} != {expected.dtype}")
    failures.extend(_finiteness(f"{name} expected", expected)[0])
    failures.extend(_finiteness(f"{name} actual", actual)[0])
    if expected.shape == actual.shape and expected.dtype == actual.dtype and not torch.equal(expected, actual):
        failures.append(f"{name} values differ")
    if failures:
        raise AssertionError("\n".join(failures))


def assert_bit_identical(expected: torch.Tensor, actual: torch.Tensor, *, name: str = "determinism") -> None:
    """Require finite tensors with identical metadata and bit patterns."""
    assert_equal(expected, actual, name=name)
    expected_bytes = expected.detach().contiguous().reshape(-1).view(torch.uint8)
    actual_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(expected_bytes, actual_bytes):
        raise AssertionError(f"{name} bit patterns differ")
