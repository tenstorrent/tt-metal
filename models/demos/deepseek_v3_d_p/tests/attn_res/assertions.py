# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Numerical assertions shared by the AttnRes tests, in three strengths.

`assert_accurate` compares an implementation against an oracle: finite, same shape, PCC at
or above a threshold. `assert_equal` requires identical values, and `assert_bit_identical`
identical bytes. Signed zero is the only thing separating those two: `-0.0` and `0.0` compare
equal as values and differ in the sign bit. NaN does not separate them — `torch.equal` is
false wherever a NaN appears, so the value comparison rejects it before the bytes are read.

The strength is chosen by what the two tensors are, not by how close they are expected to
be. Anything measured against a host oracle crossed a dtype and a different multiply order
to get there, so it can only be `assert_accurate`. Anything measured against another run of
the same implementation crossed neither, so it has no tolerance to spend: a repeat, a trace
replay, or a borrowed input read back afterwards is exact or it is a bug.

`assert_accurate` deliberately does not require matching dtypes, unlike its two siblings. A
device read returns bf16 and its oracle is computed in fp32 — requiring equality there would
force the oracle down to the width being measured.
"""

from __future__ import annotations

import torch


def _finiteness(name: str, tensor: torch.Tensor) -> list[str]:
    element_count = tensor.numel()
    nan_count = int(torch.isnan(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    positive_inf_count = int(torch.isposinf(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    negative_inf_count = int(torch.isneginf(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    non_finite_count = nan_count + positive_inf_count + negative_inf_count
    if non_finite_count == 0:
        return []
    return [
        f"{name} finiteness: non_finite={non_finite_count}/{element_count}, "
        f"nan={nan_count}/{element_count}, +inf={positive_inf_count}/{element_count}, "
        f"-inf={negative_inf_count}/{element_count}"
    ]


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected_flat = expected.detach().float().reshape(-1)
    actual_flat = actual.detach().float().reshape(-1)
    if torch.equal(expected_flat, actual_flat):
        return 1.0
    expected_centered = expected_flat.double() - expected_flat.double().mean()
    actual_centered = actual_flat.double() - actual_flat.double().mean()
    denominator = torch.linalg.vector_norm(expected_centered) * torch.linalg.vector_norm(actual_centered)
    return float(torch.dot(expected_centered, actual_centered) / denominator)


def _metadata_failures(name: str, expected: torch.Tensor, actual: torch.Tensor, match_dtype: bool) -> list[str]:
    failures = []
    if expected.shape != actual.shape:
        failures.append(f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if match_dtype and expected.dtype != actual.dtype:
        failures.append(f"{name} dtype {actual.dtype} != {expected.dtype}")
    failures.extend(_finiteness(f"{name} expected", expected))
    failures.extend(_finiteness(f"{name} actual", actual))
    return failures


def assert_accurate(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    name: str = "accuracy",
    pcc_threshold: float = 0.9999,
) -> float:
    """Require a finite, same-shaped tensor correlating with `expected` at `pcc_threshold`.

    Returns the measured PCC so a caller tracking a worst case over many sites can keep it.
    """
    failures = _metadata_failures(name, expected, actual, match_dtype=False)
    if failures:
        raise AssertionError("\n".join(failures))

    pcc = _pcc(expected, actual)
    max_abs = float((expected.float() - actual.float()).abs().max()) if expected.numel() else 0.0
    # Negated rather than `pcc < threshold`: a tensor with no variance correlates to NaN,
    # and every ordered comparison against NaN is false, so that form would pass it.
    if not pcc >= pcc_threshold:
        raise AssertionError(f"{name}: PCC {pcc:.7f} < {pcc_threshold}, max_abs={max_abs:.6e}")
    return pcc


def assert_equal(expected: torch.Tensor, actual: torch.Tensor, *, name: str = "equality") -> None:
    """Require finite tensors with identical shape, dtype, and values."""
    failures = _metadata_failures(name, expected, actual, match_dtype=True)
    if not failures and not torch.equal(expected, actual):
        max_abs = float((expected.float() - actual.float()).abs().max()) if expected.numel() else 0.0
        differing = int((expected != actual).sum().item())
        failures.append(f"{name} values differ: {differing}/{expected.numel()} elements, max_abs={max_abs:.6e}")
    if failures:
        raise AssertionError("\n".join(failures))


def assert_bit_identical(expected: torch.Tensor, actual: torch.Tensor, *, name: str = "determinism") -> None:
    """Require finite tensors with identical metadata and bit patterns."""
    assert_equal(expected, actual, name=name)
    expected_bytes = expected.detach().contiguous().reshape(-1).view(torch.uint8)
    actual_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(expected_bytes, actual_bytes):
        raise AssertionError(f"{name} bit patterns differ on tensors whose values compare equal")
