# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light numerical assertions shared by KDA tests."""

from __future__ import annotations
from collections.abc import Callable, Sequence

import torch
import ttnn


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


def collect_accuracy_and_determinism_results(
    device: ttnn.Device,
    run: Callable[[], Sequence[ttnn.Tensor]],
    *,
    count: int = 3,
) -> tuple[tuple[ttnn.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
    """Run repeatedly, retaining only first outputs and one device-side mismatch marker."""
    if count <= 1:
        raise ValueError("count must be greater than one")

    reference_outputs = tuple(run())
    if not reference_outputs:
        raise ValueError("run must return at least one output")
    mismatch_scratch = tuple(
        ttnn.empty(
            output.shape,
            dtype=ttnn.bfloat16,
            layout=output.layout,
            device=device,
            memory_config=output.memory_config(),
        )
        for output in reference_outputs
    )
    mismatch_marker = None
    for _ in range(1, count):
        outputs = tuple(run())
        if len(outputs) != len(reference_outputs):
            for output in outputs:
                ttnn.deallocate(output)
            raise ValueError("run returned a different number of outputs")
        for reference, output, scratch in zip(reference_outputs, outputs, mismatch_scratch, strict=True):
            if (
                output.shape != reference.shape
                or output.dtype != reference.dtype
                or output.layout != reference.layout
                or output.memory_config() != reference.memory_config()
            ):
                for repeat_output in outputs:
                    ttnn.deallocate(repeat_output)
                raise ValueError("run returned output with different metadata")
            ttnn.ne(reference, output, dtype=ttnn.bfloat16, output_tensor=scratch)
            current_mismatch = ttnn.max(scratch)
            ttnn.deallocate(output)
            if mismatch_marker is None:
                mismatch_marker = current_mismatch
            else:
                updated_marker = ttnn.maximum(mismatch_marker, current_mismatch)
                ttnn.deallocate(mismatch_marker)
                ttnn.deallocate(current_mismatch)
                mismatch_marker = updated_marker

    assert mismatch_marker is not None
    reference_outputs_host = tuple(ttnn.to_torch(output).clone() for output in reference_outputs)
    mismatch_marker_host = ttnn.to_torch(mismatch_marker).clone()
    for scratch in mismatch_scratch:
        ttnn.deallocate(scratch)
    ttnn.deallocate(mismatch_marker)
    return reference_outputs, reference_outputs_host, mismatch_marker_host
