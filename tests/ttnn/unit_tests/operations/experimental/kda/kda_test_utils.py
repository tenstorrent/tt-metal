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


def _relative_rmse(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """RMS error as a fraction of the signal's own RMS.

    PCC is a correlation, so it is dominated by the bulk of a tensor and can stay
    above 0.999 while a small region is badly wrong -- eleven wrong rows in 1280
    passed a PCC gate here. Relative RMSE grows with the error's total energy, so
    it registers a localised fault that PCC averages away, and being normalised it
    needs no per-tensor threshold.
    """
    if not expected.numel():
        return 0.0
    difference = (expected.float() - actual.float()).pow(2).mean().sqrt()
    scale = expected.float().pow(2).mean().sqrt()
    return float(difference / scale) if float(scale) > 0 else float(difference)


def assert_accurate(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    name: str = "accuracy",
    pcc_threshold: float = 0.999,
    rmse_threshold: float = 0.05,
    linf_threshold: float | None = None,
) -> float:
    """Require matching metadata, finite tensors, PCC, relative RMSE and relative L-inf.

    Three metrics because they fail differently. PCC is a correlation and RMSE is a
    mean, so both average over the whole tensor: eleven wrong rows in 1280 held PCC
    at 0.9996 and lifted relative RMSE only from 1.3e-2 to 1.6e-2, inside the BF16
    output's own noise. Relative L-inf does not average, and on the same data it
    separated cleanly -- 1.6e-2 when correct against 1.3e-1 to 2.7e-1 when a few
    rows were fed the wrong carry. That is the failure mode this layer keeps
    producing, so the peak matters more than the mean.

    Relative L-inf is always reported but gated only when ``linf_threshold`` is
    given, because a peak divided by a whole-tensor RMS
    depends on how concentrated the signal is: correct runs here span 1.4e-2 on a
    convolution carry to 1.6e+0 on a recurrent state, so no single bound is both
    safe and useful. Pass the clean value you measured for the tensor at hand.
    """
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
    rmse = _relative_rmse(expected, actual)
    max_abs = float((expected.float() - actual.float()).abs().max()) if expected.numel() else 0.0
    scale = float(expected.float().pow(2).mean().sqrt()) if expected.numel() else 0.0
    linf = max_abs / scale if scale > 0 else max_abs
    print(expected_summary)
    print(actual_summary)
    print(f"{name}: PCC={pcc:.6f}, rel_RMSE={rmse:.3e}, rel_Linf={linf:.3e}, max_abs={max_abs:.6e}")
    if pcc < pcc_threshold:
        raise AssertionError(f"{name} PCC {pcc:.6f} < {pcc_threshold}")
    if rmse > rmse_threshold:
        raise AssertionError(f"{name} relative RMSE {rmse:.3e} > {rmse_threshold:.3e}")
    if linf_threshold is not None and linf > linf_threshold:
        raise AssertionError(
            f"{name} relative L-inf {linf:.3e} > {linf_threshold:.3e} (max_abs {max_abs:.3e}) -- "
            "a localised fault, typically a few rows given the wrong carry"
        )
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
