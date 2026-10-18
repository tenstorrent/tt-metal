# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only acceptance metrics; never call inside a measured device pass."""
import math

import torch


def _flat_chunk(tensor, start, stop):
    if tensor.is_contiguous():
        return tensor.view(-1)[start:stop]
    # Avoid a whole-tensor contiguous copy for permuted reference states.
    flat_indices = torch.arange(start, stop, dtype=torch.int64)
    indices = []
    for size in reversed(tensor.shape):
        indices.append(flat_indices.remainder(size))
        flat_indices = torch.div(flat_indices, size, rounding_mode="floor")
    return tensor[tuple(reversed(indices))]


@torch.no_grad()
def pcc_metrics(expected, actual, *, chunk_size=262144):
    """Return centered FP64 PCC and error diagnostics using bounded buffers.

    Both inputs must be real CPU tensors with the same number of elements.
    Logical flattening preserves order even for noncontiguous tensors. Means
    are computed first, then centered covariance in a second streaming pass;
    this avoids cancellation from subtracting large raw second moments.
    Float32 corrcoef can falsely pass the0.995 gate at target B32 prefill sizes.
    Nonfinite or zero-variance inputs fail explicitly instead of hiding NaNs.
    """
    if expected.device.type != "cpu" or actual.device.type != "cpu":
        raise ValueError("Acceptance metrics require CPU tensors outside device execution")
    if expected.is_complex() or actual.is_complex():
        raise ValueError("PCC requires real-valued tensors")
    n = expected.numel()
    if n != actual.numel() or n < 2 or chunk_size < 1:
        raise ValueError("PCC requires matching nontrivial inputs and a positive chunk size")
    sum_a = sum_b = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        a = _flat_chunk(expected, start, stop).to(torch.float64)
        b = _flat_chunk(actual, start, stop).to(torch.float64)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("PCC inputs must be finite")
        sum_a += a.sum().item()
        sum_b += b.sum().item()
    mean_a, mean_b = sum_a / n, sum_b / n
    aa = bb = ab = squared_error = max_error = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        # Out-of-place centering is required when a FP64 slice aliases input.
        a = _flat_chunk(expected, start, stop).to(torch.float64)
        b = _flat_chunk(actual, start, stop).to(torch.float64)
        diff = a - b
        squared_error += (diff * diff).sum().item()
        max_error = max(max_error, diff.abs().max().item())
        a = a - mean_a
        b = b - mean_b
        aa += (a * a).sum().item()
        bb += (b * b).sum().item()
        ab += (a * b).sum().item()
    if aa == 0 or bb == 0:
        raise ValueError("PCC is undefined for constant tensors")
    score = ab / math.sqrt(aa * bb)
    if not math.isfinite(score):
        raise ValueError("PCC calculation produced a nonfinite value")
    return {
        "pcc": score,
        "pcc_method": "two_pass_centered_fp64",
        "pcc_elements": n,
        "pcc_chunk_size": chunk_size,
        "rmse": math.sqrt(squared_error / n),
        "max_abs_error": max_error,
        "exact_equal": max_error == 0.0,
    }


def pcc(expected, actual):
    return pcc_metrics(expected, actual)["pcc"]
