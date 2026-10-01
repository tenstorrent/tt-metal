# SPDX-License-Identifier: Apache-2.0
"""Numerical comparison independent of either accelerator runtime."""
from dataclasses import asdict, dataclass
import math
import torch


@dataclass(frozen=True)
class TensorMetrics:
    passed: bool
    pcc: float | None
    relative_rms: float | None
    max_abs: float | None
    reason: str | None = None

    def to_dict(self):
        return asdict(self)


def compare_tensors(reference, actual, *, min_pcc=0.99, max_relative_rms=None, max_abs=None):
    """Compare all elements; invalid/empty/shape-mismatched tensors always fail.

    Equal constants have PCC 1; unequal constants have PCC 0, even when both
    arrays have the same zero variance. Optional error bounds catch scaling or
    offset errors which Pearson correlation alone cannot detect.
    """
    if not math.isfinite(min_pcc) or not -1 <= min_pcc <= 1:
        raise ValueError("min_pcc must be finite and in [-1, 1]")
    for bound in (max_relative_rms, max_abs):
        if bound is not None and (not math.isfinite(bound) or bound < 0):
            raise ValueError("error bounds must be finite and nonnegative")
    reference, actual = torch.as_tensor(reference), torch.as_tensor(actual)
    if reference.shape != actual.shape:
        return TensorMetrics(False, None, None, None, "shape mismatch")
    if reference.numel() == 0:
        return TensorMetrics(False, None, None, None, "empty tensor")
    if reference.is_complex() or actual.is_complex():
        raise ValueError("complex tensors are unsupported")
    ref = reference.detach().cpu().double().flatten()
    out = actual.detach().cpu().double().flatten()
    if not torch.isfinite(ref).all() or not torch.isfinite(out).all():
        return TensorMetrics(False, None, None, None, "nonfinite tensor")
    # Scale first so finite large values do not overflow variance or dot products.
    scale = max(ref.abs().max().item(), out.abs().max().item()) or 1.0
    ref_scaled, out_scaled = ref / scale, out / scale
    error = out_scaled - ref_scaled
    abs_error = error.abs().max().item() * scale
    if not math.isfinite(abs_error):
        return TensorMetrics(False, None, None, None, "error magnitude exceeds float64 range")
    ref_rms = ref_scaled.square().mean().sqrt().item()
    error_rms = error.square().mean().sqrt().item()
    relative = error_rms / ref_rms if ref_rms else (0.0 if error_rms == 0 else None)
    x, y = ref_scaled - ref_scaled.mean(), out_scaled - out_scaled.mean()
    norm = x.norm().item() * y.norm().item()
    pcc = max(-1.0, min(1.0, torch.dot(x, y).item() / norm)) if norm else (1.0 if torch.equal(ref, out) else 0.0)
    if relative is not None and not math.isfinite(relative):
        return TensorMetrics(False, pcc, None, abs_error, "relative RMS exceeds float64 range")
    passed = pcc >= min_pcc
    if max_relative_rms is not None:
        passed &= relative is not None and relative <= max_relative_rms
    if max_abs is not None:
        passed &= abs_error <= max_abs
    return TensorMetrics(bool(passed), pcc, relative, abs_error, None if passed else "numerical threshold failed")
