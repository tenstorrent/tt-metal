# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA accuracy against an FP64 golden, judged by a FlashKDA-anchored error budget.

The quantity is the normalized RMS error ``‖actual − golden‖₂ / ‖golden‖₂`` (NRMSE), taken
per slice — per head for the recurrent state, per token for the output — rather than over
the whole tensor, so one bad head or one late token is not averaged away. All arithmetic is
FP64. A slice whose golden is near zero would report its own magnitude rather than an
error, so every denominator is floored at ``floor_fraction`` of the median slice norm.

The gate is relative to a budget, normally FlashKDA's NRMSE against the same golden on the
same input and slice: a slice passes when ``error <= (1 + margin) * budget + epsilon``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

FLOOR_FRACTION = 0.01


def _as_float64(name: str, tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().to(torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} has {int((~torch.isfinite(tensor)).sum())} non-finite values")
    return tensor


def slice_nrmse(
    actual: torch.Tensor,
    golden: torch.Tensor,
    keep_dims: Sequence[int],
    floor_fraction: float = FLOOR_FRACTION,
) -> torch.Tensor:
    """NRMSE reduced over every dimension except ``keep_dims``, one value per kept slice.

    ``keep_dims=()`` gives the whole-tensor NRMSE. Non-finite ``actual`` raises.
    """
    if actual.shape != golden.shape:
        raise ValueError(f"actual shape {tuple(actual.shape)} != golden shape {tuple(golden.shape)}")
    actual = _as_float64("actual", actual)
    golden = _as_float64("golden", golden)
    keep = sorted(dim % golden.ndim for dim in keep_dims)
    reduce = tuple(dim for dim in range(golden.ndim) if dim not in keep)
    error = (actual - golden).square().sum(dim=reduce).sqrt()
    norm = golden.square().sum(dim=reduce).sqrt()
    floor = floor_fraction * norm.median() if norm.numel() > 1 else torch.zeros((), dtype=torch.float64)
    return error / torch.maximum(norm, floor).clamp_min(torch.finfo(torch.float64).tiny)


def position_profile(
    actual: torch.Tensor,
    golden: torch.Tensor,
    token_dim: int,
    bucket: int,
) -> torch.Tensor:
    """NRMSE of consecutive ``bucket``-token windows along ``token_dim``; the last may be partial."""
    tokens = golden.shape[token_dim]
    return torch.stack(
        [
            slice_nrmse(
                actual.narrow(token_dim, start, min(bucket, tokens - start)),
                golden.narrow(token_dim, start, min(bucket, tokens - start)),
                keep_dims=(),
            )
            for start in range(0, tokens, bucket)
        ]
    )


@dataclass(frozen=True)
class GateResult:
    """Per-slice verdict of a measured error against its budget."""

    error: torch.Tensor
    limit: torch.Tensor

    @property
    def passed(self) -> bool:
        return bool((self.error <= self.limit).all())

    def describe(self, name: str) -> str:
        worst = int((self.error / self.limit).argmax())
        verdict = "PASS" if self.passed else f"FAIL in {int((self.error > self.limit).sum())} of {self.error.numel()}"
        return (
            f"{name}: {verdict}; worst slice {worst} error {self.error.flatten()[worst]:.3e} "
            f"limit {self.limit.flatten()[worst]:.3e}"
        )


def gate(error: torch.Tensor, budget: torch.Tensor, margin: float, epsilon: float) -> GateResult:
    """Compare per-slice errors with ``(1 + margin) * budget + epsilon``; shapes must match."""
    if error.shape != budget.shape:
        raise ValueError(f"error shape {tuple(error.shape)} != budget shape {tuple(budget.shape)}")
    if margin < 0 or epsilon < 0:
        raise ValueError(f"margin {margin} and epsilon {epsilon} must be nonnegative")
    return GateResult(error=error.to(torch.float64), limit=(1 + margin) * budget.to(torch.float64) + epsilon)
