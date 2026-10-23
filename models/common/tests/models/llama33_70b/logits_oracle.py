# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Numerical oracle helpers for Llama-3.3 batched-prefill tests."""

from __future__ import annotations

import torch


def assert_rowwise_logits_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    min_row_pcc: float,
    max_abs: float,
    require_exact_top1: bool = True,
    max_top1_mismatches: int | None = None,
    expected_top1_in_actual_topk: int | None = None,
    min_topk_overlap: int | None = None,
    isclose_atol: float | None = None,
    isclose_rtol: float | None = None,
    max_isclose_failure_fraction: float | None = None,
) -> None:
    """Require every batch row to preserve logits shape, quality, and ranking."""

    if require_exact_top1 and (max_top1_mismatches is not None or expected_top1_in_actual_topk is not None):
        raise ValueError("exact top-1 and top-k containment are mutually exclusive")
    if min_topk_overlap is not None and expected_top1_in_actual_topk is None:
        raise ValueError("min_topk_overlap requires expected_top1_in_actual_topk")
    isclose_options = (isclose_atol, isclose_rtol, max_isclose_failure_fraction)
    if any(value is not None for value in isclose_options) and not all(value is not None for value in isclose_options):
        raise ValueError("isclose_atol, isclose_rtol, and max_isclose_failure_fraction must be supplied together")

    if actual.shape != expected.shape:
        raise AssertionError(f"logits shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    if actual.ndim < 2:
        raise AssertionError(f"logits must have batch and vocabulary dimensions, got {tuple(actual.shape)}")

    actual_rows = actual.detach().float().reshape(actual.shape[0], -1)
    expected_rows = expected.detach().float().reshape(expected.shape[0], -1)
    if not torch.isfinite(actual_rows).all() or not torch.isfinite(expected_rows).all():
        raise AssertionError("logits contain non-finite values")

    actual_centered = actual_rows - actual_rows.mean(dim=1, keepdim=True)
    expected_centered = expected_rows - expected_rows.mean(dim=1, keepdim=True)
    denominator = actual_centered.norm(dim=1) * expected_centered.norm(dim=1)
    numerator = (actual_centered * expected_centered).sum(dim=1)
    row_pcc = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.where(
            torch.all(actual_rows == expected_rows, dim=1),
            torch.ones_like(denominator),
            torch.zeros_like(denominator),
        ),
    )
    row_max_abs = (actual_rows - expected_rows).abs().amax(dim=1)
    actual_top1 = actual_rows.argmax(dim=1)
    expected_top1 = expected_rows.argmax(dim=1)

    failures = []
    bad_pcc = torch.nonzero(row_pcc < min_row_pcc, as_tuple=False).reshape(-1)
    if bad_pcc.numel():
        failures.append(
            f"row PCC below {min_row_pcc}: "
            + ", ".join(f"row {row}: {row_pcc[row].item():.8f}" for row in bad_pcc.tolist())
        )
    bad_max_abs = torch.nonzero(row_max_abs > max_abs, as_tuple=False).reshape(-1)
    if bad_max_abs.numel():
        failures.append(
            f"row max-abs above {max_abs}: "
            + ", ".join(f"row {row}: {row_max_abs[row].item():.8f}" for row in bad_max_abs.tolist())
        )
    if require_exact_top1 and not torch.equal(actual_top1, expected_top1):
        disagreement = torch.nonzero(actual_top1 != expected_top1, as_tuple=False)
        failures.append(f"top-1 mismatch at {disagreement.tolist()}")
    if max_top1_mismatches is not None:
        mismatch_count = int((actual_top1 != expected_top1).sum().item())
        if mismatch_count > int(max_top1_mismatches):
            failures.append(f"top-1 mismatch count {mismatch_count} exceeds {max_top1_mismatches}")
    if expected_top1_in_actual_topk is not None:
        topk = int(expected_top1_in_actual_topk)
        if topk <= 0 or topk > actual_rows.shape[1]:
            raise ValueError(f"top-k must be in [1, {actual_rows.shape[1]}], got {topk}")
        actual_topk = actual_rows.topk(topk, dim=1).indices
        expected_topk = expected_rows.topk(topk, dim=1).indices
        expected_top1_rows = expected_top1.unsqueeze(1)
        missing_top1 = torch.nonzero(~(actual_topk == expected_top1_rows).any(dim=1), as_tuple=False).reshape(-1)
        if missing_top1.numel():
            failures.append(f"expected top-1 missing from actual top-{topk} at rows {missing_top1.tolist()}")
        if min_topk_overlap is not None:
            minimum = int(min_topk_overlap)
            if minimum <= 0 or minimum > topk:
                raise ValueError(f"min_topk_overlap must be in [1, {topk}], got {minimum}")
            overlaps = (actual_topk.unsqueeze(2) == expected_topk.unsqueeze(1)).any(dim=2).sum(dim=1)
            bad_overlap = torch.nonzero(overlaps < minimum, as_tuple=False).reshape(-1)
            if bad_overlap.numel():
                failures.append(
                    f"top-{topk} overlap below {minimum}: "
                    + ", ".join(f"row {row}: {overlaps[row].item()}" for row in bad_overlap.tolist())
                )
    if max_isclose_failure_fraction is not None:
        close = torch.isclose(actual_rows, expected_rows, atol=float(isclose_atol), rtol=float(isclose_rtol))
        failure_fraction = float((~close).float().mean().item())
        if failure_fraction > float(max_isclose_failure_fraction):
            row_fractions = (~close).float().mean(dim=1)
            failures.append(
                f"isclose failure fraction {failure_fraction:.8f} exceeds {max_isclose_failure_fraction}; "
                + ", ".join(f"row {row}: {value.item():.8f}" for row, value in enumerate(row_fractions))
            )

    if failures:
        raise AssertionError("logits parity failed; " + "; ".join(failures))
