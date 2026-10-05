# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""R9 accuracy gates for the GDN baseline, on the linear-attention family's shared ``assert_accurate``.

Gates per tensor: finite values and matching metadata, PCC >= 0.9995 (the requirement), relative RMSE <= 0.0316
and output norm ratio within 2%. PCC is invariant to a global scale; relative RMSE ``||a - e|| / ||e||`` is not,
and 0.0316 = sqrt(2 * (1 - 0.9995)) is the relative error that unbiased noise at the PCC bar produces, so it adds
scale sensitivity without tightening the bar. The norm ratio isolates a pure scale error. Relative L-inf (peak
error over the reference RMS) is reported, not gated: its clean value depends on how concentrated each tensor is
(see ``assert_accurate``), and this baseline has no measured clean value to derive a bound from.
"""

from __future__ import annotations

import torch

from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import _pcc, assert_accurate

PCC_THRESHOLD = 0.9995
RMSE_THRESHOLD = (2 * (1 - PCC_THRESHOLD)) ** 0.5
NORM_RATIO_TOLERANCE = 0.02


def measure(expected: torch.Tensor, actual: torch.Tensor, name: str) -> dict:
    """Gate one tensor; returns metrics and failures instead of raising, so a sweep records every tensor."""
    expected = expected.float()
    actual = actual.float()
    result = {"name": name, "shape": list(expected.shape), "failures": []}
    try:
        assert_accurate(expected, actual, name=name, pcc_threshold=PCC_THRESHOLD, rmse_threshold=RMSE_THRESHOLD)
    except AssertionError as error:
        result["failures"].append(str(error).splitlines()[0])
    finite = bool(torch.isfinite(actual).all())
    result["finite"] = finite
    if finite:
        e_norm = float(expected.norm())
        difference = expected - actual
        scale = float(expected.pow(2).mean().sqrt())
        result["rel_rmse"] = float(difference.norm()) / e_norm if e_norm > 0 else float(difference.norm())
        result["max_abs"] = float(difference.abs().max())
        result["rel_linf"] = result["max_abs"] / scale if scale > 0 else result["max_abs"]
        result["norm_ratio"] = float(actual.norm()) / e_norm if e_norm > 0 else float("nan")
        result["pcc"] = _pcc(expected, actual)
        if not abs(result["norm_ratio"] - 1.0) <= NORM_RATIO_TOLERANCE:
            result["failures"].append(
                f"{name} norm ratio {result['norm_ratio']:.4f} outside 1 ± {NORM_RATIO_TOLERANCE}"
            )
    result["passed"] = not result["failures"]
    return result


def per_head_rel_rmse(expected: torch.Tensor, actual: torch.Tensor) -> list[float]:
    """Relative RMSE per leading index (V head of a [Nv, K, V] state); a head whose reference is ~0 (strong decay)
    reports its absolute RMS error instead."""
    expected = expected.float().flatten(1)
    actual = actual.float().flatten(1)
    numerator = (expected - actual).norm(dim=1)
    denominator = expected.norm(dim=1)
    return [float(n / d) if d > 1e-30 else float(n) for n, d in zip(numerator, denominator)]
