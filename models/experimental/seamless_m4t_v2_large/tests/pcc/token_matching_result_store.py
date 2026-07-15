# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared in-process store for token-matching sweep summary rows.

Kept free of ttnn/torch imports so pytest conftest and test helpers always share one list
(--import-mode=importlib can otherwise duplicate heavy modules).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenMatchingResult:
    label: str
    steps: int
    top1_pct: float
    top5_pct: float
    top1_threshold_pct: float
    top5_threshold_pct: float
    passed: bool


_RESULTS: list[TokenMatchingResult] = []


def clear_token_matching_results() -> None:
    _RESULTS.clear()


def get_token_matching_results() -> list[TokenMatchingResult]:
    return list(_RESULTS)


def record_token_matching_result(
    *,
    label: str,
    steps: int,
    top1_pct: float,
    top5_pct: float,
    top1_threshold: float,
    top5_threshold: float,
) -> None:
    top1_thr_pct = top1_threshold * 100.0
    top5_thr_pct = top5_threshold * 100.0
    _RESULTS.append(
        TokenMatchingResult(
            label=label,
            steps=steps,
            top1_pct=top1_pct,
            top5_pct=top5_pct,
            top1_threshold_pct=top1_thr_pct,
            top5_threshold_pct=top5_thr_pct,
            passed=top1_pct >= top1_thr_pct and top5_pct >= top5_thr_pct,
        )
    )


def print_token_matching_summary() -> None:
    """Fallback print when pytest terminal reporter is unavailable."""
    results = get_token_matching_results()
    if not results:
        return
    print("\n" + "=" * 72)
    print("Seamless M4T v2 E2E token matching summary")
    print("=" * 72)
    header = f"{'Label':<18} {'Steps':>5}  {'Top1':>7}  {'Top5':>7}  {'Thresholds':>12}  {'Status':>6}"
    print(header)
    print("-" * len(header))
    for row in results:
        thresholds = f"{row.top1_threshold_pct:.0f}%/{row.top5_threshold_pct:.0f}%"
        status = "PASS" if row.passed else "FAIL"
        print(
            f"{row.label:<18} {row.steps:>5}  {row.top1_pct:>6.2f}%  {row.top5_pct:>6.2f}%  "
            f"{thresholds:>12}  {status:>6}"
        )
    passed = sum(1 for r in results if r.passed)
    print("-" * len(header))
    print(f"Total: {passed}/{len(results)} passed\n")
