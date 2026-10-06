# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the heavyweight golden's mismatch report.

No kernel, no device. The report has to agree with ``passed_test`` about which datums
fail, or it ranks noise above the datum that actually broke the test. Non-finite values
are where that is easiest to get wrong: NaN and ``inf - inf`` survive a subtraction as
NaN, which compares false against everything.
"""

import math
import re

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generator.heavyweight.mismatch import describe_mismatch
from helpers.utils import _MXFP_COMPARE_PARAMS, _mxfp_block_aware_compare

N = 64
SUMMARY = re.compile(
    r"(\d+) / (\d+) datums differ(?:, of which (\d+) (?:exceed|are outside))?"
)


def _summary(golden, actual, output_format=DataFormat.MxFp8R):
    report = describe_mismatch(golden, actual, output_format=output_format)
    differ, _, over = SUMMARY.search(report).groups()
    return report, int(differ), None if over is None else int(over)


def _pair(golden_value, actual_value):
    golden = torch.ones(N)
    actual = torch.ones(N)
    golden[5], actual[5] = golden_value, actual_value
    return golden, actual


NAN, INF = math.nan, math.inf


@pytest.mark.parametrize(
    "golden_value, actual_value",
    [(NAN, 1.0), (1.0, NAN), (INF, 1.0), (INF, -INF), (NAN, INF)],
    ids=["nan_golden", "nan_device", "inf_vs_finite", "inf_vs_minus_inf", "nan_vs_inf"],
)
def test_a_non_finite_mismatch_is_counted_and_listed(golden_value, actual_value):
    report, differ, over = _summary(*_pair(golden_value, actual_value))
    assert (differ, over) == (1, 1)
    worst_row = report.splitlines()[4].split()
    assert worst_row[0] == "5"


@pytest.mark.parametrize("value", [NAN, INF, -INF], ids=["nan", "inf", "minus_inf"])
def test_matching_non_finites_agree(value):
    _, differ, over = _summary(*_pair(value, value))
    assert (differ, over) == (0, 0)


def test_a_non_mx_format_counts_non_finite_mismatches_too():
    """A NaN golden against a finite device value is a failure, not just a
    difference: isclose rejects it and the both-NaN carve-out does not apply."""
    _, differ, over = _summary(*_pair(NAN, 1.0), output_format=DataFormat.Float16_b)
    assert (differ, over) == (1, 1)


@pytest.mark.parametrize(
    "output_format", list(_MXFP_COMPARE_PARAMS), ids=lambda f: f.name
)
def test_the_failure_count_matches_the_comparator(output_format):
    """Finite noise of a few steps plus planted non-finite pairs, both matching and not."""
    g = torch.Generator().manual_seed(0)
    golden = torch.randn(1024, generator=g) * 4
    actual = golden * (1 + torch.randn(1024, generator=g) * 0.2)
    golden[[3, 40, 77]] = torch.tensor([NAN, INF, INF])
    actual[[3, 40, 77]] = torch.tensor([NAN, INF, -INF])  # match, match, mismatch
    golden[100], actual[100] = 2.0, NAN

    mantissa_bits, max_steps, max_normal, min_subnormal = _MXFP_COMPARE_PARAMS[
        output_format
    ]
    valid = _mxfp_block_aware_compare(
        golden, actual, mantissa_bits, max_steps, max_normal, min_subnormal
    )
    _, _, over = _summary(golden, actual, output_format)
    assert over == int((~valid).sum())


# ---------------------------------------------------------------------------
# The no-lattice branch: Float16/Float16_b, which is what the device tests use


def test_a_format_without_a_lattice_counts_the_datums_that_actually_fail():
    """Differing is not failing. `passed_test` judges these formats with
    torch.isclose at the format's tolerance, so the report has to count the
    same thing or the top of its table reads as the cause when it is noise."""
    golden = torch.ones(N, dtype=torch.bfloat16)
    actual = golden.clone()
    actual[0] = 2.0  # outside atol=0.05
    actual[1] = 1.0 + 2**-7  # one bf16 ULP: differs, but passes
    report, differ, over = _summary(golden, actual, output_format=DataFormat.Float16_b)
    assert (differ, over) == (2, 1)
    assert "atol=0.05" in report and "rtol=0.05" in report
    assert "threshold 0.99" in report


def test_the_no_lattice_branch_says_the_ranking_can_mislead():
    """With rtol in play a large passing datum can outrank a small failing one,
    so the report says so rather than letting the order imply a verdict."""
    golden = torch.ones(N, dtype=torch.bfloat16)
    actual = golden.clone()
    actual[0] = 1.5
    report, _, _ = _summary(golden, actual, output_format=DataFormat.Float16_b)
    assert "by absolute error" in report
    assert "can outrank" in report
