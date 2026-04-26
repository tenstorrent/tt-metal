# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for the comp_pcc bug fix (issues #24928 / #25137).

Bug: comp_pcc returned 1.0 (false PASS) when one tensor was constant,
because np.corrcoef produces a fully-masked matrix in that case and
the MaskedConstant fallback incorrectly returned 1.0.

Run with:
    pytest tests/ttnn/unit_tests/test_comp_pcc_fix.py -v -s
"""

import torch
from models.common.utility_functions import comp_pcc


def print_result(label, golden, calculated, pcc_threshold, passed, pcc_value):
    print(f"\n{'='*60}")
    print(f"TEST : {label}")
    print(f"  dtype      : {golden.dtype}")
    print(f"  golden     : {golden.flatten()[:8].tolist()}{'...' if golden.numel() > 8 else ''}")
    print(f"  calculated : {calculated.flatten()[:8].tolist()}{'...' if calculated.numel() > 8 else ''}")
    print(f"  threshold  : {pcc_threshold}")
    print(f"  PCC result : {pcc_value:.6f}" if isinstance(pcc_value, float) else f"  PCC result : {pcc_value}")
    print(f"  passed     : {passed}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# THE ORIGINAL BUG CASES — these SHOULD FAIL (were incorrectly PASSING before)
# ---------------------------------------------------------------------------


def test_golden_constant_calculated_random_should_fail():
    """
    BUG CASE: golden is all-same value, calculated is random.
    PCC is undefined. Must NOT pass at threshold 0.99.
    Before fix: returned 1.0 (false PASS). After fix: allclose fallback → FAIL.
    """
    torch.manual_seed(42)
    golden = torch.full((100,), 1.0)
    calculated = torch.rand((100,))

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("golden=constant, calculated=random [SHOULD FAIL]", golden, calculated, 0.99, passed, pcc_val)

    assert (
        not passed
    ), f"BUG REGRESSION: passed with PCC={pcc_val:.4f} when golden is constant and calculated is random."


def test_calculated_constant_golden_random_should_fail():
    """
    BUG CASE: calculated is all-same value, golden is random. Same bug, other direction.
    """
    torch.manual_seed(0)
    golden = torch.rand((100,))
    calculated = torch.full((100,), 0.5)

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("golden=random, calculated=constant [SHOULD FAIL]", golden, calculated, 0.99, passed, pcc_val)

    assert (
        not passed
    ), f"BUG REGRESSION: passed with PCC={pcc_val:.4f} when calculated is constant and golden is random."


def test_exact_bug_from_issue_24928():
    """
    Exact reproduction from issue #24928.
    assert_with_pcc(torch.full((100,), 1.0), torch.rand((100,)), 1.0) should FAIL.
    """
    torch.manual_seed(7)
    golden = torch.full((100,), 1.0)
    calculated = torch.rand((100,))

    passed, pcc_val = comp_pcc(golden, calculated, pcc=1.0)
    print_result("Issue #24928 exact repro [SHOULD FAIL]", golden, calculated, 1.0, passed, pcc_val)

    assert not passed, f"BUG REGRESSION: issue #24928 exact repro still returns PCC={pcc_val:.4f} and passes."


# ---------------------------------------------------------------------------
# CORRECT BEHAVIOR — these SHOULD PASS after the fix (no regression)
# ---------------------------------------------------------------------------


def test_identical_tensors_should_pass():
    torch.manual_seed(1)
    golden = torch.rand((50,))
    calculated = golden.clone()

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.9999)
    print_result("identical tensors [SHOULD PASS]", golden, calculated, 0.9999, passed, pcc_val)

    assert passed, f"comp_pcc failed on identical tensors. PCC={pcc_val:.6f}"


def test_both_constant_same_value_should_pass():
    golden = torch.full((30,), 3.14)
    calculated = torch.full((30,), 3.14)

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("both constant same value [SHOULD PASS]", golden, calculated, 0.99, passed, pcc_val)

    assert passed, f"comp_pcc failed when both tensors are constant with same value. PCC={pcc_val:.6f}"


def test_both_constant_different_values_should_fail():
    golden = torch.full((30,), 1.0)
    calculated = torch.full((30,), 2.0)

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("both constant different values [SHOULD FAIL]", golden, calculated, 0.99, passed, pcc_val)

    assert not passed, f"comp_pcc passed when both tensors are constant with different values. PCC={pcc_val:.6f}"


def test_highly_correlated_should_pass():
    torch.manual_seed(2)
    golden = torch.randn((200,))
    calculated = golden + torch.randn((200,)) * 0.001

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("highly correlated small noise [SHOULD PASS]", golden, calculated, 0.99, passed, pcc_val)

    assert passed, f"comp_pcc failed on highly correlated tensors. PCC={pcc_val:.6f}"


def test_uncorrelated_random_should_fail():
    torch.manual_seed(3)
    golden = torch.randn((200,))
    calculated = torch.randn((200,))

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("uncorrelated random [SHOULD FAIL]", golden, calculated, 0.99, passed, pcc_val)

    assert not passed, f"comp_pcc passed on uncorrelated tensors. PCC={pcc_val:.6f}"


def test_both_nan_should_pass():
    golden = torch.full((20,), float("nan"))
    calculated = torch.full((20,), float("nan"))

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("both all-NaN [SHOULD PASS by convention]", golden, calculated, 0.99, passed, pcc_val)

    assert passed, f"comp_pcc failed when both tensors are all NaN. PCC={pcc_val}"


def test_one_nan_should_fail():
    torch.manual_seed(4)
    golden = torch.full((20,), float("nan"))
    calculated = torch.rand((20,))

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("one all-NaN [SHOULD FAIL]", golden, calculated, 0.99, passed, pcc_val)

    assert not passed, f"comp_pcc passed when one tensor is all NaN. PCC={pcc_val}"


def test_2d_constant_should_fail():
    torch.manual_seed(5)
    golden = torch.full((8, 16), 0.0)
    calculated = torch.rand((8, 16))

    passed, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
    print_result("2D constant vs random [SHOULD FAIL]", golden, calculated, 0.99, passed, pcc_val)

    assert not passed, f"BUG REGRESSION: 2D constant vs random passed with PCC={pcc_val:.4f}"
