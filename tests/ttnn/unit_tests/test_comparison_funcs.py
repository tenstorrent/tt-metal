# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Host-only tests for ttnn/tt_lib/_internal/comparison_funcs.py (#62403).

The PCC must match a float64 np.corrcoef of the same data, the nan / constant /
single-element special cases must keep their results, and comp_equal and
comp_allclose must only build the atol/rtol/PCC message when the check fails.
"""

import math
import re

import numpy as np
import pytest
import torch

from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs


def _noisy(shape, dtype, noise=0.01, seed=0):
    gen = torch.Generator().manual_seed(seed)
    golden = torch.randn(shape, generator=gen)
    calculated = golden + noise * torch.randn(shape, generator=gen)
    return golden.to(dtype), calculated.to(dtype)


def _pcc(golden, calculated):
    return comparison_funcs.get_atol_rtol_pcc(golden, calculated)[2]


def _reference_pcc(golden, calculated):
    """float64 np.corrcoef after the nan/inf -> 0 replacement comparison_funcs applies."""
    golden = torch.nan_to_num(golden.double(), nan=0.0, posinf=0.0, neginf=0.0)
    calculated = torch.nan_to_num(calculated.double(), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.corrcoef(golden.flatten().numpy(), calculated.flatten().numpy())[0, 1])


def _fail_if_called(*args, **kwargs):
    raise AssertionError("get_atol_rtol_pcc must not run when the check passes")


class TestPccValue:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize("noise", [0.01, 1.0, 100.0], ids=["high_pcc", "mid_pcc", "uncorrelated"])
    def test_matches_float64_corrcoef(self, dtype, noise):
        golden, calculated = _noisy((64, 128), dtype, noise=noise)
        pcc = _pcc(golden, calculated)
        assert isinstance(pcc, float)
        assert pcc == pytest.approx(_reference_pcc(golden, calculated), abs=1e-12)

    def test_anti_correlated(self):
        golden, calculated = _noisy((4096,), torch.bfloat16)
        pcc = _pcc(golden, -calculated)
        assert pcc < -0.99
        assert pcc == pytest.approx(_reference_pcc(golden, -calculated), abs=1e-12)

    def test_nan_and_inf_are_zeroed_before_correlating(self):
        golden, calculated = _noisy((64, 128), torch.float32)
        golden.view(-1)[::97] = float("nan")
        calculated.view(-1)[5::89] = float("inf")
        calculated.view(-1)[7::101] = float("-inf")
        assert _pcc(golden, calculated) == pytest.approx(_reference_pcc(golden, calculated), abs=1e-12)

    def test_inf_of_both_signs(self):
        # +inf and -inf in one tensor sum to nan; that must still take the zeroing path.
        golden, calculated = _noisy((256,), torch.float32)
        golden[1] = float("inf")
        golden[2] = float("-inf")
        assert _pcc(golden, calculated) == pytest.approx(_reference_pcc(golden, calculated), abs=1e-12)

    def test_integer_inputs(self):
        gen = torch.Generator().manual_seed(0)
        golden = torch.randint(-100, 100, (64, 64), generator=gen, dtype=torch.int32)
        calculated = golden + torch.randint(-2, 3, (64, 64), generator=gen, dtype=torch.int32)
        assert _pcc(golden, calculated) == pytest.approx(_reference_pcc(golden, calculated), abs=1e-12)

    def test_complex_inputs(self):
        golden, calculated = _noisy((64, 64), torch.complex64)
        expected = _reference_pcc(torch.view_as_real(golden), torch.view_as_real(calculated))
        assert _pcc(golden, calculated) == pytest.approx(expected, abs=1e-12)

    def test_non_contiguous_inputs(self):
        golden, calculated = _noisy((64, 128), torch.float32)
        assert _pcc(golden.t(), calculated.t()) == pytest.approx(
            _pcc(golden.t().contiguous(), calculated.t().contiguous()), abs=1e-15
        )

    def test_same_numel_different_rank(self):
        golden, calculated = _noisy((1, 1, 32, 64), torch.bfloat16)
        calculated = calculated.reshape(32, 64)
        assert _pcc(golden, calculated) == pytest.approx(_reference_pcc(golden, calculated), abs=1e-12)

    def test_large_magnitude_float32(self):
        golden, calculated = _noisy((64, 128), torch.float32)
        golden, calculated = golden * 1e19, calculated * 1e19
        assert _pcc(golden, calculated) == pytest.approx(_reference_pcc(golden, calculated), abs=1e-12)

    def test_large_magnitude_float64(self):
        # Without the n - 1 normalization, the variance product of 1e76 values overflows float64.
        golden, calculated = _noisy((4096,), torch.float64)
        golden, calculated = golden * 1e76, calculated * 1e76
        assert _pcc(golden, calculated) == pytest.approx(_reference_pcc(golden, calculated), abs=1e-12)

    def test_tiny_magnitude_uncorrelated_float32_fails(self):
        # float32 products of 1e-25 values underflow to zero; the float64 PCC must not.
        golden, calculated = _noisy((4096,), torch.float32, noise=100.0)
        golden, calculated = golden * 1e-25, calculated * 1e-25
        passing, output = comparison_funcs.comp_pcc(golden, calculated)
        assert not passing, output

    def test_non_finite_pcc_falls_back_to_allclose(self):
        # float64 values of 1e-160 underflow the variance product to zero.
        golden, calculated = _noisy((4096,), torch.float64, noise=100.0)
        golden, calculated = golden * 1e-160, calculated * 1e-160
        pcc = _pcc(golden, calculated)
        assert pcc == float(torch.allclose(golden, calculated, rtol=1e-05, atol=1e-04))


class TestSpecialCases:
    def test_both_all_nan(self):
        nan = torch.full((32, 32), float("nan"))
        assert _pcc(nan, nan.clone()) == 1.0
        assert comparison_funcs.comp_pcc(nan, nan.clone())[0]

    @pytest.mark.parametrize("nan_side", ["golden", "calculated"])
    def test_one_all_nan(self, nan_side):
        nan, values = torch.full((32, 32), float("nan")), torch.randn(32, 32)
        golden, calculated = (nan, values) if nan_side == "golden" else (values, nan)
        assert _pcc(golden, calculated) == 0.0
        assert not comparison_funcs.comp_pcc(golden, calculated)[0]

    def test_identical(self):
        golden, _ = _noisy((64, 64), torch.bfloat16)
        assert _pcc(golden, golden.clone()) == 1.0

    def test_identical_with_nan_at_same_positions(self):
        golden, _ = _noisy((64, 64), torch.bfloat16)
        golden.view(-1)[::7] = float("nan")
        assert _pcc(golden, golden.clone()) == 1.0

    @pytest.mark.parametrize(
        "calculated, atol, expected",
        [(1.0 + 1e-6, 1e-4, 1.0), (2.0, 1e-4, 0.0), (1.0 + 5e-4, 1e-4, 0.0), (1.0 + 5e-4, 1e-3, 1.0)],
    )
    def test_single_element_uses_callers_tolerance(self, calculated, atol, expected):
        result = comparison_funcs.get_atol_rtol_pcc(torch.tensor([1.0]), torch.tensor([calculated]), atol=atol)
        assert result[2] == expected

    @pytest.mark.parametrize("constant_side", ["golden", "calculated"])
    def test_constant_vs_random_fails(self, constant_side):
        constant, values = torch.ones(32, 32), torch.randn(32, 32)
        golden, calculated = (constant, values) if constant_side == "golden" else (values, constant)
        assert _pcc(golden, calculated) == 0.0
        assert not comparison_funcs.comp_pcc(golden, calculated)[0]

    def test_constant_vs_same_constant(self):
        assert _pcc(torch.full((32, 32), 3.0), torch.full((32, 32), 3.0)) == 1.0

    def test_constant_vs_other_constant(self):
        assert _pcc(torch.full((32, 32), 3.0), torch.full((32, 32), 3.5)) == 0.0

    @pytest.mark.parametrize("atol, expected", [(1e-3, True), (1e-5, False)])
    def test_allclose_and_pcc_forwards_tolerances(self, atol, expected):
        golden, calculated = torch.full((32,), 1.0), torch.full((32,), 1.0 + 5e-4)
        assert comparison_funcs.comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=atol)[0] == expected


class TestInputs:
    def test_inputs_are_not_modified(self):
        golden, calculated = _noisy((64, 64), torch.float64)
        golden[0, 0] = float("nan")
        calculated[1, 1] = float("inf")
        golden_copy, calculated_copy = golden.clone(), calculated.clone()
        _pcc(golden, calculated)
        assert torch.equal(golden.isnan(), golden_copy.isnan())
        assert torch.equal(golden.nan_to_num(), golden_copy.nan_to_num())
        assert torch.equal(calculated, calculated_copy)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_requires_grad_inputs(self, dtype):
        golden, calculated = _noisy((64, 64), dtype)
        expected = _pcc(golden, calculated)
        assert _pcc(golden.clone().requires_grad_(True), calculated) == expected
        assert _pcc(golden.clone().requires_grad_(True) * 1.0, calculated) == expected

    def test_comp_pcc_casts_calculated_to_golden_dtype(self):
        golden, calculated = _noisy((64, 64), torch.float32)
        passing, output = comparison_funcs.comp_pcc(golden, calculated.bfloat16())
        assert passing, output


class TestMessages:
    MESSAGE = re.compile(r"^Max ATOL Delta: \S+, Max RTOL Delta: \S+, PCC: (\S+)")

    def test_comp_pcc_message_format(self):
        golden, calculated = _noisy((64, 64), torch.bfloat16)
        passing, output = comparison_funcs.comp_pcc(golden, calculated)
        assert passing
        match = self.MESSAGE.fullmatch(output)
        assert match, output
        assert float(match.group(1)) == _pcc(golden, calculated)

        passing, output = comparison_funcs.comp_pcc(golden, calculated, pcc=0.99999)
        assert not passing
        assert self.MESSAGE.match(output) and output.endswith(", PCC check failed"), output

    def test_comp_equal_pass_skips_pcc(self, monkeypatch):
        golden, _ = _noisy((64, 64), torch.bfloat16)
        monkeypatch.setattr(comparison_funcs, "get_atol_rtol_pcc", _fail_if_called)
        assert comparison_funcs.comp_equal(golden, golden.clone()) == (True, "Equal check passed")

    def test_comp_equal_fail_message(self):
        golden, calculated = _noisy((64, 64), torch.bfloat16)
        passing, output = comparison_funcs.comp_equal(golden, calculated)
        assert not passing
        assert self.MESSAGE.match(output) and output.endswith(", Equal check failed"), output

    def test_comp_equal_unsqueezes_golden(self, monkeypatch):
        golden, _ = _noisy((32, 32), torch.float32)
        monkeypatch.setattr(comparison_funcs, "get_atol_rtol_pcc", _fail_if_called)
        assert comparison_funcs.comp_equal(golden, golden.reshape(1, 1, 32, 32))[0]

    def test_comp_allclose_pass_skips_pcc(self, monkeypatch):
        golden, calculated = _noisy((64, 64), torch.bfloat16)
        monkeypatch.setattr(comparison_funcs, "get_atol_rtol_pcc", _fail_if_called)
        assert comparison_funcs.comp_allclose(golden, calculated, rtol=0.1, atol=0.1) == (
            True,
            "Allclose check passed",
        )
        # Same element count but a different rank still takes the fast path.
        assert comparison_funcs.comp_allclose(golden.reshape(1, 64, 64), calculated, rtol=0.1, atol=0.1)[0]

    def test_comp_allclose_fail_message(self):
        golden, calculated = _noisy((64, 64), torch.bfloat16)
        passing, output = comparison_funcs.comp_allclose(golden, calculated)
        assert not passing
        assert self.MESSAGE.match(output) and output.endswith(", Allclose check failed"), output

    def test_comp_allclose_broadcast_only_match_is_not_a_pass(self, expect_error):
        # torch.allclose broadcasts [1, 32] against [32, 32]; that size mismatch must not pass silently.
        row = torch.randn(1, 32)
        with expect_error(RuntimeError, "inconsistent tensor size"):
            comparison_funcs.comp_allclose(row.expand(32, 32).contiguous(), row)

    @pytest.mark.parametrize("comp", ["comp_equal", "comp_allclose"])
    def test_empty_tensors_pass(self, comp):
        empty = torch.empty(0, 32)
        assert getattr(comparison_funcs, comp)(empty, empty.clone())[0]


class TestOtherHelpers:
    def test_comp_and_get_pcc_returns_float(self):
        golden, calculated = _noisy((64, 64), torch.bfloat16)
        passing, _, pcc = comparison_funcs.comp_and_get_pcc(golden, calculated)
        assert passing
        assert isinstance(pcc, float) and pcc == _pcc(golden, calculated)

    def test_comp_pcc_list_uses_min_pcc(self):
        good = _noisy((32, 32), torch.bfloat16)
        bad = _noisy((32, 32), torch.bfloat16, noise=100.0, seed=1)
        assert comparison_funcs.comp_pcc_list([good[0]], [good[1]])[0]
        assert not comparison_funcs.comp_pcc_list([good[0], bad[0]], [good[1], bad[1]])[0]

    def test_comp_equal_list(self):
        golden, calculated = _noisy((32, 32), torch.bfloat16)
        assert comparison_funcs.comp_equal_list([golden], [golden.clone()])[0]
        assert not comparison_funcs.comp_equal_list([golden], [calculated])[0]

    def test_pcc_is_finite_for_ordinary_inputs(self):
        golden, calculated = _noisy((1, 1, 1024, 1024), torch.bfloat16)
        assert math.isfinite(_pcc(golden, calculated))
