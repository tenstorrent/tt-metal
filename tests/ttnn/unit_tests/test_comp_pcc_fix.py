# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for the comp_pcc constant-tensor false-pass fix.

Before the fix, np.ma.corrcoef on a constant tensor returned a masked matrix
and np.min returned the diagonal 1.0 instead of the undefined off-diagonal
value, causing a silent false-pass for one-constant-vs-random comparisons.
"""

import math
import pytest
import torch

from models.common.utility_functions import comp_pcc


class TestCompPccConstantTensor:
    """comp_pcc must not return 1.0 when one tensor is constant and the other varies."""

    def test_one_constant_vs_random_fails(self):
        """Golden is constant, calculated is random: should fail (not silently pass at 1.0)."""
        golden = torch.ones(32, 32)
        calculated = torch.randn(32, 32)
        passing, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
        assert not passing, f"Expected failure but got passing=True, pcc={pcc_val}"
        # pcc_val is the allclose result (0.0), not the false 1.0
        assert pcc_val != 1.0, f"pcc_val should not be 1.0 for non-matching constant vs random"

    def test_random_vs_one_constant_fails(self):
        """Calculated is constant, golden is random: should fail."""
        golden = torch.randn(32, 32)
        calculated = torch.ones(32, 32) * 5.0
        passing, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
        assert not passing, f"Expected failure but got passing=True, pcc={pcc_val}"

    def test_both_constant_same_value_passes(self):
        """Both tensors identical constants: should pass (allclose is True)."""
        golden = torch.full((16, 16), 3.14)
        calculated = torch.full((16, 16), 3.14)
        passing, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
        assert passing, f"Expected pass for identical constants but got passing=False, pcc={pcc_val}"

    def test_both_constant_different_values_fails(self):
        """Both tensors constant but different values: should fail (allclose is False)."""
        golden = torch.full((16, 16), 1.0)
        calculated = torch.full((16, 16), 2.0)
        passing, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
        assert not passing, f"Expected failure for mismatched constants but got passing=True, pcc={pcc_val}"

    def test_normal_nonconstant_tensors_pass(self):
        """Sanity: normal correlated tensors should still pass."""
        golden = torch.linspace(0, 1, 1024)
        calculated = golden + torch.randn(1024) * 0.001
        passing, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
        assert passing, f"Expected high-PCC tensors to pass but got passing=False, pcc={pcc_val}"
        assert pcc_val > 0.99, f"Expected pcc > 0.99, got {pcc_val}"

    def test_rtol_atol_forwarded_for_close_constants(self):
        """Caller-supplied rtol/atol must reach the allclose fallback for constant tensors."""
        golden = torch.full((32,), 1.0)
        calculated = torch.full((32,), 1.0 + 5e-4)  # within atol=1e-3 but outside default atol=1e-4
        # With tight defaults it should fail
        passing_tight, _ = comp_pcc(golden, calculated, pcc=0.99, rtol=1e-05, atol=1e-04)
        assert not passing_tight, "Expected failure with tight atol"
        # With relaxed atol it should pass
        passing_relaxed, _ = comp_pcc(golden, calculated, pcc=0.99, rtol=1e-05, atol=1e-03)
        assert passing_relaxed, "Expected pass with relaxed atol"

    def test_nan_pcc_returns_float_not_masked(self):
        """pcc_val must be a plain Python float in all cases (no MaskedConstant)."""
        golden = torch.ones(64)
        calculated = torch.ones(64) * 2.0
        passing, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
        assert isinstance(pcc_val, float), f"Expected float, got {type(pcc_val)}: {pcc_val}"
        assert not math.isnan(pcc_val), "pcc_val should not be NaN"

    def test_large_magnitude_float32_tensors_use_float64_correlation(self):
        """ldexp-scale outputs overflow float32 sum-of-squares; PCC must not fall back to allclose."""
        torch.manual_seed(0)
        golden = torch.randn(64, 128, dtype=torch.float32) * 1e19
        calculated = golden + torch.randn(64, 128, dtype=torch.float32) * 1e15
        passing, pcc_val = comp_pcc(golden, calculated, pcc=0.99)
        assert passing, f"Expected high-PCC large-magnitude tensors to pass, got pcc={pcc_val}"
        assert pcc_val > 0.99, f"Expected pcc > 0.99, got {pcc_val}"
