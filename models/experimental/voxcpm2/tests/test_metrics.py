# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
from models.experimental.voxcpm2.validation.metrics import affine_diagnostics, compare_tensors


class TestMetrics(unittest.TestCase):
    def test_affine_gain_and_offset(self):
        reference = torch.arange(8, dtype=torch.float64)
        result = affine_diagnostics(reference, reference * 2.5 - 0.75)
        self.assertAlmostEqual(result["gain"], 2.5)
        self.assertAlmostEqual(result["offset"], -0.75)
        self.assertAlmostEqual(result["residual_relative_rms"], 0)
        # Diagnostics do not relax the original amplitude-sensitive gate.
        self.assertFalse(compare_tensors(reference, reference * 2.5 - 0.75, max_relative_rms=0.01).passed)

    def test_affine_residual_is_not_hidden_by_gain_fit(self):
        reference = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        residual = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        result = affine_diagnostics(reference, reference * 2 + 3 + residual)
        self.assertAlmostEqual(result["gain"], 2)
        self.assertAlmostEqual(result["offset"], 3)
        self.assertAlmostEqual(result["residual_relative_rms"], 3**0.5)

    def test_affine_constant_reference_is_undefined(self):
        result = affine_diagnostics(torch.ones(3), torch.arange(3))
        self.assertEqual(result, {"gain": None, "offset": None, "residual_relative_rms": None})

    def test_affine_rejects_invalid_inputs(self):
        for reference, actual in (
            (torch.ones(2), torch.ones(1, 2)),
            (torch.empty(0), torch.empty(0)),
            (torch.ones(2), torch.tensor([float("nan"), 1])),
            (torch.tensor([float("inf"), 1]), torch.ones(2)),
            (torch.ones(2, dtype=torch.complex64), torch.ones(2)),
        ):
            with self.subTest(reference=reference, actual=actual), self.assertRaises(ValueError):
                affine_diagnostics(reference, actual)

    def test_affine_large_finite_values_do_not_overflow(self):
        reference = torch.tensor([-1e308, 0, 1e308], dtype=torch.float64)
        result = affine_diagnostics(reference, reference * 0.5)
        self.assertAlmostEqual(result["gain"], 0.5)
        self.assertAlmostEqual(result["offset"], 0)
        self.assertAlmostEqual(result["residual_relative_rms"], 0)

    def test_scale_and_offset(self):
        ref = torch.arange(8).float()
        result = compare_tensors(ref, 2 * ref + 1)
        self.assertAlmostEqual(result.pcc, 1)
        self.assertGreater(result.relative_rms, 1)
        self.assertEqual(result.max_abs, 8)
        self.assertFalse(compare_tensors(ref, 2 * ref + 1, max_relative_rms=0.01).passed)

    def test_constants(self):
        self.assertTrue(compare_tensors(torch.zeros(3), torch.zeros(3)).passed)
        result = compare_tensors(torch.zeros(3), torch.ones(3))
        self.assertFalse(result.passed)
        self.assertEqual(result.pcc, 0)
        self.assertIsNone(result.relative_rms)
        self.assertFalse(compare_tensors(torch.ones(3), torch.full((3,), 2.0)).passed)

    def test_nonfinite(self):
        for invalid in (float("nan"), float("inf")):
            actual = torch.tensor([invalid, 1])
            self.assertFalse(compare_tensors(actual, actual).passed)
            self.assertEqual(compare_tensors(torch.ones(2), actual).reason, "nonfinite tensor")

    def test_shape_empty_and_anticorrelation(self):
        self.assertEqual(compare_tensors(torch.ones(2), torch.ones(1, 2)).reason, "shape mismatch")
        self.assertEqual(compare_tensors(torch.empty(0), torch.empty(0)).reason, "empty tensor")
        self.assertAlmostEqual(compare_tensors(torch.arange(4), -torch.arange(4)).pcc, -1)

    def test_bad_thresholds(self):
        for kwargs in ({"min_pcc": float("nan")}, {"min_pcc": 2}, {"max_abs": -1}):
            with self.assertRaises(ValueError):
                compare_tensors(torch.ones(2), torch.ones(2), **kwargs)

    def test_large_finite_values_do_not_overflow_pcc(self):
        ref = torch.tensor([1e308, -1e308], dtype=torch.float64)
        self.assertTrue(compare_tensors(ref, ref).passed)
        self.assertFalse(compare_tensors(ref, -ref).passed)
