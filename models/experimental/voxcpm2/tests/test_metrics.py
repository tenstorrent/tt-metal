# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
from models.experimental.voxcpm2.validation.metrics import compare_tensors

class TestMetrics(unittest.TestCase):
    def test_scale_and_offset(self):
        ref = torch.arange(8).float()
        result = compare_tensors(ref, 2 * ref + 1)
        self.assertAlmostEqual(result.pcc, 1)
        self.assertGreater(result.relative_rms, 1)
        self.assertEqual(result.max_abs, 8)
        self.assertFalse(compare_tensors(ref, 2 * ref + 1, max_relative_rms=.01).passed)

    def test_constants(self):
        self.assertTrue(compare_tensors(torch.zeros(3), torch.zeros(3)).passed)
        result = compare_tensors(torch.zeros(3), torch.ones(3))
        self.assertFalse(result.passed)
        self.assertEqual(result.pcc, 0)
        self.assertIsNone(result.relative_rms)
        self.assertFalse(compare_tensors(torch.ones(3), torch.full((3,), 2.)).passed)

    def test_nonfinite(self):
        for invalid in (float('nan'), float('inf')):
            actual = torch.tensor([invalid, 1])
            self.assertFalse(compare_tensors(actual, actual).passed)
            self.assertEqual(compare_tensors(torch.ones(2), actual).reason, 'nonfinite tensor')

    def test_shape_empty_and_anticorrelation(self):
        self.assertEqual(compare_tensors(torch.ones(2), torch.ones(1, 2)).reason, 'shape mismatch')
        self.assertEqual(compare_tensors(torch.empty(0), torch.empty(0)).reason, 'empty tensor')
        self.assertAlmostEqual(compare_tensors(torch.arange(4), -torch.arange(4)).pcc, -1)

    def test_bad_thresholds(self):
        for kwargs in ({'min_pcc': float('nan')}, {'min_pcc': 2}, {'max_abs': -1}):
            with self.assertRaises(ValueError):
                compare_tensors(torch.ones(2), torch.ones(2), **kwargs)

    def test_large_finite_values_do_not_overflow_pcc(self):
        ref = torch.tensor([1e308, -1e308], dtype=torch.float64)
        self.assertTrue(compare_tensors(ref, ref).passed)
        self.assertFalse(compare_tensors(ref, -ref).passed)
