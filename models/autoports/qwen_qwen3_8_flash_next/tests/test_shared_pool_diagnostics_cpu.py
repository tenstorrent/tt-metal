# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of mismatch diagnostics; importing the probe opens no device."""

import unittest

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_shared_pool_generator import (
    decode_geometry_comparison,
    tensor_comparison,
)


class DiagnosticsTest(unittest.TestCase):
    def test_equal_bf16_maxima_use_matching_greedy_tie_break(self):
        values = torch.arange(256, dtype=torch.float32).reshape(1, -1).to(torch.bfloat16)
        values[0, 3] = values[0, 201] = 512
        result = decode_geometry_comparison(values, values.clone())
        self.assertTrue(result["exact"])
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertTrue(result["passed"])
        changed = values.clone()
        changed[0, 201] = 516
        result = decode_geometry_comparison(values, changed)
        self.assertEqual(result["top1_agreement"], 0.0)
        self.assertFalse(result["passed"])

    def test_small_error_does_not_pass_exact_gate_despite_equal_top1(self):
        left = torch.tensor([1.0, 3.0, 2.0])
        right = torch.tensor([1.0, 3.0, 2.125])
        result = tensor_comparison(left, right, logits=True)
        self.assertFalse(result["exact"])
        self.assertTrue(result["top1_equal"])
        self.assertEqual(result["max_abs"], 0.125)
        self.assertEqual(result["different_elements"], 1)
        self.assertGreater(result["pcc"], 0.99)
        self.assertEqual(result["largest_differences"][0]["flat_index"], 2)

    def test_nonfinite_and_constant_state_are_reported_without_invalid_json_numbers(self):
        import json

        constant = tensor_comparison(torch.zeros(4), torch.zeros(4))
        self.assertTrue(constant["exact"])
        self.assertIsNone(constant["pcc"])
        bad = tensor_comparison(torch.tensor([float("nan"), 2.0]), torch.tensor([1.0, 2.0]))
        self.assertFalse(bad["exact"])
        self.assertEqual(bad["nonfinite_expected"], 1)
        json.dumps(bad, allow_nan=False)

    def test_shape_mismatch_remains_explicit_failure(self):
        result = tensor_comparison(torch.zeros(2), torch.zeros(1))
        self.assertFalse(result["exact"])
        self.assertEqual(result["reason"], "shape_mismatch")


if __name__ == "__main__":
    unittest.main()
