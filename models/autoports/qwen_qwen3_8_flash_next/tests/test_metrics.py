# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only PCC regressions. Run directly to avoid repo device conftest imports."""
import json
import sys
import unittest

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics


class TestPCCMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_fp64_oracle_noncontiguous_and_input_immutability(self):
        generator = torch.Generator().manual_seed(41)
        a = torch.randn(257, 129, dtype=torch.float64, generator=generator).T
        b = a * 1.7 + 0.1 * torch.randn(a.shape, dtype=torch.float64, generator=generator) + 2
        original_a, original_b = a.clone(), b.clone()
        oracle = torch.corrcoef(torch.stack([a.reshape(-1), b.reshape(-1)]))[0, 1].item()
        result = pcc_metrics(a, b, chunk_size=7777)
        self.assertLess(abs(result["pcc"] - oracle), 1e-13)
        self.assertTrue(torch.equal(a, original_a) and torch.equal(b, original_b))
        self.assertEqual(pcc_metrics(a, a)["exact_equal"], True)

    def test_large_bf16_false_float32_gate_pass(self):
        n = 32 * 129 * 10240
        a = torch.randn(n, generator=torch.Generator().manual_seed(501)).to(torch.bfloat16)
        b = (a.float() + 0.101 * torch.randn(n, generator=torch.Generator().manual_seed(502))).to(torch.bfloat16)
        old = torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item()
        result = pcc_metrics(a, b)
        self.assertLess(result["pcc"], 0.995)
        self.assertAlmostEqual(result["pcc"], 0.9949361954551471, places=12)
        self.assertFalse(result["exact_equal"])
        # This backend-specific failure is recorded, while the regression gate
        # remains the accurate result if a future Torch fixes FP32 covariance.
        print(
            json.dumps(
                {
                    "control": "large_BF16_gate_regression",
                    "legacy_pcc": old,
                    "legacy_false_pass": old >= 0.995,
                    "threshold": 0.995,
                    **result,
                }
            ),
            flush=True,
        )

    def test_invalid_inputs_rejected(self):
        for a, b in [
            (torch.ones(3), torch.ones(3)),
            (torch.tensor([1.0, float("nan")]), torch.arange(2.0)),
            (torch.arange(2.0), torch.tensor([1.0, float("inf")])),
            (torch.arange(2.0), torch.arange(3.0)),
        ]:
            with self.assertRaises(ValueError):
                pcc_metrics(a, b)


if __name__ == "__main__":
    assert "ttnn" not in sys.modules, "CPU metrics tests must not import TTNN"
    unittest.main()
