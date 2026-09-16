# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU fixture/oracle checks. This file never imports TTNN or a device fixture."""
import unittest
import torch
from tests.ttnn.unit_tests.operations.sdpa.standard_exp_test_utils import (
    make_fixture,
    reference,
    metrics,
    passes_false_gate,
)


class FixtureTests(unittest.TestCase):
    # Deterministic, BF16-exact tensors prevent host precision from masquerading as exp error.
    def test_fixture_is_repeatable_and_bf16_exact(self):
        for chunks in (1, 2):
            first = make_fixture(chunks)
            second = make_fixture(chunks)
            self.assertEqual(
                [tuple(t.shape) for t in first],
                [(1, 4, 128, 128), (1, 1, chunks * 512, 128), (1, 1, chunks * 512, 128), (1, 1, 128, chunks * 512)],
            )
            for a, b in zip(first, second):
                self.assertEqual(a.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.isfinite(first[0]).all())

    # The two-chunk case must contain -inf, a fully masked later chunk, and valid rows.
    def test_mask_geometry_never_has_an_all_masked_query(self):
        q, k, v, mask = make_fixture(2)
        self.assertTrue(torch.isneginf(mask).any())
        self.assertTrue(torch.isfinite(mask).any(-1).all())
        self.assertTrue(torch.isneginf(mask[0, 0, :64, 512:]).all())
        self.assertTrue(torch.isfinite(mask[0, 0, 64:, 512:768]).all())
        self.assertTrue(torch.isneginf(mask[0, 0, 64:, 768:]).all())

    # The explicit FP32 oracle must agree with an independent float64 score construction
    # and PyTorch's math attention for all four scale/chunk cases.
    def test_oracle_matches_independent_math(self):
        for chunks in (1, 2):
            q, k, v, mask = make_fixture(chunks)
            for scale in (1.0, 128**-0.5):
                actual = reference(q, k, v, mask, scale)
                scores = q.double()[..., :1] * k.double()[..., 0].unsqueeze(-2)
                expected = torch.softmax(scores * scale + mask.double(), -1) @ v.double()
                torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=2e-6)
                builtin = torch.nn.functional.scaled_dot_product_attention(
                    q.float(),
                    k.float().repeat_interleave(4, 1),
                    v.float().repeat_interleave(4, 1),
                    attn_mask=mask.float(),
                    scale=scale,
                )
                torch.testing.assert_close(actual, builtin, rtol=1e-5, atol=2e-6)

    # BF16 output storage alone must fit the frozen False gate on each head.
    def test_output_quantization_fits_frozen_gate(self):
        for chunks in (1, 2):
            values = make_fixture(chunks)
            for scale in (1.0, 128**-0.5):
                golden = reference(*values, scale)
                for head in range(4):
                    self.assertTrue(passes_false_gate(metrics(golden[:, head], golden[:, head].bfloat16())))

    # Omitting or doubling nonunit attention scale must be observable at the unchanged gate.
    def test_fixture_detects_missing_or_double_scale(self):
        for chunks in (1, 2):
            values = make_fixture(chunks)
            golden = reference(*values, 128**-0.5)
            for wrong_scale in (1.0, 2 * 128**-0.5):
                changed = reference(*values, wrong_scale)
                self.assertTrue(any(not passes_false_gate(metrics(golden[:, h], changed[:, h])) for h in range(4)))

    # Ignoring the mask exposes distinct later values and must fail the same frozen gate.
    def test_fixture_detects_mask_leakage(self):
        q, k, v, mask = make_fixture(2)
        for scale in (1.0, 128**-0.5):
            golden = reference(q, k, v, mask, scale)
            changed = reference(q, k, v, torch.zeros_like(mask), scale)
            self.assertTrue(any(not passes_false_gate(metrics(golden[:, h], changed[:, h])) for h in range(4)))

    # A constant or nonfinite reference must be rejected rather than producing a misleading PCC.
    def test_metrics_reject_degenerate_and_nonfinite_values(self):
        for bad in (torch.ones(8), torch.full((8,), float("nan"))):
            with self.assertRaises(ValueError):
                metrics(bad, bad)
        m = metrics(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0]))
        self.assertAlmostEqual(m["pcc"], 1.0)
        self.assertAlmostEqual(m["nl2"], 1.0)
        self.assertFalse(passes_false_gate(m))


if __name__ == "__main__":
    unittest.main()
