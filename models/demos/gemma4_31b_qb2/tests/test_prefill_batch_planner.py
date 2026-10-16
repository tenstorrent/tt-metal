# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device-free admission checks for native prefill numerical boundaries."""

import random
import unittest

from models.demos.gemma4_31b_qb2.tt.prefill_batch import config_family, padded_span, plan_prefill_batches


class PrefillBatchPlannerTests(unittest.TestCase):
    def test_long_sliding_chunks_are_not_subdivided(self):
        # Splitting a native 1024-row sliding query adds masked SDPA blocks and
        # changes finite-precision results, even with algebraically equal masks.
        for batch, span in ((3, 2048), (5, 1536), (7, 1152)):
            groups = plan_prefill_batches([span] * batch, 8, long_spans=True)
            self.assertTrue(all(len(group) <= 2 for group in groups))
            wide = plan_prefill_batches([span] * batch, 8, long_spans=True, packed_rows=8192)
            self.assertEqual(wide, [list(range(batch))])

    def test_measured_padding_regression_uses_serial_fallback(self):
        self.assertEqual(plan_prefill_batches([257, 769], 8), [[0], [1]])
        # These short partial requests already require 256 native rows each.
        self.assertEqual(plan_prefill_batches([129] * 8, 8), [list(range(8))])

    def test_original_matmul_families_remain_separate(self):
        lengths = [128, 129, 256, 257, 512, 769, 897, 1024]
        groups = plan_prefill_batches(lengths, 8)
        for group in groups:
            self.assertEqual(len({config_family(lengths[row]) for row in group}), 1)

    def test_randomized_coverage_and_capacity(self):
        rng = random.Random(9472)
        for budget in (2048, 8192):
            for maximum in range(1, 33):
                for long_spans in (False, True):
                    for _ in range(100):
                        lengths = [rng.randint(1, 8192) for _ in range(rng.randint(1, 40))]
                        groups = plan_prefill_batches(lengths, maximum, long_spans=long_spans, packed_rows=budget)
                        self.assertEqual(sorted(row for group in groups for row in group), list(range(len(lengths))))
                        for group in groups:
                            self.assertTrue(1 <= len(group) <= maximum)
                            if len(group) == 1:
                                continue
                            span = max(padded_span(lengths[row]) for row in group)
                            self.assertLessEqual(span * len(group), 8192)
                            native_rows = sum(padded_span(lengths[row]) for row in group)
                            self.assertLessEqual(4 * span * len(group), 5 * native_rows)
                            self.assertLessEqual(min(span, 1024) * len(group), budget)
                            self.assertEqual(len({config_family(lengths[row]) for row in group}), 1)
                            if span > 1024:
                                self.assertTrue(long_spans)
                                self.assertEqual(len({padded_span(lengths[row]) for row in group}), 1)

    def test_invalid_admission_parameters(self):
        for maximum in (0, 33):
            with self.assertRaises(ValueError):
                plan_prefill_batches([128], maximum)
        for capacity in (0, 127, 8193):
            with self.assertRaises(ValueError):
                plan_prefill_batches([128], 8, packed_rows=capacity)
        for lengths in ([0], [-1], [1.5], [True]):
            with self.assertRaises(ValueError):
                plan_prefill_batches(lengths, 8)


if __name__ == "__main__":
    unittest.main()
