# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only regression for nonfinite PCC values in acceptance vectors."""

import unittest

from models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder import _pcc_values_pass


class PCCGateTests(unittest.TestCase):
    def test_nonfinite_after_a_passing_prefix_is_rejected(self):
        # This ordering previously passed min(values) >= threshold.
        for nonfinite in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(nonfinite=nonfinite):
                self.assertFalse(_pcc_values_pass([0.999989, nonfinite, 0.999980]))
                self.assertFalse(_pcc_values_pass([nonfinite, 0.999989]))

    def test_threshold_and_empty_vectors(self):
        self.assertTrue(_pcc_values_pass([0.995, 1.0]))
        self.assertFalse(_pcc_values_pass([0.995, 0.994999]))
        self.assertFalse(_pcc_values_pass([]))


if __name__ == "__main__":
    unittest.main()
