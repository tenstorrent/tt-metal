# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run directly with Python: no repo conftest or device imports."""

import unittest

from benchmark_work import WorkAccounting, kv_read_tokens, payload


class WorkTests(unittest.TestCase):
    def setUp(self):
        self.work = WorkAccounting()

    def test_useful_prefill_matches_independent_layer_audit(self):
        # Accepted whole-layer audit, then last-token full vocabulary projection.
        expected = 25 * 882489950208 + 5 * 1215408635904 + 2 * 2816 * 262144
        self.assertEqual(self.work.prefill([4096])["useful_flops"], expected)
        self.assertEqual(self.work.prefill([4096] * 32)["useful_flops"], 32 * expected)

    def test_bfp_exponents_and_padding(self):
        self.assertEqual(payload(1, 32, "bfloat4_b"), 576)
        self.assertEqual(payload(32, 64, "bfloat8_b"), 2176)

    def test_kv_boundary_rounding(self):
        self.assertEqual(kv_read_tokens(4095, True), 1024)
        self.assertEqual(kv_read_tokens(4096, True), 1152)
        self.assertEqual(kv_read_tokens(4222, False), 4224)
        self.assertEqual(kv_read_tokens(4223, True), 1024)
        self.assertEqual(kv_read_tokens(0, False), 32)

    def test_batch_reuses_only_head_not_decoder_weights(self):
        one = self.work.decode([4096])["terms"]
        many = self.work.decode([4096] * 32)["terms"]
        self.assertEqual(many["lm_head_weights"], one["lm_head_weights"])
        self.assertEqual(many["indexed_top8_expert_weights"], 32 * one["indexed_top8_expert_weights"])
        self.assertEqual(many["qkv_output_weights"], 32 * one["qkv_output_weights"])

    def test_inactive_rows_do_not_execute_decoder(self):
        one = self.work.decode([4096])["terms"]
        partial = self.work.decode([4096], batch_slots=32)["terms"]
        self.assertEqual(one["kv_reads"], partial["kv_reads"])
        self.assertEqual(partial["embedding_rows"], 32 * one["embedding_rows"])

    def test_127_steps_not_128(self):
        steps = [self.work.decode([p]) for p in range(4096, 4223)]
        self.assertEqual(len(steps), 127)
        self.assertEqual(sum(s["terms"]["kv_reads"] for s in steps), 127 * steps[0]["terms"]["kv_reads"])
        self.assertGreater(steps[-1]["terms"]["metadata_estimate"], steps[0]["terms"]["metadata_estimate"])

    def test_reject_unsupported(self):
        for kwargs in ({"layers": 1}, {"mesh": (1, 1)}, {"context": 4096}):
            with self.assertRaises(ValueError):
                WorkAccounting(**kwargs)
        for pos in (-1, 262144, 1.5):
            with self.assertRaises(ValueError):
                self.work.decode([pos])
        with self.assertRaises(ValueError):
            self.work.decode([4096] * 33)

    def test_peak_is_four_asics(self):
        peaks = self.work.peaks()
        self.assertEqual(peaks["peak_flops_per_s"], 4 * 663552e9)
        self.assertEqual(peaks["peak_dram_bytes_per_s"], 2048e9)


if __name__ == "__main__":
    unittest.main()
