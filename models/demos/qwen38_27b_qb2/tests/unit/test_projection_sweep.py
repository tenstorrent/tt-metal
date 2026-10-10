# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported geometry and unqualified projection timing claims."""

import copy
import unittest

from models.demos.qwen38_27b_qb2.tests.projection_sweep import ROLES, candidates, compare, geometry


class ProjectionSweepTest(unittest.TestCase):
    def test_plan_is_bounded_and_covers_controls(self):
        for role, spec in ROLES.items():
            configs = candidates(role)
            self.assertEqual(len(configs), 10)
            self.assertEqual(configs[0], dict(readers=2, cores=8, block=spec["block"]))
            self.assertEqual(len({tuple(c.values()) for c in configs}), 10)
            for config in configs:
                dims = geometry(role, config)
                self.assertEqual(dims["padded_n"] % (8 * config["readers"] * 32), 0)
                self.assertGreaterEqual(dims["padded_n"], spec["n"])

    def test_rejects_unsupported_reader_count_and_partial_k_block(self):
        for config in (
            dict(readers=4, cores=8, block=6),
            dict(readers=2, cores=8, block=5),
            dict(readers=2, cores=8, block=8),
        ):
            with self.assertRaises(ValueError):
                geometry("output", config)

    def test_extra_padding_is_charged(self):
        two, three = (geometry("output", dict(readers=r, cores=8, block=6)) for r in (2, 3))
        self.assertEqual(two["padded_n"], 5120)
        self.assertEqual(three["padded_n"], 5376)
        self.assertGreater(three["encoded_weight_bytes"], two["encoded_weight_bytes"])

    def rows(self):
        return [
            dict(samples_us=[v] * 5, output_sha256=["a"] * 4, accuracy_passed=True, changed_input_trace_passed=True)
            for v in (100, 80, 101)
        ]

    def test_reduction_projects_only_the_correct_layer_count(self):
        result = compare(*self.rows(), role="output")
        self.assertTrue(result["comparison_qualified"])
        self.assertAlmostEqual(result["projected_model_saving_ms"], 0.984)
        self.assertFalse(result["full_model_measured"])
        self.assertFalse(result["promoted_to_serving"])

    def test_rejects_drift_changed_control_and_accuracy_failures(self):
        changes = [
            (2, "samples_us", [110] * 5),
            (2, "output_sha256", ["b"] * 4),
            (1, "accuracy_passed", False),
            (1, "changed_input_trace_passed", False),
        ]
        for index, key, value in changes:
            rows = copy.deepcopy(self.rows())
            rows[index][key] = value
            result = compare(*rows, role="down")
            self.assertFalse(result["comparison_qualified"])
            self.assertIsNone(result["speedup"])
            self.assertIsNone(result["projected_model_saving_ms"])

    def test_rejects_incomplete_or_nonfinite_samples(self):
        for samples in ([1] * 4, [float("nan")] * 5, [0] * 5):
            rows = self.rows()
            rows[1]["samples_us"] = samples
            with self.assertRaises(ValueError):
                compare(*rows, role="down")
