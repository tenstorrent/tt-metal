# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported geometry and unqualified projection timing claims."""

import copy
import unittest

from models.demos.qwen38_27b_qb2.demo.run_projection_sweep import validate_report
from models.demos.qwen38_27b_qb2.tests.projection_sweep import (
    BATCHES,
    BOUNDARIES,
    ROLES,
    candidates,
    compare,
    geometry,
    input_contract,
)


class ProjectionSweepTest(unittest.TestCase):
    def test_compact_boundary_does_not_expand_users_into_tile_planes(self):
        compact = input_contract(16, "output", "compact_l1")
        public = input_contract(16, "output", "public_dram")
        self.assertEqual(compact["shape"], [1, 1, 16, 1536])
        self.assertEqual(compact["padded_shape"], [1, 1, 32, 1536])
        self.assertEqual(public["padded_shape"], [16, 32, 1536])
        self.assertEqual(compact["memory"], "l1")
        self.assertTrue(compact["keep_sharded"])

    def test_layouts_cannot_be_mixed_in_one_bracket(self):
        rows = self.rows()
        rows[1]["input_layout"] = "compact_l1"
        with self.assertRaisesRegex(ValueError, "input layout"):
            compare(*rows, role="output")

    def complete_report(self):
        cases, comparisons = [], []
        for batch in BATCHES:
            for role, layout in BOUNDARIES:
                group = []
                for config in (*candidates(role), candidates(role)[0]):
                    row = dict(
                        self.rows()[0],
                        batch=batch,
                        role=role,
                        input_layout=layout,
                        input_contract=input_contract(batch, role, layout),
                        config=config,
                    )
                    group.append(row)
                cases.extend(group)
                comparisons.extend(
                    dict(
                        batch=batch,
                        role=role,
                        input_layout=layout,
                        config=r["config"],
                        **compare(group[0], r, group[-1], role=role),
                    )
                    for r in group[1:-1]
                )
        return dict(state="completed", passed=True, cleanup_completed=True, cases=cases, comparisons=comparisons)

    def test_report_requires_separate_complete_producer_boundaries(self):
        report = self.complete_report()
        self.assertEqual(len(report["cases"]), 62)
        self.assertEqual(len(validate_report(report)), 50)
        for change in ("missing_layout", "wrong_geometry", "mixed_layout", "wrong_comparison"):
            damaged = copy.deepcopy(report)
            if change == "missing_layout":
                damaged["cases"] = [r for r in damaged["cases"] if r["input_layout"] != "compact_l1"]
            elif change == "wrong_geometry":
                damaged["cases"][0]["input_contract"] = input_contract(16, "output", "public_dram")
            elif change == "mixed_layout":
                damaged["cases"][1]["input_layout"] = "public_dram"
            else:
                damaged["comparisons"][0]["speedup"] = 2.0
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_report(damaged)

    def test_plan_is_bounded_and_covers_controls(self):
        for role, spec in ROLES.items():
            configs = candidates(role)
            count = 10 if role == "output" else 8
            self.assertEqual(len(configs), count)
            self.assertEqual(configs[0], dict(readers=2, cores=8, block=spec["block"]))
            self.assertEqual(len({tuple(c.values()) for c in configs}), count)
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
