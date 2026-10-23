# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest

from core_sweep import coarse_cases, coarse_counts, load_observations, refine_cases
from noc_routing_model import predict

T = json.loads(Path(__file__).with_name("topology_bh.json").read_text())


def observation(cores, time, payload=128, estimate=None, **extra):
    case = dict(cores=cores, traffic="write", payload_bytes=payload, batch=16, mode="noc0", banks="balanced")
    case.update(extra)
    row = dict(case=case, measured_us=time)
    if estimate is not None:
        row["prediction"] = dict(estimated_us=estimate)
    return row


class CoreSweepTests(unittest.TestCase):
    def test_coarse_counts_cover_cap_and_scale(self):
        self.assertEqual(coarse_counts(110), [1, 2, 4, 8, 16, 32, 64, 96, 110])
        self.assertEqual(coarse_counts(1), [1])
        for limit in [2, 3, 9, 64, 150, 300]:
            counts = coarse_counts(limit)
            self.assertEqual(counts, sorted(set(counts)))
            self.assertEqual((counts[0], counts[-1]), (1, limit))

    def test_cartesian_product_and_hardware_cap(self):
        cases, plan = coarse_cases(T, maximum=1000)
        self.assertEqual(len(cases), 9 * 2 * 2 * 3 * 2)
        self.assertEqual(plan["core_limit"], 110)
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in cases}), len(cases))
        cases, _ = coarse_cases(
            T, maximum=3, traffic=["write"], payloads=[128], batches=[16], modes=["noc0"], banks=["balanced"]
        )
        self.assertEqual([c["cores"] for c in cases], [1, 2, 3])
        self.assertGreater(predict(T, cases[-1])["estimated_us"], 0)

    def test_incompatible_combinations_are_recorded(self):
        cases, plan = coarse_cases(T, maximum=1, traffic=["read", "mixed"], batches=[1, 16])
        self.assertTrue(plan["skipped_combinations"])
        self.assertTrue(all(c["mode"] == "split" for c in cases if c["traffic"] == "mixed"))
        self.assertFalse(any(c["batch"] == 1 and c["mode"] == "split" for c in cases))
        with self.assertRaisesRegex(ValueError, "no valid"):
            coarse_cases(T, traffic=["mixed"], modes=["noc0"])

    def test_invalid_probe_sizes_rejected(self):
        for kwargs in [dict(payloads=[1]), dict(payloads=[32768]), dict(batches=[3]), dict(maximum=0)]:
            with self.assertRaises(ValueError):
                coarse_cases(T, **kwargs)

    def test_saturation_bisects_without_repeating_measured_counts(self):
        rows = [observation(8, 2), observation(16, 4), observation(32, 8)]
        cases, plan = refine_cases(T, rows)
        self.assertEqual([c["cores"] for c in cases], [24, 12])
        self.assertIn("bandwidth_saturation", plan["decisions"][0]["reasons"])
        self.assertEqual(plan["decisions"][0]["relative_bandwidth_gain_per_relative_core_gain"], 0)
        # Accumulated results split the remaining gaps, never resubmit core 24.
        again, _ = refine_cases(T, rows + [observation(24, 6)])
        self.assertNotIn(24, [c["cores"] for c in again])

    def test_linear_bandwidth_scaling_does_not_refine(self):
        cases, _ = refine_cases(T, [observation(8, 2), observation(16, 2)])
        self.assertEqual(cases, [])

    def test_error_trigger_without_latency_or_saturation_trigger(self):
        cases, plan = refine_cases(T, [observation(8, 2, estimate=1), observation(16, 2, estimate=1)])
        self.assertEqual(cases[0]["cores"], 12)
        self.assertEqual(plan["decisions"][0]["reasons"], ["prediction_error"])

    def test_budget_is_fair_across_separate_families(self):
        rows = [observation(n, n, payload=p) for p in [128, 1024] for n in [8, 16, 32]]
        cases, plan = refine_cases(T, rows, budget=2)
        self.assertEqual({c["payload_bytes"] for c in cases}, {128, 1024})
        self.assertEqual(plan["deferred_intervals"], 2)
        self.assertEqual(refine_cases(T, rows, budget=2), refine_cases(T, list(reversed(rows)), budget=2))

    def test_no_cross_family_interpolation_or_extrapolation(self):
        cases, _ = refine_cases(T, [observation(8, 2), observation(16, 4, payload=1024)])
        self.assertEqual(cases, [])
        cases, _ = refine_cases(T, [observation(8, 2), observation(9, 4)])
        self.assertEqual(cases, [])

    def test_repeated_samples_are_aggregated(self):
        rows = [observation(8, 2), observation(8, 2.1), observation(16, 4)]
        cases, _ = refine_cases(T, rows)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["cores"], 12)

    def test_reject_incompatible_or_invalid_observations(self):
        for row in [
            observation(8, 0),
            observation(8, float("nan")),
            observation(8, 2, correct=False),
            observation(8, 2, pages=32),
            observation(8, 2, clock_mhz=1000),
            observation(8, 2, placement="column"),
            observation(8, 2, logical_coords=[[0, 0]]),
        ]:
            with self.assertRaises(ValueError):
                refine_cases(T, [row])
        with self.assertRaises(ValueError):
            refine_cases(T, [])

    def test_existing_probe_and_validator_formats(self):
        root = Path(__file__).parent
        rows = load_observations(root / "read_probe_bh.jsonl")
        cases, _ = refine_cases(T, rows)
        self.assertTrue(cases)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "validation.json"
            for value in [[observation(8, 2), observation(16, 4)], {"cases": [observation(8, 2), observation(16, 4)]}]:
                p.write_text(json.dumps(value))
                cases, _ = refine_cases(T, load_observations(p))
                self.assertEqual(cases[0]["cores"], 12)


if __name__ == "__main__":
    unittest.main()
