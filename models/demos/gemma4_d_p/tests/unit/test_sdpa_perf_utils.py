# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks; runnable with unittest without loading repository pytest fixtures."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from models.demos.gemma4_d_p.tests import sdpa_perf_utils as utils
from models.demos.gemma4_d_p.tests import sweep_sdpa_perf as sweep


class LayoutTests(unittest.TestCase):
    def test_cache_ownership_and_roundtrip(self):
        # Three groups, two ranks, two rows per local slab.
        chronological = torch.arange(12).reshape(1, 1, 12, 1)
        packed = utils.cache_to_rank_major(chronological, cp=2, slab=2)
        self.assertEqual(packed.flatten().tolist(), [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11])
        self.assertTrue(torch.equal(utils.cache_to_chronological(packed, cp=2, slab=2), chronological))

    def test_rotated_full_and_partial_query_mapping(self):
        for length in (96, 128):
            q = torch.arange(96, 96 + length).reshape(1, 1, length, 1)
            packed, positions = utils.pack_queries(q, prefix=96, cp=2, slab=64)
            self.assertEqual(sum(map(len, positions)), length)
            for rank, pos in enumerate(positions):
                self.assertTrue(torch.equal(packed[0, 0, rank * 64 : rank * 64 + len(pos), 0], pos))
                self.assertTrue(torch.all(pos % 128 // 64 == rank))
        self.assertEqual(positions[0].tolist(), list(range(128, 192)))
        self.assertEqual(positions[1].tolist(), list(range(96, 128)) + list(range(192, 224)))

    def test_reference_masks_future_and_outside_window(self):
        # Uniform logits make expected values the arithmetic mean of allowed rows.
        q = torch.zeros(1, 4, 3, 2)
        k = torch.zeros(1, 2, 9, 2)
        v = torch.arange(9).reshape(1, 1, 9, 1).expand(1, 2, 9, 2).float().clone()
        v[:, 1] += 100
        positions = torch.tensor([2, 7, 8])
        for window, means in ((None, [1, 3.5, 4]), (3, [1, 6, 7])):
            result = utils.reference_attention(q, k, v, positions, window, block=2)
            self.assertTrue(torch.allclose(result[0, 0, :, 0], torch.tensor(means, dtype=torch.float32)))
            self.assertTrue(torch.allclose(result[0, 3, :, 0], torch.tensor(means, dtype=torch.float32) + 100))

    def test_accuracy_rejects_nan_and_detects_error(self):
        expected = torch.tensor([1.0, 2.0, 4.0])
        pcc, rmse = utils.accuracy_metrics(expected, expected)
        self.assertAlmostEqual(pcc, 1)
        self.assertEqual(rmse, 0)
        self.assertGreater(utils.accuracy_metrics(expected, expected + 1)[1], 0.9)
        with self.assertRaises(AssertionError):
            utils.accuracy_metrics(expected, torch.tensor([1.0, float("nan"), 4.0]))


class TimingTests(unittest.TestCase):
    def record(
        self, chip, duration, runtime=7, source="ttnn/operations/transformer/sdpa/device/kernels/ring_joint_reader.cpp"
    ):
        return dict(chip_id=chip, runtime_id=runtime, duration_ns=duration, kernel_sources=(source,))

    def test_slowest_chip_and_unrelated_program_filter(self):
        records = [self.record(3, 50), self.record(9, 80), self.record(3, 999, 6, "slice.cpp")]
        self.assertEqual(utils.sdpa_duration_ns(records, {3, 9}), 80)

    def test_missing_duplicate_and_multiple_invocations_fail(self):
        cases = [
            [self.record(3, 50)],
            [self.record(3, 50), self.record(3, 51), self.record(9, 80)],
            [self.record(3, 50), self.record(9, 80), self.record(3, 70, runtime=8)],
            [self.record(3, float("nan")), self.record(9, 80)],
        ]
        for records in cases:
            with self.subTest(records=records), self.assertRaises(AssertionError):
                utils.sdpa_duration_ns(records, {3, 9})

    def test_median_and_nearest_rank_p90(self):
        self.assertEqual(utils.timing_summary([x * 1000 for x in range(1, 21)]), (10.5, 18))
        with self.assertRaises(ValueError):
            utils.timing_summary([1000] * 19)


class DriverTests(unittest.TestCase):
    def test_default_does_not_launch_any_process(self):
        with (
            patch.object(sweep.subprocess, "Popen", side_effect=AssertionError("device launch")),
            patch.object(sweep.subprocess, "check_output", side_effect=AssertionError("unexpected subprocess")),
            redirect_stdout(StringIO()) as out,
        ):
            self.assertEqual(sweep.main(["--baseline-only"]), 0)
        self.assertIn("Plan only", out.getvalue())
        self.assertTrue((sweep.ROOT / "models/demos/gemma4_d_p/tests/test_sdpa_perf.py").exists())

    def test_sweep_geometry_and_one_field_at_a_time(self):
        self.assertEqual(len(utils.tilings("global")), 12)
        self.assertEqual(len(utils.tilings("swa")), 2)
        winner = utils.Candidate(q=128, k=512, math_approx=True)
        for config in utils.family_candidates(winner, "exp_approx"):
            self.assertEqual((config.q, config.k, config.math_approx), (128, 512, True))

    def test_accuracy_failures_never_win(self):
        self.assertEqual(
            sweep.best(
                [
                    {"status": "accuracy_failed", "median_us": 1},
                    {"status": "passed", "median_us": 5},
                ]
            )["median_us"],
            5,
        )

    def test_csv_quotes_config_and_errors(self):
        import csv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            row = {"layer": "global", "config": {"q": 64, "k": 256}, "error": "unsupported, reason\nmore detail"}
            sweep.append_csv(path, row)
            with path.open() as stream:
                saved = next(csv.DictReader(stream))
            self.assertEqual(saved["error"], row["error"])

    def test_mocked_driver_confirms_in_fresh_processes(self):
        calls = []

        def fake_process(command, env, log_path, timeout):
            calls.append((command, env))
            Path(env["GEMMA4_SDPA_RESULT"]).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "median_us": 10,
                        "p90_us": 12,
                        "pcc": 1.0,
                        "rmse": 0.0,
                    }
                )
            )
            return 0

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sweep, "run_process", side_effect=fake_process),
            patch.object(sweep.fcntl, "flock"),
            patch.object(sweep.subprocess, "check_output", side_effect=["commit\n", b"diff"]),
            redirect_stdout(StringIO()),
        ):
            result = sweep.main(
                [
                    "--run-device",
                    "--baseline-only",
                    "--layers",
                    "global",
                    "--prefixes",
                    "0",
                    "--output",
                    directory,
                ]
            )
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual([env["GEMMA4_SDPA_SEED"] for _, env in calls], ["1234", "5678", "5678"])
            self.assertTrue(all(env["GEMMA4_SDPA_REPEATS"] == "20" for _, env in calls))
            self.assertTrue(all(env["GEMMA4_SDPA_CANDIDATE"] == utils.baseline("global").to_json() for _, env in calls))
            self.assertEqual(json.loads((Path(directory) / "winners.json").read_text())[0]["speedup"], 1)

    def test_mocked_timeout_stops_entire_sweep(self):
        import csv

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sweep, "run_process", return_value="timeout") as run,
            patch.object(sweep.fcntl, "flock"),
            patch.object(sweep.subprocess, "check_output", side_effect=["commit\n", b"diff"]),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "Stopping sweep after timeout"):
                sweep.main(["--run-device", "--output", directory])
            self.assertEqual(run.call_count, 1)
            with (Path(directory) / "results.csv").open() as stream:
                self.assertEqual(next(csv.DictReader(stream))["status"], "timeout")

    def test_pytest_internal_error_stops_even_after_accuracy_failure(self):
        def fake_process(command, env, log_path, timeout):
            Path(env["GEMMA4_SDPA_RESULT"]).write_text(json.dumps({"status": "accuracy_failed"}))
            return 3

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sweep, "run_process", side_effect=fake_process) as run,
            patch.object(sweep.fcntl, "flock"),
            patch.object(sweep.subprocess, "check_output", side_effect=["commit\n", b"diff"]),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "Stopping sweep after error"):
                sweep.main(["--run-device", "--output", directory])
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
