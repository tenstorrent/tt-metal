# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for context scaling and real timing/control boundaries."""

import copy
import importlib
import json
import sys
import unittest
from pathlib import Path


class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = importlib.import_module(
            "models.demos.llama_3p1_8b_d_p.tests.performance.long_context_performance_helpers"
        )

    def sample(self, tokens=4096, slot=0, phase="measured", repetition=0, duration=1.0):
        chunks = []
        elapsed = 10.0
        for start in range(0, tokens, 1024):
            chunks.append(
                dict(
                    start=start,
                    end=start + 1024,
                    chunk_start=elapsed,
                    forward_start=elapsed + 0.25,
                    forward_end=elapsed + duration,
                )
            )
            elapsed += duration
        return self.h.request_sample(
            tokens=tokens,
            slot=slot,
            phase=phase,
            repetition=repetition,
            prompt_start=10.0,
            prompt_end=elapsed + 0.5,
            chunks=chunks,
            programs_before=7,
            programs_after=7,
        )

    # More chunks must change the throughput denominator and preserve each chunk's own clocks.
    def test_context_length_changes_denominator_and_chunk_count(self):
        for tokens in (4096, 8192, 16384, 32768, 65536, 131072):
            sample = self.sample(tokens)
            self.assertEqual(len(sample["chunks"]), tokens // 1024)
            self.assertEqual(sample["tokens"], tokens)
            self.assertEqual(sample["tokens_per_second_per_user"], tokens / (tokens / 1024 + 0.5))
            self.assertEqual(sample["forward_wall_seconds"], tokens / 1024 * 0.75)

    # Missing, repeated or reordered chunks must fail, even if their total duration looks plausible.
    def test_rejects_incomplete_or_reordered_chunk_coverage(self):
        good = self.sample()
        for indexes in ([0, 1, 2], [0, 1, 2, 2], [1, 0, 2, 3]):
            bad = copy.deepcopy(good)
            bad["clock_timestamps"]["chunks"] = [bad["clock_timestamps"]["chunks"][i] for i in indexes]
            with self.assertRaises(ValueError):
                self.h.validate_sample(bad)
        bad = copy.deepcopy(good)
        bad["clock_timestamps"]["chunks"][1]["chunk_start"] = 10.1
        with self.assertRaises(ValueError):
            self.h.validate_sample(bad)

    # Warmups must remain excluded, and a missing request or compilation during measurement must fail.
    def test_sequential_slot_schedule_excludes_warmups_and_requires_stable_programs(self):
        samples = [
            self.sample(slot=s, phase=p, repetition=n, duration=5.0 if p == "warmup" else 1.0 + s)
            for s, p, n in self.h.request_schedule()
        ]
        result = self.h.aggregate(samples)
        self.assertEqual([g["prompt_wall_seconds"]["median"] for g in result["per_slot"]], [4.5, 8.5])
        self.assertEqual([g["measured_requests"] for g in result["per_slot"]], [3, 3])
        self.assertEqual(len(result["warmups"]), 2)
        for changed in (samples[:-1], [samples[1], samples[0]] + samples[2:]):
            with self.assertRaises(ValueError):
                self.h.aggregate(changed)
        changed = copy.deepcopy(samples)
        changed[-1]["programs_after"] = 8
        with self.assertRaises(ValueError):
            self.h.aggregate(changed)
        changed = copy.deepcopy(samples)
        changed[-1] = self.sample(8192, slot=1, repetition=2)
        with self.assertRaises(ValueError):
            self.h.aggregate(changed)

    # Deferred upload/model work must complete inside the correct timers; expensive readback is excluded.
    def test_runner_times_completion_and_excludes_readback_and_cleanup(self):
        now = [0.0]
        pending = [7.0]
        events = []
        retained = []
        saved = []

        def sync():
            now[0] += pending[0]
            pending[0] = 0
            events.append("sync")

        def upload(start, end):
            events.append(("upload", start, end))
            pending[0] += 2.0
            return ("tokens", start)

        def forward(token, start, end):
            self.assertEqual(pending[0], 0.0, "Upload must complete before forward starts")
            events.append(("forward", start, end))
            pending[0] += 3.0
            return ("output", start)

        def inspect(output, start, end):
            self.assertEqual(len([e for e in events if isinstance(e, tuple) and e[0] == "forward"]), 4)
            events.append(("inspect", start, end))
            now[0] += 100.0
            return [start]

        def release(value):
            events.append(("release", value))
            now[0] += 10.0

        sample, checks, readback = self.h.run_request(
            tokens=4096,
            slot=0,
            phase="measured",
            repetition=0,
            upload=upload,
            forward=forward,
            synchronize=sync,
            program_count=lambda: 5,
            inspect=inspect,
            release=release,
            clock=lambda: now[0],
            retained=retained,
            on_sample=saved.append,
        )
        self.assertEqual(saved, [sample])
        self.assertEqual(sample["prompt_wall_seconds"], 20.0)
        self.assertEqual(sample["forward_wall_seconds"], 12.0)
        self.assertEqual([c["upload_and_sync_wall_seconds"] for c in sample["chunks"]], [2.0] * 4)
        self.assertEqual(sample["tokens_per_second_per_user"], 4096 / 20.0)
        self.assertEqual(readback, 400.0)
        self.assertEqual(checks, [0, 1024, 2048, 3072])
        self.assertEqual(retained, [])
        self.assertEqual(now[0], 507.0)

    # A failed forward must leave uploaded buffers reachable by the caller's existing finally cleanup.
    def test_failed_forward_keeps_owned_resources_reachable(self):
        retained = []

        def fail(*args):
            raise RuntimeError("synthetic forward failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
            self.h.run_request(
                tokens=4096,
                slot=0,
                phase="warmup",
                repetition=0,
                upload=lambda start, end: ("token", start),
                forward=fail,
                synchronize=lambda: None,
                program_count=lambda: 0,
                inspect=lambda *args: [],
                release=lambda value: None,
                clock=lambda: 1.0,
                retained=retained,
            )
        self.assertEqual(retained, [("token", 0)])

    # Existing raw 2K clocks must reproduce their original samples exactly without rerunning a model.
    def test_saved_2k_sample_parity(self):
        path = Path(__file__).parent / "fixtures" / "saved_2k_timing.json"
        report = json.loads(path.read_text())
        for sample in report["samples"]:
            self.h.validate_sample(sample)
        result = self.h.aggregate(report["samples"])
        self.assertEqual(result["per_slot"], report["summary"]["per_slot"])
        self.assertEqual(result["warmups"], report["summary"]["warmups"])

    # Completion validation must cover every chip of every 4K chunk, not only the old 2K prefix.
    def test_report_validation_rejects_missing_or_duplicate_late_chunk_output(self):
        from models.demos.llama_3p1_8b_d_p.tests.performance.long_context_performance_validation import report_reasons
        from models.demos.llama_3p1_8b_d_p.tests.performance.synthetic_report import build_report

        original = json.loads((Path(__file__).parent / "fixtures" / "saved_2k_timing.json").read_text())
        report = build_report()
        self.assertEqual(report_reasons(report, 4096), [])
        # A 4K metadata envelope cannot legitimize shorter 2K timing/throughput samples.
        mixed = copy.deepcopy(report)
        mixed["samples"] = copy.deepcopy(original["samples"])
        mixed["summary"] = self.h.aggregate(mixed["samples"])
        self.assertIn("Timing sample context differs from requested context", report_reasons(mixed, 4096))
        missing = copy.deepcopy(report)
        missing["output_checks"].pop()
        self.assertIn("Missing/duplicate all32 logits evidence", report_reasons(missing, 4096))
        duplicate = copy.deepcopy(report)
        duplicate["output_checks"][-1] = duplicate["output_checks"][0]
        self.assertIn("Missing/duplicate all32 logits evidence", report_reasons(duplicate, 4096))
        wrong = copy.deepcopy(report)
        wrong["output_checks"][-1]["sha256"] = "0" * 64
        self.assertIn("Logits finite/shape/repeat digest failure", report_reasons(wrong, 4096))

    # Invalid benchmark input must refuse import before Torch/TTNN or mesh creation.
    def test_closed_contract_stops_before_native_import(self):
        import os
        import runpy
        import sys
        from unittest.mock import patch

        from models.demos.llama_3p1_8b_d_p.tests.performance.performance_config import load_config

        path = Path(__file__).parent
        with self.assertRaisesRegex(RuntimeError, "explicit performance"):
            load_config(None)
        with patch.dict(
            os.environ, {"LLAMA_LONG_CONTEXT_PERF_CONFIG": str(path / "fixtures" / "invalid_context.json")}
        ):
            with self.assertRaisesRegex(ValueError, "context"):
                runpy.run_path(str(path / "test_long_context_performance.py"))
        self.assertNotIn("ttnn", sys.modules)
        self.assertNotIn("torch", sys.modules)

    # A single retained-output release failure must not retry prior successes or skip later owned cleanup/reporting.
    def test_cleanup_release_failure_drains_once_and_persists_report(self):
        retained = ["first", "failing", "last"]
        events = []
        report = {"status": "started"}
        persisted = []

        def release(value):
            events.append(("release", value))
            if value == "failing":
                raise RuntimeError("synthetic release failure")

        actions = [
            ("cache.k.deallocate", lambda: events.append("cache.k")),
            ("cache.v.deallocate", lambda: events.append("cache.v")),
            ("model.close", lambda: events.append("model.close")),
            ("device.synchronize", lambda: events.append("device.synchronize")),
        ]

        def sequence():
            try:
                self.h.release_owned_resources(retained, release)
            finally:
                self.h.finalize_owned_resources(
                    retained=retained,
                    release=release,
                    actions=actions,
                    report=report,
                    persist=lambda: persisted.append(copy.deepcopy(report)),
                    primary_error=sys.exc_info()[1],
                )

        with self.assertRaises(self.h.CleanupError):
            sequence()
        self.assertEqual(
            events,
            [
                ("release", "first"),
                ("release", "failing"),
                ("release", "last"),
                "cache.k",
                "cache.v",
                "model.close",
                "device.synchronize",
            ],
        )
        self.assertEqual(retained, [])
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["cleanup_errors"][0]["action"], "retained[1]")

    # A model-close failure must still synchronize and write the final report before cleanup failure escapes.
    def test_cleanup_model_close_failure_still_synchronizes_and_writes(self):
        events = []
        report = {"status": "started"}

        def fail_close():
            events.append("model.close")
            raise RuntimeError("synthetic close failure")

        with self.assertRaises(self.h.CleanupError):
            self.h.finalize_owned_resources(
                retained=[],
                release=lambda value: events.append(("release", value)),
                actions=[
                    ("model.close", fail_close),
                    ("device.synchronize", lambda: events.append("device.synchronize")),
                ],
                report=report,
                persist=lambda: events.append(("persist", copy.deepcopy(report))),
                primary_error=None,
            )
        self.assertEqual(events[:2], ["model.close", "device.synchronize"])
        self.assertEqual(events[2][0], "persist")
        self.assertEqual(events[2][1]["cleanup_errors"][0]["action"], "model.close")

    # Cleanup faults after a forward exception must be reported without replacing that original execution error.
    def test_primary_forward_error_survives_cleanup_and_report_write(self):
        class ForwardError(RuntimeError):
            pass

        retained = ["output"]
        events = []
        report = {"status": "started"}

        def fail_release(value):
            raise RuntimeError("release during failure")

        def fail_close():
            raise RuntimeError("close during failure")

        def sequence():
            try:
                raise ForwardError("synthetic forward failure")
            finally:
                self.h.finalize_owned_resources(
                    retained=retained,
                    release=fail_release,
                    actions=[
                        ("model.close", fail_close),
                        ("device.synchronize", lambda: events.append("device.synchronize")),
                    ],
                    report=report,
                    persist=lambda: events.append(("persist", copy.deepcopy(report))),
                    primary_error=sys.exc_info()[1],
                )

        with self.assertRaisesRegex(ForwardError, "synthetic forward failure"):
            sequence()
        self.assertEqual(retained, [])
        self.assertEqual(events[0], "device.synchronize")
        self.assertEqual(events[1][0], "persist")
        self.assertEqual(
            [failure["action"] for failure in events[1][1]["cleanup_errors"]],
            ["retained[0]", "model.close"],
        )


if __name__ == "__main__":
    unittest.main()
