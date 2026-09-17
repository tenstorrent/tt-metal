# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for untimed phase markers and warmup-only stack timers."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path


class ProgressTests(unittest.TestCase):
    def setUp(self):
        prefix = "models.demos.llama_3p1_8b_d_p.tests.performance."
        self.h = importlib.import_module(prefix + "long_context_performance_helpers")
        self.p = importlib.import_module(prefix + "request_progress")

    def exercise(self, *, phase="warmup", fail_forward=False, fail_event=None, observed=True):
        now, pending, retained, events, timers, saved = [0.0], [7.0], [], [], [], []

        def synchronize():
            now[0] += pending[0]
            pending[0] = 0
            events.append("sync")

        def upload(start, end):
            pending[0] += 2
            events.append("upload")
            return ("tokens", start)

        def forward(token, start, end):
            events.append("forward")
            if fail_forward:
                raise RuntimeError("model failed")
            pending[0] += 3
            return ("output", start)

        def inspect(output, start, end):
            now[0] += 100
            events.append("inspect")
            return [start]

        def release(value):
            now[0] += 10
            events.append("release")

        def emit(event, **identity):
            events.append(event)
            now[0] += 1000  # Deliberately expensive I/O must not enter either timer.
            if event == fail_event:
                raise RuntimeError("marker failed")

        class Timer:
            def dump_traceback_later(self, seconds, *, repeat, exit):
                timers.append(("arm", seconds, repeat, exit))
                now[0] += 1000

            def cancel_dump_traceback_later(self):
                timers.append(("cancel",))
                now[0] += 1000

        kwargs = dict(
            tokens=4096,
            slot=0,
            phase=phase,
            repetition=0,
            upload=upload,
            forward=forward,
            synchronize=synchronize,
            program_count=lambda: 5,
            inspect=inspect,
            release=release,
            clock=lambda: now[0],
            retained=retained,
            on_sample=saved.append,
        )
        state = dict(events=events, timers=timers, retained=retained, saved=saved)
        try:
            if observed:
                state["result"] = self.p.run_observed_request(
                    self.h.run_request, emit=emit, warmup_stack_seconds=600, stack_handler=Timer(), **kwargs
                )
            else:
                state["result"] = self.h.run_request(**kwargs)
        except RuntimeError as error:
            state["error"] = str(error)
        return state

    # Expensive marker/timer work must not change prompt, forward, H2D or readback durations.
    def test_phase_markers_stay_outside_all_timed_intervals(self):
        plain, observed = self.exercise(observed=False), self.exercise()
        a, b = plain["result"][0], observed["result"][0]
        for key in ("prompt_wall_seconds", "forward_wall_seconds", "tokens_per_second_per_user"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(b["prompt_wall_seconds"], 20)
        self.assertEqual(b["forward_wall_seconds"], 12)
        self.assertEqual(observed["result"][2], 400)
        self.assertEqual([x["upload_and_sync_wall_seconds"] for x in b["chunks"]], [2] * 4)
        events = observed["events"]
        self.assertLess(events.index("request_begin"), events.index("upload"))
        self.assertLess(max(i for i, x in enumerate(events) if x == "forward"), events.index("forward_complete"))
        self.assertLess(events.index("forward_complete"), events.index("inspect"))
        self.assertGreater(
            events.index("readback_release_complete"), max(i for i, x in enumerate(events) if x == "release")
        )
        self.assertEqual(observed["saved"], [b])
        self.assertEqual(observed["retained"], [])

    # A timer is permitted during excluded warmups and must never arm for measured requests.
    def test_timer_is_warmup_only_and_cancelled(self):
        self.assertEqual(self.exercise()["timers"], [("arm", 600, False, False), ("cancel",)])
        self.assertEqual(self.exercise(phase="measured")["timers"], [])

    # A model failure must propagate, cancel the timer and never emit a forward-complete event.
    def test_forward_failure_does_not_claim_completion_or_take_cleanup_ownership(self):
        result = self.exercise(fail_forward=True)
        self.assertEqual(result["error"], "model failed")
        self.assertNotIn("forward_complete", result["events"])
        self.assertNotIn("readback_release_complete", result["events"])
        self.assertEqual(result["timers"][-1], ("cancel",))
        self.assertEqual(result["retained"], [("tokens", 0)])  # The existing caller owns cleanup.

    # Marker failure after forward must preserve all caller-owned resources and skip readback.
    def test_marker_failure_preserves_original_exception_and_borrowed_resources(self):
        result = self.exercise(fail_event="forward_complete")
        self.assertEqual(result["error"], "marker failed")
        self.assertNotIn("inspect", result["events"])
        self.assertNotIn("readback_release_complete", result["events"])
        self.assertEqual(len(result["retained"]), 8)
        freed = []
        self.h.release_owned_resources(result["retained"], freed.append)
        self.assertEqual(len(set(freed)), 8)
        self.assertEqual(result["retained"], [])
        self.assertEqual(result["timers"][-1], ("cancel",))

    # Progress records identify phases and requests but can never serve as execution acceptance.
    def test_json_progress_is_not_an_acceptance_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            self.p.emit_progress(path, "forward_complete", slot=1, phase="warmup", repetition=0, tokens=4096)
            row = json.loads(path.read_text())
            self.assertEqual(row["event"], "forward_complete")
            self.assertEqual(row["slot"], 1)
            self.assertFalse(row["execution_verified"])
            self.assertFalse(row["full_model_accepted"])
            self.assertEqual(row["scope"], "phase_progress_not_acceptance")
            self.assertIn("utc", row)

    # Bad timer settings must fail before any model/marker call; the default disables dumping.
    def test_timer_policy_rejects_nonfinite_or_negative_settings(self):
        self.assertEqual(self.p.parse_warmup_stack_seconds(None), 0)
        for value in ("nan", "inf", "-1", "bad"):
            with self.assertRaises(ValueError):
                self.p.parse_warmup_stack_seconds(value)
