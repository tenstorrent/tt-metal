# SPDX-License-Identifier: Apache-2.0
"""Host-only idle export/state transition coverage; no TTNN imports."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "idle_phases", Path(__file__).resolve().parents[1] / "tt/benchmark_phases.py"
)
phases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phases)


class IdleExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with patch.object(phases.threading.Thread, "start"):
            self.recorder = phases.PhaseRecorder(
                Path(self.directory.name) / "phase.jsonl",
                {"layers": [0]},
                auto_export_interval=1,
                retain_decode_tokens=True,
            )
        self.addCleanup(self.recorder.server.close)
        self.addCleanup(lambda: self.recorder.export_stream.close() if self.recorder.export_stream else None)
        self.addCleanup(lambda: Path(self.recorder.socket_path).unlink(missing_ok=True))

    def frames(self):
        return [json.loads(line) for line in self.recorder.path.read_text().splitlines()]

    def test_idle_stable_manual_drain_explicit(self):
        r = self.recorder
        r._export(automatic=True)
        before = r.path.stat()
        for _ in range(4):
            r._export(automatic=True)
        after = r.path.stat()
        self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))
        r.command("drain")
        self.assertEqual(len(self.frames()), 2)

    def test_state_only_changes_export_then_settle(self):
        r = self.recorder
        r._export(automatic=True)
        changes = [
            lambda: r.identity.update(layers=[0, 1]),
            lambda: r.errors.append("synthetic diagnostic error"),
            lambda: setattr(r, "truncated", True),
            lambda: setattr(r, "token_diagnostic_disabled", True),
            lambda: setattr(r, "token_diagnostic_failures", 1),
            lambda: setattr(r, "token_diagnostic_reason", "synthetic"),
            lambda: setattr(r, "position_identity_uncertain", True),
            lambda: r.pending.update({9: {}}),
            lambda: r.pending.clear(),
            lambda: r.unbound.append((9, object())),
            lambda: r.unbound.clear(),
        ]
        for change in changes:
            previous = len(self.frames())
            change()
            r._export(automatic=True)
            self.assertEqual(len(self.frames()), previous + 1)
            r._export(automatic=True)
            self.assertEqual(len(self.frames()), previous + 1)
        self.assertFalse(self.frames()[-1]["token_retention"]["pending_incomplete"])
        self.assertTrue(self.frames()[-1]["token_retention"]["loss"])

    def test_token_events_and_final_empty_metadata_survive(self):
        r = self.recorder
        r.events.extend(
            [{"event": "submit", "step_id": 1}, {"event": "complete", "step_id": 1, "sampled_decode_token_ids": [42]}]
        )
        r.export_chunk_events = 1
        r._export(automatic=True)
        r._export(automatic=True)
        r._export(automatic=True)
        frames = self.frames()
        self.assertEqual(
            [e for f in frames for e in f["events"]],
            [{"event": "submit", "step_id": 1}, {"event": "complete", "step_id": 1, "sampled_decode_token_ids": [42]}],
        )
        self.assertEqual(frames[-1]["buffered_events"], 0)
        before = r.path.stat()
        r._export(automatic=True)
        self.assertEqual(r.path.stat().st_size, before.st_size)
        r.command("reset")
        r._export(automatic=True)
        self.assertEqual(self.frames()[-1]["window_start_ns"], r.window_start_ns)

    def test_metadata_change_during_write_is_not_acknowledged_early(self):
        r = self.recorder
        original = r._write_export

        def write_and_mutate(result):
            original(result)
            r.errors.append("changed during I/O")

        with patch.object(r, "_write_export", side_effect=write_and_mutate):
            r._export(automatic=True)
        self.assertEqual(self.frames()[-1]["errors"], [])
        r._export(automatic=True)
        self.assertEqual(self.frames()[-1]["errors"], ["changed during I/O"])

    def test_same_count_new_events_never_deduplicated(self):
        r = self.recorder
        for step in range(3):
            r.events.append({"step_id": step})
            r._export(automatic=True)
        self.assertEqual([e["step_id"] for f in self.frames() for e in f["events"]], [0, 1, 2])

    def test_failed_write_never_acknowledged_as_exported(self):
        r = self.recorder
        with patch.object(r, "_write_export", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):
                r._export(automatic=True)
        self.assertTrue(r.export_failed)
        self.assertTrue(r.truncated)
        self.assertIsNone(r.last_export_state)


if __name__ == "__main__":
    unittest.main()
