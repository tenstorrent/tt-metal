# SPDX-License-Identifier: Apache-2.0
"""CPU-only collector tests; direct unittest execution avoids TT pytest fixtures."""

import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tt" / "benchmark_phases.py"
spec = importlib.util.spec_from_file_location("gemma4_benchmark_phases_host_tests", SOURCE)
phases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phases)


class PhaseRecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.log_path = Path(cls.directory.name) / "phases.jsonl"
        cls.recorder = phases.PhaseRecorder(str(cls.log_path), {"layers": [0, 1]})

    @classmethod
    def tearDownClass(cls):
        # Control socket belongs solely to this test; daemon ends with unittest.
        Path(cls.recorder.socket_path).unlink(missing_ok=True)
        cls.directory.cleanup()

    def command(self, command):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(self.recorder.socket_path)
            client.sendall((json.dumps({"command": command}) + "\n").encode())
            received = b""
            while not received.endswith(b"\n"):
                part = client.recv(65536)
                self.assertTrue(part, "collector closed without a complete JSON response")
                received += part
        return json.loads(received)

    def setUp(self):
        self.assertNotIn("error", self.command("reset"))

    def test_overlapping_reused_device_output_retains_distinct_host_reads(self):
        recorder = self.recorder
        device, first_host, second_host = object(), object(), object()
        first = recorder.begin("decode", batch_size=2, sampling_mode="device", positions=[20, 30])
        recorder.submitted_output(first, device)
        recorder.bind_readback(device, first_host)
        second = recorder.begin("decode", batch_size=2, sampling_mode="device", positions=[21, 31])
        recorder.submitted_output(second, device)
        recorder.bind_readback(device, second_host)
        recorder.complete_output(first_host)
        recorder.complete_output(second_host)
        result = self.command("drain")
        completed = [event for event in result["events"] if event["event"] == "complete"]
        self.assertEqual([event["step_id"] for event in completed], [first, second])
        self.assertEqual([event["positions"] for event in completed], [[20, 30], [21, 31]])
        self.assertEqual(result["pending_steps"], 0)
        self.assertEqual(result["errors"], [])
        # No later forward is needed to export the final readback completion.
        self.assertGreaterEqual(result["boundary_ns"], completed[-1]["monotonic_ns"])

    def test_position_shadow_ignores_stale_inputs_and_resets_on_reload(self):
        recorder = self.recorder
        self.assertEqual(recorder.decode_positions([5, -1], True), [5, -1])
        self.assertEqual(recorder.decode_positions([5, -1], False), [6, -1])
        self.assertEqual(recorder.decode_positions([5, -1], False), [7, -1])
        self.assertEqual(recorder.decode_positions([20, 30], True), [20, 30])
        self.assertEqual(recorder.decode_positions([0, 0], False), [21, 31])

    def test_reset_refuses_pending_work(self):
        step = self.recorder.begin("prefill", batch_size=1, sampling_mode="host_compatibility", starts=[0], ends=[4096])
        try:
            result = self.command("reset")
            self.assertIn("pending work", result["error"])
            self.assertEqual(self.command("check")["pending_steps"], 1)
        finally:
            self.recorder.complete(step)
        self.assertNotIn("error", self.command("reset"))

    def test_socket_drain_persists_and_clears_completed_events(self):
        recorder = self.recorder
        device = object()
        step = recorder.begin("decode", batch_size=1, sampling_mode="host_compatibility")
        recorder.submitted_output(step, device)
        # Existing synchronous conversion has no separate readback hook.
        recorder.complete_output(device)
        result = self.command("drain")
        self.assertEqual([event["event"] for event in result["events"]], ["submit", "complete"])
        self.assertEqual(json.loads(self.log_path.read_text().splitlines()[-1]), result)
        self.assertEqual(self.command("flush")["events"], [])
        self.assertEqual(self.command("check")["pending_steps"], 0)


class AutoExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "phase.jsonl"

    def recorder(self, **kwargs):
        recorder = phases.PhaseRecorder(self.path, {}, auto_export_interval=kwargs.pop("interval", 3600), **kwargs)
        self.addCleanup(lambda: Path(recorder.socket_path).unlink(missing_ok=True))
        self.addCleanup(lambda: recorder.export_stream.close() if recorder.export_stream else None)
        return recorder

    def step(self, recorder, **kwargs):
        step = recorder.begin("decode", batch_size=1, sampling_mode="device", **kwargs)
        recorder.complete(step)

    def test_template_paths_are_unique_and_private_without_overwrite(self):
        from unittest.mock import patch

        template = str(Path(self.directory.name) / "phase-{pid}-{time_ns}.jsonl")
        with patch.object(phases.time, "time_ns", side_effect=[100, 101]):
            first = phases.PhaseRecorder(template, {}, auto_export_interval=3600)
            second = phases.PhaseRecorder(template, {}, auto_export_interval=3600)
        for recorder in (first, second):
            self.addCleanup(lambda r=recorder: Path(r.socket_path).unlink(missing_ok=True))
            self.addCleanup(lambda r=recorder: r.export_stream.close() if r.export_stream else None)
            self.step(recorder)
            recorder._export(automatic=True)
            self.assertEqual(recorder.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(json.loads(recorder.path.read_text())["events"]), 2)
        self.assertNotEqual(first.path, second.path)
        self.assertNotEqual(first.socket_path, second.socket_path)
        self.assertNotIn("{", str(first.path))
        self.assertEqual(first.socket_path, phases.socket_path(first.path))
        before = second.path.read_bytes()
        self.step(first)
        first._export(automatic=True)
        self.assertEqual(second.path.read_bytes(), before)

    def test_background_export_creates_private_file_without_client(self):
        import time

        recorder = self.recorder(interval=0.01)
        self.step(recorder)
        deadline = time.monotonic() + 2
        while not self.path.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        with recorder.export_lock:
            records = [json.loads(line) for line in self.path.read_text().splitlines()]
            self.assertEqual(len([e for r in records for e in r["events"]]), 2)
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
            recorder.file_exhausted = True  # stop periodic writes before fixture teardown

    def test_concurrent_completion_and_drains_have_no_loss_or_duplicates(self):
        from concurrent.futures import ThreadPoolExecutor

        recorder = self.recorder(export_chunk_events=3)

        def produce():
            for _ in range(100):
                self.step(recorder)

        def export():
            for _ in range(40):
                recorder._export(automatic=True)

        def manual():
            for _ in range(20):
                recorder.command("drain")

        with ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(fn) for fn in (produce, export, manual)]
            for future in futures:
                future.result(timeout=5)
        recorder.command("drain")
        events = [e for line in self.path.read_text().splitlines() for e in json.loads(line)["events"]]
        self.assertEqual(len(events), 200)
        self.assertEqual(len({(e["step_id"], e["event"]) for e in events}), 200)
        self.assertEqual(recorder.command("check")["errors"], [])

    def test_disk_write_does_not_hold_submission_lock(self):
        import threading

        recorder = self.recorder()
        self.step(recorder)
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()
        original = recorder._write_export

        def slow_write(result):
            entered.set()
            self.assertTrue(release.wait(3))
            original(result)

        recorder._write_export = slow_write
        writer = threading.Thread(target=recorder._export)
        writer.start()
        self.assertTrue(entered.wait(2))
        producer = threading.Thread(target=lambda: (self.step(recorder), completed.set()))
        producer.start()
        try:
            self.assertTrue(completed.wait(1), "submission blocked on file IO")
        finally:
            release.set()
            writer.join(3)
            producer.join(3)
        recorder.command("drain")
        events = [e for line in self.path.read_text().splitlines() for e in json.loads(line)["events"]]
        self.assertEqual(len(events), 4)

    def test_file_cap_has_explicit_terminal_record(self):
        recorder = self.recorder(max_file_bytes=4096)
        self.step(recorder, positions=list(range(2000)))
        recorder._export(automatic=True)
        self.assertLessEqual(self.path.stat().st_size, 4096)
        terminal = json.loads(self.path.read_text().splitlines()[-1])
        self.assertTrue(terminal["truncated"])
        size = self.path.stat().st_size
        for _ in range(100):
            self.step(recorder)
        recorder._export(automatic=True)
        self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual(len(recorder.events), 0)
        self.assertTrue(recorder.command("check")["errors"])

    def test_buffer_overflow_is_persisted_as_invalid_window(self):
        recorder = self.recorder(max_events=2)
        self.step(recorder)
        self.step(recorder)
        self.assertEqual(len(recorder.events), 2)
        recorder._export(automatic=True)
        result = json.loads(self.path.read_text())
        self.assertTrue(result["truncated"])
        self.assertIn("event buffer overflow", result["errors"])
        with self.assertRaises(RuntimeError):
            recorder.command("reset")

    def test_existing_file_refused_and_events_preserved(self):
        recorder = self.recorder()
        self.path.write_text("existing private evidence")
        self.step(recorder)
        with self.assertRaises(FileExistsError):
            recorder._export(automatic=True)
        self.assertEqual(self.path.read_text(), "existing private evidence")
        self.assertEqual(len(recorder.events), 2)
        self.assertTrue(recorder.command("check")["truncated"])
        self.assertTrue(recorder.command("check")["errors"])
        with self.assertRaises(RuntimeError):
            recorder.command("drain")

    def test_short_writes_are_completed_and_write_failure_invalidates_window(self):
        recorder = self.recorder()
        recorder._export(automatic=True)
        stream = recorder.export_stream

        class ShortWriter:
            def tell(self):
                return stream.tell()

            def write(self, data):
                return stream.write(data[:17])

            def close(self):
                stream.close()

        recorder.export_stream = ShortWriter()
        self.step(recorder)
        recorder._export(automatic=True)
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(records[-1]["events"]), 2)
        self.step(recorder)

        def fail_write(data):
            raise OSError("disk full")

        recorder.export_stream.write = fail_write
        with self.assertRaisesRegex(OSError, "disk full"):
            recorder._export(automatic=True)
        self.assertEqual(len(recorder.events), 2)
        self.assertTrue(recorder.command("check")["truncated"])
        self.assertIn("phase export failed; timing window invalid", recorder.command("check")["errors"])

    def test_symlink_refused_and_accounting_error_not_cleared(self):
        recorder = self.recorder()
        target = self.path.with_name("target")
        target.write_text("unchanged")
        self.path.symlink_to(target)
        self.step(recorder)
        with self.assertRaises(FileExistsError):
            recorder._export(automatic=True)
        self.assertEqual(target.read_text(), "unchanged")
        with self.assertRaises(RuntimeError):
            recorder.command("reset")


if __name__ == "__main__":
    unittest.main(verbosity=2)
