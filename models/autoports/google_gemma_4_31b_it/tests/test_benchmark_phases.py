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


if __name__ == "__main__":
    unittest.main(verbosity=2)
