#!/usr/bin/env python3
"""Real Linux own-CPU-child pidfd controls, before any model entry."""
import json
import os
from pathlib import Path
import platform
import signal
import select
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "formatter_stock_pair"))
import formatter_ci_binding as ci
import owned_seed_copy as owned

CAPTURED = []


class LinuxPidfd(unittest.TestCase):
    def child(self):
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", "import sys; sys.stdin.buffer.read()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        saved = owned.identity(process.pid)
        self.assertTrue(saved)
        CAPTURED.append({"pid": process.pid, "birth_ticks": saved["birth_ticks"], "uid": os.getuid()})
        return process, saved

    def close(self, process, saved):
        if process.stdin and not process.stdin.closed:
            process.stdin.close()  # Ordinary EOF only, never numeric-PID signalling.
        process.wait(timeout=3)
        self.assertFalse(owned.same_process(saved))

    def test_real_handle_term_and_positive_reap(self):
        process, saved = self.child()
        reaper = threading.Thread(target=process.wait, daemon=True)
        reaper.start()
        try:
            ci.stop_child_pidfd(saved, os.getuid(), 3)
            reaper.join(timeout=1)
            self.assertFalse(reaper.is_alive())
            self.assertEqual(process.returncode, -signal.SIGTERM)
        finally:
            self.close(process, saved)

    def test_real_handle_changed_birth_or_uid_refuses(self):
        for bad_birth, bad_uid in [(True, False), (False, True)]:
            process, saved = self.child()
            try:
                wrong = {**saved, "birth_ticks": saved["birth_ticks"] + 1} if bad_birth else saved
                with self.assertRaises(AssertionError):
                    ci.stop_child_pidfd(wrong, os.getuid() + int(bad_uid), 3)
                self.assertIsNone(process.poll(), "Refused child was signalled")
            finally:
                self.close(process, saved)

    def test_real_exit_between_validation_and_send(self):
        process, saved = self.child()
        original_signal = signal.pidfd_send_signal
        boundary_seen = []

        def exit_at_send(descriptor, signum):
            process.stdin.close()
            process.wait(timeout=3)
            boundary_seen.append(descriptor)
            return original_signal(descriptor, signum)  # Actual kernel ESRCH on the pinned exited child.

        try:
            with patch.object(signal, "pidfd_send_signal", side_effect=exit_at_send):
                ci.stop_child_pidfd(saved, os.getuid(), 3)
            self.assertEqual(len(boundary_seen), 1)
            self.assertEqual(process.returncode, 0)
        finally:
            self.close(process, saved)

    def test_real_exit_during_handle_capture(self):
        process, saved = self.child()
        original_open = os.pidfd_open

        def exit_at_open(pid, flags):
            descriptor = original_open(pid, flags)
            process.stdin.close()
            process.wait(timeout=3)
            return descriptor

        try:
            with patch.object(os, "pidfd_open", side_effect=exit_at_open):
                ci.stop_child_pidfd(saved, os.getuid(), 3)
            self.assertEqual(process.returncode, 0)
        finally:
            self.close(process, saved)

    def test_real_zombie_wait_is_bounded_and_requires_reap(self):
        process, saved = self.child()
        descriptor = os.pidfd_open(process.pid, 0)
        try:
            process.stdin.close()
            self.assertTrue(select.select([descriptor], [], [], 3)[0])
            self.assertEqual(owned.identity(process.pid)["state"], "Z")
            # Readable pidfd is exit, not immediate /proc disappearance.
            with self.assertRaises(AssertionError):
                ci.stop_child_pidfd(saved, os.getuid(), 0.05)
            self.assertTrue(owned.same_process(saved))

            def delayed_reap():
                time.sleep(0.15)
                process.wait()

            reaper = threading.Thread(target=delayed_reap, daemon=True)
            reaper.start()
            ci.stop_child_pidfd(saved, os.getuid(), 3)
            reaper.join(timeout=1)
            self.assertFalse(reaper.is_alive())
            self.assertEqual(process.returncode, 0)
        finally:
            os.close(descriptor)
            self.close(process, saved)


if __name__ == "__main__":
    assert (
        platform.system() == "Linux" and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
    ), "Real Linux pidfds required; no platform/PID fallback"
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LinuxPidfd))
    evidence = Path("evidence")
    evidence.mkdir(exist_ok=True)
    (evidence / "linux-pidfd-controls.json").write_text(
        json.dumps(
            {
                "platform": platform.system(),
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "captured_own_cpu_children": CAPTURED,
                "passed": result.wasSuccessful() and not result.skipped,
                "no_devices_models_or_credentials": True,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 2)
