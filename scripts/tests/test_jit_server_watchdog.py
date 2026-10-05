#!/usr/bin/env python3
"""Isolated watchdog installation and polling tests (no real systemd changes)."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from test_update_llk_server import UpdateServerFixture

WATCHDOG = Path(__file__).resolve().parents[1] / "jit_server_watchdog.py"


class WatchdogTests(UpdateServerFixture):
    def setUp(self):
        super().setUp()
        shutil.copy2(WATCHDOG, self.root / "scripts/jit_server_watchdog.py")
        self.watch_state = self.root / ".git/jit-server-watchdog"
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.env["PATH"] = str(self.bin) + os.pathsep + self.env["PATH"]
        self.env["XDG_CONFIG_HOME"] = str(self.base / "config")
        self.stub_systemd(False)
        self.addCleanup(self.stop_watchdog)

    def stub_systemd(self, available):
        script = self.bin / "systemctl"
        script.write_text(
            """#!/usr/bin/python3
import pathlib, sys
base = pathlib.Path(__file__).parent
with (base / 'calls').open('a') as f: f.write(' '.join(sys.argv[1:]) + '\\n')
args = sys.argv[2:]
"""
            + (
                """if args[0] == 'show-environment': sys.exit(0)
if args[0] == 'is-active':
 print('active' if (base / 'enabled').exists() else 'inactive')
 sys.exit(0 if (base / 'enabled').exists() else 3)
if args[0] == 'enable': (base / 'enabled').touch()
if args[0] == 'disable': (base / 'enabled').unlink(missing_ok=True)
"""
                if available
                else "sys.exit(1)\n"
            )
        )
        script.chmod(0o755)

    def watch(self, *args):
        return self.command(self.root, sys.executable, "scripts/jit_server_watchdog.py", *args, check=False)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def stop_watchdog(self):
        worker_file = self.watch_state / "worker.json"
        if worker_file.exists():
            record = json.loads(worker_file.read_text())
            try:
                os.kill(record["pid"], signal.SIGTERM)
                deadline = time.monotonic() + 5
                while worker_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                if worker_file.exists():
                    os.kill(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass

    def wait_for_check(self):
        last = self.watch_state / "last-check.json"
        deadline = time.monotonic() + 10
        while not last.exists():
            if time.monotonic() >= deadline:
                self.fail((self.watch_state / "watchdog.log").read_text())
            time.sleep(0.05)
        return json.loads(last.read_text())

    def test_loop_install_is_idempotent_and_stop_preserves_server(self):
        result = self.watch("install")
        self.assert_success(result)
        self.assertIn("detached polling loop", result.stdout)
        record = (self.watch_state / "worker.json").read_text()
        self.assert_success(self.watch("install"))
        self.assertEqual((self.watch_state / "worker.json").read_text(), record)
        config = json.loads((self.watch_state / "config.json").read_text())
        self.assertEqual(config["interval"], 300)
        self.assertEqual(config["backend"], "loop")
        self.assertEqual((self.watch_state / "config.json").stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.root / "state/build-ran").exists())
        self.assert_success(self.watch("stop"))
        self.stop_watchdog()
        self.assertIsNone(self.old.poll())

    def test_loop_runs_update_and_records_success(self):
        self.assert_success(self.watch("install", "--interval", "1", "--", "--enable-ccache"))
        last = self.wait_for_check()
        self.assertEqual(last["exit_code"], 0)
        self.old.wait(timeout=5)
        self.assertEqual((self.root / "state/build-args").read_text(), "--enable-ccache\n")
        self.assertIn("Server ready", (self.watch_state / "watchdog.log").read_text())

    def test_loop_retries_failed_build_on_later_tick(self):
        (self.root / "state/fail-build").touch()
        self.assert_success(self.watch("install", "--interval", "1"))
        first = self.wait_for_check()
        self.assertNotEqual(first["exit_code"], 0)
        self.assertIsNone(self.old.poll())
        (self.root / "state/fail-build").unlink()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            last = json.loads((self.watch_state / "last-check.json").read_text())
            if last["exit_code"] == 0:
                break
            time.sleep(0.1)
        else:
            self.fail((self.watch_state / "watchdog.log").read_text())
        self.assertIn("Retrying an unfinished update", (self.watch_state / "watchdog.log").read_text())
        self.old.wait(timeout=5)

    def test_systemd_install_and_stop_use_user_timer(self):
        self.stub_systemd(True)
        result = self.watch("install")
        self.assert_success(result)
        config = json.loads((self.watch_state / "config.json").read_text())
        self.assertEqual(config["backend"], "systemd")
        units = self.base / "config/systemd/user"
        service = next(units.glob("*.service"))
        timer = next(units.glob("*.timer"))
        self.assertIn("KillMode=process", service.read_text())
        self.assertIn("OnUnitInactiveSec=300s", timer.read_text())
        self.assertIn("OnActiveSec=300s", timer.read_text())
        calls = (self.bin / "calls").read_text()
        self.assertIn("--user enable --now", calls)
        self.assert_success(self.watch("install"))
        self.assertEqual((self.bin / "calls").read_text().count("--user enable --now"), 1)
        self.assert_success(self.watch("stop"))
        self.assertIn("--user disable --now", (self.bin / "calls").read_text())
        self.assertIsNone(self.old.poll())
        if shutil.which("systemd-analyze"):
            verified = subprocess.run(
                ["systemd-analyze", "verify", str(service), str(timer)], capture_output=True, text=True
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_remote_error_is_logged_without_stopping_server(self):
        self.assert_success(self.watch("install"))
        self.git(self.root, "remote", "set-url", "origin", str(self.base / "missing"))
        result = self.watch("check")
        self.assertNotEqual(result.returncode, 0)
        last = json.loads((self.watch_state / "last-check.json").read_text())
        self.assertNotEqual(last["exit_code"], 0)
        self.assertIn("ERROR:", (self.watch_state / "watchdog.log").read_text())
        self.assertIsNone(self.old.poll())

    def test_status_before_install_and_invalid_interval(self):
        result = self.watch("status")
        self.assert_success(result)
        self.assertIn("not installed", result.stdout)
        result = self.watch("install", "--interval", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.watch_state / "config.json").exists())

    def test_worker_does_not_duplicate_existing_loop(self):
        self.assert_success(self.watch("install"))
        record = (self.watch_state / "worker.json").read_text()
        self.assert_success(self.watch("run"))
        self.assertEqual((self.watch_state / "worker.json").read_text(), record)

    def test_invalid_build_options_do_not_install_a_timer(self):
        for option in ("--help", "--cle", "--configure-only"):
            with self.subTest(option=option):
                result = self.watch("install", "--", option)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.watch_state / "config.json").exists())
                self.assertFalse((self.watch_state / "worker.json").exists())
