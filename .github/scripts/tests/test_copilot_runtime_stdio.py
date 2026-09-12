#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for Copilot's EAGAIN crash and the Bash startup integration."""

import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import tempfile
import time
import unittest


RELAY = Path(__file__).resolve().parents[1] / "copilot-runtime-stdio.py"
BURST = """
import os, sys
for fd in (1, 2):
    os.set_blocking(fd, False)
try:
    for _ in range(256):
        os.write(1, b'o' * 4096)
        os.write(2, b'e' * 4096)
except BlockingIOError:
    os._exit(134)
sys.exit(17)
"""


class RuntimeOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="copilot stdio ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(os.environ, RUNNER_TEMP=str(self.root))
        for name in ("BASH_ENV", "COPILOT_AGENT_RUNTIME_VERSION", "TT_COPILOT_STDIO_SPOOLED"):
            self.env.pop(name, None)

    def test_reproduces_eagain_without_relay(self):
        with subprocess.Popen(
            [sys.executable, "-c", BURST], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env
        ) as child:
            # Deliberately leave both output pipes unread until the writer exits.
            self.assertEqual(child.wait(timeout=5), 134)
            child.communicate(timeout=5)

    def test_burst_survives_backpressure_and_preserves_output_and_status(self):
        with subprocess.Popen(
            [sys.executable, str(RELAY), "--", sys.executable, "-c", BURST],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        ) as child:
            time.sleep(0.2)  # A slow Actions log consumer must not kill the runtime.
            stdout, stderr = child.communicate(timeout=10)
            self.assertEqual(child.returncode, 17)
            self.assertEqual(stdout, (b"o" * 4096 + b"e" * 4096) * 256)
            self.assertEqual(stderr, b"")
        self.assertEqual(list(self.root.iterdir()), [])  # Spools are private and unlinked.

    def test_workflow_command_markers_stay_ordered_with_stderr(self):
        command = """
import os
os.write(1, b'::stop-commands::fixture-token\\n')
for _ in range(256):
    os.write(2, b'diagnostic\\n' * 100)
os.write(1, b'::fixture-token::\\n')
"""
        result = subprocess.run(
            [sys.executable, str(RELAY), "--", sys.executable, "-c", command],
            capture_output=True,
            env=self.env,
            timeout=5,
            check=True,
        )
        self.assertEqual(
            result.stdout,
            b"::stop-commands::fixture-token\n" + b"diagnostic\n" * 25600 + b"::fixture-token::\n",
        )

    def test_preserves_stdin_arguments_and_signal_exit(self):
        result = subprocess.run(
            [
                sys.executable,
                str(RELAY),
                "--",
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); print(sys.stdin.read(), end='')",
                "argument with spaces",
            ],
            input=b"stdin fixture\n",
            capture_output=True,
            env=self.env,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"argument with spaces\nstdin fixture\n")
        result = subprocess.run(
            [sys.executable, str(RELAY), "--", "bash", "-c", "kill -TERM $$"],
            capture_output=True,
            env=self.env,
            timeout=5,
        )
        self.assertEqual(result.returncode, 128 + signal.SIGTERM)

    def test_cancellation_finishes_when_log_consumer_stops_reading(self):
        with subprocess.Popen(
            [sys.executable, str(RELAY), "--", sys.executable, "-c", BURST],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        ) as child:
            time.sleep(0.2)
            child.terminate()
            self.assertEqual(child.wait(timeout=7), 128 + signal.SIGTERM)
            child.communicate(timeout=5)

    def test_forwards_cancellation(self):
        ready = self.root / "ready"
        command = "import pathlib,time; pathlib.Path('ready').touch(); time.sleep(60)"
        with subprocess.Popen(
            [sys.executable, str(RELAY), "--", sys.executable, "-c", command],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        ) as child:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists())
            child.terminate()
            child.communicate(timeout=7)
            self.assertEqual(child.returncode, 128 + signal.SIGTERM)

    def install(self):
        previous = self.root / "previous.sh"
        previous.write_text('export PREVIOUS_LOADED="${PREVIOUS_LOADED:-}x"\n')
        env_file = self.root / "github-env"
        env = dict(self.env, BASH_ENV=str(previous), GITHUB_ENV=str(env_file))
        subprocess.run([sys.executable, str(RELAY), "--install"], env=env, check=True, capture_output=True)
        env["BASH_ENV"] = env_file.read_text().strip().split("=", 1)[1]
        return env

    def test_bash_startup_only_wraps_runtime_and_preserves_previous_startup(self):
        env = self.install()
        script = self.root / "step.sh"
        script.write_text(
            f"{shlex.quote(sys.executable)} -c 'import os,stat; print(stat.S_ISREG(os.fstat(1).st_mode))'\n"
            'printf "%s\\n" "$PREVIOUS_LOADED" "$1"\n'
        )
        command = ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(script), "space argument"]
        ordinary = subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=5)
        self.assertEqual(ordinary.stdout, "False\nx\nspace argument\n")
        env["COPILOT_AGENT_RUNTIME_VERSION"] = "test-runtime"
        wrapped = subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=5)
        self.assertEqual(wrapped.stdout, "True\nx\nspace argument\n")
        # The generated runtime's errexit/pipefail semantics survive re-exec.
        script.write_text("false | true\nprintf 'should not execute'\n")
        failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(failed.stdout, "")


if __name__ == "__main__":
    unittest.main()
