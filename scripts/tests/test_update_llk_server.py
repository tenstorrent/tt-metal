#!/usr/bin/env python3
"""Linux integration tests; only temporary Git repositories and dummy servers are used.

Run: python3 -m unittest discover -s scripts/tests -p 'test_update_llk_server.py' -v
"""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "update_llk_server.py"
SUBMODULE = "tt_metal/third_party/tt_ops_code_gen"
spec = importlib.util.spec_from_file_location("update_llk_server", SCRIPT)
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


class UpdateServerFixture(unittest.TestCase):
    def command(self, cwd, *args, check=True):
        return subprocess.run(args, cwd=cwd, env=self.env, text=True, capture_output=True, check=check, timeout=20)

    def git(self, cwd, *args):
        return self.command(cwd, "git", *args).stdout.strip()

    def commit(self, repo, filename, content):
        (repo / filename).write_text(content)
        self.git(repo, "add", filename)
        self.git(repo, "commit", "-qm", filename)
        return self.git(repo, "rev-parse", "HEAD")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.env = dict(
            os.environ,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL="/dev/null",
            GIT_AUTHOR_NAME="Test",
            GIT_AUTHOR_EMAIL="test@example.invalid",
            GIT_COMMITTER_NAME="Test",
            GIT_COMMITTER_EMAIL="test@example.invalid",
            GIT_ALLOW_PROTOCOL="file",
        )
        self.parent_remote = self.base / "parent-remote"
        self.sub_remote = self.base / "sub-remote"
        for repo, branch in ((self.parent_remote, "llk_helper_library"), (self.sub_remote, "main")):
            self.git(self.base, "init", "-q", "-b", branch, str(repo))
            self.commit(repo, "version", "initial\n")
        self.git(self.parent_remote, "submodule", "add", str(self.sub_remote), SUBMODULE)
        self.git(self.parent_remote, "commit", "-qam", "add submodule")
        self.root = self.base / "checkout"
        self.git(self.base, "clone", "-q", "--recurse-submodules", str(self.parent_remote), str(self.root))
        self.sub = self.root / SUBMODULE
        (self.root / "scripts").mkdir()
        shutil.copy2(SCRIPT, self.root / "scripts/update_llk_server.py")
        (self.root / ".gitignore").write_text("/old/\n/build/\n/state/\n")
        (self.root / "state").mkdir()
        (self.root / "server.py").write_text(
            """import json, os, pathlib, socket, sys
state = pathlib.Path('state')
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('127.0.0.1', int(os.environ['TT_METAL_JIT_SERVER_ENDPOINT'].rsplit(':', 1)[1])))
s.listen()
(state / ('pid-' + str(os.getpid()))).write_text(json.dumps({'marker': os.environ['TEST_MARKER'], 'args': sys.argv[1:]}))
print('server started', flush=True)
while True:
 c, _ = s.accept()
 c.close()
"""
        )
        (self.root / "build_metal.sh").write_text(
            """#!/bin/sh
printf '%s\\n' "$@" > state/build-args
touch state/build-ran
if [ -f state/fail-build ]; then exit 42; fi
if [ -f state/missing-binary ]; then rm -f build/tools/jit_compile_server; fi
if [ -f state/bad-binary ]; then printf '#!/bin/sh\\nexit 23\\n' > build/tools/jit_compile_server; fi
"""
        )
        (self.root / "build_metal.sh").chmod(0o755)
        self.git(self.root, "add", ".")
        self.git(self.root, "commit", "-qm", "test harness")
        # Publish fixture additions so local HEAD is an ancestor of remote HEAD.
        self.git(self.parent_remote, "fetch", str(self.root), "llk_helper_library")
        self.git(self.parent_remote, "merge", "--ff-only", "FETCH_HEAD")
        self.parent_latest = self.commit(self.parent_remote, "version", "latest parent\n")
        self.sub_latest = self.commit(self.sub_remote, "version", "latest submodule\n")
        for folder in ("old/tools", "build/tools"):
            path = self.root / folder
            path.mkdir(parents=True)
            shutil.copy2("/usr/bin/python3", path / "jit_compile_server")
        self.old = None
        self.addCleanup(self.stop_servers)
        self.start_server()

    def start_server(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        env = dict(self.env, TT_METAL_JIT_SERVER_ENDPOINT=f"0.0.0.0:{self.port}", TEST_MARKER="preserved")
        with open(self.root / "state/server.log", "ab") as log:
            self.old = subprocess.Popen(
                [str(self.root / "old/tools/jit_compile_server"), "server.py", "kept-argument"],
                cwd=self.root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
        deadline = time.monotonic() + 5
        while not (self.root / f"state/pid-{self.old.pid}").exists():
            if self.old.poll() is not None or time.monotonic() > deadline:
                self.fail("Dummy server failed to start: " + (self.root / "state/server.log").read_text())
            time.sleep(0.02)

    def stop_servers(self):
        # Match the unique temporary executable path, including replacements that failed readiness.
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                args = (proc / "cmdline").read_bytes().split(b"\0")
                if (
                    args
                    and args[0].startswith(os.fsencode(self.root) + b"/")
                    and args[0].endswith(b"/jit_compile_server")
                ):
                    os.kill(int(proc.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        if self.old is not None:
            self.old.wait(timeout=5)
        lock = Path(tempfile.gettempdir()) / (
            "update-llk-server-" + hashlib.sha256(os.fsencode(self.root)).hexdigest()[:16] + ".lock"
        )
        lock.unlink(missing_ok=True)

    def update(self, *args):
        return self.command(self.root, sys.executable, "scripts/update_llk_server.py", *args, check=False)

    def assert_not_restarted(self, result, built=False):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIsNone(self.old.poll(), result.stdout + result.stderr)
        self.assertEqual((self.root / "state/build-ran").exists(), built)
        self.assertEqual(len(list((self.root / "state").glob("pid-*"))), 1)


class UpdateServerTests(UpdateServerFixture):
    def test_success_updates_both_repositories_and_preserves_server_settings(self):
        result = self.update("--", "--enable-ccache")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git(self.root, "rev-parse", "HEAD"), self.parent_latest)
        self.assertEqual(self.git(self.sub, "rev-parse", "HEAD"), self.sub_latest)
        self.old.wait(timeout=5)
        records = list((self.root / "state").glob("pid-*"))
        self.assertEqual(len(records), 2)
        new = next(p for p in records if p.name != f"pid-{self.old.pid}")
        self.assertEqual(json.loads(new.read_text()), {"marker": "preserved", "args": ["kept-argument"]})
        self.assertEqual(os.readlink(f"/proc/{new.name[4:]}/exe"), str(self.root / "build/tools/jit_compile_server"))
        self.assertEqual((self.root / "state/server.log").read_text().count("server started"), 2)
        self.assertEqual((self.root / "state/build-args").read_text(), "--enable-ccache\n")

    def test_build_failure_keeps_old_server(self):
        (self.root / "state/fail-build").touch()
        self.assert_not_restarted(self.update(), built=True)

    def test_failed_build_is_retried_without_force(self):
        marker = self.root / ".git/llk-server-update-pending"
        (self.root / "state/fail-build").touch()
        self.assert_not_restarted(self.update(), built=True)
        self.assertTrue(marker.exists())
        (self.root / "state/fail-build").unlink()
        result = self.update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Retrying an unfinished update", result.stdout)
        self.assertFalse(marker.exists())
        self.old.wait(timeout=5)

    def bring_parent_up_to_date(self):
        self.git(self.root, "fetch", "origin", "llk_helper_library")
        self.git(self.root, "merge", "--ff-only", "origin/llk_helper_library")

    def test_unchanged_parent_skips_fetch_build_restart_even_with_new_submodule(self):
        self.bring_parent_up_to_date()
        before = self.git(self.sub, "rev-parse", "HEAD")
        result = self.update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("unchanged; skipping", result.stdout)
        self.assertNotIn("git fetch", result.stdout)
        self.assertNotEqual(before, self.sub_latest)
        self.assertEqual(self.git(self.sub, "rev-parse", "HEAD"), before)
        self.assertFalse((self.root / "state/build-ran").exists())
        self.assertIsNone(self.old.poll())

    def test_unchanged_parent_needs_neither_clean_files_nor_running_server(self):
        self.bring_parent_up_to_date()
        (self.root / "version").write_text("local edits")
        self.old.terminate()
        self.old.wait(timeout=5)
        result = self.update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("unchanged; skipping", result.stdout)
        self.assertFalse((self.root / "state/build-ran").exists())

    def test_force_retries_failed_build_after_parent_has_advanced(self):
        (self.root / "state/fail-build").touch()
        self.assert_not_restarted(self.update(), built=True)
        (self.root / "state/fail-build").unlink()
        result = self.update("--force", "--", "--enable-ccache")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Server ready", result.stdout)
        self.old.wait(timeout=5)

    def test_remote_check_failure_is_not_treated_as_up_to_date(self):
        self.git(self.root, "remote", "set-url", "origin", str(self.base / "missing"))
        result = self.update()
        self.assert_not_restarted(result)
        self.assertNotIn("unchanged; skipping", result.stdout)

    def test_missing_remote_branch_is_reported(self):
        self.git(self.parent_remote, "branch", "-m", "renamed")
        result = self.update()
        self.assert_not_restarted(result)
        self.assertNotIn("unchanged; skipping", result.stdout)

    def test_missing_binary_keeps_old_server(self):
        (self.root / "state/missing-binary").touch()
        self.assert_not_restarted(self.update(), built=True)

    def test_replacement_startup_failure_is_reported(self):
        (self.root / "state/bad-binary").touch()
        result = self.update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("status 23", result.stderr)
        self.old.wait(timeout=5)

    def test_dirty_parent_is_rejected(self):
        (self.root / "version").write_text("local edits")
        self.assert_not_restarted(self.update())

    def test_dirty_submodule_is_rejected(self):
        (self.sub / "version").write_text("local edits")
        self.assert_not_restarted(self.update())

    def test_staged_changes_are_rejected(self):
        (self.root / "version").write_text("staged edits")
        self.git(self.root, "add", "version")
        self.assert_not_restarted(self.update())

    def test_diverged_parent_is_rejected(self):
        self.commit(self.root, "local", "local commit")
        self.assert_not_restarted(self.update())

    def test_diverged_submodule_does_not_advance_parent(self):
        before = self.git(self.root, "rev-parse", "HEAD")
        self.commit(self.sub, "local", "local commit")
        self.assert_not_restarted(self.update())
        self.assertEqual(self.git(self.root, "rev-parse", "HEAD"), before)

    def test_fetch_failure_keeps_server_and_parent_head(self):
        before = self.git(self.root, "rev-parse", "HEAD")
        self.git(self.sub, "remote", "set-url", "origin", str(self.base / "missing"))
        self.assert_not_restarted(self.update())
        self.assertEqual(self.git(self.root, "rev-parse", "HEAD"), before)

    def test_wrong_parent_branch_is_rejected(self):
        self.git(self.root, "checkout", "-qb", "wrong")
        self.assert_not_restarted(self.update())

    def test_wrong_submodule_branch_is_rejected(self):
        self.git(self.sub, "checkout", "-qb", "wrong")
        self.assert_not_restarted(self.update())

    def test_submodule_main_branch_is_supported(self):
        self.git(self.sub, "checkout", "main")
        result = self.update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git(self.sub, "rev-parse", "HEAD"), self.sub_latest)

    def test_no_server_is_rejected_before_fetch(self):
        before = self.git(self.root, "rev-parse", "HEAD")
        self.old.terminate()
        self.old.wait(timeout=5)
        result = self.update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("found 0", result.stderr)
        self.assertEqual(self.git(self.root, "rev-parse", "HEAD"), before)
        self.assertFalse((self.root / "state/build-ran").exists())

    def test_multiple_servers_are_rejected(self):
        original = self.old
        self.start_server()
        try:
            result = self.update()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("found 2", result.stderr)
            self.assertIsNone(original.poll())
            self.assertIsNone(self.old.poll())
            self.assertFalse((self.root / "state/build-ran").exists())
        finally:
            original.terminate()
            original.wait(timeout=5)

    def test_concurrent_update_is_rejected(self):
        lock = Path(tempfile.gettempdir()) / (
            "update-llk-server-" + hashlib.sha256(os.fsencode(self.root)).hexdigest()[:16] + ".lock"
        )
        with open(lock, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.update()
            self.assert_not_restarted(result)
            self.assertIn("already running", result.stderr)

    def test_untracked_file_conflict_stops_before_build(self):
        self.commit(self.parent_remote, "new-file", "remote content")
        (self.root / "new-file").write_text("untracked local content")
        self.assert_not_restarted(self.update())
        self.assertEqual((self.root / "new-file").read_text(), "untracked local content")

    def test_missing_submodule_is_rejected(self):
        (self.sub / ".git").unlink()
        self.assert_not_restarted(self.update())

    def test_changed_process_identity_is_not_signalled(self):
        server = list(updater.find_server(self.root))
        self.addCleanup(server[4].close)
        self.addCleanup(server[5].close)
        server[6] = "invalid-start-time"
        with mock.patch.object(updater.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "Original server exited"):
                updater.restart(self.root, server)
            kill.assert_not_called()
        self.assertIsNone(self.old.poll())

    def test_shutdown_timeout_does_not_start_replacement(self):
        server = updater.find_server(self.root)
        self.addCleanup(server[4].close)
        self.addCleanup(server[5].close)
        with (
            mock.patch.object(updater.os, "kill") as kill,
            mock.patch.object(updater.time, "monotonic", side_effect=[0, 31]),
            mock.patch.object(updater.subprocess, "Popen") as start,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                updater.restart(self.root, server)
            kill.assert_called_once_with(self.old.pid, signal.SIGTERM)
            start.assert_not_called()
        self.assertIsNone(self.old.poll())

    def test_readiness_timeout_is_reported(self):
        server = updater.find_server(self.root)
        self.addCleanup(server[4].close)
        self.addCleanup(server[5].close)
        with (
            mock.patch.object(updater.os, "kill"),
            mock.patch.object(updater, "alive", side_effect=[True, False]),
            mock.patch.object(updater.time, "monotonic", side_effect=[0, 0, 31]),
            mock.patch.object(updater.subprocess, "Popen") as start,
        ):
            start.return_value.pid = 999999
            with self.assertRaisesRegex(RuntimeError, "did not become reachable"):
                updater.restart(self.root, server)
            start.assert_called_once()
        self.assertIsNone(self.old.poll())

    def test_unsafe_and_nonbuilding_options_are_rejected(self):
        for flag in (
            "--clean",
            "--configure-only",
            "--build-packages",
            "--help",
            "-h",
            "--cle",
            "--configure-on",
            "--build-pack",
            "-ch",
            "--hel",
        ):
            with self.subTest(flag=flag):
                self.assert_not_restarted(self.update("--", flag))


if __name__ == "__main__":
    unittest.main()
