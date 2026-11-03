# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only lifecycle checks; no vLLM or TTNN imports or devices."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_server as server


class OwnershipTests(unittest.TestCase):
    def state(self, pid, argv=None):
        return dict(
            pid=pid,
            argv=argv or Path(f"/proc/{pid}/cmdline").read_bytes().decode().strip("\0").split("\0"),
            process_identity=server.process_identity(pid),
        )

    def test_recycled_pid_and_changed_command_rejected(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            state = self.state(child.pid, child.args)
            self.assertTrue(server.owned_process(state))
            state["process_identity"]["start_ticks"] -= 1
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                server.owned_process(state)
            state = self.state(child.pid, child.args)
            state["argv"] = ["another server"]
            with self.assertRaisesRegex(RuntimeError, "command changed"):
                server.owned_process(state)
            state = self.state(child.pid, child.args)
            state.pop("process_identity")
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                server.owned_process(state)
        finally:
            child.terminate()
            child.wait(timeout=5)
        self.assertFalse(server.owned_process(state))

    def test_live_unowned_state_never_signalled_or_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "owned.json"
            state = self.state(os.getpid())
            state_path.write_text(json.dumps(state))
            with patch.object(server, "STATE", state_path), patch.object(server, "BENCH", root), patch.object(
                server.os, "killpg"
            ) as kill:
                with self.assertRaisesRegex(RuntimeError, "ownership"):
                    server.stop()
                kill.assert_not_called()
                self.assertTrue(state_path.exists())

    def test_escalation_checks_group_after_leader_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "owned.json"
            state_path.write_text(json.dumps(dict(pid=12345)))
            with patch.object(server, "STATE", state_path), patch.object(server, "BENCH", root), patch.object(
                server, "owned_process", return_value=True
            ), patch.object(server.os, "killpg") as kill, patch.object(server.time, "sleep") as sleep:
                server.stop()
                self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGTERM))
                self.assertEqual(kill.call_args_list[-1].args, (12345, signal.SIGKILL))
                self.assertEqual(sleep.call_count, 8)
                self.assertTrue((root / "stopped-server-12345.json").exists())

    def test_absent_leader_surviving_group_retains_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "owned.json"
            state_path.write_text(json.dumps(dict(pid=12345)))
            with patch.object(server, "STATE", state_path), patch.object(server, "BENCH", root), patch.object(
                server, "process_identity", return_value=None
            ), patch.object(server.os, "killpg") as kill:
                with self.assertRaisesRegex(RuntimeError, "group survives"):
                    server.stop()
                kill.assert_called_once_with(12345, 0)
                self.assertTrue(state_path.exists())

    def test_leader_exits_between_identity_and_command_reads(self):
        state = dict(pid=12345, process_identity={"process_group": 12345})
        with patch.object(server, "process_identity", return_value=state["process_identity"]), patch.object(
            Path, "read_bytes", side_effect=FileNotFoundError
        ), patch.object(server.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "group survives"):
                server.owned_process(state)
            kill.assert_called_once_with(12345, 0)

    def test_sigterm_during_startup_cleans_detached_host_stub(self):
        # Run the real hook entry point with its server launcher replaced by a
        # plain sleeping Python process. A mocked health probe sends SIGTERM to
        # the helper itself, exercising its installed handler and cleanup.
        source = str(Path(server.__file__).resolve())
        script = r"""import json, os, runpy, signal, subprocess, sys, threading, time
from pathlib import Path
from unittest.mock import patch
real_popen=subprocess.Popen
real_write=Path.write_text
mode=sys.argv[3]
child=None
class CancelProbe:
    def __call__(self, *args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
def launch(*args, **kwargs):
    global child
    child=real_popen([sys.executable,"-c","import time; time.sleep(60)"], **kwargs)
    # Keep the state command exact even though the launcher is a host stub.
    args[0][:]=[sys.executable,"-c","import time; time.sleep(60)"]
    return child
def write(path, *args, **kwargs):
    if path.name == "owned.tmp" and mode == "state_write":
        raise OSError("test state persistence failure")
    result=real_write(path,*args,**kwargs)
    if path.name == "owned.tmp" and mode == "masked_cancel":
        os.kill(os.getpid(),signal.SIGTERM)
    return result
namespace=runpy.run_path(sys.argv[1])
main=namespace["main"]
g=main.__globals__
g["BENCH"]=Path(sys.argv[2]); g["STATE"]=Path(sys.argv[2])/"owned.json"
sys.argv=[sys.argv[1],"--max-num-seqs","32","--base-url","http://unused","--output",str(Path(sys.argv[2])/"identity.json")]
try:
    with patch.object(subprocess,"Popen",launch), patch.object(g["requests"],"get",CancelProbe()), patch.object(Path,"write_text",write):
        g["entrypoint"]()
except (InterruptedError,OSError):
    pass
else:
    raise RuntimeError("cancellation was swallowed")
if child is None:
    raise RuntimeError("stub was not launched")
child.wait(timeout=1)
if child.returncode != -signal.SIGTERM:
    raise RuntimeError(f"stub failed graceful SIGTERM: {child.returncode}")
if g["STATE"].exists():
    raise RuntimeError("owned state was not retired")
print("detached stub terminated and state retired")
"""
        for mode in ("cancel", "state_write", "masked_cancel"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, "-c", script, source, directory, mode], capture_output=True, text=True, timeout=12
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("state retired", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
