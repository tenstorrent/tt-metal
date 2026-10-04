"""Adapter lifecycle around sealed stock commands; no command/request/numerical change."""
import json
from pathlib import Path
import subprocess
import sys
import time

import owned_seed_copy as owned


class StockProcesses:
    def __init__(self, original, output):
        self.original, self.output = original, Path(output)
        self.real_identity = original.process_identity
        self.real_close = original.close_server
        self.first_server = {}

    def identity(self, pid):
        # The stock initial assertion receives the actual identity captured and
        # persisted while the child was gated. Later checks read /proc normally.
        return self.first_server.pop(pid, None) or self.real_identity(pid)

    def spawn(self, command, role, **kwargs):
        assert "stdin" not in kwargs and kwargs.get("start_new_session") is True
        helper = Path(__file__).with_name("owned_exec.py")
        process = subprocess.Popen(
            [sys.executable, "-B", str(helper), *command], stdin=subprocess.PIPE, text=True, **kwargs
        )
        saved = None
        try:
            saved = self.real_identity(process.pid)
            assert saved and saved["session"] == process.pid
            owned.save(
                self.output / (role + "-ownership.json"),
                {"owned": saved, "operation_directory": str(self.output), "released": False},
            )
            process.stdin.write("GO\n")
            process.stdin.flush()
            process.stdin.close()
            owned.save(
                self.output / (role + "-ownership.json"),
                {"owned": saved, "operation_directory": str(self.output), "released": True},
            )
            return process, saved
        except BaseException:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if saved:
                assert self.original.close_server(
                    process, saved, {process.pid: saved}, self.output / (role + "-cleanup.json")
                )
                assert process.poll() is not None
            else:
                # No GO was sent without identity. EOF + waitpid closes the
                # exact direct child without signalling any unverified PID.
                code = process.wait(timeout=10)
                owned.save(
                    self.output / (role + "-cleanup.json"),
                    {"identity_capture_failed": True, "reaped": True, "escalated": False, "exit_code": code},
                )
            raise

    def close_server(self, process, saved, tracked, path):
        result = self.real_close(process, saved, tracked, path)
        reaped = process.poll() is not None
        receipt = json.loads(Path(path).read_text())
        receipt["direct_child_reaped"] = reaped
        receipt["owned_birth_captured_before_start"] = saved
        owned.save(path, receipt)
        assert reaped, "Owned stock child was not positively reaped"
        return result

    def Popen(self, command, **kwargs):
        if len(command) > 1 and command[1] == "/pair/plugin/examples/server_example_tt.py":
            process, saved = self.spawn(command, "server", **kwargs)
            self.first_server[process.pid] = saved
            return process
        # Stock tar verification is filesystem-only. Scientific phase commands
        # use run() below; there is no generic model-launch fallback here.
        return subprocess.Popen(command, **kwargs)

    def run_owned(self, command, log, env, cwd, seconds):
        process, saved = None, None
        with Path(log).open("w") as stream:
            process, saved = self.spawn(
                command,
                Path(log).stem,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            tracked = {process.pid: saved}
            try:
                deadline = time.monotonic() + seconds
                while process.poll() is None:
                    self.original.track_server(process, saved, tracked)
                    assert time.monotonic() < deadline, "Stock phase bounded deadline"
                    time.sleep(0.1)
                return process.returncode
            finally:
                assert self.original.close_server(
                    process, saved, tracked, self.output / (Path(log).stem + "-cleanup.json")
                ), "Stock phase escalation is failed cleanup"
                assert process.poll() is not None, "Stock phase direct child must be positively reaped"

    def __getattr__(self, name):
        return getattr(subprocess, name)
