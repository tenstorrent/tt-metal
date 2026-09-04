"""Exercise the real launcher with a fake CLI; never start a paid model."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("codex_sandbox", SCRIPTS / "codex_sandbox.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)

FAKE_CODEX = r"""
import json
import os
from pathlib import Path
import sys
import time

record = {"args": sys.argv[1:], "cwd": os.getcwd(), "home": os.environ.get("CODEX_HOME")}
if sys.argv[1:2] == ["sandbox"]:
    assert sys.argv[1:] == ["sandbox", "-c", 'sandbox_mode="workspace-write"', "--", "/bin/sh", "-c", "exit 0"]
    assert sys.stdin.read() == ""
    Path("probe.json").write_text(json.dumps(record))
    if os.environ.get("PROBE_SLEEP"):
        time.sleep(30)
    print(os.environ.get("PROBE_ERROR", ""), file=sys.stderr)
    raise SystemExit(int(os.environ.get("PROBE_STATUS", "0")))
record["prompt_files_at_start"] = [path.name for path in Path(os.environ["TMPDIR"]).glob("autodebug-prompt.*")]
record["prompt"] = sys.stdin.read()
Path("session.json").write_text(json.dumps(record))
raise SystemExit(int(os.environ.get("SESSION_STATUS", "0")))
"""


class SandboxTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "checkout with spaces"
        self.workspace.mkdir()
        (self.workspace / "source path").mkdir()
        self.prompt_dir = self.root / "prompt files"
        self.prompt_dir.mkdir()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("codex", "claude"):
            executable = bin_dir / name
            executable.write_text(f"#!{sys.executable}\n" + FAKE_CODEX)
            executable.chmod(0o755)
        self.env = os.environ.copy()
        for name in list(self.env):
            if name.startswith("AUTODEBUG_") or name == "CLAUDECODE":
                self.env.pop(name)
        self.env.update(
            PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}",
            CODEX_HOME=str(self.root / "config"),
            TMPDIR=str(self.prompt_dir),
        )

    def launch(self, *, agent="codex", shell="bash", focus_paths=(), **env):
        return subprocess.run(
            [
                shell,
                str(SCRIPTS / "autodebug.sh"),
                "--agent",
                agent,
                "--model",
                "test-model",
                "--effort",
                "high",
                "--",
                *focus_paths,
                "Investigate the sample failure",
            ],
            cwd=self.workspace,
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=10,
        )

    def session(self):
        return json.loads((self.workspace / "session.json").read_text())

    def test_success_keeps_sandbox_even_when_fallback_authorized(self):
        for opt_in in ("0", "1"):
            with self.subTest(opt_in=opt_in):
                result = self.launch(AUTODEBUG_ALLOW_UNSANDBOXED=opt_in)
                self.assertEqual(result.returncode, 0, result.stderr)
                session = self.session()
                self.assertIn("workspace-write", session["args"])
                self.assertNotIn("danger-full-access", session["args"])
                self.assertEqual(session["cwd"], str(self.workspace))
                self.assertEqual(session["home"], self.env["CODEX_HOME"])
                self.assertIn("test-model", session["args"])
                self.assertIn("model_reasoning_effort=high", session["args"])
                self.assertIn("Investigate the sample failure", session["prompt"])
                self.assertNotIn("WARNING", result.stderr)
                probe = json.loads((self.workspace / "probe.json").read_text())
                self.assertEqual(probe["cwd"], session["cwd"])
                self.assertEqual(probe["home"], session["home"])

    def test_known_failure_stops_by_default(self):
        result = self.launch(
            PROBE_STATUS="1",
            PROBE_ERROR="bwrap: Failed to make / slave: Permission denied",
            SLURM_JOB_ID="1234",
            container="docker",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no model was started", result.stderr)
        self.assertFalse((self.workspace / "session.json").exists())

    def test_authorized_known_failures_warn_and_replace_approval_flags(self):
        for error in (
            "bwrap: Failed to make / slave: Permission denied",
            "bwrap: setting up uid map: Permission denied",
            "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted",
            "bwrap: Creating new namespace failed: Operation not permitted",
        ):
            with self.subTest(error=error):
                result = self.launch(AUTODEBUG_ALLOW_UNSANDBOXED="1", PROBE_STATUS="1", PROBE_ERROR=error)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = self.session()["args"]
                self.assertEqual(args[:3], ["--ask-for-approval", "never", "exec"])
                self.assertIn("danger-full-access", args)
                self.assertNotIn("--approve-for-me", args)
                self.assertNotIn("workspace-write", args)
                self.assertIn(error, result.stderr)
                self.assertIn("WARNING", result.stderr)
                self.assertIn("mounted/shared data", result.stderr)

    def test_unrelated_errors_never_enable_fallback(self):
        for error in (
            "invalid config",
            "No space left on device",
            "Permission denied",
            "error: unrecognized subcommand 'sandbox'",
            "bwrap: execvp /bin/true: Permission denied",
        ):
            with self.subTest(error=error):
                result = self.launch(AUTODEBUG_ALLOW_UNSANDBOXED="1", PROBE_STATUS="1", PROBE_ERROR=error)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse((self.workspace / "session.json").exists())

    def test_invalid_opt_in_stops_before_probe(self):
        for value in ("yes", "true", "", "2"):
            with self.subTest(value=value):
                result = self.launch(AUTODEBUG_ALLOW_UNSANDBOXED=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be 0 or 1", result.stderr)
                self.assertFalse((self.workspace / "probe.json").exists())
                self.assertFalse((self.workspace / "session.json").exists())

    def test_timeout_never_enables_fallback(self):
        real_popen = subprocess.Popen
        started = []

        def start(*args, **kwargs):
            process = real_popen(*args, cwd=self.workspace, **kwargs)
            started.append(process)
            return process

        with patch.dict(os.environ, {**self.env, "PROBE_SLEEP": "1", "AUTODEBUG_ALLOW_UNSANDBOXED": "1"}):
            with patch("subprocess.Popen", side_effect=start):
                with self.assertRaisesRegex(SystemExit, "timed out"):
                    PREFLIGHT.select_sandbox(timeout=0.2)
                self.assertIsNotNone(started[0].returncode)

    def test_missing_binary_never_enables_fallback(self):
        with patch.dict(os.environ, AUTODEBUG_ALLOW_UNSANDBOXED="1"):
            with patch("subprocess.Popen", side_effect=FileNotFoundError("codex missing")):
                with self.assertRaisesRegex(SystemExit, "cannot run sandbox preflight"):
                    PREFLIGHT.select_sandbox()

    def test_claude_does_not_run_codex_preflight(self):
        result = self.launch(agent="claude", PROBE_STATUS="1", AUTODEBUG_ALLOW_UNSANDBOXED="invalid")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.workspace / "probe.json").exists())
        self.assertIn("--permission-mode", self.session()["args"])
        self.assertIn("auto", self.session()["args"])

    def test_child_failure_is_not_retried_unsandboxed(self):
        result = self.launch(AUTODEBUG_ALLOW_UNSANDBOXED="1", SESSION_STATUS="7")
        self.assertEqual(result.returncode, 7)
        self.assertIn("workspace-write", self.session()["args"])
        self.assertNotIn("WARNING", result.stderr)

    def test_both_backends_receive_investigator_role(self):
        for agent in ("codex", "claude"):
            with self.subTest(agent=agent):
                result = self.launch(agent=agent)
                self.assertEqual(result.returncode, 0, result.stderr)
                prompt = self.session()["prompt"]
                self.assertTrue(prompt.startswith("You are the AutoDebug investigator,"))
                self.assertIn("Do not invoke the AutoDebug launcher again.", prompt)
                self.assertIn("Investigate the sample failure", prompt)

    def test_system_bash_handles_focus_and_agent_names(self):
        # /bin/bash is Bash 3.2 on the macOS CI runner.
        for agent in ("CoDeX", "ClAuDe"):
            for focus_paths in ((), ("source path",)):
                with self.subTest(agent=agent, focus_paths=focus_paths):
                    result = self.launch(agent=agent, shell="/bin/bash", focus_paths=focus_paths)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr, "")
                    prompt = self.session()["prompt"]
                    self.assertIn("Investigate the sample failure", prompt)
                    self.assertEqual("`source path`" in prompt, bool(focus_paths))

    def test_prompt_is_unlinked_before_child_starts(self):
        for shell in ("bash", "/bin/bash"):
            for agent in ("codex", "claude"):
                for status in ("0", "7"):
                    with self.subTest(shell=shell, agent=agent, status=status):
                        result = self.launch(agent=agent, shell=shell, SESSION_STATUS=status)
                        self.assertEqual(result.returncode, int(status), result.stderr)
                        session = self.session()
                        self.assertIn("Investigate the sample failure", session["prompt"])
                        self.assertEqual(session["prompt_files_at_start"], [])
                        self.assertEqual(list(self.prompt_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
