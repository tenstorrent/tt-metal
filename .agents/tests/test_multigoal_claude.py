#!/usr/bin/env python3
"""Tests for .agents/scripts/multigoal-claude against a fake `claude` CLI.

No hardware, no API calls, no tokens: tests/claude is a shim that replays canned
stream-json events (see fake_claude.py) so the runner's state machine, gate
policy, manifest bookkeeping, and recovery paths can be exercised deterministically.

Run:  python3 .agents/tests/test_multigoal_claude.py
"""
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

TESTS_DIR = pathlib.Path(__file__).resolve().parent
AGENTS_DIR = TESTS_DIR.parent
RUNNER = AGENTS_DIR / "scripts" / "multigoal-claude"

COMPLETE = {"status": "complete", "summary": "all requirements met"}
BLOCKED = {"status": "blocked", "summary": "cannot proceed", "blocker": "device is on fire"}


class MultigoalClaudeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="multigoal-claude-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ws = self.tmp / "ws"
        (self.ws / ".agents" / "skills").mkdir(parents=True)
        self.fake_dir = self.tmp / "fake"
        (self.fake_dir / "responses").mkdir(parents=True)
        self.log_dir = self.tmp / "logs"
        self.prompts_dir = self.tmp / "prompts"
        self.prompts_dir.mkdir()

    # -- helpers -------------------------------------------------------------
    def add_skill(self, name: str, description: str = "") -> pathlib.Path:
        skill = self.ws / ".agents" / "skills" / name / "SKILL.md"
        skill.parent.mkdir(parents=True, exist_ok=True)
        desc = f"description: {description}\n" if description else ""
        skill.write_text(f"---\nname: {name}\n{desc}---\n# {name}\n")
        return skill

    def add_prompt(self, stem: str, body: str, check: str | None = None) -> pathlib.Path:
        prompt = self.prompts_dir / f"{stem}.txt"
        prompt.write_text(body)
        if check is not None:
            (self.prompts_dir / f"{stem}.check.sh").write_text(check)
        return prompt

    def queue_response(self, n: int, **spec) -> None:
        (self.fake_dir / "responses" / f"{n:03d}.json").write_text(json.dumps(spec))

    def calls(self) -> list[dict]:
        path = self.fake_dir / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def manifest(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for raw in (self.log_dir / "manifest.txt").read_text().splitlines():
            if "=" in raw and not raw.startswith("- "):
                key, value = raw.split("=", 1)
                values[key] = value
        return values

    def argv_for(self, *stems: str, extra: list[str] | None = None) -> list[str]:
        argv = [
            sys.executable,
            str(RUNNER),
            "--repo",
            str(self.ws),
            "--log-dir",
            str(self.log_dir),
            "--claude-bin",
            str(TESTS_DIR / "claude"),
            "--replace",
            "HF_MODEL=org/Test-Model",
        ]
        argv.extend(extra or [])
        argv.extend(str(self.prompts_dir / f"{stem}.txt") for stem in stems)
        return argv

    def env_for(self) -> dict[str, str]:
        env = os.environ.copy()
        env["FAKE_CLAUDE_DIR"] = str(self.fake_dir)
        env["ANTHROPIC_API_KEY"] = "sk-test-leak-canary"
        env["ANTHROPIC_AUTH_TOKEN"] = "tok-test-leak-canary"
        return env

    def run_runner(self, *stems: str, extra: list[str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(self.argv_for(*stems, extra=extra), env=self.env_for(), capture_output=True, text=True)

    # -- core flow -----------------------------------------------------------
    def test_two_stages_complete(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.add_prompt("02-b", "/goal Stage two for HF_MODEL.\n")
        self.queue_response(1, verdict=COMPLETE)
        self.queue_response(2, verdict=COMPLETE)
        proc = self.run_runner("01-a", "02-b")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        manifest = self.manifest()
        self.assertEqual(manifest["stage_1_terminal_status"], "complete")
        self.assertEqual(manifest["stage_2_terminal_status"], "complete")
        self.assertEqual(manifest["runner"], "claude-code")
        self.assertIn("| 01-01-a | complete | none |", (self.log_dir / "STATUS.md").read_text())
        stdin = self.calls()[0]["stdin"]
        self.assertIn("org/Test-Model", stdin)
        self.assertNotIn("HF_MODEL", stdin)

    def test_prompt_wrapper_and_skill_context(self) -> None:
        self.add_skill("tt-device-usage", "Use TT devices safely and recover hangs.")
        self.add_prompt("01-a", "/goal Use $tt-device-usage and $nonexistent-skill for HF_MODEL.\n")
        self.queue_response(1, verdict=COMPLETE)
        proc = self.run_runner("01-a")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        stdin = self.calls()[0]["stdin"]
        self.assertIn("===== GOAL =====", stdin)
        self.assertIn("- $tt-device-usage: read .agents/skills/tt-device-usage/SKILL.md", stdin)
        self.assertIn("Use TT devices safely and recover hangs.", stdin)
        self.assertIn("Referenced skills with no SKILL.md found", stdin)
        self.assertLess(stdin.index("===== GOAL ====="), stdin.index("Terminal contract"))
        self.assertEqual(self.manifest()["stage_1_missing_skills"], "nonexistent-skill")

    def test_verdict_schema_is_passed(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(1, verdict=COMPLETE)
        self.run_runner("01-a")
        schema = json.loads(self.calls()[0]["json_schema"])
        self.assertEqual(schema["properties"]["status"]["enum"], ["complete", "blocked"])
        self.assertEqual(schema["required"], ["status", "summary"])

    def test_blocked_verdict_stops_with_exit_3(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.add_prompt("02-b", "/goal Stage two for HF_MODEL.\n")
        self.queue_response(1, verdict=BLOCKED)
        proc = self.run_runner("01-a", "02-b")
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(self.manifest()["stage_1_terminal_status"], "blocked")
        self.assertIn("device is on fire", proc.stderr)
        self.assertEqual(len(self.calls()), 1, "stage 2 must not launch")

    def test_missing_verdict_stops_the_pipeline(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(1, verdict=None, result_text="I did lots of work and it all passed!")
        proc = self.run_runner("01-a")
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotEqual(self.manifest()["stage_1_terminal_status"], "complete")

    # -- auth hygiene --------------------------------------------------------
    def test_api_key_never_reaches_the_agent(self) -> None:
        """An ambient key would silently move an unattended run onto metered billing."""
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(1, verdict=COMPLETE)
        proc = self.run_runner("01-a")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        call = self.calls()[0]
        self.assertFalse(call["api_key_leaked"])
        self.assertFalse(call["auth_token_leaked"])

    def test_claude_config_dir_recorded(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(1, verdict=COMPLETE)
        home = self.tmp / "claude-home"
        proc = self.run_runner("01-a", extra=["--claude-config-dir", str(home)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.manifest()["claude_config_dir"], str(home))

    def test_missing_claude_bin_fails_at_launch(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        proc = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--repo",
                str(self.ws),
                "--log-dir",
                str(self.log_dir),
                "--claude-bin",
                "definitely-not-a-real-binary-xyz",
                "--replace",
                "HF_MODEL=org/Test-Model",
                str(self.prompts_dir / "01-a.txt"),
            ],
            env=self.env_for(),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not found on PATH", proc.stderr)
        self.assertEqual(self.calls(), [], "no agent should have been launched")

    # -- infra terminal classification ---------------------------------------
    def test_max_turns_is_distinct_and_resumable(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(
            1,
            subtype="error_max_turns",
            is_error=True,
            terminal_reason="max_turns",
            errors=["Reached maximum number of turns (1)"],
            exit_code=1,
        )
        proc = self.run_runner("01-a")
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(self.manifest()["stage_1_terminal_status"], "maxTurns")
        # detail must come from the error variant's errors[] array
        self.assertIn("Reached maximum number of turns", proc.stderr)

    def test_schema_retry_exhausted_is_recoverable(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(
            1,
            subtype="error_max_structured_output_retries",
            is_error=True,
            terminal_reason="structured_output_retry_exhausted",
            errors=["Could not produce conforming output"],
            exit_code=1,
        )
        proc = self.run_runner("01-a")
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(self.manifest()["stage_1_terminal_status"], "blocked")
        self.assertIn("Could not produce conforming output", proc.stderr)

    def test_usage_limit_maps_to_exit_3(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(
            1,
            subtype="error_during_execution",
            is_error=True,
            errors=["Your usage limit reached; resets at 3pm"],
            exit_code=1,
        )
        proc = self.run_runner("01-a")
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(self.manifest()["stage_1_terminal_status"], "usageLimited")

    def test_execution_error_maps_to_exit_5(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(
            1,
            subtype="error_during_execution",
            is_error=True,
            errors=["API connection torn down"],
            exit_code=1,
        )
        proc = self.run_runner("01-a")
        self.assertEqual(proc.returncode, 5, proc.stderr)
        self.assertEqual(self.manifest()["stage_1_terminal_status"], "turnFailed")
        self.assertIn("API connection torn down", proc.stderr)

    # -- check gates ---------------------------------------------------------
    def test_check_passes_after_remediation(self) -> None:
        check = (
            "#!/usr/bin/env bash\n"
            'count_file="$MULTIGOAL_LOG_DIR/check_count"\n'
            'n=$(cat "$count_file" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$count_file"\n'
            "[ $n -ge 2 ] && exit 0\n"
            "echo 'artifact missing: readiness table' >&2\n"
            "exit 1\n"
        )
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n", check=check)
        self.queue_response(1, verdict=COMPLETE)
        self.queue_response(2, verdict=COMPLETE)
        proc = self.run_runner("01-a", extra=["--replace", "MODEL_DIR=models/autoports/test"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        manifest = self.manifest()
        self.assertEqual(manifest["stage_1_check"], "pass")
        self.assertEqual(manifest["stage_1_check_attempt_1"], "exit1")
        self.assertEqual(manifest["stage_1_check_attempt_2"], "exit0")
        self.assertIn("stage_1_remediation_1_session_id", manifest)
        remediation_stdin = self.calls()[1]["stdin"]
        self.assertIn("FAILED (exit 1, ADVISORY)", remediation_stdin)
        self.assertIn("treat it as a bug report", remediation_stdin)

    def test_critical_check_exits_6(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n", check="#!/usr/bin/env bash\nexit 2\n")
        self.queue_response(1, verdict=COMPLETE)
        self.queue_response(2, verdict=COMPLETE)
        proc = self.run_runner("01-a", extra=["--replace", "MODEL_DIR=models/autoports/test"])
        self.assertEqual(proc.returncode, 6, proc.stderr)
        self.assertEqual(self.manifest()["stage_1_check"], "critical-fail")

    def test_broken_check_exits_7_after_one_retry(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n", check="#!/usr/bin/env bash\nexit 3\n")
        self.queue_response(1, verdict=COMPLETE)
        proc = self.run_runner("01-a", extra=["--replace", "MODEL_DIR=models/autoports/test"])
        self.assertEqual(proc.returncode, 7, proc.stderr)
        manifest = self.manifest()
        self.assertEqual(manifest["stage_1_check"], "check-error(exit3)")
        self.assertEqual(manifest["stage_1_check_attempt_1_retry"], "exit3")

    def test_hung_check_is_killed_and_treated_as_broken(self) -> None:
        """Without --check-timeout a hung check script hangs the whole pipeline."""
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n", check="#!/usr/bin/env bash\nsleep 60\n")
        self.queue_response(1, verdict=COMPLETE)
        proc = self.run_runner("01-a", extra=["--replace", "MODEL_DIR=models/autoports/test", "--check-timeout", "1"])
        self.assertEqual(proc.returncode, 7, proc.stderr)
        self.assertEqual(self.manifest()["stage_1_check"], "check-error(exit124)")

    def test_model_dir_required_when_checks_exist(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n", check="#!/usr/bin/env bash\nexit 0\n")
        proc = self.run_runner("01-a")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("MODEL_DIR is required", proc.stderr)

    # -- timeouts, resume, crash recovery ------------------------------------
    def test_stage_timeout_fires_during_total_silence(self) -> None:
        """The child emits init then stops emitting anything. An inline timeout check
        inside the stdout loop can never fire here because the loop blocks in
        readline(); the watchdog must kill the process."""
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(1, hang=True)
        proc = self.run_runner("01-a", extra=["--stage-timeout", "2"])
        self.assertEqual(proc.returncode, 5, proc.stderr)
        self.assertEqual(self.manifest()["stage_1_terminal_status"], "turnFailed")
        self.assertIn("stage-timeout", proc.stderr)

    def test_session_id_pinned_and_never_combined_with_resume(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(1, verdict=COMPLETE)
        proc = self.run_runner("01-a")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        call = self.calls()[0]
        self.assertIsNotNone(call["session_id_flag"])
        self.assertIsNone(call["resume"])
        self.assertEqual(self.manifest()["stage_1_session_id"], call["session_id_flag"])

    def test_crash_mid_stage_still_leaves_a_resume_anchor(self) -> None:
        """The whole point of --resume-stage is recovering a run that died mid-stage.
        If the session id is only recorded after the goal returns, a crash leaves
        nothing to resume from."""
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(1, hang=True)
        proc = subprocess.Popen(
            self.argv_for("01-a"),
            env=self.env_for(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        manifest_path = self.log_dir / "manifest.txt"
        session_id = None
        for _ in range(100):  # up to ~10s
            if manifest_path.exists() and "stage_1_session_id" in self.manifest():
                session_id = self.manifest()["stage_1_session_id"]
                break
            time.sleep(0.1)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        self.assertIsNotNone(session_id, "manifest had no session id when the runner was killed")

        self.queue_response(2, verdict=COMPLETE)
        resumed = self.run_runner("01-a", extra=["--resume-stage", "1"])
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(self.calls()[1]["resume"], session_id)
        self.assertIsNone(self.calls()[1]["session_id_flag"])
        self.assertEqual(self.manifest()["stage_1_resume_1_terminal_status"], "complete")

    def test_resume_stage_continues_recorded_session(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.add_prompt("02-b", "/goal Stage two for HF_MODEL.\n")
        self.queue_response(1, verdict=BLOCKED)
        proc = self.run_runner("01-a", "02-b")
        self.assertEqual(proc.returncode, 3, proc.stderr)
        original = self.manifest()["stage_1_session_id"]
        self.queue_response(2, verdict=COMPLETE)
        self.queue_response(3, verdict=COMPLETE)
        proc = self.run_runner("01-a", "02-b", extra=["--resume-stage", "1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        calls = self.calls()
        self.assertEqual(calls[1]["resume"], original)
        self.assertIn("Continue this existing multigoal stage", calls[1]["stdin"])
        self.assertIn("===== GOAL (original) =====", calls[1]["stdin"])
        self.assertIsNone(calls[2]["resume"], "later stages get fresh sessions")
        manifest = self.manifest()
        self.assertEqual(manifest["stage_1_resume_1_session_id"], original)
        self.assertEqual(manifest["stage_2_terminal_status"], "complete")

    # -- passthrough and large prompts ---------------------------------------
    def test_settings_and_claude_arg_passthrough(self) -> None:
        self.add_prompt("01-a", "/goal Stage one for HF_MODEL.\n")
        self.queue_response(1, verdict=COMPLETE)
        proc = self.run_runner(
            "01-a",
            # flag-shaped values need the = form, else argparse eats them as options
            extra=["--settings", '{"env":{"X":"1"}}', "--claude-arg=--max-turns", "--claude-arg=500"],
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        call = self.calls()[0]
        self.assertEqual(call["settings"], '{"env":{"X":"1"}}')
        self.assertIn("--max-turns", call["argv"])
        self.assertIn("500", call["argv"])

    def test_oversized_prompt_does_not_deadlock(self) -> None:
        # Well past the ~64 KiB OS pipe buffer (writing stdin inline would
        # deadlock) while staying inside MAX_GOAL_OBJECTIVE_CHARS.
        filler = "x" * 90_000
        self.add_prompt("01-a", f"/goal Stage one for HF_MODEL.\n\n{filler}\n")
        self.queue_response(1, verdict=COMPLETE)
        proc = self.run_runner("01-a")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertGreater(self.calls()[0]["stdin_len"], 90_000)

    def test_dry_run_over_the_real_stage_prompts(self) -> None:
        prompts = sorted((AGENTS_DIR / "prompts" / "model_bringup_multigoal").glob("*.txt"))
        self.assertGreater(len(prompts), 0)
        proc = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--dry-run",
                "--repo",
                str(AGENTS_DIR.parent),
                "--log-dir",
                str(self.log_dir),
                "--replace",
                "HF_MODEL=org/Test-Model",
                "--replace",
                "MODEL_DIR=models/autoports/org_test_model",
                *[str(p) for p in prompts],
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        manifest = self.manifest()
        for index in range(1, len(prompts) + 1):
            self.assertEqual(manifest[f"stage_{index}_dry_run"], "true")


if __name__ == "__main__":
    unittest.main(verbosity=2)
