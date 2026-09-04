from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from unittest import mock


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "multigoal"
LOADER = importlib.machinery.SourceFileLoader("tt_metal_multigoal", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
MULTIGOAL = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = MULTIGOAL
LOADER.exec_module(MULTIGOAL)


class ShellProfileConfigTests(unittest.TestCase):
    def test_parse_args_applies_the_default(self) -> None:
        with mock.patch.object(sys, "argv", ["multigoal", "goal.txt"]):
            args = MULTIGOAL.parse_args()

        self.assertEqual(args.config, ["shell_environment_policy.experimental_use_profile=false"])

    def test_shell_profile_loading_is_disabled_by_default(self) -> None:
        self.assertEqual(
            MULTIGOAL.with_shell_profile_default([]),
            ["shell_environment_policy.experimental_use_profile=false"],
        )

    def test_explicit_shell_profile_setting_is_preserved(self) -> None:
        config = ["model_reasoning_effort=high", "shell_environment_policy.experimental_use_profile=true"]

        self.assertEqual(MULTIGOAL.with_shell_profile_default(config), config)

    def test_unrelated_config_overrides_are_preserved(self) -> None:
        self.assertEqual(
            MULTIGOAL.with_shell_profile_default(["model_reasoning_effort=high"]),
            [
                "shell_environment_policy.experimental_use_profile=false",
                "model_reasoning_effort=high",
            ],
        )


class PersistentLogTests(unittest.TestCase):
    def setUp(self) -> None:
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.repo = self.root / "workspace"
        self.repo.mkdir()
        self.codex_home = self.root / "codex-home"
        self.tmp_dir = self.root / "volatile-tmp"
        self.tmp_dir.mkdir()
        (self.tmp_dir / "scratch.txt").write_text("temporary data")
        self.prompt = self.root / "goal.txt"
        self.prompt.write_text("/goal Complete this model bringup stage.\n")
        self.run_id = "20260904T120000Z"
        self.log_dir = self.repo / "bringup" / "artifacts" / "multigoal-runs" / self.run_id
        stack.enter_context(
            mock.patch.dict(
                os.environ,
                {"CODEX_HOME": str(self.codex_home), "TMPDIR": str(self.tmp_dir)},
                clear=True,
            )
        )
        stack.enter_context(mock.patch.object(MULTIGOAL, "timestamp", return_value=self.run_id))
        stack.enter_context(redirect_stderr(io.StringIO()))

    def run_main(self, *options: str) -> None:
        argv = ["multigoal", str(self.prompt), "--repo", str(self.repo), "--codex-bin", "unused", *options]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(MULTIGOAL.main(), 0)

    def test_default_logs_are_written_under_the_selected_repo(self) -> None:
        self.run_main("--dry-run")

        manifest = MULTIGOAL.read_manifest(self.log_dir / "manifest.txt")
        self.assertEqual(manifest["codex_home"], str(self.codex_home))
        self.assertEqual(manifest["stage_1_dry_run"], "true")
        self.assertFalse((self.codex_home / "multigoal-runs").exists())
        self.assertFalse((self.tmp_dir / "codex-multigoal-runs").exists())

    def test_explicit_codex_home_keeps_logs_in_the_workspace(self) -> None:
        explicit_home = self.root / "explicit-codex-home"
        self.run_main("--dry-run", "--codex-home", str(explicit_home))

        manifest = MULTIGOAL.read_manifest(self.log_dir / "manifest.txt")
        self.assertEqual(manifest["codex_home"], str(explicit_home))
        self.assertFalse((explicit_home / "multigoal-runs").exists())

    def test_environment_override_is_a_parent_directory(self) -> None:
        log_root = self.root / "custom-logs"
        with mock.patch.dict(os.environ, {"RUN_MULTIGOAL_LOG_DIR": str(log_root)}):
            self.run_main("--dry-run")

        self.assertTrue((log_root / self.run_id / "manifest.txt").is_file())
        self.assertFalse(self.log_dir.exists())

    def test_log_dir_overrides_environment_and_is_used_exactly(self) -> None:
        log_root = self.root / "custom-logs"
        explicit_dir = self.root / "chosen-run"
        with mock.patch.dict(os.environ, {"RUN_MULTIGOAL_LOG_DIR": str(log_root)}):
            self.run_main("--dry-run", "--log-dir", str(explicit_dir))

        self.assertTrue((explicit_dir / "manifest.txt").is_file())
        self.assertFalse(log_root.exists())
        self.assertFalse(self.log_dir.exists())

    def test_resume_after_temporary_storage_is_cleared(self) -> None:
        def interrupt_goal(*args, on_thread_started):
            on_thread_started("stage-4-thread")
            raise RuntimeError("simulated interruption")

        with mock.patch.object(MULTIGOAL, "AppServerClient"), mock.patch.object(
            MULTIGOAL, "execute_goal", side_effect=interrupt_goal
        ) as start_goal:
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_main("--start-index", "4")

            manifest = self.log_dir / "manifest.txt"
            original_manifest = manifest.read_text()
            self.assertEqual(MULTIGOAL.read_manifest(manifest)["stage_4_thread_id"], "stage-4-thread")
            shutil.rmtree(self.tmp_dir)

            with mock.patch.object(MULTIGOAL, "execute_resumed_goal", return_value=("complete", None)) as resume_goal:
                self.run_main("--start-index", "4", "--resume-stage", "4", "--log-dir", str(self.log_dir))

            start_goal.assert_called_once()
            resume_goal.assert_called_once()
            self.assertEqual(resume_goal.call_args.args[4], "stage-4-thread")
            self.assertTrue(manifest.read_text().startswith(original_manifest))
            self.assertEqual(MULTIGOAL.read_manifest(manifest)["stage_4_resume_1_terminal_status"], "complete")


if __name__ == "__main__":
    unittest.main()
