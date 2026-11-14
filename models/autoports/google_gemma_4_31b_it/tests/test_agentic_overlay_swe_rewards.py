# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "overlay", Path(__file__).parents[1] / "tools/agentic_overlay_swe_rewards.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.jobs, self.tasks, self.recovery = root / "job", root / "tasks", root / "recovery"
        trial = self.jobs / "task__trial"
        for directory in (trial / "verifier", self.tasks / "task/tests", self.recovery / "task"):
            directory.mkdir(parents=True)
        source = trial / "result.json"
        native = dict(
            task_name="task",
            trial_name="task__trial",
            finished_at="2026-01-01T00:00:01Z",
            verifier_result={"rewards": {"reward": 0}},
        )
        source.write_text(json.dumps(native))
        (trial / "verifier/test-stdout.txt").write_text("original log")
        (self.tasks / "task/tests/config.json").write_text("{}")
        directory = self.recovery / "task"
        (directory / "parser.py").write_text("parser")
        (directory / "test-output.original.txt").write_text("test output")
        (directory / "report.json").write_text(
            json.dumps({"task": {"resolved": True, "patch_successfully_applied": True}})
        )
        self.evidence = dict(
            status="parsed",
            task="task",
            trial="task__trial",
            raw_result=str(source),
            raw_reward=native["verifier_result"],
            recovered_reward=1,
            returncode=0,
            raw_log_sha256=module.file_sha(trial / "verifier/test-stdout.txt"),
            config_sha256=module.file_sha(self.tasks / "task/tests/config.json"),
            parser_sha256=module.file_sha(directory / "parser.py"),
            test_output_sha256=module.file_sha(directory / "test-output.original.txt"),
        )
        self.save()
        attempt = dict(
            task="task", trial_name="task__trial", source_result=str(source), solved=False, rewards={"reward": 0}
        )
        self.raw = dict(
            score={"solved": 0, "n": 2, "complete": False},
            attempts=[attempt],
            tasks=[dict(attempt), {"task": "missing", "solved": None}],
        )

    def save(self):
        (self.recovery / "task/provenance.json").write_text(json.dumps(self.evidence))
        (self.recovery / "recovery-summary.json").write_text(json.dumps({"rows": [self.evidence]}))

    def apply(self):
        return module.overlay(self.raw, self.recovery, self.jobs, self.tasks)

    def test_overlay_preserves_raw_and_missing_reward(self):
        result = self.apply()
        self.assertFalse(self.raw["tasks"][0]["solved"])
        self.assertTrue(result["tasks"][0]["solved"])
        self.assertIsNone(result["tasks"][1]["solved"])
        self.assertEqual(result["score"]["unresolved"], ["missing"])
        self.assertFalse(result["score"]["complete"])
        self.assertIsNone(result["score"]["interval_95"])
        self.assertEqual(result["tasks"][0]["rewards"], {"reward": 0})

    def test_rejects_changed_raw_log(self):
        (self.jobs / "task__trial/verifier/test-stdout.txt").write_text("different log")
        with self.assertRaisesRegex(ValueError, "Raw log hash"):
            self.apply()

    def test_rejects_changed_config(self):
        (self.tasks / "task/tests/config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "config hash"):
            self.apply()

    def test_rejects_other_run(self):
        self.evidence["raw_result"] = str(self.jobs / "other/result.json")
        self.save()
        with self.assertRaisesRegex(ValueError, "different job"):
            self.apply()

    def test_rejects_report_status_conflict(self):
        (self.recovery / "task/report.json").write_text(
            json.dumps({"task": {"resolved": False, "patch_successfully_applied": True}})
        )
        with self.assertRaisesRegex(ValueError, "reward mismatch"):
            self.apply()


if __name__ == "__main__":
    unittest.main()
