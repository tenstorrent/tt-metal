import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
WORKFLOW = ROOT / ".github" / "workflows" / "llk-state-audit.yaml"
LLK_GITIGNORE = ROOT / "tt_metal" / "tt-llk" / ".gitignore"


class LlkStateWorkflowTest(unittest.TestCase):
    def test_workflow_runs_tests_model_and_artifact_drift_checks(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("tt_metal/tt-llk/**", text)
        self.assertIn("python3 -m unittest discover", text)
        self.assertIn("python3 -m tools.llk_state_audit check --root .", text)
        self.assertIn("python3 -m tools.llk_state_audit verify --root .", text)
        self.assertIn("working-directory: tt_metal/tt-llk", text)

    def test_generated_state_map_csv_is_not_ignored(self) -> None:
        text = LLK_GITIGNORE.read_text(encoding="utf-8")
        self.assertIn(
            "!docs/llk_state_audit/llk_state_map.csv",
            text.splitlines(),
        )


if __name__ == "__main__":
    unittest.main()
