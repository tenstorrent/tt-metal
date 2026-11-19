# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_time_budget


def write_test(path: Path, team) -> None:
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "sample test",
                    "team": team,
                    "budget_type": "pr_gate",
                    "skus": {"wh_n150": {"timeout": 1}},
                }
            ]
        )
    )


class TeamValidationTests(unittest.TestCase):
    def test_invalid_team_is_reported_without_charging_a_budget(self) -> None:
        for team in (None, 7, ["infra"], ""):
            with self.subTest(team=team), tempfile.TemporaryDirectory() as temp_dir:
                write_test(Path(temp_dir) / "sample.yaml", team)
                problems = []

                buckets = verify_time_budget.collect(temp_dir, {}, problems)

                self.assertFalse(buckets)
                self.assertEqual(len(problems), 1)
                self.assertEqual(problems[0]["kind"], "bad_team")
                self.assertEqual(problems[0]["found"], repr(team))

                body = verify_time_budget.comment_body(problems)
                self.assertIn("### Invalid `team` (1)", body)
                self.assertIn(f"got `{team!r}`", body)

    def test_valid_team_is_charged_to_its_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_test(Path(temp_dir) / "sample.yaml", "infra")
            problems = []

            buckets = verify_time_budget.collect(temp_dir, {}, problems)

            self.assertFalse(problems)
            self.assertEqual(dict(buckets), {("infra", "pr_gate", "wh_n150"): {"sample.yaml": 1}})


if __name__ == "__main__":
    unittest.main()
