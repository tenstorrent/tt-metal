from __future__ import annotations

from pathlib import Path

from tools.ci.m5_manage_issue_lifecycle import candidate_github_owners_from_text, parse_codeowners


def test_candidate_github_owners_from_text_uses_last_matching_rule(tmp_path: Path) -> None:
    codeowners = tmp_path / "CODEOWNERS"
    codeowners.write_text(
        "\n".join(
            [
                "ttnn/* @alice",
                "ttnn/special/* @bob",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rules = parse_codeowners(codeowners)
    text = "Failure path: ttnn/special/op/foo.py"
    owners = candidate_github_owners_from_text(text, rules)
    assert owners == ["bob"]
