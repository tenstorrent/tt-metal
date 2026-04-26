from __future__ import annotations

from pathlib import Path

from tools.ci.m5_manage_issue_lifecycle import (
    candidate_github_owners_from_text,
    issue_numbers_from_text,
    message_replies,
    parse_json_after_marker,
    parse_codeowners,
)


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


def test_parse_json_after_marker_success() -> None:
    text = 'noise\n===FINAL_OWNER_CLAIM_JSON===\n{"claimed": true, "slack_user_id": "U123"}\n'
    parsed = parse_json_after_marker(text, "===FINAL_OWNER_CLAIM_JSON===")
    assert parsed["claimed"] is True
    assert parsed["slack_user_id"] == "U123"


def test_parse_json_after_marker_missing_marker_raises() -> None:
    text = '{"claimed": false}'
    try:
        parse_json_after_marker(text, "===FINAL_OWNER_CLAIM_JSON===")
    except ValueError as exc:
        assert "marker not found" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing marker")


def test_issue_numbers_from_text_parses_issue_dump_urls() -> None:
    text = (
        "Investigate https://github.com/ebanerjeeTT/issue_dump/issues/851 and "
        "https://github.com/ebanerjeeTT/issue_dump/issues/858"
    )
    assert issue_numbers_from_text(text) == [851, 858]


def test_message_replies_supports_legacy_replies_key() -> None:
    msg = {"replies": [{"ts": "1.0", "user": "U1", "text": "working on it"}]}
    out = message_replies(msg)
    assert len(out) == 1
    assert out[0]["user"] == "U1"
