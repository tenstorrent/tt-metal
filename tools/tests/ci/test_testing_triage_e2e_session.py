from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools/ci/testing_triage_e2e_session.py"
    spec = importlib.util.spec_from_file_location("testing_triage_e2e_session", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_issue_number_from_text() -> None:
    mod = _load_module()
    text = "Issue: https://github.com/ebanerjeeTT/issue_dump/issues/1234"
    assert mod.parse_issue_number(text) == 1234


def test_build_bot_response_fix_request_contains_mock_pr() -> None:
    mod = _load_module()
    text = mod.build_bot_response(
        issue_number=77,
        progress={"defer_disable": False},
        fix_request={"requested": True},
        mock_github_owner="test-owner",
    )
    assert "Mock draft PR" in text
    assert "mock-77" in text
