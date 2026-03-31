from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_m4_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools/ci/m4_create_issues_and_notify.py"
    spec = importlib.util.spec_from_file_location("m4_create_issues_and_notify", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_batch_agent_json_accepts_raw_json_without_marker():
    mod = _load_m4_module()
    text = (
        '{"decisions":[{"workflow_name":"wf","job_name":"job","job_urls":["u1","u2","u3"],'
        '"deterministic":true,"confidence":"high","signature":"abc","error_excerpt":"x",'
        '"reason":"ok","create_issue":true,"draft_slack":true,"issue_title":"t","issue_body":"b","slack_text":"s"}]}'
    )
    decisions = mod.parse_batch_agent_json(text)
    assert isinstance(decisions, list)
    assert len(decisions) == 1
    assert decisions[0]["workflow_name"] == "wf"


def test_parse_batch_agent_json_missing_marker_has_actionable_error():
    mod = _load_m4_module()
    with pytest.raises(ValueError) as exc:
        mod.parse_batch_agent_json("agent output without marker and without json payload")
    message = str(exc.value)
    assert "marker not found" in message
    assert "output excerpt" in message
