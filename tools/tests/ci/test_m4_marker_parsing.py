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


def test_find_existing_issue_for_job_identity_matches_newly_created_body_marker():
    mod = _load_m4_module()
    workflow_name = "aggregate-workflow-data"
    job_name = "Galaxy Qwen3-32B long context demo tests"
    job_key = mod.job_identity_key(workflow_name, job_name)
    marker = mod.issue_job_identity_marker(job_key)
    open_issues = [{"url": "https://github.com/ebanerjeeTT/issue_dump/issues/999", "body": f"foo\n{marker}\nbar"}]
    existing = mod.find_existing_issue_for_job_identity(
        open_issues,
        workflow_name=workflow_name,
        job_name=job_name,
    )
    assert existing == "https://github.com/ebanerjeeTT/issue_dump/issues/999"


def test_find_existing_issue_for_title_matches_case_insensitive_exact():
    mod = _load_m4_module()
    open_issues = [
        {
            "url": "https://github.com/ebanerjeeTT/issue_dump/issues/854",
            "title": "[CI] Galaxy Qwen3-32B long context demo: trace buffer size exceeds allocated region",
        }
    ]
    existing = mod.find_existing_issue_for_title(
        open_issues,
        "[ci] galaxy qwen3-32b long context demo: trace buffer size exceeds allocated region",
    )
    assert existing == "https://github.com/ebanerjeeTT/issue_dump/issues/854"


def test_slack_lookup_by_full_name_exact_and_unique_word_match():
    mod = _load_m4_module()
    members = [
        {
            "id": "U1",
            "name": "utku",
            "profile": {"display_name": "Utku Aydonat", "real_name": "Utku Aydonat"},
        },
        {
            "id": "U2",
            "name": "other",
            "profile": {"display_name": "Ali User", "real_name": "Ali User"},
        },
    ]
    assert mod.slack_lookup_by_full_name("Utku Aydonat", members) == "U1"
    assert mod.slack_lookup_by_full_name("Utku", members) == "U1"


def test_slack_lookup_by_username_accepts_tt_suffix_variants():
    mod = _load_m4_module()
    members = [
        {
            "id": "U1",
            "name": "aliu",
            "profile": {"display_name": "Aliu", "real_name": "Ali User"},
        }
    ]
    assert mod.slack_lookup_by_username("", "aliuTT", members) == "U1"


def test_owners_from_workflow_name_matches_codeowners_workflow_rules(tmp_path):
    mod = _load_m4_module()
    wf_root = tmp_path / ".github" / "workflows"
    wf_root.mkdir(parents=True)
    (wf_root / "t3000-unit-tests-impl.yaml").write_text("name: test\n", encoding="utf-8")
    rules = [(".github/workflows/t3000-unit-tests-impl.yaml", ["aliuTT", "cfjchu"])]
    owners = mod.owners_from_workflow_name(
        "T3K T3000 unit tests",
        rules=rules,
        workflow_root=wf_root,
    )
    assert owners == {"aliuTT", "cfjchu"}


def test_render_owner_mentions_caps_to_top_three(monkeypatch):
    mod = _load_m4_module()

    monkeypatch.setattr(mod, "parse_codeowners", lambda _path: [("tests/", ["u1", "u2", "u3", "u4"])])
    monkeypatch.setattr(mod, "extract_repo_paths", lambda _text: ["tests/foo.py"])
    monkeypatch.setattr(mod, "owners_for_paths", lambda _paths, _rules: {"u1", "u2", "u3", "u4"})
    monkeypatch.setattr(mod, "owners_from_workflow_name", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        mod,
        "github_user_info",
        lambda _token, username: {"login": username, "name": username, "email": ""},
    )
    username_to_uid = {"u1": "U1", "u2": "U2", "u3": "U3", "u4": "U4"}
    monkeypatch.setattr(mod, "slack_lookup_by_username", lambda *_args: username_to_uid[_args[1]])
    monkeypatch.setattr(mod, "slack_lookup_by_full_name", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "recent_author_emails_for_paths", lambda _paths: set())
    monkeypatch.setattr(mod, "slack_lookup_by_email", lambda *_args, **_kwargs: None)

    mentions, unresolved, selection = mod.render_owner_mentions(
        issue_token="x",
        slack_token="y",
        workflow_name="wf",
        job_name="job",
        text_sources=["x"],
        members_cache=[],
    )
    tokens = [tok for tok in mentions.split(" ") if tok.strip()]
    assert len(tokens) == 3
    assert unresolved == []
    assert selection["selected_owner_count"] == 3
