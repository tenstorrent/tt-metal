"""Tests for LLM tool-use parsing (does not call the API)."""

from bug_checker.llm import LLMSession


def _session() -> LLMSession:
    return LLMSession.__new__(LLMSession)


def test_build_findings_empty():
    s = _session()
    assert s._build_findings({"findings": []}, "test-rule", "warning") == []


def test_build_findings_no_findings_key():
    s = _session()
    assert s._build_findings({}, "test-rule", "warning") == []


def test_build_findings_single():
    s = _session()
    tool_input = {
        "findings": [
            {
                "file": "src/foo.cpp",
                "line": 42,
                "message": "Buffer size mismatch between sender and receiver",
                "suggested_fix": None,
            }
        ]
    }
    findings = s._build_findings(tool_input, "test-rule", "blocking")
    assert len(findings) == 1
    f = findings[0]
    assert f.file == "src/foo.cpp"
    assert f.line == 42
    assert f.message == "Buffer size mismatch between sender and receiver"
    assert f.severity == "blocking"
    assert f.suggested_fix is None


def test_build_findings_multiple():
    s = _session()
    tool_input = {
        "findings": [
            {
                "file": "a.cpp",
                "line": 10,
                "message": "First issue",
                "suggested_fix": "auto x = 1;",
            },
            {"file": "b.cpp", "line": 20, "message": "Second issue"},
        ]
    }
    findings = s._build_findings(tool_input, "rule-1", "warning")
    assert len(findings) == 2
    assert findings[0].file == "a.cpp"
    assert findings[0].suggested_fix == "auto x = 1;"
    assert findings[1].file == "b.cpp"
    assert findings[1].suggested_fix is None


def test_build_findings_suggested_fix_empty_string_normalized():
    # Empty string suggested_fix should be normalized to None
    s = _session()
    tool_input = {
        "findings": [
            {"file": "a.cpp", "line": 1, "message": "Bug", "suggested_fix": ""}
        ]
    }
    findings = s._build_findings(tool_input, "rule-1", "blocking")
    assert findings[0].suggested_fix is None


def test_build_findings_preserves_rule_id_and_severity():
    s = _session()
    tool_input = {"findings": [{"file": "x.cpp", "line": 5, "message": "issue"}]}
    findings = s._build_findings(tool_input, "my-rule", "warning")
    assert findings[0].rule_id == "my-rule"
    assert findings[0].severity == "warning"


def test_session_has_no_messages_state():
    """LLMSession must not carry a messages field — each analyze_rule call is stateless."""
    s = _session()
    assert not hasattr(s, "messages"), (
        "LLMSession.messages was re-introduced; each analyze_rule call must be "
        "a fresh single-turn request with no shared history."
    )
