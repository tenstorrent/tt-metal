"""Tests for LLM response parsing (does not call the API)."""

from bug_checker.llm import Finding, LLMSession


def test_parse_findings_no_findings():
    session = LLMSession.__new__(LLMSession)
    result = session._parse_findings("NO_FINDINGS", "test-rule", "warning")
    assert result == []


def test_parse_findings_single():
    session = LLMSession.__new__(LLMSession)
    text = (
        "I found an issue:\n\n"
        "```finding\n"
        "file: src/foo.cpp\n"
        "line: 42\n"
        "message: Buffer size mismatch between sender and receiver\n"
        "suggested_fix: NONE\n"
        "```\n"
    )
    findings = session._parse_findings(text, "test-rule", "blocking")
    assert len(findings) == 1
    f = findings[0]
    assert f.file == "src/foo.cpp"
    assert f.line == 42
    assert f.message == "Buffer size mismatch between sender and receiver"
    assert f.severity == "blocking"
    assert f.suggested_fix is None


def test_parse_findings_multiple():
    session = LLMSession.__new__(LLMSession)
    text = (
        "Found two issues:\n\n"
        "```finding\n"
        "file: a.cpp\n"
        "line: 10\n"
        "message: First issue\n"
        "suggested_fix: auto x = 1;\n"
        "```\n\n"
        "```finding\n"
        "file: b.cpp\n"
        "line: 20\n"
        "message: Second issue\n"
        "suggested_fix: NONE\n"
        "```\n"
    )
    findings = session._parse_findings(text, "rule-1", "warning")
    assert len(findings) == 2
    assert findings[0].file == "a.cpp"
    assert findings[0].suggested_fix == "auto x = 1;"
    assert findings[1].file == "b.cpp"
    assert findings[1].suggested_fix is None


def test_parse_findings_malformed_ignored():
    session = LLMSession.__new__(LLMSession)
    text = "```finding\n" "line: 5\n" "message: missing file field\n" "```\n"
    findings = session._parse_findings(text, "rule-1", "warning")
    assert len(findings) == 0
