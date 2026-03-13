"""LLM abstraction layer — Claude via Anthropic API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


@dataclass
class Finding:
    rule_id: str
    file: str
    line: int
    message: str
    severity: str  # "blocking" | "warning"
    suggested_fix: Optional[str] = None


@dataclass
class LLMSession:
    """Manages a conversation with Claude for one rule or group of rules."""

    model: str = ""
    messages: list[dict] = field(default_factory=list)
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        if not self.model:
            self.model = os.environ.get("BUG_CHECKER_MODEL", DEFAULT_MODEL)
        if self._client is None:
            if anthropic is None:
                raise ImportError("The 'anthropic' package is required. Install it with: pip install anthropic")
            api_key = os.environ.get("BUG_CHECKER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Set BUG_CHECKER_API_KEY or ANTHROPIC_API_KEY environment variable")
            self._client = anthropic.Anthropic(api_key=api_key)

    def analyze_rule(
        self,
        rule_content: str,
        rule_id: str,
        severity: str,
        suggest_fix: bool,
        diff: str,
        context_files: dict[str, str] | None = None,
    ) -> list[Finding]:
        """Run a single rule against the diff. Returns findings."""
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(rule_content, rule_id, severity, suggest_fix, diff, context_files)

        self.messages.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=self.messages,
        )

        assistant_text = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_text})

        return self._parse_findings(assistant_text, rule_id, severity)

    def _build_system_prompt(self) -> str:
        return (
            "You are a code review assistant specialized in finding bugs in C++ and Python code "
            "for the Tenstorrent tt-metal project. You analyze PR diffs against known bug patterns.\n\n"
            "When you find a potential bug, report it in the following structured format, one per finding:\n\n"
            "```finding\n"
            "file: <relative file path>\n"
            "line: <line number in the new code>\n"
            "message: <clear explanation of the bug>\n"
            "suggested_fix: <code fix, or NONE if not requested>\n"
            "```\n\n"
            "If you find no issues matching the rule, respond with exactly:\n"
            "NO_FINDINGS\n\n"
            "Be precise. Only report findings that clearly match the described bug pattern. "
            "Do not report style issues or speculative problems."
        )

    def _build_user_message(
        self,
        rule_content: str,
        rule_id: str,
        severity: str,
        suggest_fix: bool,
        diff: str,
        context_files: dict[str, str] | None,
    ) -> str:
        parts = [
            f"## Rule: {rule_id} (severity: {severity})\n",
            rule_content,
            "\n## PR Diff\n",
            f"```diff\n{diff}\n```\n",
        ]

        if context_files:
            parts.append("\n## Additional Context Files\n")
            for path, content in context_files.items():
                parts.append(f"### {path}\n```\n{content}\n```\n")

        if suggest_fix:
            parts.append("\nPlease include a suggested_fix for each finding. " "Show the corrected code snippet.\n")
        else:
            parts.append("\nDo NOT include suggested fixes. Set suggested_fix to NONE.\n")

        return "\n".join(parts)

    def _parse_findings(self, response_text: str, rule_id: str, severity: str) -> list[Finding]:
        """Parse structured findings from the LLM response."""
        if "NO_FINDINGS" in response_text and "```finding" not in response_text:
            return []

        findings = []
        blocks = response_text.split("```finding")
        for block in blocks[1:]:  # skip text before first finding block
            end = block.find("```")
            if end == -1:
                content = block
            else:
                content = block[:end]

            finding = self._parse_finding_block(content, rule_id, severity)
            if finding:
                findings.append(finding)

        return findings

    def _parse_finding_block(self, block: str, rule_id: str, severity: str) -> Finding | None:
        """Parse a single finding block."""
        fields: dict[str, str] = {}
        current_key = None
        current_lines: list[str] = []

        for line in block.strip().splitlines():
            stripped = line.strip()
            # Check if this line starts a new field
            for key in ("file", "line", "message", "suggested_fix"):
                if stripped.startswith(f"{key}:"):
                    if current_key:
                        fields[current_key] = "\n".join(current_lines).strip()
                    current_key = key
                    current_lines = [stripped[len(key) + 1 :].strip()]
                    break
            else:
                if current_key:
                    current_lines.append(stripped)

        if current_key:
            fields[current_key] = "\n".join(current_lines).strip()

        file_path = fields.get("file", "").strip()
        line_str = fields.get("line", "0").strip()
        message = fields.get("message", "").strip()

        if not file_path or not message:
            return None

        try:
            line_num = int(line_str)
        except ValueError:
            line_num = 0

        suggested_fix = fields.get("suggested_fix", "").strip()
        if suggested_fix == "NONE":
            suggested_fix = None

        return Finding(
            rule_id=rule_id,
            file=file_path,
            line=line_num,
            message=message,
            severity=severity,
            suggested_fix=suggested_fix,
        )
