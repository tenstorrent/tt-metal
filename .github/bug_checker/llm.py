"""LLM abstraction layer — Claude via Anthropic API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from bug_checker.logger import logger

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

REPORT_FINDINGS_TOOL = {
    "name": "report_findings",
    "description": (
        "Report all bugs found in the PR diff that match the rule pattern. "
        "Call with an empty findings list if no issues are found."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "description": "Bugs found. Empty list if none.",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Relative file path where the bug is located",
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number in the new (post-diff) code",
                        },
                        "message": {
                            "type": "string",
                            "description": "Clear explanation of the bug and why it matches the pattern",
                        },
                        "suggested_fix": {
                            "type": "string",
                            "description": "Corrected code snippet, or null if not requested",
                        },
                    },
                    "required": ["file", "line", "message"],
                },
            }
        },
        "required": ["findings"],
    },
}


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
    """Thin wrapper over the Anthropic client for running rule analyses.

    Each call to analyze_rule is a fully independent, stateless API request.
    No conversation history is retained between calls.
    """

    model: str = ""
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        if not self.model:
            self.model = os.environ.get("BUG_CHECKER_MODEL", DEFAULT_MODEL)
        if self._client is None:
            if anthropic is None:
                raise ImportError(
                    "The 'anthropic' package is required. Install it with: pip install anthropic"
                )
            api_key = os.environ.get("BUG_CHECKER_API_KEY") or os.environ.get(
                "ANTHROPIC_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "Set BUG_CHECKER_API_KEY or ANTHROPIC_API_KEY environment variable"
                )
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
        """Run a single rule against the diff. Returns findings.

        Each call is independent — no state is shared with previous calls.
        """
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(
            rule_content, rule_id, severity, suggest_fix, diff, context_files
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=[REPORT_FINDINGS_TOOL],
            tool_choice={"type": "tool", "name": "report_findings"},
        )

        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_use_block is None:
            logger.warning(
                f"Rule {rule_id}: expected tool_use response, got none — skipping"
            )
            return []

        return self._build_findings(tool_use_block.input, rule_id, severity)

    def _build_system_prompt(self) -> str:
        return (
            "You are a code review assistant specialized in finding bugs in C++ and Python code "
            "for the Tenstorrent tt-metal project. You analyze PR diffs against known bug patterns.\n\n"
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
            parts.append(
                "\nPlease include a suggested_fix for each finding. Show the corrected code snippet.\n"
            )
        else:
            parts.append(
                "\nDo NOT include suggested fixes. Leave suggested_fix null.\n"
            )

        return "\n".join(parts)

    def _build_findings(
        self, tool_input: dict, rule_id: str, severity: str
    ) -> list[Finding]:
        """Build Finding objects from a validated tool use input dict."""
        findings = []
        for item in tool_input.get("findings", []):
            suggested_fix = item.get("suggested_fix") or None
            findings.append(
                Finding(
                    rule_id=rule_id,
                    file=item["file"],
                    line=item.get("line", 0),
                    message=item["message"],
                    severity=severity,
                    suggested_fix=suggested_fix,
                )
            )
        return findings
