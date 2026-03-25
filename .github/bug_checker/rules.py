"""Rule loading, parsing, and targeting logic."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from bug_checker.logger import logger


RULES_DIR = Path(__file__).resolve().parent / "rules"
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.yaml"


@dataclass
class Rule:
    id: str
    file: str
    severity: str  # "blocking" | "warning"
    suggest_fix: bool
    model: Optional[str]
    group: Optional[str]
    paths: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    content: str = ""

    def match_reason(self, changed_files: list[str], pr_labels: list[str]) -> str | None:
        """Return a human-readable reason this rule matched, or None if it did not.

        Checks path globs first, then PR labels. Returns the first match found.
        """
        for pattern in self.paths:
            for f in changed_files:
                if fnmatch.fnmatch(f, pattern):
                    return f"path {f!r} matches glob {pattern!r}"
        for label in self.labels:
            if label in pr_labels:
                return f"label {label!r}"
        return None

    def matches_pr(self, changed_files: list[str], pr_labels: list[str]) -> bool:
        """Return True if this rule should run for the given PR."""
        return self.match_reason(changed_files, pr_labels) is not None


def load_manifest() -> dict:
    """Load and return the parsed manifest.yaml."""
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def load_rules() -> list[Rule]:
    """Load all rules from the manifest and their markdown content.

    Warns about orphan rules — rules with neither paths nor labels that can
    never be selected for any PR.
    """
    manifest = load_manifest()
    rules = []
    for rule_id, config in manifest.get("rules", {}).items():
        rule_file = RULES_DIR / config["file"]
        content = rule_file.read_text() if rule_file.exists() else ""
        rule = Rule(
            id=rule_id,
            file=config["file"],
            severity=config.get("severity", "warning"),
            suggest_fix=config.get("suggest_fix", False),
            model=config.get("model"),
            group=config.get("group"),
            paths=config.get("paths", []),
            labels=config.get("labels", []),
            content=content,
        )
        if not rule.paths and not rule.labels:
            logger.warning(
                f"Rule {rule_id!r} has no paths or labels and can never be selected. "
                "Add at least one path glob or label to manifest.yaml."
            )
        rules.append(rule)
    return rules


def select_rules(rules: list[Rule], changed_files: list[str], pr_labels: list[str]) -> list[Rule]:
    """Filter rules to only those that match the PR, logging the reason for each match."""
    selected = []
    for rule in rules:
        reason = rule.match_reason(changed_files, pr_labels)
        if reason:
            logger.debug(f"Rule {rule.id!r} selected: {reason}")
            selected.append(rule)
    return selected


def group_rules(rules: list[Rule]) -> list[list[Rule]]:
    """Group rules by their group name. Ungrouped rules become single-item lists."""
    groups: dict[Optional[str], list[Rule]] = {}
    isolated: list[list[Rule]] = []
    for rule in rules:
        if rule.group is None:
            isolated.append([rule])
        else:
            groups.setdefault(rule.group, []).append(rule)
    return isolated + list(groups.values())
