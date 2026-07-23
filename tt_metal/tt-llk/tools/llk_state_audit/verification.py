"""Reviewed links between audit claims and LLK state-transition tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inventory import AuditModelError

_SCOPES = {
    "runtime_reconfigure",
    "restore",
    "canonical_baseline",
    "synchronization",
}
_ENTRY_KEYS = {
    "path",
    "architectures",
    "functions",
    "families",
    "domains",
    "validation_scope",
}
_ARCHITECTURES = {"wormhole_b0", "blackhole", "quasar"}


def load_verification_manifest(
    root: Path | str,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate the reviewed test-to-audit mapping."""
    root_path = Path(root).resolve()
    manifest_path = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("verification_manifest.json")
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditModelError(f"cannot load verification manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "tests"}
        or manifest["schema_version"] != 1
        or not isinstance(manifest["tests"], list)
        or not manifest["tests"]
    ):
        raise AuditModelError(
            "verification manifest requires schema_version 1 and tests"
        )
    seen_paths: set[str] = set()
    for entry in manifest["tests"]:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise AuditModelError("verification entry has an invalid schema")
        if (
            not isinstance(entry["path"], str)
            or not entry["path"]
            or entry["path"] in seen_paths
        ):
            raise AuditModelError("verification test paths must be unique strings")
        seen_paths.add(entry["path"])
        if not (root_path / entry["path"]).is_file():
            raise AuditModelError(f"verification test does not exist: {entry['path']}")
        if entry["validation_scope"] not in _SCOPES:
            raise AuditModelError("verification entry has an unknown validation scope")
        if not _string_list(entry["architectures"]) or not set(
            entry["architectures"]
        ).issubset(_ARCHITECTURES):
            raise AuditModelError("verification entry has invalid architectures")
        for key in ("functions", "families", "domains"):
            if not _string_list(entry[key]):
                raise AuditModelError(f"verification entry requires non-empty {key}")
    return manifest


def validate_verification_manifest(
    manifest: dict[str, Any],
    audit_data: dict[str, Any],
) -> None:
    """Require reviewed function/domain claims to exist in the current audit."""
    definitions = {
        (item["architecture"], item["name"])
        for item in audit_data["inventory"]["functions"]
    }
    domains: dict[tuple[str, str], set[str]] = {}
    for effect in audit_data["effects"]:
        domains.setdefault(
            (effect["architecture"], effect["function"]),
            set(),
        ).add(effect["domain"])
    for entry in manifest["tests"]:
        for architecture in entry["architectures"]:
            missing_functions = [
                function
                for function in entry["functions"]
                if (architecture, function) not in definitions
            ]
            if missing_functions:
                raise AuditModelError(
                    "verification manifest references missing functions: "
                    f"{architecture}/{missing_functions}"
                )
            mapped_domains = set().union(
                *(
                    domains.get((architecture, function), set())
                    for function in entry["functions"]
                )
            )
            missing_domains = set(entry["domains"]) - mapped_domains
            if missing_domains:
                raise AuditModelError(
                    "verification manifest domains are not audit-backed: "
                    f"{architecture}/{entry['path']}/{sorted(missing_domains)}"
                )


def _string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item) for item in value)
    )
