# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicit DB-backed additions for missing canonical headers, never overwrites.

An approved substitution relaxes historical dependency identity, not source,
golden-outcome, cache, or review gates. Approval text is an audit attestation,
not authentication. The caller must obtain actual user authorization.
"""

from pathlib import Path

from tools.generic_op_to_factory.export_run import ExportError, _hash_file, _safe_path, verify_export


def resolve(specifications, tree):
    """Verify immutable donor exports and reject replacements of pinned files."""
    if not isinstance(specifications, list):
        raise ExportError("dependency_substitutions must be an explicit list")
    required = {"export", "source_path", "destination", "approval", "reason", "limitations"}
    resolved = []
    destinations = set()
    for specification in specifications:
        if not isinstance(specification, dict) or specification.keys() != required:
            raise ExportError("Dependency substitution has missing or unknown keys")
        if any(not isinstance(value, str) or not value.strip() for value in specification.values()):
            raise ExportError("Dependency substitution needs nonempty paths, approval, reason and limitations")
        package = Path(specification["export"])
        if not package.is_absolute() or package.resolve() != package:
            raise ExportError("Dependency export must be an absolute, unredirected path")
        source = _safe_path(specification["source_path"])
        destination = _safe_path(specification["destination"])
        if (
            not destination.startswith("ttnn/cpp/ttnn/kernel_lib/")
            or Path(destination).suffix not in (".h", ".hpp", ".inl")
            or Path(source).suffix not in (".h", ".hpp", ".inl")
        ):
            raise ExportError("Dependency substitutions only add missing canonical kernel_lib headers")
        if any(
            destination == name or destination.startswith(name + "/") or name.startswith(destination + "/")
            for name in tree.entries
        ):
            raise ExportError(f"Dependency destination conflicts with a pinned Git entry: {destination}")
        if any(
            destination == name or destination.startswith(name + "/") or name.startswith(destination + "/")
            for name in destinations
        ):
            raise ExportError("Duplicate or colliding dependency destinations")
        destinations.add(destination)
        manifest = verify_export(package)
        entries = [entry for entry in manifest["files"] if entry["path"] == source]
        if len(entries) != 1 or not entries[0].get("materialized"):
            raise ExportError("Dependency source must be materialized text from a verified DB export")
        entry = entries[0]
        resolved.append(
            {
                **specification,
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
                "database": manifest["database"],
                "run_id": manifest["run_id"],
                "table": entry["table"],
                "row_id": entry["row_id"],
                "database_name": entry["database_name"],
                "donor_snapshot_sha256": manifest["snapshot_sha256"],
                "historical_dependency_identity_verified": False,
            }
        )
    return resolved


def check_runtime(resolved, runtime, *, installed):
    """Reject occupied targets before install and any drift after installation."""
    runtime = Path(runtime)
    if runtime.resolve() != runtime:
        raise ExportError("Dependency runtime is redirected")
    for entry in resolved:
        destination = runtime / entry["destination"]
        if destination.resolve() != destination:
            raise ExportError("Dependency destination is redirected")
        if installed:
            if not destination.is_file() or _hash_file(destination) != (entry["sha256"], entry["size_bytes"]):
                raise ExportError(f"Installed dependency substitution changed: {entry['destination']}")
        elif destination.exists() or destination.is_symlink():
            raise ExportError(f"Dependency destination exists; refusing overwrite: {entry['destination']}")


def check_declared_headers(resolved, runtime, tree):
    """Do not let untracked shared helpers bypass substitution scope/provenance."""
    runtime = Path(runtime)
    allowed = set(tree.entries) | {entry["destination"] for entry in resolved}
    for path in (runtime / "ttnn/cpp/ttnn/kernel_lib").rglob("*"):
        if (path.is_file() or path.is_symlink()) and path.relative_to(runtime).as_posix() not in allowed:
            raise ExportError(f"Unrecorded canonical dependency needs explicit substitution: {path}")


def install(resolved, runtime):
    """Install exact donor bytes exclusively; partial failures remain inspectable."""
    runtime = Path(runtime)
    check_runtime(resolved, runtime, installed=False)
    for entry in resolved:
        source = Path(entry["export"]) / entry["source_path"]
        if _hash_file(source) != (entry["sha256"], entry["size_bytes"]):
            raise ExportError("Dependency source changed before installation")
        destination = runtime / entry["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.resolve() != destination.parent:
            raise ExportError("Dependency parent is redirected")
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())
    check_runtime(resolved, runtime, installed=True)


def scope(resolved):
    if resolved:
        return "User-approved substituted dependencies; not exact historical runtime reproduction"
    return "Recorded dependencies; runtime and outcome verification remain required"
