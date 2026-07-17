#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-specific validation logic for Phase-1 tracer manifests.

This module holds the reusable validation core used by the CLI entry point
(``validate_trace_manifest.py``). It performs the fail-fast checks that keep
malformed manifests from surfacing as noisy ``pytest`` errors deep inside a
device run:

  1. Top-level structure (``records`` list; consistent ``num_records`` /
     ``input_shape`` when present).
  2. Per-record required keys and types; ``idx`` matches position.
  3. Supported ``kind`` values and per-kind required ``params`` (sourced from
     ``tracer_op_specs`` so the validator and the runtime harness agree).
  4. ``in_shape`` / ``out_shape`` are length-4 lists of positive ints.
  5. Artifact existence (``in_path`` / ``out_path`` / ``w_path`` / ``b_path``),
     resolved manifest-relative the same way the runtime harness resolves them.
  6. Shape consistency: the on-disk tensor shape matches the recorded shape.

``torch`` is imported lazily and only for the optional shape-consistency check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from tracer_op_specs import SUPPORTED_KINDS, is_supported, required_params


# Required per-record keys and their expected JSON types.
REQUIRED_RECORD_KEYS = {
    "idx": int,
    "name": str,
    "kind": str,
    "params": dict,
    "in_shape": list,
    "out_shape": list,
    "in_path": str,
    "out_path": str,
}

ShapeLoader = Callable[[Path], Tuple[int, ...]]

_MISSING = object()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


class Report:
    """Accumulates validation findings so all problems surface in one pass."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.records = 0
        self.artifacts_checked = 0
        self.shape_checks = 0

    def error(self, context: str, message: str) -> None:
        self.errors.append(f"{context}: {message}")

    @property
    def ok(self) -> bool:
        return not self.errors


def resolve_artifact_path(manifest_path: Path, artifact_path: str) -> Path:
    """Resolve an artifact path exactly as the runtime harness does.

    Mirrors ``tracer_test_harness._resolve_artifact_path`` (Option 1:
    manifest-relative) so this preflight predicts what the harness will load:
      1. Absolute paths are used as-is.
      2. Relative paths are resolved against the manifest's directory.
    """
    p = Path(str(artifact_path).strip())
    if p.is_absolute():
        return p
    return Path(manifest_path).resolve().parent / p


def _default_shape_loader(path: Path) -> Tuple[int, ...]:
    """Load a ``.pt`` artifact and return its tensor shape (mirrors the harness)."""
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, torch.Tensor):
                return tuple(value.shape)
    raise TypeError(f"unsupported tensor payload type: {type(obj).__name__}")


def _validate_shape_field(record: dict, key: str, context: str, report: Report) -> Optional[list]:
    """Validate a shape field is a length-4 list of positive ints. Returns it if list-of-ints."""
    shape = record.get(key)
    if not isinstance(shape, list):
        # Type error already reported by required-key validation.
        return None
    if len(shape) != 4:
        report.error(context, f"'{key}' must have 4 dimensions (NCHW), got {shape!r}")
    if not all(_is_positive_int(d) for d in shape):
        report.error(context, f"'{key}' must contain positive ints, got {shape!r}")
        return None
    return shape


def _validate_artifact(
    manifest_path: Path,
    record: dict,
    key: str,
    context: str,
    report: Report,
    expected_shape: Optional[list],
    shape_loader: Optional[ShapeLoader],
) -> None:
    """Check that an artifact exists and (for in/out) matches its recorded shape."""
    value = record.get(key)
    if value is None:
        return
    if not isinstance(value, str):
        report.error(context, f"'{key}' must be a string path or null, got {type(value).__name__}")
        return

    report.artifacts_checked += 1
    resolved = resolve_artifact_path(manifest_path, value)
    if not resolved.exists():
        report.error(context, f"artifact '{key}' not found: {value!r} (resolved: {resolved})")
        return

    if shape_loader is None or expected_shape is None:
        return

    report.shape_checks += 1
    try:
        actual = shape_loader(resolved)
    except Exception as exc:  # noqa: BLE001 - surface any load failure as a validation error
        report.error(context, f"failed to load tensor '{key}' at {resolved}: {exc}")
        return
    if tuple(actual) != tuple(expected_shape):
        manifest_key = "in_shape" if key == "in_path" else "out_shape"
        report.error(
            context,
            f"'{key}' tensor shape {tuple(actual)} does not match manifest "
            f"'{manifest_key}' {tuple(expected_shape)}",
        )


def _validate_record(
    manifest_path: Path,
    position: int,
    record: Any,
    report: Report,
    shape_loader: Optional[ShapeLoader],
) -> None:
    context = f"record[{position}]"
    if not isinstance(record, dict):
        report.error(context, f"must be an object, got {type(record).__name__}")
        return
    report.records += 1

    # 2) Required keys and types.
    for key, expected_type in REQUIRED_RECORD_KEYS.items():
        value = record.get(key, _MISSING)
        if value is _MISSING:
            report.error(context, f"missing required key '{key}'")
            continue
        if expected_type is int:
            if not _is_int(value):
                report.error(context, f"'{key}' must be an int, got {type(value).__name__}")
        elif not isinstance(value, expected_type):
            report.error(context, f"'{key}' must be a {expected_type.__name__}, got {type(value).__name__}")

    # idx should match the record's position in the list.
    idx = record.get("idx")
    if _is_int(idx) and idx != position:
        report.error(context, f"'idx' ({idx}) does not match record position ({position})")

    # w_path / b_path are optional but must be string-or-null when present.
    for key in ("w_path", "b_path"):
        value = record.get(key, _MISSING)
        if value is not _MISSING and value is not None and not isinstance(value, str):
            report.error(context, f"'{key}' must be a string path or null, got {type(value).__name__}")

    # 3) Supported kind + per-kind required params (shared with the harness).
    kind = record.get("kind")
    if isinstance(kind, str):
        if not is_supported(kind):
            report.error(context, f"unsupported 'kind' {kind!r}; expected one of {sorted(SUPPORTED_KINDS)}")
        needed = required_params(kind)
        if needed:
            params = record.get("params")
            if isinstance(params, dict):
                missing = [p for p in needed if p not in params]
                if missing:
                    report.error(context, f"{kind} 'params' missing {missing}")

    # 4) Shape well-formedness.
    in_shape = _validate_shape_field(record, "in_shape", context, report)
    out_shape = _validate_shape_field(record, "out_shape", context, report)

    # 5/6) Artifact existence + shape consistency.
    _validate_artifact(manifest_path, record, "in_path", context, report, in_shape, shape_loader)
    _validate_artifact(manifest_path, record, "out_path", context, report, out_shape, shape_loader)
    _validate_artifact(manifest_path, record, "w_path", context, report, None, shape_loader)
    _validate_artifact(manifest_path, record, "b_path", context, report, None, shape_loader)


def validate_manifest(
    manifest_path: Any,
    *,
    check_shapes: bool = True,
    shape_loader: Optional[ShapeLoader] = None,
) -> Report:
    """Validate a Phase-1 tracer manifest, returning an accumulated ``Report``.

    Args:
        manifest_path: Path to ``manifest.json``.
        check_shapes: When True, compare on-disk tensor shapes to recorded ones.
        shape_loader: Optional callable ``(path) -> shape`` (used for testing and
            to avoid importing torch). Defaults to a ``torch.load``-based loader.
    """
    report = Report()
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        report.error("manifest", f"file not found: {manifest_path}")
        return report
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.error("manifest", f"not valid JSON: {exc}")
        return report
    if not isinstance(data, dict):
        report.error("manifest", f"top-level must be a JSON object, got {type(data).__name__}")
        return report

    records = data.get("records")
    if not isinstance(records, list):
        report.error("manifest", "missing or invalid 'records' (must be a list)")
        return report

    # 1) Top-level metadata consistency.
    num_records = data.get("num_records")
    if num_records is not None and (not _is_int(num_records) or num_records != len(records)):
        report.error("manifest", f"'num_records' ({num_records!r}) does not match len(records) ({len(records)})")
    input_shape = data.get("input_shape")
    if input_shape is not None and not (
        isinstance(input_shape, list) and len(input_shape) == 4 and all(_is_int(d) for d in input_shape)
    ):
        report.error("manifest", f"'input_shape' must be a list of 4 ints, got {input_shape!r}")

    # Resolve the shape loader (torch is only needed here).
    if check_shapes and shape_loader is None:
        try:
            import torch  # noqa: F401
        except ImportError:
            report.error(
                "shape-check",
                "torch is required for tensor shape consistency checks; " "install torch or pass --no-shape-check",
            )
        else:
            shape_loader = _default_shape_loader
    active_loader = shape_loader if check_shapes else None

    for position, record in enumerate(records):
        _validate_record(manifest_path, position, record, report, active_loader)

    return report


def print_resolved(manifest_path: Path, limit: int) -> None:
    """Print resolved artifact paths for the first ``limit`` records."""
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    records = data.get("records", []) if isinstance(data, dict) else []

    print("")
    print(f"Resolved artifact paths (first {limit} records):")
    if not records:
        print("  (no records found)")
        return
    for record in records[:limit]:
        if not isinstance(record, dict):
            continue
        idx = record.get("idx", "?")
        name = record.get("name", "?")
        kind = record.get("kind", "?")
        print(f"  [{idx}] {name} ({kind})")
        for key in ("in_path", "out_path", "w_path", "b_path"):
            value = record.get(key)
            if value is None:
                continue
            resolved = resolve_artifact_path(Path(manifest_path), str(value))
            marker = "OK " if resolved.exists() else "MISSING"
            print(f"        {key:8} [{marker}] {resolved}")


def format_report(manifest_path: Any, report: Report) -> str:
    """Build the human-readable validation report."""
    lines = [
        "Phase-1 tracer manifest validation",
        f"  Manifest:          {manifest_path}",
        f"  Records:           {report.records}",
        f"  Artifacts checked: {report.artifacts_checked}",
        f"  Shape checks:      {report.shape_checks}",
        f"  Errors:            {len(report.errors)}",
    ]
    if report.errors:
        lines.append("")
        lines.append("Errors:")
        for err in report.errors:
            lines.append(f"  - {err}")
    lines.append("")
    lines.append(f"  Decision:          {'pass' if report.ok else 'fail'}")
    return "\n".join(lines) + "\n"
