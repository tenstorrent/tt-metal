#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared op specifications and manifest-record validation for the bringup tools.

Single source of truth for:

  * ``Record`` -- the typed representation of one manifest entry that both the
    runtime harness (``tracer_test_harness.py``) and the manifest validator
    (``trace_manifest_validation.py``) operate on. The harness loads a manifest
    into ``Record`` objects; the validator parses raw JSON into ``Record`` and
    checks it, so the same type drives both paths.
  * ``OpSpec`` / ``OP_SPECS`` -- the op kinds the Phase-1 tracer records, the
    ``params`` each op dereferences, and which kinds the harness can replay.
  * ``shared_validate_record`` -- the record-level checks (supported kind,
    well-formed 4D shapes, required params present) that must hold both at
    validation time and at replay time, so the two never drift apart.

Kept dependency-light on purpose: stdlib only (no ``torch`` / ``ttnn``), so the
validator can import it without pulling heavy runtime dependencies and the
harness can import it cheaply as a sibling module.

The recorded kinds mirror the "interesting" leaf modules hooked by
``phase1_record_ops.py``. ``runnable`` marks the subset the harness can actually
replay on device today (the rest are valid to appear in a manifest but skipped).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


# ── Op registry ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpSpec:
    """Static description of a traced op kind.

    Attributes:
        kind: The module type name recorded by the tracer (e.g. ``"Conv2d"``).
        required_params: ``params`` keys the harness dereferences to replay the
            op; the validator checks these are present. These are exactly the
            extra ``params`` a given op (like ``Conv2d``) reads at runtime.
        uses_weight: Whether the op carries a weight artifact (``w_path``).
        uses_bias: Whether the op carries a bias artifact (``b_path``).
        runnable: Whether ``tracer_test_harness`` can replay this kind on device.
    """

    kind: str
    required_params: Tuple[str, ...] = ()
    uses_weight: bool = False
    uses_bias: bool = False
    runnable: bool = False


# Registry of all kinds the Phase-1 tracer records (phase1_record_ops.py, the
# "interesting" module tuple). The harness only replays the ``runnable`` subset.
OP_SPECS: Dict[str, OpSpec] = {
    "Conv2d": OpSpec(
        kind="Conv2d",
        required_params=(
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
        ),
        uses_weight=True,
        uses_bias=True,
        runnable=True,
    ),
    "ConvTranspose2d": OpSpec(kind="ConvTranspose2d", uses_weight=True, uses_bias=True),
    "GroupNorm": OpSpec(kind="GroupNorm", uses_weight=True, uses_bias=True),
    "BatchNorm2d": OpSpec(kind="BatchNorm2d", uses_weight=True, uses_bias=True),
    "ReLU": OpSpec(kind="ReLU", runnable=True),
    "MaxPool2d": OpSpec(kind="MaxPool2d"),
    "Upsample": OpSpec(kind="Upsample"),
}

# Kinds valid to appear in a manifest (superset of the runnable kinds).
SUPPORTED_KINDS = frozenset(OP_SPECS)


def get_spec(kind: str) -> Optional[OpSpec]:
    """Return the ``OpSpec`` for ``kind``, or ``None`` if unsupported."""
    return OP_SPECS.get(kind)


def is_supported(kind: str) -> bool:
    """Whether ``kind`` is a recognized (recordable) op kind."""
    return kind in OP_SPECS


def is_runnable(kind: str) -> bool:
    """Whether the harness can replay ``kind`` on device."""
    spec = OP_SPECS.get(kind)
    return bool(spec and spec.runnable)


def required_params(kind: str) -> Tuple[str, ...]:
    """Return the required ``params`` keys for ``kind`` (empty if none/unknown)."""
    spec = OP_SPECS.get(kind)
    return spec.required_params if spec is not None else ()


def missing_params(kind: str, params: Optional[Mapping[str, Any]]) -> List[str]:
    """Return the ``required_params`` for ``kind`` that are absent from ``params``."""
    present = params or {}
    return [p for p in required_params(kind) if p not in present]


# ── Manifest record type ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Record:
    """One manifest entry, as loaded/validated by the harness and the validator.

    ``in_shape`` / ``out_shape`` are 4D NCHW. ``w_path`` / ``b_path`` are optional
    (present for ops that carry weights/bias). This is the single schema both the
    replay path and the preflight validator share.
    """

    idx: int
    kind: str
    in_shape: List[int]
    out_shape: List[int]
    params: Dict[str, Any]
    in_path: str
    out_path: str
    name: str = ""
    w_path: Optional[str] = None
    b_path: Optional[str] = None


# Required per-record keys and their expected JSON types (structural contract).
REQUIRED_RECORD_KEYS: Dict[str, type] = {
    "idx": int,
    "name": str,
    "kind": str,
    "params": dict,
    "in_shape": list,
    "out_shape": list,
    "in_path": str,
    "out_path": str,
}

_MISSING = object()


def is_int(value: Any) -> bool:
    """True for a real int (bools are rejected)."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_positive_int(value: Any) -> bool:
    return is_int(value) and value > 0


def is_valid_shape(shape: Any) -> bool:
    """True when ``shape`` is a length-4 list/tuple of positive ints (NCHW)."""
    return isinstance(shape, (list, tuple)) and len(shape) == 4 and all(is_positive_int(d) for d in shape)


def record_from_mapping(raw: Mapping[str, Any], idx: int) -> Record:
    """Build a ``Record`` from a raw manifest dict (lenient; never raises).

    Mirrors the loader the harness historically used. Malformed fields are
    coerced to safe defaults so callers can validate the resulting ``Record``
    with :func:`shared_validate_record` rather than crashing on bad input.
    """
    raw = raw if isinstance(raw, Mapping) else {}
    in_shape = raw.get("in_shape")
    out_shape = raw.get("out_shape")
    params = raw.get("params")
    return Record(
        idx=idx,
        name=str(raw.get("name", "")),
        kind=str(raw.get("kind")),
        in_shape=list(in_shape) if isinstance(in_shape, (list, tuple)) else [],
        out_shape=list(out_shape) if isinstance(out_shape, (list, tuple)) else [],
        params=dict(params) if isinstance(params, dict) else {},
        in_path=str(raw.get("in_path")),
        out_path=str(raw.get("out_path")),
        w_path=raw.get("w_path"),
        b_path=raw.get("b_path"),
    )


def load_manifest(manifest_path: Any) -> List[Record]:
    """Load a manifest JSON file into a list of ``Record`` objects."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = data.get("records", []) if isinstance(data, dict) else []
    return [record_from_mapping(r, i) for i, r in enumerate(records)]


# ── Shared record validation ─────────────────────────────────────────────────


def validate_record_mapping(raw: Any, *, position: int) -> List[str]:
    """Structural validation of a raw manifest record dict.

    Checks it is an object with the required keys of the expected JSON types,
    that ``idx`` matches ``position``, and that optional ``w_path`` / ``b_path``
    are string-or-null. Returns human-readable error messages (no context
    prefix); an empty list means the structure is sound enough to build a
    :class:`Record`.
    """
    errors: List[str] = []
    if not isinstance(raw, dict):
        errors.append(f"must be an object, got {type(raw).__name__}")
        return errors

    for key, expected_type in REQUIRED_RECORD_KEYS.items():
        value = raw.get(key, _MISSING)
        if value is _MISSING:
            errors.append(f"missing required key '{key}'")
            continue
        if expected_type is int:
            if not is_int(value):
                errors.append(f"'{key}' must be an int, got {type(value).__name__}")
        elif not isinstance(value, expected_type):
            errors.append(f"'{key}' must be a {expected_type.__name__}, got {type(value).__name__}")

    idx = raw.get("idx")
    if is_int(idx) and idx != position:
        errors.append(f"'idx' ({idx}) does not match record position ({position})")

    for key in ("w_path", "b_path"):
        value = raw.get(key, _MISSING)
        if value is not _MISSING and value is not None and not isinstance(value, str):
            errors.append(f"'{key}' must be a string path or null, got {type(value).__name__}")

    return errors


def shared_validate_record(record: Record) -> List[str]:
    """Record-level checks shared by the manifest validator and ``run_record``.

    These are the semantic guarantees the replay path relies on, hoisted so the
    preflight validator enforces exactly the same rules:

      * ``kind`` is a supported op kind.
      * ``in_shape`` / ``out_shape`` are well-formed 4D positive-int shapes.
      * the op's required ``params`` (e.g. ``Conv2d``'s kernel/stride/...) are
        present.

    Returns human-readable error messages (no context prefix); empty means valid.
    """
    errors: List[str] = []

    if not is_supported(record.kind):
        errors.append(f"unsupported 'kind' {record.kind!r}; expected one of {sorted(SUPPORTED_KINDS)}")

    for key, shape in (("in_shape", record.in_shape), ("out_shape", record.out_shape)):
        if not is_valid_shape(shape):
            errors.append(f"'{key}' must be a length-4 list of positive ints (NCHW), got {shape!r}")

    missing = missing_params(record.kind, record.params)
    if missing:
        errors.append(f"{record.kind} 'params' missing {missing}")

    return errors
