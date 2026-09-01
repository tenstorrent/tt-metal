#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Contract for portable cluster health records: ``exabox.cluster_health.v1``.

Every launcher (CLI, scheduled job, or other wrapper) must emit the same
object. This module is stdlib-only and performs no filesystem I/O.

Status values are a closed enum. Physical analyzer exit code 0 maps to
``passed`` in report adapters (phase 02); this module does not interpret codes.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

SCHEMA_ID = "exabox.cluster_health.v1"

TEST_TYPES: frozenset[str] = frozenset({"physical", "fabric", "recover", "dispatch", "host"})
STATUSES: frozenset[str] = frozenset({"passed", "failed", "skipped", "degraded"})

REQUIRED_FIELDS: tuple[str, ...] = ("schema", "ts", "test_type", "status", "hosts")

OPTIONAL_STRING_FIELDS: frozenset[str] = frozenset(
    {
        "cluster",
        "source",
        "triggered_by",
        "trigger_kind",
        "orchestrator_id",
        "artifact_uri",
        "record_uri",
        "record_id",
    }
)

TOPOLOGY_KEYS: frozenset[str] = frozenset({"instance_paths", "physical", "rank_bindings"})
PHYSICAL_KEYS: frozenset[str] = frozenset({"hostname", "aisle", "rack", "shelf_u"})
RANK_BINDING_KEYS: frozenset[str] = frozenset({"rank", "mesh_id", "mesh_host_rank", "host"})

ALLOWED_TOP_LEVEL: frozenset[str] = (
    frozenset(REQUIRED_FIELDS)
    | OPTIONAL_STRING_FIELDS
    | frozenset({"analyzer_code", "topology", "duration_s", "labels"})
)

CLUSTER_HEALTH_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SCHEMA_ID,
    "title": "Exabox cluster health record v1",
    "type": "object",
    "additionalProperties": False,
    "required": list(REQUIRED_FIELDS),
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "ts": {"type": "string", "format": "date-time", "description": "RFC3339 UTC"},
        "test_type": {"type": "string", "enum": sorted(TEST_TYPES)},
        "status": {
            "type": "string",
            "enum": sorted(STATUSES),
            "description": "Physical analyzer 0 -> passed (mapped by report adapters).",
        },
        "hosts": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
            "description": "Source of truth: all machines in this run.",
        },
        "analyzer_code": {"type": "integer"},
        "topology": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "instance_paths": {"type": "array", "items": {"type": "string"}},
                "physical": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["hostname"],
                        "properties": {
                            "hostname": {"type": "string", "minLength": 1},
                            "aisle": {"type": "string"},
                            "rack": {"type": ["integer", "string"]},
                            "shelf_u": {"type": ["integer", "string"]},
                        },
                    },
                },
                "rank_bindings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["rank", "mesh_id"],
                        "properties": {
                            "rank": {"type": "integer"},
                            "mesh_id": {"type": "integer"},
                            "mesh_host_rank": {
                                "type": "integer",
                                "description": "Optional, matching tt-run: single-host meshes omit it.",
                            },
                            "host": {"type": "string"},
                        },
                    },
                },
            },
        },
        "cluster": {"type": "string"},
        "source": {"type": "string"},
        "triggered_by": {"type": "string"},
        "trigger_kind": {"type": "string"},
        "orchestrator_id": {"type": "string"},
        "artifact_uri": {"type": "string"},
        "record_uri": {"type": "string", "description": "Absolute path after file write"},
        "record_id": {"type": "string"},
        "duration_s": {
            "type": "number",
            "minimum": 0,
            "description": "Finite non-negative seconds; NaN and Infinity are rejected.",
        },
        "labels": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Opaque low-cardinality extras (e.g. quad, superpod).",
        },
    },
}


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _require_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"must be an object, got {type(value).__name__}")
    return value


def _require_str(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(path, f"must be a string, got {type(value).__name__}")
    if not allow_empty and value == "":
        _fail(path, "must be a non-empty string")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or isinstance(value, float)


def _parse_rfc3339_utc(value: str, path: str) -> None:
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{path}: must be RFC3339 UTC, got {value!r}") from exc
    if dt.tzinfo is None:
        _fail(path, "must include a UTC offset")
    offset = dt.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        _fail(path, "must be UTC")


def _validate_hosts(hosts: Any) -> None:
    if not isinstance(hosts, list):
        _fail("hosts", f"must be an array, got {type(hosts).__name__}")
    if not hosts:
        _fail("hosts", "must be a non-empty array")
    for i, host in enumerate(hosts):
        _require_str(host, f"hosts[{i}]")


def _validate_string_list(value: Any, path: str) -> None:
    if not isinstance(value, list):
        _fail(path, f"must be an array, got {type(value).__name__}")
    for i, item in enumerate(value):
        _require_str(item, f"{path}[{i}]", allow_empty=True)


def _validate_physical_item(item: Any, path: str) -> None:
    obj = _require_dict(item, path)
    extra = set(obj) - PHYSICAL_KEYS
    if extra:
        _fail(path, f"unknown keys: {sorted(extra)}")
    _require_str(obj.get("hostname"), f"{path}.hostname")
    if "aisle" in obj:
        _require_str(obj["aisle"], f"{path}.aisle", allow_empty=True)
    if "rack" in obj and not (_is_int(obj["rack"]) or isinstance(obj["rack"], str)):
        _fail(f"{path}.rack", "must be an integer or string")
    if "shelf_u" in obj and not (_is_int(obj["shelf_u"]) or isinstance(obj["shelf_u"], str)):
        _fail(f"{path}.shelf_u", "must be an integer or string")


def _validate_rank_binding(item: Any, path: str) -> None:
    obj = _require_dict(item, path)
    extra = set(obj) - RANK_BINDING_KEYS
    if extra:
        _fail(path, f"unknown keys: {sorted(extra)}")
    for key in ("rank", "mesh_id"):
        if key not in obj:
            _fail(path, f"missing {key}")
        if not _is_int(obj[key]):
            _fail(f"{path}.{key}", "must be an integer")
    if "mesh_host_rank" in obj and not _is_int(obj["mesh_host_rank"]):
        _fail(f"{path}.mesh_host_rank", "must be an integer")
    if "host" in obj:
        _require_str(obj["host"], f"{path}.host", allow_empty=True)


def _validate_topology(topology: Any) -> None:
    obj = _require_dict(topology, "topology")
    extra = set(obj) - TOPOLOGY_KEYS
    if extra:
        _fail("topology", f"unknown keys: {sorted(extra)}")
    if "instance_paths" in obj:
        _validate_string_list(obj["instance_paths"], "topology.instance_paths")
    if "physical" in obj:
        physical = obj["physical"]
        if not isinstance(physical, list):
            _fail("topology.physical", f"must be an array, got {type(physical).__name__}")
        for i, item in enumerate(physical):
            _validate_physical_item(item, f"topology.physical[{i}]")
    if "rank_bindings" in obj:
        bindings = obj["rank_bindings"]
        if not isinstance(bindings, list):
            _fail("topology.rank_bindings", f"must be an array, got {type(bindings).__name__}")
        for i, item in enumerate(bindings):
            _validate_rank_binding(item, f"topology.rank_bindings[{i}]")


def _validate_labels(labels: Any) -> None:
    obj = _require_dict(labels, "labels")
    for key, value in obj.items():
        if not isinstance(key, str) or key == "":
            _fail("labels", "keys must be non-empty strings")
        _require_str(value, f"labels.{key}", allow_empty=True)


def validate_record(record: Any, *, file_written: bool = False) -> None:
    """Validate a cluster health record.

    Raises ValueError with a ``field: message`` path on the first failure.

    ``file_written=True`` requires ``record_id`` and an absolute ``record_uri``.
    ``file_written=False`` forbids both (stdout-only / dry-run).
    """
    obj = _require_dict(record, "$")
    extra = set(obj) - ALLOWED_TOP_LEVEL
    if extra:
        _fail("$", f"unknown keys: {sorted(extra)}")
    if "scope" in obj:
        _fail("scope", "forbidden")

    for fname in REQUIRED_FIELDS:
        if fname not in obj:
            _fail(fname, "required")

    if obj["schema"] != SCHEMA_ID:
        _fail("schema", f"must be {SCHEMA_ID!r}")

    ts = _require_str(obj["ts"], "ts")
    _parse_rfc3339_utc(ts, "ts")

    test_type = _require_str(obj["test_type"], "test_type")
    if test_type not in TEST_TYPES:
        _fail("test_type", f"must be one of {sorted(TEST_TYPES)}")

    status = _require_str(obj["status"], "status")
    if status not in STATUSES:
        _fail("status", f"must be one of {sorted(STATUSES)}")

    _validate_hosts(obj["hosts"])

    if "analyzer_code" in obj and not _is_int(obj["analyzer_code"]):
        _fail("analyzer_code", "must be an integer")

    if "topology" in obj:
        _validate_topology(obj["topology"])

    for fname in OPTIONAL_STRING_FIELDS:
        if fname in obj:
            _require_str(obj[fname], fname)

    if "duration_s" in obj:
        duration = obj["duration_s"]
        if not _is_number(duration) or duration < 0 or not math.isfinite(duration):
            _fail("duration_s", "must be a finite number >= 0")

    if "labels" in obj:
        _validate_labels(obj["labels"])

    has_record_id = "record_id" in obj
    has_record_uri = "record_uri" in obj
    if file_written:
        if not has_record_id:
            _fail("record_id", "required after file write")
        if not has_record_uri:
            _fail("record_uri", "required after file write")
        uri = obj["record_uri"]
        if not uri.startswith("/"):
            _fail("record_uri", "must be an absolute path")
    else:
        if has_record_id:
            _fail("record_id", "must be omitted for stdout-only records")
        if has_record_uri:
            _fail("record_uri", "must be omitted for stdout-only records")


def loads_and_validate(text: str, *, file_written: bool = False) -> dict[str, Any]:
    """Parse one JSON object (compact stdout/file line or pretty fixture) and validate it."""
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"$: invalid JSON: {exc}") from exc
    validate_record(record, file_written=file_written)
    return record
