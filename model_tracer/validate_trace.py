#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone, fail-fast validator for traced operation manifests.

Trace failures (missing files, malformed manifests, inconsistent shapes)
currently surface late during pytest or CI replay, producing noisy errors that
obscure the root cause. This tool validates a trace manifest up-front so users
can fix problems locally before triggering long test runs.

It checks the elements called out in the tracing issue:

  1. Required fields and supported enum values in the master JSON.
  2. Artifact path resolution and file existence (no silent "degraded mode").
  3. Tensor shape consistency (internal well-formedness and, when inline tensor
     ``values`` are present, that the element count matches ``original_shape``).
  4. Optional: print resolved artifact info for the first N records.

The master JSON here is ``ttnn_operations_master.json``; when it is not passed
explicitly it is resolved the same way ``MasterConfigLoader`` resolves it, so
this tool catches the exact case where the loader would otherwise fall back to
empty configs. An optional trace-selection registry YAML can also be validated
(``status`` enum and ``targets`` -> ``registry`` cross references).

Usage:
    python model_tracer/validate_trace.py --manifest <master.json>
    python model_tracer/validate_trace.py --manifest <master.json> --print-resolved 5
    python model_tracer/validate_trace.py --manifest <master.json> \\
        --registry model_tracer/trace_selection_registry.yaml
    python model_tracer/validate_trace.py            # resolve master JSON automatically

Exit codes:
    0  Manifest is valid (no errors; no warnings unless --strict)
    1  Validation failed, or the manifest could not be found / parsed
"""

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# Required per-execution ``machine_info`` fields. Mirrors the set enforced only
# at DB-load time in tests/sweep_framework/load_ttnn_ops_data_v2.py so the same
# guarantees can be checked standalone, before any DB load or pytest run.
REQUIRED_MACHINE_INFO_FIELDS = (
    "board_type",
    "device_series",
    "card_count",
    "mesh_device_shape",
    "device_count",
)

# Lifecycle enum for registry entries (the issue's "kind" values).
VALID_REGISTRY_STATUSES = {"draft", "active", "deprecated"}

# Enum-like string fields on a tensor argument and the prefix each must carry.
TENSOR_ENUM_PREFIXES = {
    "layout": "Layout.",
    "storage_type": "StorageType.",
    "original_dtype": "DataType.",
}
MEMORY_CONFIG_ENUM_PREFIXES = {
    "buffer_type": "BufferType.",
    "memory_layout": "TensorMemoryLayout.",
}


def get_base_dir():
    """Resolve the tt-metal base directory.

    Resolution order (matches master_config_loader_v2.get_base_dir):
      1. Walk up from this file to find ``model_tracer/traced_operations``
      2. ``TT_METAL_HOME`` env var (validated to contain that marker)
      3. ``PYTHONPATH`` entries
      4. Current working directory
    """
    marker = os.path.join("model_tracer", "traced_operations")

    def walk_up(start_dir):
        current = os.path.abspath(start_dir)
        while True:
            if os.path.isdir(os.path.join(current, marker)):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent

    base = walk_up(os.path.dirname(os.path.abspath(__file__)))
    if base:
        return base

    tt_metal_home = os.environ.get("TT_METAL_HOME", "").strip()
    if tt_metal_home and os.path.isdir(os.path.join(tt_metal_home, marker)):
        return tt_metal_home

    for path in os.environ.get("PYTHONPATH", "").split(":"):
        if path:
            base = walk_up(path)
            if base:
                return base

    base = walk_up(os.getcwd())
    return base or os.getcwd()


def resolve_master_path(explicit_path):
    """Resolve the master JSON path without touching the filesystem.

    Order mirrors MasterConfigLoader.__init__:
      1. Explicit ``--manifest`` argument
      2. ``TTNN_MASTER_JSON_PATH`` environment variable
      3. Canonical ``model_tracer/traced_operations/ttnn_operations_master.json``

    Returns ``(resolved_path, source_description)``. Existence is checked
    separately so a missing path can be reported as an actionable error.
    """
    if explicit_path:
        return str(explicit_path), "--manifest argument"
    env_path = os.environ.get("TTNN_MASTER_JSON_PATH")
    if env_path:
        return env_path, "TTNN_MASTER_JSON_PATH env var"
    canonical = os.path.join(get_base_dir(), "model_tracer", "traced_operations", "ttnn_operations_master.json")
    return canonical, "canonical default path"


class Report:
    """Accumulates validation findings so all problems surface in one pass."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.operations = 0
        self.configurations = 0
        self.executions = 0
        self.tensor_args = 0

    def error(self, context, message):
        self.errors.append(f"{context}: {message}")

    def warn(self, context, message):
        self.warnings.append(f"{context}: {message}")

    @property
    def passed(self):
        return not self.errors


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_shape_string(value):
    """Parse a shape rendered as a string, e.g. ``"[1, 8]"`` -> ``[1, 8]``.

    Returns the parsed list on success or ``None`` if it cannot be parsed as a
    flat list/tuple of ints.
    """
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)) and all(isinstance(d, int) for d in parsed):
        return list(parsed)
    return None


def _flatten_count(values):
    """Count scalar leaves in a (possibly nested) list of tensor values."""
    if not isinstance(values, (list, tuple)):
        return 1
    total = 0
    for item in values:
        total += _flatten_count(item)
    return total


def _validate_tensor_argument(arg_name, arg, context, report):
    """Validate shape consistency and enum values for a single tensor argument."""
    report.tensor_args += 1
    ctx = f"{context} arg '{arg_name}'"

    shape = arg.get("original_shape")
    if shape is None:
        report.error(ctx, "tensor is missing 'original_shape'")
        return
    # An empty shape is a valid 0-d (scalar) tensor; only the element type of a
    # non-empty shape must be positive ints.
    if not isinstance(shape, list):
        report.error(ctx, f"'original_shape' must be a list, got {shape!r}")
        return
    if not all(_is_positive_int(d) for d in shape):
        report.error(ctx, f"'original_shape' must contain positive ints, got {shape!r}")
        return

    logical_shape = arg.get("logical_shape")
    if logical_shape is not None:
        if not isinstance(logical_shape, list) or not all(_is_positive_int(d) for d in logical_shape):
            report.error(ctx, f"'logical_shape' must be a list of positive ints, got {logical_shape!r}")

    placement = arg.get("tensor_placement")
    if isinstance(placement, dict):
        mesh = placement.get("mesh_device_shape")
        if isinstance(mesh, str) and _parse_shape_string(mesh) is None:
            report.error(ctx, f"'tensor_placement.mesh_device_shape' is not a valid shape: {mesh!r}")

    # Enum-like string fields (only validated when present and non-null).
    for field, prefix in TENSOR_ENUM_PREFIXES.items():
        val = arg.get(field)
        if val is not None and not (isinstance(val, str) and val.startswith(prefix)):
            report.error(ctx, f"'{field}' should start with '{prefix}', got {val!r}")

    mem = arg.get("memory_config")
    if isinstance(mem, dict):
        for field, prefix in MEMORY_CONFIG_ENUM_PREFIXES.items():
            val = mem.get(field)
            if val is not None and not (isinstance(val, str) and val.startswith(prefix)):
                report.error(ctx, f"'memory_config.{field}' should start with '{prefix}', got {val!r}")

    # Inline serialized tensor values (only present when value serialization was
    # enabled during tracing) must match the declared shape.
    values = arg.get("values")
    if values is not None:
        expected = math.prod(shape)
        actual = _flatten_count(values)
        if actual != expected:
            report.error(
                ctx,
                f"inline 'values' element count {actual} does not match "
                f"original_shape {shape} (expected {expected})",
            )


def _validate_execution(execution, context, report, metadata_trace_uid):
    """Validate required fields on a single execution record."""
    if not isinstance(execution, dict):
        report.error(context, f"execution must be an object, got {type(execution).__name__}")
        return
    report.executions += 1

    if not execution.get("source"):
        report.error(context, "missing 'execution.source'")

    # Provenance: fresh traces carry 'trace_uid'; DB-reconstructed traces carry
    # 'trace_run_ids'. Accept either (or a manifest-level default trace_uid).
    trace_uid = execution.get("trace_uid") or metadata_trace_uid
    trace_run_ids = execution.get("trace_run_ids")
    has_run_ids = isinstance(trace_run_ids, list) and len(trace_run_ids) > 0
    if not trace_uid and not has_run_ids:
        report.error(context, "missing provenance: 'execution.trace_uid' or non-empty 'execution.trace_run_ids'")

    machine_info = execution.get("machine_info")
    if not isinstance(machine_info, dict):
        report.error(context, "missing or invalid 'execution.machine_info'")
        return
    for field in REQUIRED_MACHINE_INFO_FIELDS:
        value = machine_info.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            report.error(context, f"missing 'machine_info.{field}'")


def validate_master_data(data, report):
    """Validate the structure and required fields of a loaded master JSON."""
    if not isinstance(data, dict):
        report.error("master JSON", f"top-level must be an object, got {type(data).__name__}")
        return
    operations = data.get("operations")
    if not isinstance(operations, dict):
        report.error("master JSON", "missing or invalid 'operations' object")
        return
    if not operations:
        report.warn("master JSON", "'operations' is empty (no traced configurations)")

    metadata_trace_uid = None
    if isinstance(data.get("metadata"), dict):
        metadata_trace_uid = data["metadata"].get("trace_uid")

    for op_name, op_data in operations.items():
        report.operations += 1
        if not isinstance(op_data, dict):
            report.error(f"operation '{op_name}'", "must be an object")
            continue
        configs = op_data.get("configurations")
        if not isinstance(configs, list):
            report.error(f"operation '{op_name}'", "missing or invalid 'configurations' list")
            continue
        for index, config in enumerate(configs):
            if not isinstance(config, dict):
                report.error(f"operation '{op_name}' config[{index}]", "must be an object")
                continue
            # Thread the manifest-level default trace_uid through provenance checks.
            _validate_configuration(op_name, index, config, report, metadata_trace_uid)


def _validate_configuration(op_name, index, config, report, metadata_trace_uid):
    config_id = config.get("config_id", index)
    context = f"operation '{op_name}' config_id={config_id}"
    report.configurations += 1

    config_hash = config.get("config_hash")
    if not config_hash or not isinstance(config_hash, str):
        report.error(context, "missing or invalid 'config_hash'")

    arguments = config.get("arguments")
    if not isinstance(arguments, dict):
        report.error(context, "missing or invalid 'arguments'")
    else:
        for arg_name, arg in arguments.items():
            if isinstance(arg, dict) and (arg.get("type") == "ttnn.Tensor" or "original_shape" in arg):
                _validate_tensor_argument(arg_name, arg, context, report)

    executions = config.get("executions")
    if not isinstance(executions, list) or len(executions) == 0:
        report.error(context, "missing or empty 'executions'")
    else:
        for exec_index, execution in enumerate(executions):
            exec_ctx = f"{context} execution[{exec_index}]"
            _validate_execution(execution, exec_ctx, report, metadata_trace_uid)


def validate_registry(registry_path, report):
    """Validate a trace-selection registry YAML (status enum + cross references)."""
    if yaml is None:
        report.error("registry", "PyYAML is required to validate a registry. Install it with: pip install pyyaml")
        return
    if not os.path.exists(registry_path):
        report.error("registry", f"file not found: {registry_path}")
        return
    try:
        with open(registry_path) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        report.error("registry", f"failed to parse YAML: {exc}")
        return
    if not isinstance(data, dict):
        report.error("registry", "top-level must be a mapping with 'targets' and 'registry'")
        return

    registry = data.get("registry") or []
    if not isinstance(registry, list):
        report.error("registry", "'registry' must be a list")
        registry = []

    known_trace_ids = set()
    for index, entry in enumerate(registry):
        if not isinstance(entry, dict):
            report.error(f"registry entry[{index}]", "must be a mapping")
            continue
        trace_id = entry.get("trace_id")
        if not isinstance(trace_id, int) or isinstance(trace_id, bool):
            report.error(f"registry entry[{index}]", f"'trace_id' must be an int, got {trace_id!r}")
        else:
            known_trace_ids.add(trace_id)
        status = entry.get("status")
        if status not in VALID_REGISTRY_STATUSES:
            report.error(
                f"registry trace_id={trace_id}",
                f"invalid 'status' {status!r}; expected one of {sorted(VALID_REGISTRY_STATUSES)}",
            )

    # Cross-reference: pinned trace IDs in targets must exist in the registry.
    targets = data.get("targets") or {}
    if isinstance(targets, dict):
        for scope, entries in targets.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                pinned = entry.get("trace")
                if pinned is None:
                    continue
                pinned_ids = pinned if isinstance(pinned, list) else [pinned]
                for tid in pinned_ids:
                    if isinstance(tid, int) and not isinstance(tid, bool) and tid not in known_trace_ids:
                        report.error(
                            f"targets.{scope}",
                            f"pinned trace {tid} is not present in 'registry'",
                        )


def print_resolved(data, limit):
    """Print resolved artifact info for the first ``limit`` configurations."""
    operations = data.get("operations", {}) if isinstance(data, dict) else {}
    print("")
    print(f"Resolved artifact paths (first {limit} configurations):")
    shown = 0
    for op_name, op_data in operations.items():
        if shown >= limit:
            break
        if not isinstance(op_data, dict):
            continue
        for config in op_data.get("configurations", []):
            if shown >= limit:
                break
            if not isinstance(config, dict):
                continue
            shown += 1
            config_id = config.get("config_id", "?")
            config_hash = config.get("config_hash", "?")
            executions = config.get("executions", []) or []
            sources = sorted({e.get("source", "?") for e in executions if isinstance(e, dict)})
            hw = ""
            if executions and isinstance(executions[0], dict):
                mi = executions[0].get("machine_info", {}) or {}
                hw = (
                    f"{mi.get('board_type', '?')}/{mi.get('device_series', '?')} "
                    f"cards={mi.get('card_count', '?')} mesh={mi.get('mesh_device_shape', '?')}"
                )
            print(f"  [{shown}] {op_name} config_id={config_id}")
            print(f"        config_hash: {config_hash}")
            print(f"        hardware:    {hw}")
            for src in sources:
                print(f"        source:      {src}")
    if shown == 0:
        print("  (no configurations found)")


def format_report(resolved_path, path_source, report, strict):
    """Build the human-readable validation report."""
    lines = [
        "Trace manifest validation",
        f"  Manifest:        {resolved_path}",
        f"  Resolved via:    {path_source}",
        f"  Operations:      {report.operations}",
        f"  Configurations:  {report.configurations}",
        f"  Executions:      {report.executions}",
        f"  Tensor args:     {report.tensor_args}",
        f"  Errors:          {len(report.errors)}",
        f"  Warnings:        {len(report.warnings)}",
    ]
    if report.errors:
        lines.append("")
        lines.append("Errors:")
        for err in report.errors:
            lines.append(f"  - {err}")
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warn in report.warnings:
            lines.append(f"  - {warn}")

    failed = bool(report.errors) or (strict and bool(report.warnings))
    lines.append("")
    lines.append(f"  Decision:        {'fail' if failed else 'pass'}")
    return "\n".join(lines) + "\n", failed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the master JSON to validate. If omitted, resolves via "
        "TTNN_MASTER_JSON_PATH or the canonical traced_operations path.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional trace_selection_registry.yaml to validate (status enum "
        "and targets -> registry cross references).",
    )
    parser.add_argument(
        "--print-resolved",
        type=int,
        metavar="N",
        default=0,
        help="Print resolved artifact info for the first N configurations.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (non-zero exit).",
    )
    args = parser.parse_args(argv)

    resolved_path, path_source = resolve_master_path(args.manifest)

    if not os.path.exists(resolved_path):
        print(f"Error: master JSON not found: {resolved_path} (resolved via {path_source})", file=sys.stderr)
        print(
            "  Re-trace with model_tracer/generic_ops_tracer.py or reconstruct with "
            "tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-manifest, "
            "or pass --manifest explicitly.",
            file=sys.stderr,
        )
        return 1

    try:
        with open(resolved_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: {resolved_path} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    report = Report()
    validate_master_data(data, report)

    if args.registry is not None:
        validate_registry(str(args.registry), report)

    report_text, failed = format_report(resolved_path, path_source, report, args.strict)

    print("")
    print("=" * 60)
    print(report_text, end="")
    print("=" * 60)

    if args.print_resolved > 0:
        print_resolved(data, args.print_resolved)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
