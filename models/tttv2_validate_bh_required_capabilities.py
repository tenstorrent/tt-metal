#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate immutable TTTv2 BlackHole required-capability contracts.

The qualification path is deliberately host-only and dependency-free: it
imports neither TTNN, model packages, nor third-party schema libraries.  The
checked-in JSON Schema remains the declarative specification; this module
implements fail-closed validation of its exact contract shape plus relational
invariants that JSON Schema cannot conveniently express across arrays.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).resolve().parent
DEFAULT_SCHEMA = MODELS_DIR / "tttv2_bh_required_capabilities.schema.json"
DEFAULT_CONTRACTS = (
    MODELS_DIR / "tttv2_llama3_8b_bh_required_capabilities.json",
    MODELS_DIR / "tttv2_qwen3_32b_bh_required_capabilities.json",
    MODELS_DIR / "tttv2_llama33_70b_bh_required_capabilities.json",
)

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_INSTANCE_SCHEMA = "tttv2_bh_required_capabilities.schema.json"
_RESOLUTION = "PASS_OR_APPROVED_UNSUPPORTED_GEOMETRY"
_OPTIMIZATION_PROFILES = {"performance", "accuracy"}
_TRACE_MODES = {"none", "decode_only", "all"}
_CAPABILITIES = {"functional", "token_accuracy", "determinism", "performance", "quality"}
_CONTEXT_SUBCASES = {"long_prefill", "chunked_prefill", "cached_prefill"}

_ROOT_KEYS = (
    "$schema",
    "schema_version",
    "contract_id",
    "model",
    "authority",
    "scope",
    "geometries",
    "demo_requirements",
    "serving_requirements",
    "cross_cutting_requirements",
    "excluded_or_deferred",
)
_MODEL_KEYS = ("package", "hf_model_id", "demo_entry_point")
_DEMO_KEYS = (
    "id",
    "geometry_id",
    "profile",
    "demo_case",
    "source_parameter_id",
    "node_id",
    "batch_size",
    "decode_tokens",
    "repeat_batches",
    "report_perf",
    "dp",
    "trace_mode",
    "context_bucket",
    "cache_protocol",
    "capabilities",
    "required_resolution",
    "acceptance_condition",
)

# These identities select immutable, model-specific requirements.  In
# particular, model.package must never be usable to bypass those checks.
_REQUIRED_CONTRACT_IDENTITIES = {
    "tttv2_llama3_8b_bh_required_capabilities_v1": {
        "package": "llama3_8b",
        "hf_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "demo_entry_point": "models/common/tests/demos/llama3_8b/demo.py",
    },
    "tttv2_qwen3_32b_bh_required_capabilities_v1": {
        "package": "qwen3_32b",
        "hf_model_id": "Qwen/Qwen3-32B",
        "demo_entry_point": "models/common/tests/demos/qwen3_32b/demo.py",
    },
    "tttv2_llama33_70b_bh_required_capabilities_v1": {
        "package": "llama33_70b",
        "hf_model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "demo_entry_point": "models/common/tests/demos/llama33_70b/demo.py",
    },
}
_QWEN_CONTRACT_ID = "tttv2_qwen3_32b_bh_required_capabilities_v1"
_LLAMA3_8B_CONTRACT_ID = "tttv2_llama3_8b_bh_required_capabilities_v1"
_LLAMA33_70B_CONTRACT_ID = "tttv2_llama33_70b_bh_required_capabilities_v1"

# The digest covers every executable workload selector in each demo row.  Row
# order is intentionally ignored; row identity and contents are not.  The
# explicit ID sets retain actionable missing/extra diagnostics, while the
# digest fails closed on changes to any listed field.
_IMMUTABLE_DEMO_MANIFEST_FIELDS = (
    "id",
    "geometry_id",
    "profile",
    "demo_case",
    "source_parameter_id",
    "node_id",
    "batch_size",
    "decode_tokens",
    "repeat_batches",
    "report_perf",
    "dp",
    "trace_mode",
    "context_bucket",
    "cache_protocol",
    "capabilities",
    "required_resolution",
    "acceptance_condition",
)
_LLAMA_DEMO_MANIFESTS = {
    _LLAMA3_8B_CONTRACT_ID: {
        "ids": frozenset(
            {
                "p150.accuracy.batch_1",
                "p150.accuracy.batch_32",
                "p150.accuracy.eval_32_repeat_1",
                "p150.accuracy.eval_32_repeat_3",
                "p150.accuracy.seeded_cross_cardinality",
                "p150.accuracy.token_accuracy",
                "p150.performance.batch_1",
                "p150.performance.batch_32",
                "p150.performance.eval_32_repeat_1",
                "p150.performance.eval_32_repeat_3",
                "p150.performance.seeded_cross_cardinality",
                "p150.performance.token_accuracy",
                "p150x4.accuracy.batch_1",
                "p150x4.accuracy.batch_32",
                "p150x4.accuracy.ci_b1_dp4",
                "p150x4.accuracy.eval_32_repeat_1",
                "p150x4.accuracy.eval_32_repeat_3",
                "p150x4.accuracy.seeded_cross_cardinality",
                "p150x4.accuracy.token_accuracy",
                "p150x4.performance.batch_1",
                "p150x4.performance.batch_32",
                "p150x4.performance.ci_b1_dp4",
                "p150x4.performance.eval_32_repeat_1",
                "p150x4.performance.eval_32_repeat_3",
                "p150x4.performance.seeded_cross_cardinality",
                "p150x4.performance.token_accuracy",
                "p300.accuracy.ci_b1_dp2",
                "p300.performance.ci_b1_dp2",
            }
        ),
        "sha256": "ddfe8b75427e2d1720e48811e7d97eaaaebcf290f2fcbba093cb6ece15384a39",
    },
    _LLAMA33_70B_CONTRACT_ID: {
        "ids": frozenset(
            {
                "p150x4.accuracy.batch_32_ci",
                "p150x4.accuracy.eval_32",
                "p150x4.accuracy.eval_32_perf_report",
                "p150x4.accuracy.token_accuracy",
                "p150x4.performance.batch_32_ci",
                "p150x4.performance.eval_32",
                "p150x4.performance.eval_32_perf_report",
                "p150x4.performance.token_accuracy",
            }
        ),
        "sha256": "2da2ae1d31e1f5eaab7cf989ea8cc4277712f3da699c8d5d961315267e58fedd",
    },
}
_QWEN_DEMO_MANIFEST = {
    "p150x4.performance.token_accuracy": (
        "performance",
        "token-accuracy",
        "token-accuracy",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b[performance-token-accuracy-P150x4]",
    ),
    "p150x4.accuracy.token_accuracy": (
        "accuracy",
        "token-accuracy",
        "token-accuracy",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b[accuracy-token-accuracy-P150x4]",
    ),
    "p150x4.performance.eval_32": (
        "performance",
        "eval-32",
        "eval-32",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b[performance-eval-32-P150x4]",
    ),
    "p150x4.accuracy.eval_32": (
        "accuracy",
        "eval-32",
        "eval-32",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b[accuracy-eval-32-P150x4]",
    ),
    "p150x4.performance.eval_32_perf_report": (
        "performance",
        "eval-32-perf-report",
        "eval-32-perf-report",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b[performance-eval-32-perf-report-P150x4]",
    ),
    "p150x4.accuracy.eval_32_perf_report": (
        "accuracy",
        "eval-32-perf-report",
        "eval-32-perf-report",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b[accuracy-eval-32-perf-report-P150x4]",
    ),
    "p150x4.performance.batch_32_ci": (
        "performance",
        "batch-32-ci",
        "batch-32-ci",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b[performance-batch-32-ci-P150x4]",
    ),
    "p150x4.accuracy.batch_32_ci": (
        "accuracy",
        "batch-32-ci",
        "batch-32-ci",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b[accuracy-batch-32-ci-P150x4]",
    ),
    "p150x4.accuracy.seeded_cross_cardinality": (
        "accuracy",
        "seeded-cross-cardinality",
        "seeded-cross-cardinality",
        "models/common/tests/demos/qwen3_32b/demo.py::test_qwen3_32b_p150x4_seeded_cross_cardinality[P150x4]",
    ),
}
_QWEN_DEMO_MANIFEST_SHA256 = "81371403fab67ffa4bdc02f491154aa484d4ca74a534919031aeed21e05021db"


class CapabilityContractValidationError(ValueError):
    """Raised when a required-capability contract violates its declaration."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapabilityContractValidationError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise CapabilityContractValidationError(f"{path}: top-level JSON value must be an object")
    return value


def load_schema(schema_path: Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    """Load the declarative schema and fail if it is not Draft 2020-12."""

    schema = _read_json(schema_path)
    if schema.get("$schema") != _DRAFT_2020_12:
        raise CapabilityContractValidationError(f"{schema_path}: $schema must declare JSON Schema Draft 2020-12")
    if schema.get("$id") != _INSTANCE_SCHEMA:
        raise CapabilityContractValidationError(f"{schema_path}: unexpected or missing $id")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise CapabilityContractValidationError(f"{schema_path}: root schema must be a closed object")
    if set(schema.get("required", ())) != set(_ROOT_KEYS):
        raise CapabilityContractValidationError(f"{schema_path}: root required keys disagree with stdlib validator")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or set(properties) != set(_ROOT_KEYS):
        raise CapabilityContractValidationError(f"{schema_path}: root properties disagree with stdlib validator")
    model_schema = properties.get("model", {})
    if set(model_schema.get("required", ())) != set(_MODEL_KEYS):
        raise CapabilityContractValidationError(f"{schema_path}: model required keys disagree with stdlib validator")
    demo_schema = schema.get("$defs", {}).get("demoRequirement", {})
    if set(demo_schema.get("required", ())) != set(_DEMO_KEYS):
        raise CapabilityContractValidationError(f"{schema_path}: demo required keys disagree with stdlib validator")
    cache_schema = demo_schema.get("properties", {}).get("cache_protocol")
    expected_cache_schema = {
        "type": "array",
        "prefixItems": [{"const": "cold_write"}, {"const": "warm_read"}],
        "items": False,
        "minItems": 2,
        "maxItems": 2,
    }
    if cache_schema != expected_cache_schema:
        raise CapabilityContractValidationError(
            f"{schema_path}: cache_protocol declaration disagrees with exact cold_write/warm_read protocol"
        )
    return schema


def _closed_object(
    value: Any,
    path: str,
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(value, dict):
        return None, [f"{path} must be an object"]
    required_set = set(required)
    allowed = required_set | set(optional)
    errors = [f"{path} missing required key {key}" for key in sorted(required_set - value.keys())]
    errors.extend(f"{path} has unknown key {key}" for key in sorted(value.keys() - allowed))
    return value, errors


def _string(value: Any, path: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, str):
        return [f"{path} must be a string"]
    if nonempty and not value.strip():
        return [f"{path} must be non-empty"]
    return []


def _integer(value: Any, path: str, *, minimum: int | None = None) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, int):
        return [f"{path} must be an integer"]
    if minimum is not None and value < minimum:
        return [f"{path} must be >= {minimum}"]
    return []


def _boolean(value: Any, path: str) -> list[str]:
    return [] if isinstance(value, bool) else [f"{path} must be a boolean"]


def _enum(value: Any, path: str, allowed: set[str]) -> list[str]:
    if not isinstance(value, str) or value not in allowed:
        return [f"{path} must be one of {sorted(allowed)}"]
    return []


def _string_list(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        return [f"{path} must be an array"]
    errors = []
    if nonempty and not value:
        errors.append(f"{path} must not be empty")
    for index, item in enumerate(value):
        errors.extend(_string(item, f"{path}.{index}"))
    return errors


def _enum_list(value: Any, path: str, allowed: set[str], *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        return [f"{path} must be an array"]
    errors = []
    if nonempty and not value:
        errors.append(f"{path} must not be empty")
    for index, item in enumerate(value):
        errors.extend(_enum(item, f"{path}.{index}", allowed))
    if all(isinstance(item, str) for item in value) and len(value) != len(set(value)):
        errors.append(f"{path} must not contain duplicates")
    return errors


def _const(value: Any, path: str, expected: Any) -> list[str]:
    return [] if value == expected else [f"{path} must equal {expected!r}"]


def _duplicate_values(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _require_unique(items: Iterable[Mapping[str, Any]], key: str, label: str, errors: list[str]) -> None:
    duplicates = _duplicate_values(str(item[key]) for item in items)
    if duplicates:
        errors.append(f"duplicate {label}: {', '.join(duplicates)}")


def _validate_geometry(value: Any, path: str) -> list[str]:
    keys = ("id", "ci_sku", "mesh_name", "mesh_shape", "dies", "tp_per_lane", "dp", "fabric_config", "role")
    item, errors = _closed_object(value, path, required=keys)
    if item is None or errors:
        return errors
    for key in ("id", "ci_sku", "mesh_name"):
        errors.extend(_string(item[key], f"{path}.{key}"))
    shape = item["mesh_shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        errors.append(f"{path}.mesh_shape must contain exactly two integers")
    else:
        for index, dimension in enumerate(shape):
            errors.extend(_integer(dimension, f"{path}.mesh_shape.{index}", minimum=1))
    for key in ("dies", "tp_per_lane", "dp"):
        errors.extend(_integer(item[key], f"{path}.{key}", minimum=1))
    if item["fabric_config"] is not None:
        errors.extend(_string(item["fabric_config"], f"{path}.fabric_config"))
    errors.extend(_enum(item["role"], f"{path}.role", {"required", "additive_functional", "development_standin"}))
    return errors


def _validate_demo(value: Any, path: str) -> list[str]:
    item, errors = _closed_object(value, path, required=_DEMO_KEYS)
    if item is None or errors:
        return errors
    for key in ("id", "geometry_id", "demo_case", "source_parameter_id", "node_id", "acceptance_condition"):
        errors.extend(_string(item[key], f"{path}.{key}"))
    errors.extend(_enum(item["profile"], f"{path}.profile", _OPTIMIZATION_PROFILES))
    integer_fields = (
        ("batch_size", 1),
        ("decode_tokens", 0),
        ("repeat_batches", 1),
        ("dp", 1),
        ("context_bucket", 1),
    )
    for key, minimum in integer_fields:
        errors.extend(_integer(item[key], f"{path}.{key}", minimum=minimum))
    errors.extend(_boolean(item["report_perf"], f"{path}.report_perf"))
    errors.extend(_enum(item["trace_mode"], f"{path}.trace_mode", _TRACE_MODES))
    errors.extend(
        _enum_list(
            item["cache_protocol"],
            f"{path}.cache_protocol",
            {"cold_write", "warm_read"},
            nonempty=True,
        )
    )
    if item["cache_protocol"] != ["cold_write", "warm_read"]:
        errors.append(f"{path}.cache_protocol must equal ['cold_write', 'warm_read']")
    errors.extend(_enum_list(item["capabilities"], f"{path}.capabilities", _CAPABILITIES, nonempty=True))
    errors.extend(_const(item["required_resolution"], f"{path}.required_resolution", _RESOLUTION))
    return errors


def _validate_context(value: Any, path: str) -> list[str]:
    item, errors = _closed_object(
        value,
        path,
        required=("bucket", "required_subcases", "required_resolution", "acceptance_condition"),
        optional=("bucket_rule",),
    )
    if item is None or errors:
        return errors
    if item["bucket"] is None:
        if "bucket_rule" not in item:
            errors.append(f"{path}.bucket_rule is required when bucket is null")
    else:
        errors.extend(_integer(item["bucket"], f"{path}.bucket", minimum=1))
    if "bucket_rule" in item:
        errors.extend(_string(item["bucket_rule"], f"{path}.bucket_rule"))
    errors.extend(_enum_list(item["required_subcases"], f"{path}.required_subcases", _CONTEXT_SUBCASES, nonempty=True))
    if (
        isinstance(item["required_subcases"], list)
        and all(isinstance(subcase, str) for subcase in item["required_subcases"])
        and set(item["required_subcases"]) != _CONTEXT_SUBCASES
    ):
        errors.append(f"{path}.required_subcases must declare all long/chunked/cached-prefill cases")
    errors.extend(_const(item["required_resolution"], f"{path}.required_resolution", _RESOLUTION))
    errors.extend(_string(item["acceptance_condition"], f"{path}.acceptance_condition"))
    return errors


def _validate_serving(value: Any, path: str) -> list[str]:
    keys = (
        "row_id",
        "geometry_id",
        "profile",
        "model_optimization_profile",
        "dp",
        "trace_mode",
        "trace_prefill_buckets",
        "context_requirements",
        "cached_prefill",
        "chunked_prefill",
        "tier0_smoke",
        "tier1_performance",
        "tier2_quality",
        "required_resolution",
        "acceptance_condition",
    )
    row, errors = _closed_object(value, path, required=keys)
    if row is None or errors:
        return errors
    for key in ("row_id", "geometry_id", "acceptance_condition"):
        errors.extend(_string(row[key], f"{path}.{key}"))
    errors.extend(_enum(row["profile"], f"{path}.profile", {"decode_only", "all"}))
    errors.extend(_const(row["model_optimization_profile"], f"{path}.model_optimization_profile", "performance"))
    errors.extend(_integer(row["dp"], f"{path}.dp", minimum=1))
    errors.extend(_enum(row["trace_mode"], f"{path}.trace_mode", {"decode_only", "all"}))
    buckets = row["trace_prefill_buckets"]
    if not isinstance(buckets, list):
        errors.append(f"{path}.trace_prefill_buckets must be an array")
    else:
        for index, bucket in enumerate(buckets):
            errors.extend(_integer(bucket, f"{path}.trace_prefill_buckets.{index}", minimum=1))
        all_integer_buckets = all(isinstance(bucket, int) and not isinstance(bucket, bool) for bucket in buckets)
        if all_integer_buckets and len(buckets) != len(set(buckets)):
            errors.append(f"{path}.trace_prefill_buckets must not contain duplicates")
    contexts = row["context_requirements"]
    if not isinstance(contexts, list) or not contexts:
        errors.append(f"{path}.context_requirements must be a non-empty array")
    else:
        for index, context in enumerate(contexts):
            errors.extend(_validate_context(context, f"{path}.context_requirements.{index}"))
    errors.extend(_const(row["cached_prefill"], f"{path}.cached_prefill", "required"))
    errors.extend(_const(row["chunked_prefill"], f"{path}.chunked_prefill", "required"))
    errors.extend(_boolean(row["tier0_smoke"], f"{path}.tier0_smoke"))
    errors.extend(_const(row["tier1_performance"], f"{path}.tier1_performance", "required"))
    errors.extend(_const(row["tier2_quality"], f"{path}.tier2_quality", "required_human_review"))
    errors.extend(_const(row["required_resolution"], f"{path}.required_resolution", _RESOLUTION))
    return errors


def _validate_cross_cutting(value: Any, path: str) -> list[str]:
    item, errors = _closed_object(
        value,
        path,
        required=("id", "capability", "applies_to", "required_resolution", "acceptance_condition"),
    )
    if item is None or errors:
        return errors
    for key in ("id", "capability", "acceptance_condition"):
        errors.extend(_string(item[key], f"{path}.{key}"))
    errors.extend(_string_list(item["applies_to"], f"{path}.applies_to", nonempty=True))
    if (
        isinstance(item["applies_to"], list)
        and all(isinstance(geometry_id, str) for geometry_id in item["applies_to"])
        and len(item["applies_to"]) != len(set(item["applies_to"]))
    ):
        errors.append(f"{path}.applies_to must not contain duplicates")
    errors.extend(_const(item["required_resolution"], f"{path}.required_resolution", _RESOLUTION))
    return errors


def _validate_exclusion(value: Any, path: str) -> list[str]:
    item, errors = _closed_object(
        value,
        path,
        required=("id", "classification", "reason", "acceptance_effect"),
    )
    if item is None or errors:
        return errors
    for key in ("id", "reason", "acceptance_effect"):
        errors.extend(_string(item[key], f"{path}.{key}"))
    errors.extend(
        _enum(
            item["classification"],
            f"{path}.classification",
            {"post_v1_stretch", "out_of_scope", "retained_tttv1", "development_only"},
        )
    )
    return errors


def _demo_manifest_digest(demos: Iterable[Mapping[str, Any]]) -> str:
    """Return a stable digest over executable immutable demo fields."""

    manifest = [
        {field: row[field] for field in _IMMUTABLE_DEMO_MANIFEST_FIELDS}
        for row in sorted(demos, key=lambda item: str(item["id"]))
    ]
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _literal_parametrize_ids(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, set[str]]:
    """Extract single-argument pytest parametrize IDs without importing the demo."""

    result: dict[str, set[str]] = {}
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or _call_name(decorator.func) != "pytest.mark.parametrize":
            continue
        if len(decorator.args) < 2:
            continue
        argument = decorator.args[0]
        values = decorator.args[1]
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            continue
        if "," in argument.value or not isinstance(values, (ast.List, ast.Tuple)):
            continue
        ids: set[str] = set()
        for element in values.elts:
            parameter_value: str | None = None
            parameter_id: str | None = None
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                parameter_value = element.value
            elif isinstance(element, ast.Call) and _call_name(element.func) == "pytest.param" and element.args:
                first = element.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    parameter_value = first.value
                for keyword in element.keywords:
                    if (
                        keyword.arg == "id"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        parameter_id = keyword.value.value
            if parameter_id is not None:
                ids.add(parameter_id)
            elif parameter_value is not None:
                ids.add(parameter_value)
        if ids:
            result[argument.value] = ids
    return result


def _demo_source_errors(contract: Mapping[str, Any]) -> list[str]:
    """Check entry point and declared pytest functions/parameters host-only.

    This deliberately does not claim pytest collection parity.  Authoritative
    ``--collect-only`` remains a separate acceptance gate with the real test
    environment and device selector.
    """

    errors: list[str] = []
    entry_point = contract["model"]["demo_entry_point"]
    source_path = MODELS_DIR.parent / entry_point
    if not source_path.is_file():
        return [f"declared demo entry point does not exist: {entry_point}"]
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        return [f"cannot parse declared demo entry point {entry_point}: {error}"]

    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    parametrizations = {name: _literal_parametrize_ids(function) for name, function in functions.items()}

    for row in contract["demo_requirements"]:
        node_id = row["node_id"]
        node_path, separator, pytest_node = node_id.partition("::")
        if not separator:
            errors.append(f"demo requirement {row['id']} node_id must contain path::function")
            continue
        if node_path != entry_point:
            errors.append(
                f"demo requirement {row['id']} node path {node_path!r} does not match entry point {entry_point!r}"
            )
        function_name = pytest_node.split("[", 1)[0]
        if function_name not in functions:
            errors.append(
                f"demo requirement {row['id']} declares missing test function {function_name!r} in {entry_point}"
            )
            continue

        parameters = parametrizations[function_name]
        profiles = parameters.get("optimizations")
        if profiles is not None and row["profile"] not in profiles:
            errors.append(f"demo requirement {row['id']} profile {row['profile']!r} is not declared by {function_name}")
        source_ids: set[str] = set()
        for parameter_name, ids in parameters.items():
            if parameter_name != "optimizations":
                source_ids.update(ids)
        if source_ids and row["source_parameter_id"] not in source_ids:
            errors.append(
                f"demo requirement {row['id']} source_parameter_id {row['source_parameter_id']!r} "
                f"is not declared by {function_name}"
            )
    return errors


def _structural_errors(contract: Mapping[str, Any]) -> list[str]:
    root, errors = _closed_object(contract, "<root>", required=_ROOT_KEYS)
    if root is None or errors:
        return errors
    errors.extend(_const(root["$schema"], "$schema", _INSTANCE_SCHEMA))
    errors.extend(_const(root["schema_version"], "schema_version", 1))
    errors.extend(_string(root["contract_id"], "contract_id"))

    model, nested = _closed_object(root["model"], "model", required=_MODEL_KEYS)
    errors.extend(nested)
    if model is not None and not nested:
        for key in ("package", "hf_model_id", "demo_entry_point"):
            errors.extend(_string(model[key], f"model.{key}"))

    authority, nested = _closed_object(
        root["authority"], "authority", required=("plan", "integration_contract", "source_files")
    )
    errors.extend(nested)
    if authority is not None and not nested:
        errors.extend(_string(authority["plan"], "authority.plan"))
        errors.extend(_string(authority["integration_contract"], "authority.integration_contract"))
        errors.extend(_string_list(authority["source_files"], "authority.source_files", nonempty=True))

    scope, nested = _closed_object(
        root["scope"], "scope", required=("declaration_state", "immutability", "acceptance_rule")
    )
    errors.extend(nested)
    if scope is not None and not nested:
        errors.extend(_const(scope["declaration_state"], "scope.declaration_state", "pre_acceptance"))
        errors.extend(_string(scope["immutability"], "scope.immutability"))
        errors.extend(_string(scope["acceptance_rule"], "scope.acceptance_rule"))

    collections = (
        ("geometries", _validate_geometry, True),
        ("demo_requirements", _validate_demo, True),
        ("serving_requirements", _validate_serving, True),
        ("cross_cutting_requirements", _validate_cross_cutting, False),
        ("excluded_or_deferred", _validate_exclusion, False),
    )
    for key, validator, nonempty in collections:
        values = root[key]
        if not isinstance(values, list):
            errors.append(f"{key} must be an array")
            continue
        if nonempty and not values:
            errors.append(f"{key} must not be empty")
        for index, value in enumerate(values):
            errors.extend(validator(value, f"{key}.{index}"))
    return errors


def _semantic_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    geometries = contract["geometries"]
    demos = contract["demo_requirements"]
    serving = contract["serving_requirements"]
    cross_cutting = contract["cross_cutting_requirements"]

    contract_id = contract["contract_id"]
    expected_identity = _REQUIRED_CONTRACT_IDENTITIES.get(contract_id)
    if expected_identity is None:
        errors.append(
            f"contract_id must identify one of the three immutable BH contracts: "
            f"{', '.join(sorted(_REQUIRED_CONTRACT_IDENTITIES))}"
        )
    else:
        for key, expected in expected_identity.items():
            if contract["model"][key] != expected:
                errors.append(f"contract {contract_id} model.{key} must equal {expected!r}")

    _require_unique(geometries, "id", "geometry IDs", errors)
    _require_unique(demos, "id", "demo requirement IDs", errors)
    _require_unique(serving, "row_id", "serving row IDs", errors)
    _require_unique(cross_cutting, "id", "cross-cutting requirement IDs", errors)

    node_ids = [item["node_id"] for item in demos]
    duplicates = _duplicate_values(node_ids)
    if duplicates:
        errors.append(f"duplicate demo node IDs: {', '.join(duplicates)}")

    geometry_ids = {geometry["id"] for geometry in geometries}
    for item in demos:
        if item["geometry_id"] not in geometry_ids:
            errors.append(f"demo requirement {item['id']} references unknown geometry_id {item['geometry_id']}")
    for row in serving:
        if row["geometry_id"] not in geometry_ids:
            errors.append(f"serving row {row['row_id']} references unknown geometry_id {row['geometry_id']}")
        profile = row["profile"]
        trace_mode = row["trace_mode"]
        buckets = row["trace_prefill_buckets"]
        if profile != trace_mode:
            errors.append(f"serving row {row['row_id']} has profile={profile} but trace_mode={trace_mode}")
        if profile == "decode_only" and buckets:
            errors.append(f"decode-only serving row {row['row_id']} must not declare prefill trace buckets")
        if profile == "all" and not buckets:
            errors.append(f"all-trace serving row {row['row_id']} must declare prefill trace buckets")
    for requirement in cross_cutting:
        for geometry_id in requirement["applies_to"]:
            if geometry_id not in geometry_ids:
                errors.append(
                    f"cross-cutting requirement {requirement['id']} references unknown geometry {geometry_id}"
                )

    expected_llama_manifest = _LLAMA_DEMO_MANIFESTS.get(contract_id)
    if expected_llama_manifest is not None:
        actual_ids = {item["id"] for item in demos}
        expected_ids = expected_llama_manifest["ids"]
        missing_ids = sorted(expected_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_ids)
        model_label = expected_identity["package"] if expected_identity is not None else contract_id
        if missing_ids:
            errors.append(f"{model_label} demo manifest missing immutable rows: {', '.join(missing_ids)}")
        if extra_ids:
            errors.append(f"{model_label} demo manifest has undeclared rows: {', '.join(extra_ids)}")
        if not missing_ids and not extra_ids and _demo_manifest_digest(demos) != expected_llama_manifest["sha256"]:
            errors.append(
                f"{model_label} demo manifest does not match immutable row/node/profile/case/geometry/"
                "workload/trace/cache/capability/acceptance fields"
            )

    if contract_id == _LLAMA3_8B_CONTRACT_ID:
        cross_rows = [item for item in demos if item["demo_case"] == "seeded-cross-cardinality"]
        expected_cross_ids = {
            "p150.performance.seeded_cross_cardinality",
            "p150.accuracy.seeded_cross_cardinality",
            "p150x4.performance.seeded_cross_cardinality",
            "p150x4.accuracy.seeded_cross_cardinality",
        }
        if {item["id"] for item in cross_rows} != expected_cross_ids:
            errors.append("llama3_8b must preserve all four canonical seeded cross-cardinality rows")
        for row in cross_rows:
            acceptance = row["acceptance_condition"]
            for required_phrase in (
                "32 true active-batch-1 sequential-prefill controls",
                "exact token-ID comparisons",
                "INVARIANT",
                "BATCHED_PREFILL_REJECTED",
                "completed negative experiment disposition",
                "does not pass invariance",
                "sequential production policy to remain retained after rejection",
                "Malformed or incomplete outputs",
            ):
                if required_phrase not in acceptance:
                    errors.append(
                        "llama3_8b seeded cross-cardinality acceptance must define complete executed "
                        f"experiment semantics including {required_phrase!r}"
                    )

        cross_by_id = {item["id"]: item for item in cross_cutting}
        experiment = cross_by_id.get("cross_cardinality_invariance")
        policy = cross_by_id.get("disable_batched_prefill_policy")
        if experiment is None or policy is None:
            errors.append("llama3_8b must preserve experiment-disposition and batched-prefill-policy requirements")
        else:
            experiment_text = f"{experiment['capability']} {experiment['acceptance_condition']}"
            policy_text = f"{policy['capability']} {policy['acceptance_condition']}"
            for phrase in (
                "INVARIANT",
                "BATCHED_PREFILL_REJECTED",
                "either disposition satisfies",
                "does not pass invariance",
                "malformed or incomplete execution",
            ):
                if phrase not in experiment_text:
                    errors.append(f"llama3_8b experiment disposition must preserve phrase {phrase!r}")
            for phrase in (
                "INVARIANT",
                "BATCHED_PREFILL_REJECTED",
                "remains sequential after BATCHED_PREFILL_REJECTED",
                "independent of completed experiment disposition",
            ):
                if phrase not in policy_text:
                    errors.append(f"llama3_8b batched-prefill policy must follow verdict phrase {phrase!r}")

        performance_policy = cross_by_id.get("fail_closed_performance")
        if performance_policy is None:
            errors.append("llama3_8b must preserve the observational performance-floor policy")
        else:
            performance_text = f"{performance_policy['capability']} {performance_policy['acceptance_condition']}"
            for phrase in (
                "must not block BH model execution or observational measurement",
                "cannot establish performance acceptance",
                "Every complete declared floor remains enforced",
                "every failed meets_target result fails",
                "performance acceptance requires an independently justified, frozen floor",
                "records performance acceptance only when a complete independently frozen floor exists",
                "every declared target passes",
            ):
                if phrase not in performance_text:
                    errors.append(
                        "llama3_8b performance-floor policy must preserve observational execution and "
                        f"fail-closed acceptance phrase {phrase!r}"
                    )

    if contract_id == _LLAMA33_70B_CONTRACT_ID:
        performance_policy = {item["id"]: item for item in cross_cutting}.get("fail_closed_performance")
        if performance_policy is None:
            errors.append("llama33_70b must preserve its observational-without-floor performance policy")
        else:
            performance_text = f"{performance_policy['capability']} {performance_policy['acceptance_condition']}"
            for phrase in (
                "observational",
                "must not claim acceptance",
                "complete independently frozen floor",
                "target miss fails",
            ):
                if phrase not in performance_text:
                    errors.append(
                        "llama33_70b performance policy must preserve observational execution and "
                        f"fail-closed complete-floor semantics including {phrase!r}"
                    )

    if contract_id == _QWEN_CONTRACT_ID:
        actual_manifest = {
            item["id"]: (item["profile"], item["demo_case"], item["source_parameter_id"], item["node_id"])
            for item in demos
        }
        missing_ids = sorted(set(_QWEN_DEMO_MANIFEST) - set(actual_manifest))
        extra_ids = sorted(set(actual_manifest) - set(_QWEN_DEMO_MANIFEST))
        if missing_ids:
            errors.append(f"qwen3_32b demo manifest missing immutable rows: {', '.join(missing_ids)}")
        if extra_ids:
            errors.append(f"qwen3_32b demo manifest has undeclared rows: {', '.join(extra_ids)}")
        for row_id in sorted(set(actual_manifest) & set(_QWEN_DEMO_MANIFEST)):
            if actual_manifest[row_id] != _QWEN_DEMO_MANIFEST[row_id]:
                errors.append(f"qwen3_32b demo manifest row {row_id} does not match its immutable identity")
        if not missing_ids and not extra_ids and _demo_manifest_digest(demos) != _QWEN_DEMO_MANIFEST_SHA256:
            errors.append(
                "qwen3_32b demo manifest does not match immutable row/node/profile/case/geometry/"
                "workload/trace/cache/capability/acceptance fields"
            )

        eval_perf_rows = [item for item in demos if item["demo_case"] == "eval-32-perf-report"]
        if {item["profile"] for item in eval_perf_rows} != _OPTIMIZATION_PROFILES:
            errors.append("qwen3_32b must declare eval-32-perf-report for performance and accuracy profiles")
        for item in eval_perf_rows:
            if item["trace_mode"] != "all":
                errors.append(f"qwen3_32b eval-32-perf-report row {item['id']} must declare full trace mode")

        all_trace_rows = [row for row in serving if row["profile"] == "all"]
        if not all_trace_rows:
            errors.append("qwen3_32b must declare an all-trace serving row")
        for row in all_trace_rows:
            if row["trace_prefill_buckets"] != [128, 1024]:
                errors.append(
                    f"qwen3_32b all-trace serving row {row['row_id']} must declare model-owned "
                    "Q128/Q1024 prefill buckets"
                )

        cross_rows = [item for item in demos if item["demo_case"] == "seeded-cross-cardinality"]
        expected_cross_node = (
            "models/common/tests/demos/qwen3_32b/demo.py::" "test_qwen3_32b_p150x4_seeded_cross_cardinality[P150x4]"
        )
        if len(cross_rows) != 1:
            errors.append("qwen3_32b must declare one canonical seeded cross-cardinality node")
        else:
            row = cross_rows[0]
            if row["node_id"] != expected_cross_node:
                errors.append(f"qwen3_32b seeded cross-cardinality node_id must be {expected_cross_node}")
            if (
                row["source_parameter_id"] != "seeded-cross-cardinality"
                or row["profile"] != "accuracy"
                or row["batch_size"] != 32
                or row["decode_tokens"] != 32
                or row["trace_mode"] != "decode_only"
                or row["context_bucket"] != 1024
                or row["report_perf"] is not False
            ):
                errors.append(
                    "qwen3_32b canonical seeded cross-cardinality row must preserve the "
                    "accuracy/batch32/decode32/decode-only/Q1024/non-perf geometry"
                )
            acceptance = row["acceptance_condition"]
            for required_phrase in (
                "32 true active-batch-1 controls",
                "token-ID tuples",
                "regular-batched Q128 rows",
                "Q128 active-30/padded-32",
                "Q1024 active-2/padded-2 regular-batched",
                "captures exactly one top-k decode trace",
                "successful replay of that top-k trace key",
                "zero coverage misses",
                "zero post-activation compile rejections",
                "malformed or incomplete outputs",
                "INVARIANT",
                "BATCHED_PREFILL_REJECTED",
                "retains the sequential P150x4 policy",
            ):
                if required_phrase not in acceptance:
                    errors.append(
                        "qwen3_32b seeded cross-cardinality acceptance must define exact-token "
                        f"executed verdict semantics including {required_phrase!r}"
                    )
        cross_by_id = {item["id"]: item for item in cross_cutting}
        experiment = cross_by_id.get("cross_cardinality_invariance")
        policy = cross_by_id.get("disable_batched_prefill_policy")
        performance_policy = cross_by_id.get("fail_closed_performance")
        if experiment is None or policy is None:
            errors.append("qwen3_32b must preserve experiment-disposition and batched-prefill-policy requirements")
        else:
            experiment_text = f"{experiment['capability']} {experiment['acceptance_condition']}"
            policy_text = f"{policy['capability']} {policy['acceptance_condition']}"
            for phrase in ("INVARIANT", "BATCHED_PREFILL_REJECTED", "either disposition satisfies"):
                if phrase not in experiment_text:
                    errors.append(f"qwen3_32b experiment disposition must permit completed verdict phrase {phrase!r}")
            for phrase in ("INVARIANT", "BATCHED_PREFILL_REJECTED", "remains sequential"):
                if phrase not in policy_text:
                    errors.append(f"qwen3_32b batched-prefill policy must follow recorded verdict phrase {phrase!r}")
        if performance_policy is None:
            errors.append("qwen3_32b must preserve its observational-without-floor performance policy")
        else:
            performance_text = f"{performance_policy['capability']} {performance_policy['acceptance_condition']}"
            for phrase in (
                "observational",
                "must not claim acceptance",
                "complete independently frozen floor",
                "target miss fails",
            ):
                if phrase not in performance_text:
                    errors.append(
                        "qwen3_32b performance policy must preserve observational execution and "
                        f"fail-closed complete-floor semantics including {phrase!r}"
                    )
    if expected_identity is not None:
        errors.extend(_demo_source_errors(contract))
    return errors


def validate_contract_data(
    contract: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    source: str = "<memory>",
) -> None:
    """Validate one loaded contract using the closed stdlib implementation."""

    if schema.get("$schema") != _DRAFT_2020_12:
        raise CapabilityContractValidationError("loaded schema is not declared as Draft 2020-12")
    errors = _structural_errors(contract)
    if not errors:
        errors.extend(_semantic_errors(contract))
    if errors:
        details = "\n  - ".join(errors)
        raise CapabilityContractValidationError(f"{source}: capability contract validation failed:\n  - {details}")


def validate_contract(path: Path, schema: Mapping[str, Any]) -> None:
    validate_contract_data(_read_json(path), schema, source=str(path))


def validate_contracts(
    contract_paths: Iterable[Path] = DEFAULT_CONTRACTS,
    *,
    schema_path: Path = DEFAULT_SCHEMA,
) -> tuple[Path, ...]:
    schema = load_schema(schema_path)
    paths = tuple(contract_paths)
    for path in paths:
        validate_contract(path, schema)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contracts",
        nargs="*",
        type=Path,
        help="Contracts to validate; defaults to all three BH files",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = validate_contracts(args.contracts or DEFAULT_CONTRACTS, schema_path=args.schema)
    for path in paths:
        print(f"validated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
