#!/usr/bin/env python3
"""Validate immutable TTTv2 BlackHole required-capability contracts.

The qualification path is deliberately host-only and dependency-free: it
imports neither TTNN, model packages, nor third-party schema libraries.  The
checked-in JSON Schema remains the declarative specification; this module
implements fail-closed validation of its exact contract shape plus relational
invariants that JSON Schema cannot conveniently express across arrays.
"""

from __future__ import annotations

import argparse
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
        raise CapabilityContractValidationError(
            f"{schema_path}: $schema must declare JSON Schema Draft 2020-12"
        )
    if schema.get("$id") != _INSTANCE_SCHEMA:
        raise CapabilityContractValidationError(f"{schema_path}: unexpected or missing $id")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise CapabilityContractValidationError(f"{schema_path}: root schema must be a closed object")
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
    keys = (
        "id", "geometry_id", "profile", "demo_case", "source_parameter_id", "node_id",
        "batch_size", "decode_tokens", "repeat_batches", "report_perf", "dp", "trace_mode",
        "context_bucket", "cache_protocol", "capabilities", "required_resolution", "acceptance_condition",
    )
    item, errors = _closed_object(value, path, required=keys)
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
        "row_id", "geometry_id", "profile", "model_optimization_profile", "dp", "trace_mode",
        "trace_prefill_buckets", "context_requirements", "cached_prefill", "chunked_prefill",
        "tier0_smoke", "tier1_performance", "tier2_quality", "required_resolution", "acceptance_condition",
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
        all_integer_buckets = all(
            isinstance(bucket, int) and not isinstance(bucket, bool) for bucket in buckets
        )
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


def _structural_errors(contract: Mapping[str, Any]) -> list[str]:
    root_keys = (
        "$schema", "schema_version", "contract_id", "model", "authority", "scope", "geometries",
        "demo_requirements", "serving_requirements", "cross_cutting_requirements", "excluded_or_deferred",
    )
    root, errors = _closed_object(contract, "<root>", required=root_keys)
    if root is None or errors:
        return errors
    errors.extend(_const(root["$schema"], "$schema", _INSTANCE_SCHEMA))
    errors.extend(_const(root["schema_version"], "schema_version", 1))
    errors.extend(_string(root["contract_id"], "contract_id"))

    model, nested = _closed_object(root["model"], "model", required=("package", "hf_model_id", "demo_entry_point"))
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
