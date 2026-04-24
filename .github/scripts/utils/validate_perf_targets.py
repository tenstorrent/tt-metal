#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from models.demos.utils.model_targets import resolve_target_entry
except ModuleNotFoundError:
    # Allow direct invocation without PYTHONPATH set by adding repo root.
    REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(REPO_ROOT))
    from models.demos.utils.model_targets import resolve_target_entry


LOWER_IS_BETTER_METRICS = {
    "prefill_time_to_token",
    "compile_prefill",
    "compile_decode",
}

METRIC_NAME_MAP = {
    "prefill_t/s": ("inference_prefill", "tokens/s"),
    "prefill_time_to_token": ("inference_prefill", "time_to_token"),
    "decode_t/s": ("inference_decode", "tokens/s"),
    "decode_t/s/u": ("inference_decode", "tokens/s/user"),
    "top1": ("inference_decode", "top1_token_accuracy"),
    "top5": ("inference_decode", "top5_token_accuracy"),
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid benchmark JSON payload in {path}")
    return payload


def _measurement_lookup(benchmark_json: dict[str, Any]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for measurement in benchmark_json.get("measurements", []):
        if not isinstance(measurement, dict):
            continue
        step_name = measurement.get("step_name")
        name = measurement.get("name")
        value = measurement.get("value")
        if isinstance(step_name, str) and isinstance(name, str) and _is_number(value):
            lookup[(step_name, name)] = float(value)
    return lookup


def _extract_metric_value(metric_name: str, lookup: dict[tuple[str, str], float]) -> float | None:
    if metric_name in METRIC_NAME_MAP:
        return lookup.get(METRIC_NAME_MAP[metric_name])

    for (step_name, name), value in lookup.items():
        if name == metric_name:
            return value
        if f"{step_name}.{name}" == metric_name:
            return value
    return None


def _metric_tolerance(metric_name: str, thresholds: dict[str, Any], default_high_tolerance: float) -> float:
    explicit = thresholds.get(f"{metric_name}_tolerance")
    if _is_number(explicit):
        return float(explicit)
    generic = thresholds.get("tolerance")
    if _is_number(generic):
        return float(generic)
    return default_high_tolerance


def _check_metric(
    metric_name: str,
    expected_value: float,
    measured_value: float,
    high_tolerance: float,
) -> str | None:
    if metric_name in LOWER_IS_BETTER_METRICS:
        if measured_value > expected_value:
            return f"{metric_name}: measured={measured_value} > expected={expected_value}"
        lower_bound = expected_value * (2 - high_tolerance)
        if measured_value < lower_bound:
            return (
                f"{metric_name}: measured={measured_value} < lower_bound={lower_bound} "
                f"(expected={expected_value}, high_tolerance={high_tolerance})"
            )
        return None

    if measured_value < expected_value:
        return f"{metric_name}: measured={measured_value} < expected={expected_value}"
    upper_bound = expected_value * high_tolerance
    if measured_value > upper_bound:
        return (
            f"{metric_name}: measured={measured_value} > upper_bound={upper_bound} "
            f"(expected={expected_value}, high_tolerance={high_tolerance})"
        )
    return None


def _benchmark_files(benchmark_dir: Path) -> list[Path]:
    return sorted(benchmark_dir.glob("complete_run_*.json"))


def _validate_targets_schema(targets_yaml: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    targets = targets_yaml.get("targets")
    if not isinstance(targets, dict):
        return ["'targets' key is missing or is not a mapping"]

    for model_name, model_block in targets.items():
        if not isinstance(model_block, dict):
            errors.append(f"Model '{model_name}' must map to a dict")
            continue
        skus = model_block.get("skus", {})
        if not isinstance(skus, dict):
            errors.append(f"Model '{model_name}' has invalid 'skus' mapping")
            continue
        for sku_name, sku_block in skus.items():
            entries = sku_block.get("entries", []) if isinstance(sku_block, dict) else None
            if not isinstance(entries, list):
                errors.append(f"Model '{model_name}' sku '{sku_name}' must provide an entries list")
                continue
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    errors.append(f"Model '{model_name}' sku '{sku_name}' entry #{idx} must be a dict")
                    continue
                status = str(entry.get("status", "active")).lower()
                if status not in {"active", "todo"}:
                    errors.append(
                        f"Model '{model_name}' sku '{sku_name}' entry #{idx} has invalid status '{entry.get('status')}'"
                    )
    return errors


def _collect_active_test_combos(tests_yaml_path: Path) -> list[tuple[str, str]]:
    tests_yaml = _load_yaml(tests_yaml_path)
    if not isinstance(tests_yaml, list):
        raise ValueError(f"Invalid tests YAML format at {tests_yaml_path}: expected top-level list of tests")

    combos: list[tuple[str, str]] = []
    for test in tests_yaml:
        if not isinstance(test, dict):
            continue
        model = test.get("model")
        skus = test.get("skus", {})
        if not isinstance(model, str) or not isinstance(skus, dict):
            continue
        for sku_name, sku_cfg in skus.items():
            if not isinstance(sku_cfg, dict):
                continue
            tier = sku_cfg.get("tier")
            if isinstance(tier, int) and tier in {1, 2, 3}:
                combos.append((model, sku_name))
    return combos


def _validate_gap_coverage(
    tests_yaml_path: Path,
    targets_yaml_path: str,
) -> list[str]:
    errors: list[str] = []
    for model, sku in _collect_active_test_combos(tests_yaml_path):
        entry = resolve_target_entry(
            model_name=model,
            sku=sku,
            batch_size=None,
            seq_len=None,
            targets_yaml_path=targets_yaml_path,
            include_todo=True,
        )
        if entry is None:
            errors.append(
                f"Active test combo model={model}, sku={sku} is missing in centralized targets "
                "(must be active entry or explicit TODO)"
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark artifacts against centralized perf/accuracy targets.")
    parser.add_argument("--targets-yaml", default="models/model_targets.yaml")
    parser.add_argument("--benchmark-dir", default="generated/benchmark_data")
    parser.add_argument("--tests-yaml", default="tests/pipeline_reorg/models_e2e_tests.yaml")
    parser.add_argument("--sku", default=None, help="Override SKU for this job (recommended in CI matrix jobs)")
    parser.add_argument("--strict-missing", action="store_true", help="Fail when matching target is TODO or missing")
    parser.add_argument("--high-tol-percentage", type=float, default=1.15)
    return parser.parse_args()


def _sanitize_cli_path(
    raw_path: str,
    *,
    arg_name: str,
    expect_dir: bool = False,
    allowed_suffixes: tuple[str, ...] = (),
) -> Path:
    if "\x00" in raw_path:
        raise ValueError(f"Invalid {arg_name}: path contains null byte")

    candidate = Path(raw_path).expanduser()
    try:
        sanitized = candidate.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"Invalid {arg_name}: failed to resolve path '{raw_path}' ({exc})") from exc

    if expect_dir:
        if sanitized.exists() and not sanitized.is_dir():
            raise ValueError(f"Invalid {arg_name}: expected directory path, got file '{sanitized}'")
    elif sanitized.exists() and not sanitized.is_file():
        raise ValueError(f"Invalid {arg_name}: expected file path, got directory '{sanitized}'")

    if allowed_suffixes and sanitized.suffix.lower() not in allowed_suffixes:
        suffixes = ", ".join(allowed_suffixes)
        raise ValueError(f"Invalid {arg_name}: expected one of [{suffixes}], got '{sanitized.suffix}'")

    return sanitized


def main() -> int:
    args = parse_args()
    try:
        targets_yaml_path = _sanitize_cli_path(
            args.targets_yaml,
            arg_name="--targets-yaml",
            allowed_suffixes=(".yaml", ".yml"),
        )
        benchmark_dir = _sanitize_cli_path(
            args.benchmark_dir,
            arg_name="--benchmark-dir",
            expect_dir=True,
        )
        tests_yaml_path = _sanitize_cli_path(
            args.tests_yaml,
            arg_name="--tests-yaml",
            allowed_suffixes=(".yaml", ".yml"),
        )
    except ValueError as exc:
        print(f"::error::{exc}")
        return 1

    targets_yaml = _load_yaml(targets_yaml_path)
    if not isinstance(targets_yaml, dict):
        print(f"::error::Invalid YAML document at {targets_yaml_path}: expected top-level mapping")
        return 1
    schema_errors = _validate_targets_schema(targets_yaml)
    if schema_errors:
        for error in schema_errors:
            print(f"::error::{error}")
        return 1

    gap_errors = _validate_gap_coverage(tests_yaml_path, str(targets_yaml_path))
    if gap_errors:
        for error in gap_errors:
            print(f"::error::{error}")
        return 1

    benchmark_files = _benchmark_files(benchmark_dir)
    if not benchmark_files:
        print(f"::warning::No benchmark JSON files found under {benchmark_dir}")
        return 0

    hard_failures: list[str] = []
    missing_entries: list[str] = []

    for benchmark_file in benchmark_files:
        run = _load_json(benchmark_file)
        model_name = run.get("ml_model_name")
        if not isinstance(model_name, str):
            hard_failures.append(f"{benchmark_file.name}: missing ml_model_name")
            continue

        sku = args.sku
        if sku is None:
            device_info = run.get("device_info", {})
            if isinstance(device_info, dict):
                sku = device_info.get("card_type")
        if not isinstance(sku, str) or not sku.strip():
            hard_failures.append(f"{benchmark_file.name}: missing sku (pass --sku in workflow)")
            continue

        batch_size = run.get("batch_size")
        batch_size = int(batch_size) if _is_number(batch_size) else None
        seq_len = run.get("input_sequence_length")
        seq_len = int(seq_len) if _is_number(seq_len) else None

        entry = resolve_target_entry(
            model_name=model_name,
            sku=sku,
            batch_size=batch_size,
            seq_len=seq_len,
            targets_yaml_path=str(targets_yaml_path),
            include_todo=True,
        )
        if entry is None:
            missing_entries.append(
                f"{benchmark_file.name}: no target entry for model={model_name}, sku={sku}, batch_size={batch_size}, seq_len={seq_len}"
            )
            continue

        if str(entry.get("status", "active")).lower() == "todo":
            missing_entries.append(
                f"{benchmark_file.name}: target entry is TODO for model={model_name}, sku={sku}, batch_size={batch_size}, seq_len={seq_len}"
            )
            continue

        measured = _measurement_lookup(run)
        thresholds: dict[str, Any] = {}
        perf = entry.get("perf", {})
        accuracy = entry.get("accuracy", {})
        if isinstance(perf, dict):
            thresholds.update(perf)
        if isinstance(accuracy, dict):
            thresholds.update(accuracy)

        for metric_name, expected in thresholds.items():
            if metric_name.endswith("_tolerance") or metric_name == "tolerance":
                continue
            if not _is_number(expected):
                continue
            measured_value = _extract_metric_value(metric_name, measured)
            if measured_value is None or math.isnan(measured_value):
                hard_failures.append(
                    f"{benchmark_file.name}: metric '{metric_name}' missing in benchmark payload for "
                    f"model={model_name}, sku={sku}"
                )
                continue
            tolerance = _metric_tolerance(metric_name, thresholds, args.high_tol_percentage)
            metric_failure = _check_metric(
                metric_name=metric_name,
                expected_value=float(expected),
                measured_value=float(measured_value),
                high_tolerance=tolerance,
            )
            if metric_failure:
                hard_failures.append(f"{benchmark_file.name}: {metric_failure}")

    for failure in hard_failures:
        print(f"::error::{failure}")

    for missing in missing_entries:
        level = "::error::" if args.strict_missing else "::warning::"
        print(f"{level}{missing}")

    if hard_failures:
        return 1
    if args.strict_missing and missing_entries:
        return 1
    print(
        f"Validation completed: {len(benchmark_files)} benchmark file(s), "
        f"{len(hard_failures)} hard failures, {len(missing_entries)} missing/TODO entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
