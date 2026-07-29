#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

try:
    from models.demos.utils import model_targets
except ModuleNotFoundError:
    # Allow direct invocation without PYTHONPATH set by adding repo root.
    sys.path.insert(0, str(REPO_ROOT))
    from models.demos.utils import model_targets

LOWER_IS_BETTER_METRICS = {
    "prefill_time_to_first_token",
    "prefill_time_to_token",
    "compile_prefill",
    "compile_decode",
}

TARGETS_YAML_RELATIVE_PATH = Path("models/model_targets.yaml")
BENCHMARK_DIR_RELATIVE_PATH = Path("generated/benchmark_data")
TESTS_YAML_RELATIVE_PATH = Path("tests/pipeline_reorg/models_e2e_tests.yaml")


class PathProfile(str, Enum):
    REPO_ROOT = "repo-root"
    CURRENT_WORKING_DIRECTORY = "cwd"


METRIC_NAME_MAP = {
    "compile_prefill": ("compile_prefill", "time(s)"),
    "compile_decode": ("compile_decode", "time(s)"),
    "prefill_t/s": ("inference_prefill", "tokens/s"),
    "prefill_time_to_token": ("inference_prefill", "time_to_token"),
    "prefill_time_to_first_token": ("inference_prefill", "time_to_token"),
    "prefill_decode_t/s/u": ("inference_prefill_decode", "tokens/s/user"),
    "decode_t/s": ("inference_decode", "tokens/s"),
    "decode_t/s/u": ("inference_decode", "tokens/s/user"),
    "top1": ("inference_decode", "top1_token_accuracy"),
    "top5": ("inference_decode", "top5_token_accuracy"),
}

ALLOWED_TARGET_METRIC_NAMES = {
    "compile_decode",
    "compile_prefill",
    "decode_t/s",
    "decode_t/s/u",
    "prefill_decode_t/s/u",
    "prefill_t/s",
    "prefill_time_to_token",
    "prefill_time_to_first_token",
    "top1",
    "top5",
}

PREFILL_TIME_TO_TOKEN_KEY = "prefill_time_to_token"
PREFILL_TIME_TO_FIRST_TOKEN_KEY = "prefill_time_to_first_token"

# Accuracy target metric names, and the measurement names a token-matching run emits.
# A benchmark run is treated as an accuracy (token-matching) run iff it reports these
# measurements; otherwise it is a perf (eval) run. This lets us validate only the
# relevant metric family per run: perf numbers from token-matching runs are teacher-
# forcing artifacts (not real perf), and eval runs do not measure token accuracy.
ACCURACY_TARGET_METRIC_NAMES = {"top1", "top5"}
ACCURACY_MEASUREMENT_NAMES = {"top1_token_accuracy", "top5_token_accuracy"}

# Reverse lookup from a benchmark (step_name, measurement_name) pair back to a canonical
# target metric name. Used to report measured values that have no matching target entry so
# the summary can still surface them. The first key wins for pairs shared by multiple metric
# names (e.g. prefill_time_to_token / prefill_time_to_first_token both map to time_to_token).
_MEASUREMENT_TO_METRIC_NAME: dict[tuple[str, str], str] = {}
for _metric_name, _pair in METRIC_NAME_MAP.items():
    _MEASUREMENT_TO_METRIC_NAME.setdefault(_pair, _metric_name)


def _is_number(value: Any) -> bool:
    """Return True for int/float (but not bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_yaml(path: Path) -> Any:
    """Load YAML and normalize empty content to an empty mapping."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def _load_json(path: Path) -> dict[str, Any]:
    """Load and type-check benchmark payload JSON."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid benchmark JSON payload in {path}")
    return payload


def _measurement_lookup(benchmark_json: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Build (step_name, metric_name) -> value lookup from benchmark payload."""
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


def _is_accuracy_run(measured_lookup: dict[tuple[str, str], float]) -> bool:
    """True for token-matching (accuracy) runs, identified by top1/top5 measurements.

    The benchmark JSON carries no explicit test-type field (run_type is always
    "demo"), so we key off the measurements: only token-matching runs emit
    top1/top5 token accuracy.
    """
    return any(name in ACCURACY_MEASUREMENT_NAMES for _step, name in measured_lookup)


def _extract_metric_value(metric_name: str, lookup: dict[tuple[str, str], float]) -> float | None:
    """Resolve a metric value, failing on ambiguous unqualified metric names."""
    if metric_name in METRIC_NAME_MAP:
        return lookup.get(METRIC_NAME_MAP[metric_name])

    matches: list[tuple[str, str, float]] = []
    for (step_name, name), value in lookup.items():
        if name == metric_name:
            matches.append((step_name, name, value))
        if f"{step_name}.{name}" == metric_name:
            return value

    if not matches:
        return None
    if len(matches) > 1:
        candidates = ", ".join(sorted(f"{step}.{name}" for step, name, _ in matches))
        raise ValueError(
            f"Metric '{metric_name}' is ambiguous across multiple steps: {candidates}. "
            "Use an explicit step-qualified metric name."
        )
    return matches[0][2]


def _normalize_ttft_thresholds(
    thresholds: dict[str, Any],
    benchmark_file_name: str,
    model_name: str,
    sku: str,
) -> dict[str, Any]:
    """
    Normalize TTFT aliases in thresholds and enforce `prefill_time_to_first_token` precedence.

    `prefill_time_to_first_token` targets are stored in milliseconds and converted to seconds
    for comparison with benchmark payload `time_to_token`.
    """
    normalized_thresholds = dict(thresholds)
    has_ttft_ms = _is_number(normalized_thresholds.get(PREFILL_TIME_TO_FIRST_TOKEN_KEY))
    has_ttft_s = _is_number(normalized_thresholds.get(PREFILL_TIME_TO_TOKEN_KEY))

    if has_ttft_ms and has_ttft_s:
        print(
            "::warning::"
            f"{benchmark_file_name}: both {PREFILL_TIME_TO_FIRST_TOKEN_KEY} and {PREFILL_TIME_TO_TOKEN_KEY} are set "
            f"for model={model_name}, sku={sku}; using {PREFILL_TIME_TO_FIRST_TOKEN_KEY} and ignoring "
            f"{PREFILL_TIME_TO_TOKEN_KEY}"
        )
        normalized_thresholds.pop(PREFILL_TIME_TO_TOKEN_KEY, None)

    if has_ttft_ms:
        normalized_thresholds[PREFILL_TIME_TO_FIRST_TOKEN_KEY] = (
            float(normalized_thresholds[PREFILL_TIME_TO_FIRST_TOKEN_KEY]) / 1000.0
        )

    return normalized_thresholds


def _check_metric(
    metric_name: str,
    expected_value: float,
    measured_value: float,
    tolerance: float,
) -> str | None:
    """Compare measured vs expected using a symmetric +/- tolerance band.

    A measurement passes when it lies within +/- ``tolerance`` of ``expected_value``,
    independent of whether the metric is higher- or lower-is-better (e.g. an
    expected 100 ms TTFT with tolerance 0.1 accepts any value in [90, 110] ms).
    Outside the band it fails: on the "worse" side that is a regression, on the
    "better" side it signals the target is stale and should be refreshed. The
    lower/higher-is-better classification only affects the wording.
    """
    lower_bound = expected_value * (1 - tolerance)
    upper_bound = expected_value * (1 + tolerance)
    if lower_bound <= measured_value <= upper_bound:
        return None

    lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
    if measured_value > upper_bound:
        note = "regression" if lower_is_better else "much better than expected, update target"
        return (
            f"{metric_name}: measured={measured_value} > upper_bound={upper_bound} "
            f"(expected={expected_value}, tolerance={tolerance}) [{note}]"
        )
    note = "much better than expected, update target" if lower_is_better else "regression"
    return (
        f"{metric_name}: measured={measured_value} < lower_bound={lower_bound} "
        f"(expected={expected_value}, tolerance={tolerance}) [{note}]"
    )


def _display_values(
    metric_name: str,
    measured: float | None,
    expected: float | None,
) -> tuple[float | None, float | None, str]:
    """Return (display_measured, display_expected, unit) for the human summary.

    `prefill_time_to_first_token` targets are normalized to seconds for comparison, so both
    measured and expected are converted back to milliseconds here to match how the target is
    authored and read. Every other metric is reported in its raw benchmark unit.
    """
    if metric_name == PREFILL_TIME_TO_FIRST_TOKEN_KEY:
        disp_measured = round(measured * 1000.0, 4) if measured is not None else None
        disp_expected = round(expected * 1000.0, 4) if expected is not None else None
        return disp_measured, disp_expected, "ms"

    unit = METRIC_NAME_MAP.get(metric_name, (None, ""))[1]
    disp_measured = round(measured, 4) if measured is not None else None
    disp_expected = round(expected, 4) if expected is not None else None
    return disp_measured, disp_expected, unit


def _make_report_record(
    *,
    model_name: str,
    sku: str,
    batch_size: int | None,
    seq_len: int | None,
    metric_name: str,
    measured: float | None,
    expected: float | None,
    tolerance: float | None,
    status: str,
) -> dict[str, Any]:
    """Build one row of the perf/accuracy report."""
    disp_measured, disp_expected, unit = _display_values(metric_name, measured, expected)
    return {
        "model": model_name,
        "sku": sku,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "metric_name": metric_name,
        "measured": disp_measured,
        "expected": disp_expected,
        "tolerance": tolerance,
        "unit": unit,
        "status": status,
    }


def _benchmark_files(benchmark_dir: Path) -> list[Path]:
    """Return benchmark payload files in deterministic order."""
    return sorted(benchmark_dir.glob("complete_run_*.json"))


def _validate_targets_schema(targets_yaml: dict[str, Any]) -> list[str]:
    """Validate centralized target schema and semantic consistency."""
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
            seen_entry_dims: set[tuple[Any, Any]] = set()
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    errors.append(f"Model '{model_name}' sku '{sku_name}' entry #{idx} must be a dict")
                    continue
                status = str(entry.get("status", "active")).lower()
                if status not in {"active", "todo"}:
                    errors.append(
                        f"Model '{model_name}' sku '{sku_name}' entry #{idx} has invalid status '{entry.get('status')}'"
                    )
                dims = (entry.get("batch_size"), entry.get("seq_len"))
                if dims in seen_entry_dims:
                    errors.append(
                        f"Model '{model_name}' sku '{sku_name}' has duplicate entry for batch_size={dims[0]}, seq_len={dims[1]}"
                    )
                else:
                    seen_entry_dims.add(dims)

                for block_name in ("perf", "accuracy"):
                    block = entry.get(block_name, {})
                    if not isinstance(block, dict):
                        errors.append(
                            f"Model '{model_name}' sku '{sku_name}' entry #{idx} has non-dict '{block_name}' block"
                        )
                        continue
                    for metric_name, metric_value in block.items():
                        is_tolerance_key = model_targets.is_tolerance_key(metric_name)
                        if is_tolerance_key:
                            if not _is_number(metric_value):
                                errors.append(
                                    f"Model '{model_name}' sku '{sku_name}' entry #{idx} has non-numeric tolerance '{metric_name}'"
                                )
                            elif not (0.0 <= float(metric_value) <= 1.0):
                                errors.append(
                                    f"Model '{model_name}' sku '{sku_name}' entry #{idx} has tolerance '{metric_name}' outside [0.0, 1.0]"
                                )
                            continue
                        if metric_name not in ALLOWED_TARGET_METRIC_NAMES:
                            errors.append(
                                f"Model '{model_name}' sku '{sku_name}' entry #{idx} has unknown metric '{metric_name}'"
                            )
                            continue
                        if not _is_number(metric_value):
                            errors.append(
                                f"Model '{model_name}' sku '{sku_name}' entry #{idx} has non-numeric metric '{metric_name}'"
                            )
    return errors


def _collect_active_test_combos(tests_yaml_path: Path) -> list[tuple[str, str]]:
    """Collect all active model/SKU combos in tier-1/2/3 CI tests."""
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


def _normalize_token(value: Any) -> str:
    """Normalize string-like values for case-insensitive matching."""
    return str(value).strip().lower()


def _has_model_sku_coverage(targets_yaml: dict[str, Any], model_name: str, sku: str) -> bool:
    """Return True when centralized targets include at least one entry for model/SKU."""
    targets = targets_yaml.get("targets", {})
    if not isinstance(targets, dict):
        return False

    model_norm = _normalize_token(model_name)
    sku_norm = model_targets.normalize_sku(sku)

    for model_key, model_block in targets.items():
        if not isinstance(model_block, dict):
            continue
        aliases = model_block.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        model_matches = _normalize_token(model_key) == model_norm or any(
            _normalize_token(alias) == model_norm for alias in aliases
        )
        if not model_matches:
            continue

        skus = model_block.get("skus", {})
        if not isinstance(skus, dict):
            continue
        for sku_key, sku_block in skus.items():
            if model_targets.normalize_sku(sku_key) != sku_norm or not isinstance(sku_block, dict):
                continue
            entries = sku_block.get("entries", [])
            if not isinstance(entries, list):
                return False
            return any(isinstance(entry, dict) for entry in entries)
    return False


def _validate_gap_coverage(
    tests_yaml_path: Path,
    targets_yaml: dict[str, Any],
) -> list[str]:
    """Report active e2e model/SKU combos missing from the centralized targets.

    Returns a list of human-readable messages — these are surfaced as
    warnings (not errors) so a missing target does not fail CI. The
    intent is to nudge owners to add the entry, not to block landing.
    Only `models_e2e_tests.yaml` is walked here by design.
    """
    warnings: list[str] = []
    for model, sku in _collect_active_test_combos(tests_yaml_path):
        if not _has_model_sku_coverage(targets_yaml, model_name=model, sku=sku):
            warnings.append(
                f"Active test combo model={model}, sku={sku} is missing in centralized targets "
                "(add an active entry or explicit TODO in models/model_targets.yaml)"
            )
    return warnings


def parse_args() -> argparse.Namespace:
    """Parse and validate validator CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Validate benchmark artifacts against centralized perf/accuracy targets."
    )
    parser.add_argument(
        "--path-profile",
        choices=[profile.value for profile in PathProfile],
        default=PathProfile.REPO_ROOT.value,
        help="Path root profile used to resolve benchmark and target files",
    )
    parser.add_argument("--sku", default=None, help="Override SKU for this job (recommended in CI matrix jobs)")
    # TODO: Enable strict-missing by default in CI once model targets migration is complete.
    parser.add_argument("--strict-missing", action="store_true", help="Fail when matching target is TODO or missing")
    return parser.parse_args()


def _resolve_paths(path_profile: PathProfile) -> tuple[Path, Path, Path]:
    """Resolve target/test/benchmark paths from selected path profile."""
    base_path = REPO_ROOT if path_profile is PathProfile.REPO_ROOT else Path.cwd().resolve()
    targets_yaml_path = base_path / TARGETS_YAML_RELATIVE_PATH
    benchmark_dir = base_path / BENCHMARK_DIR_RELATIVE_PATH
    tests_yaml_path = base_path / TESTS_YAML_RELATIVE_PATH
    return targets_yaml_path, benchmark_dir, tests_yaml_path


class ValidationResult:
    """Structured findings from a validation run.

    Holds raw findings only — no printing and no exit-code policy. Callers
    (CI ``main()`` and the local pytest hook) format output and decide how to
    treat ``missing_entries`` (strict vs warning-only). ``schema_errors`` is
    fatal: when present, no benchmark comparison is performed.

    Implemented as a plain class (not a dataclass) on purpose: this module uses
    ``from __future__ import annotations`` and is loaded via
    ``importlib.util.module_from_spec`` (by the CI tests and the local pytest
    hook) without being registered in ``sys.modules``, which breaks dataclass
    field-type resolution.
    """

    def __init__(self) -> None:
        self.schema_errors: list[str] = []
        self.gap_warnings: list[str] = []
        self.hard_failures: list[str] = []
        self.missing_entries: list[str] = []
        self.reported_metrics: list[dict[str, Any]] = []
        self.num_benchmark_files: int = 0


def validate(
    targets_yaml_path: Path,
    benchmark_dir: Path,
    tests_yaml_path: Path | None = None,
    sku_override: str | None = None,
) -> ValidationResult:
    """Validate benchmark artifacts against centralized perf/accuracy targets.

    Reads files but performs no stdout side effects and makes no exit-code
    decision — it returns a :class:`ValidationResult` for the caller to act on.
    This is the shared core used by both the CI entrypoint (``main()``) and the
    local opt-in pytest hook so the two can never diverge.

    ``tests_yaml_path`` is only used for the active-combo gap report; pass
    ``None`` (e.g. for local runs) to skip it.
    """
    result = ValidationResult()

    # Keep resolver path controlled from this module to avoid passing dynamic file paths
    # into model_targets APIs (Cycode SAST: unsanitized dynamic input in file path).
    model_targets.TARGETS_YAML_PATH_DEFAULT = str(targets_yaml_path)

    targets_yaml = _load_yaml(targets_yaml_path)
    if not isinstance(targets_yaml, dict):
        result.schema_errors.append(f"Invalid YAML document at {targets_yaml_path}: expected top-level mapping")
        return result

    schema_errors = _validate_targets_schema(targets_yaml)
    if schema_errors:
        result.schema_errors = schema_errors
        return result

    if tests_yaml_path is not None and tests_yaml_path.exists():
        result.gap_warnings = _validate_gap_coverage(tests_yaml_path, targets_yaml)

    benchmark_files = _benchmark_files(benchmark_dir)
    result.num_benchmark_files = len(benchmark_files)

    for benchmark_file in benchmark_files:
        run = _load_json(benchmark_file)
        model_name = run.get("ml_model_name")
        if not isinstance(model_name, str):
            result.hard_failures.append(f"{benchmark_file.name}: missing ml_model_name")
            continue

        sku = sku_override
        if sku is None:
            device_info = run.get("device_info", {})
            if isinstance(device_info, dict):
                sku = device_info.get("card_type")
        if not isinstance(sku, str) or not sku.strip():
            result.hard_failures.append(f"{benchmark_file.name}: missing sku (pass --sku in workflow)")
            continue

        batch_size = run.get("batch_size")
        batch_size = int(batch_size) if _is_number(batch_size) else None
        seq_len = run.get("input_sequence_length")
        seq_len = int(seq_len) if _is_number(seq_len) else None

        entry = model_targets.resolve_target_entry(
            model_name=model_name,
            sku=sku,
            batch_size=batch_size,
            seq_len=seq_len,
            include_todo=True,
        )
        if entry is None:
            result.missing_entries.append(
                f"{benchmark_file.name}: no target entry for model={model_name}, sku={sku}, batch_size={batch_size}, seq_len={seq_len}"
            )
            continue

        if str(entry.get("status", "active")).lower() == "todo":
            result.missing_entries.append(
                f"{benchmark_file.name}: target entry is TODO for model={model_name}, sku={sku}, batch_size={batch_size}, seq_len={seq_len}"
            )
            continue

        measured = _measurement_lookup(run)
        is_accuracy_run = _is_accuracy_run(measured)
        thresholds: dict[str, Any] = {}
        perf = entry.get("perf", {})
        accuracy = entry.get("accuracy", {})
        if isinstance(perf, dict):
            thresholds.update(perf)
        if isinstance(accuracy, dict):
            thresholds.update(accuracy)
        thresholds = _normalize_ttft_thresholds(
            thresholds=thresholds,
            benchmark_file_name=benchmark_file.name,
            model_name=model_name,
            sku=sku,
        )

        hard_failures_prefix = (
            f"{benchmark_file.name}, model={model_name}, sku={sku}, batch_size={batch_size}, seq_len={seq_len}"
        )

        # Benchmark measurement pairs already covered by a target so the no-target pass below
        # does not re-report them.
        covered_pairs: set[tuple[str, str]] = set()
        for metric_name, expected in thresholds.items():
            if model_targets.is_tolerance_key(metric_name):
                continue
            if not _is_number(expected):
                continue
            # Validate accuracy metrics only on accuracy (token-matching) runs, and
            # perf metrics only on perf (eval) runs. This prevents teacher-forcing
            # throughput/latency from a token-matching run being checked against perf
            # targets (and vice versa).
            if (metric_name in ACCURACY_TARGET_METRIC_NAMES) != is_accuracy_run:
                continue
            if metric_name in METRIC_NAME_MAP:
                covered_pairs.add(METRIC_NAME_MAP[metric_name])
            tolerance = model_targets.resolve_metric_tolerance(
                metric_name=metric_name,
                thresholds=thresholds,
                default_tolerance=model_targets.DEFAULT_PERF_TOLERANCE,
            )
            try:
                measured_value = _extract_metric_value(metric_name, measured)
            except ValueError as exc:
                result.hard_failures.append(f"{hard_failures_prefix}: ambiguous metric '{metric_name}': {exc}")
                continue
            if measured_value is None or math.isnan(measured_value):
                result.hard_failures.append(
                    f"{hard_failures_prefix}: metric '{metric_name}' missing in benchmark payload for measured={measured}"
                )
                result.reported_metrics.append(
                    _make_report_record(
                        model_name=model_name,
                        sku=sku,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        metric_name=metric_name,
                        measured=None,
                        expected=float(expected),
                        tolerance=tolerance,
                        status="missing-measurement",
                    )
                )
                continue
            metric_failure = _check_metric(
                metric_name=metric_name,
                expected_value=float(expected),
                measured_value=float(measured_value),
                tolerance=tolerance,
            )
            if metric_failure:
                result.hard_failures.append(f"{hard_failures_prefix}: {metric_failure}")
            result.reported_metrics.append(
                _make_report_record(
                    model_name=model_name,
                    sku=sku,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    metric_name=metric_name,
                    measured=float(measured_value),
                    expected=float(expected),
                    tolerance=tolerance,
                    status="fail" if metric_failure else "pass",
                )
            )

        # Always surface measured e2e numbers, even when no target exists for them, so the
        # report is a single place to read results. Only known metrics are reported (service
        # measurements are skipped), and the accuracy/perf run split is respected.
        for pair, value in measured.items():
            if pair in covered_pairs:
                continue
            metric_name = _MEASUREMENT_TO_METRIC_NAME.get(pair)
            if metric_name is None:
                continue
            if (metric_name in ACCURACY_TARGET_METRIC_NAMES) != is_accuracy_run:
                continue
            covered_pairs.add(pair)
            result.reported_metrics.append(
                _make_report_record(
                    model_name=model_name,
                    sku=sku,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    metric_name=metric_name,
                    measured=float(value),
                    expected=None,
                    tolerance=None,
                    status="no-target",
                )
            )

    return result


_SUMMARY_HEADER = "| Model | SKU | batch | seq | Metric | Measured | Target | Tolerance | Unit | Status |"
_SUMMARY_SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def _render_summary(result: ValidationResult) -> str:
    """Render measured perf/accuracy numbers (and targets, when present) as a Markdown table.

    Always includes every reported metric so results are readable from a single CI step
    instead of scanning the full run log. Deterministically ordered for stable output.
    """
    lines = ["## Report and Validate Perf and Accuracy targets", ""]
    if not result.reported_metrics:
        lines.append("_No perf/accuracy measurements reported._")
        return "\n".join(lines) + "\n"

    def _fmt(value: Any) -> str:
        return "-" if value is None else str(value)

    lines.append(_SUMMARY_HEADER)
    lines.append(_SUMMARY_SEPARATOR)
    for record in sorted(
        result.reported_metrics,
        key=lambda r: (str(r["model"]), str(r["sku"]), str(r["metric_name"])),
    ):
        tolerance = record["tolerance"]
        tolerance_disp = "-" if tolerance is None else f"±{round(float(tolerance) * 100, 2)}%"
        lines.append(
            "| {model} | {sku} | {batch} | {seq} | {metric} | {measured} | {target} | {tol} | {unit} | {status} |".format(
                model=_fmt(record["model"]),
                sku=_fmt(record["sku"]),
                batch=_fmt(record["batch_size"]),
                seq=_fmt(record["seq_len"]),
                metric=record["metric_name"],
                measured=_fmt(record["measured"]),
                target=_fmt(record["expected"]),
                tol=tolerance_disp,
                unit=_fmt(record["unit"]),
                status=record["status"],
            )
        )
    return "\n".join(lines) + "\n"


def _write_step_summary(summary: str) -> None:
    """Append the summary to the GitHub Actions step summary file when running in CI."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(summary)
    except OSError as exc:
        print(f"::warning::Failed to write GITHUB_STEP_SUMMARY: {exc}")


def main() -> int:
    """Run centralized target validation over benchmark artifacts (CI entrypoint)."""
    args = parse_args()
    path_profile = PathProfile(args.path_profile)
    targets_yaml_path, benchmark_dir, tests_yaml_path = _resolve_paths(path_profile)

    result = validate(
        targets_yaml_path=targets_yaml_path,
        benchmark_dir=benchmark_dir,
        tests_yaml_path=tests_yaml_path,
        sku_override=args.sku,
    )

    if result.schema_errors:
        for error in result.schema_errors:
            print(f"::error::{error}")
        return 1

    for warning in result.gap_warnings:
        print(f"::warning::{warning}")

    # Always report measured perf/accuracy numbers so they are visible directly in the CI
    # step output (and the GitHub step summary) without scanning the full run log.
    summary = _render_summary(result)
    print(summary)
    _write_step_summary(summary)

    if result.num_benchmark_files == 0:
        print(f"::warning::No benchmark JSON files found under {benchmark_dir}")
        return 0

    for failure in result.hard_failures:
        print(f"::error::{failure}")

    for missing in result.missing_entries:
        level = "::error::" if args.strict_missing else "::warning::"
        print(f"{level}{missing}")

    if result.hard_failures:
        return 1
    if args.strict_missing and result.missing_entries:
        return 1
    print(
        f"Validation completed: {result.num_benchmark_files} benchmark file(s), "
        f"{len(result.hard_failures)} hard failures, {len(result.missing_entries)} missing/TODO entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
