# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a pre-run catalog of every ``perf_*.py`` report schema.

The script imports perf modules, expands their pytest parametrization, and calls
each Python test body with ``PerfConfig.run`` replaced by a collector. It does
not build kernels, start a simulator, or access a device.

Run from ``tests/python_tests``:

    python generate_perf_catalog.py --arch quasar
    python generate_perf_catalog.py --arch wormhole --output perf_catalog.wh.json
    python generate_perf_catalog.py --arch quasar --strict

The catalog contains the deterministic CSV sweep columns. Runtime-derived
profiler and hardware-counter columns are intentionally not predicted.
"""

from __future__ import annotations

import argparse
import datetime
import enum
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent
_SELF = Path(__file__).resolve()


def _canonical(value: Any) -> str:
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_canonical(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_canonical(v) for v in value) + "]"
    if hasattr(value, "input_format") and hasattr(value, "output_format"):
        input_b = getattr(value, "input_format_B", None)
        suffix = f", B={input_b.name}" if input_b is not None else ""
        return (
            f"{type(value).__name__}("
            f"{value.input_format.name}->{value.output_format.name}{suffix})"
        )
    return repr(value)


def _import_module(path: Path, arch: str) -> ModuleType:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.environ.setdefault("LLK_HOME", str(_ROOT.parent.parent))
    os.environ["CHIP_ARCH"] = arch
    dotted = ".".join(path.relative_to(_ROOT).with_suffix("").parts)
    return __import__(dotted, fromlist=["*"])


def _parameter_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, str):
        return [name.strip() for name in raw_names.split(",")]
    return [str(name) for name in raw_names]


def _parameter_values(raw_value: Any, count: int) -> tuple[Any, ...]:
    # pytest.param(...) values are stored in a ParameterSet.
    if hasattr(raw_value, "values"):
        raw_value = raw_value.values
    if count == 1:
        if isinstance(raw_value, tuple) and len(raw_value) == 1:
            return raw_value
        return (raw_value,)
    return tuple(raw_value)


def _parameter_rows(function) -> list[dict[str, Any]]:
    marks = [
        mark
        for mark in getattr(function, "pytestmark", [])
        if getattr(mark, "name", None) == "parametrize"
    ]
    rows: list[dict[str, Any]] = [{}]
    for mark in marks:
        names = _parameter_names(mark.args[0])
        mark_rows = [
            dict(zip(names, _parameter_values(value, len(names))))
            for value in mark.args[1]
        ]
        rows = [{**left, **right} for left in rows for right in mark_rows]
    return rows


def _test_functions(module: ModuleType) -> list:
    return [
        obj
        for name, obj in vars(module).items()
        if name.startswith("test")
        and callable(obj)
        and getattr(obj, "__module__", None) == module.__name__
    ]


def _marks(obj) -> list:
    marks = getattr(obj, "pytestmark", [])
    return marks if isinstance(marks, list) else [marks]


def _skip_reason(module: ModuleType, function) -> str | None:
    for mark in [*_marks(module), *_marks(function)]:
        name = getattr(mark, "name", None)
        if name == "skip":
            return mark.kwargs.get("reason", "pytest.mark.skip")
        if name != "skipif" or not mark.args:
            continue
        condition = mark.args[0]
        # String skip conditions require pytest's configured namespace. Catalog
        # collection cannot evaluate them reliably, so only resolve concrete
        # booleans (which is what the architecture skip helpers use).
        if not isinstance(condition, str) and bool(condition):
            return mark.kwargs.get("reason", "pytest.mark.skipif")
    return None


def _discover_modules() -> list[Path]:
    return [
        path
        for path in sorted(_ROOT.rglob("perf_*.py"))
        if path.resolve() != _SELF
        and path.name != "perf_hang_skips.py"
        and "__pycache__" not in path.parts
    ]


class CatalogCollector:
    def __init__(self):
        self.current_case: dict[str, Any] | None = None
        self.captures: list[dict[str, Any]] = []

    def capture(self, config, run_count=1):
        if self.current_case is None:
            raise RuntimeError("PerfConfig.run called outside a catalog case")

        sweep_header = config.get_csv_report_sweep_header()
        run_types = [run_type.name for _, _, run_type in config.run_configs]
        timing_columns = _timing_columns(run_types, include_std=run_count > 1)
        text_size_columns = [
            name for name in sweep_header if name.startswith("TEXT_SIZE(")
        ]
        config_columns = [
            name for name in sweep_header if not name.startswith("TEXT_SIZE(")
        ]
        expected_header = (
            config_columns + ["marker"] + timing_columns + text_size_columns
        )
        duplicates = sorted(
            name for name, count in Counter(expected_header).items() if count > 1
        )
        self.captures.append(
            {
                **self.current_case,
                "source": config.test_name,
                "run_count": run_count,
                "run_types": run_types,
                "sweep_header": sweep_header,
                "timing_columns": timing_columns,
                "expected_header": expected_header,
                "duplicate_columns": duplicates,
            }
        )

        from helpers.test_config import TestOutcome

        return TestOutcome()


def _timing_columns(run_types: list[str], include_std: bool) -> list[str]:
    """Return profiler columns implied by run types and repetition count."""
    timings = []
    for run_type in run_types:
        names = (
            [f"{run_type}[UNPACK]", f"{run_type}[PACK]"]
            if run_type == "L1_CONGESTION"
            else [run_type]
        )
        for name in names:
            timings.append(f"mean({name})")
            if include_std:
                timings.append(f"std({name})")
    return timings


def _schema_summary(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # CSV column order is part of the generated output, so ordered headers are
    # distinct catalog schemas even though PerfReport's contamination guard only
    # compares column membership.
    schemas: dict[tuple[str, ...], dict[str, Any]] = {}
    for capture in captures:
        key = tuple(capture["expected_header"])
        schema = schemas.setdefault(
            key,
            {
                "sweep_header": capture["sweep_header"],
                "timing_columns": capture["timing_columns"],
                "expected_header": capture["expected_header"],
                "variant_count": 0,
                "functions": set(),
                "sources": set(),
                "run_types": capture["run_types"],
                "run_counts": set(),
                "example_case": capture["case"],
                "duplicate_columns": set(),
            },
        )
        schema["variant_count"] += 1
        schema["functions"].add(capture["function"])
        schema["sources"].add(capture["source"])
        schema["run_counts"].add(capture["run_count"])
        schema["duplicate_columns"].update(capture["duplicate_columns"])

    result = []
    for schema in schemas.values():
        result.append(
            {
                **schema,
                "functions": sorted(schema["functions"]),
                "sources": sorted(schema["sources"]),
                "run_counts": sorted(schema["run_counts"]),
                "duplicate_columns": sorted(schema["duplicate_columns"]),
            }
        )
    return result


def generate_catalog(arch: str) -> dict[str, Any]:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.environ.setdefault("LLK_HOME", str(_ROOT.parent.parent))
    os.environ["CHIP_ARCH"] = arch

    from helpers.perf import PerfConfig
    from helpers.test_config import TestConfig

    # pytest_configure normally initializes these architecture-dependent fields.
    # setup_arch() only resolves constants from CHIP_ARCH; it does not touch a device.
    TestConfig.setup_arch()

    collector = CatalogCollector()
    original_run = PerfConfig.run
    original_infra_testing = TestConfig.INFRA_TESTING
    modules = []
    all_warnings = []
    total_cases = 0

    def catalog_run(config, _perf_report, run_count=1):
        return collector.capture(config, run_count)

    TestConfig.INFRA_TESTING = True
    PerfConfig.run = catalog_run
    try:
        for path in _discover_modules():
            relative_path = str(path.relative_to(_ROOT))
            module_result = {
                "path": relative_path,
                "functions": [],
                "schemas": [],
                "warnings": [],
                "errors": [],
            }
            modules.append(module_result)

            try:
                module = _import_module(path, arch)
            except BaseException as exc:
                message = f"import failed: {type(exc).__name__}: {exc}"
                module_result["errors"].append(message)
                all_warnings.append(f"{relative_path}: {message}")
                continue

            module_capture_start = len(collector.captures)
            for function in _test_functions(module):
                signature = inspect.signature(function)
                rows = _parameter_rows(function)
                function_result = {
                    "name": function.__name__,
                    "collected_cases": len(rows),
                    "captured_configs": 0,
                    "skipped_cases": 0,
                    "skip_reason": None,
                    "runtime_skip_reasons": [],
                    "errors": [],
                }
                module_result["functions"].append(function_result)
                total_cases += len(rows)

                if reason := _skip_reason(module, function):
                    function_result["skipped_cases"] = len(rows)
                    function_result["skip_reason"] = reason
                    continue

                if "perf_report" not in signature.parameters:
                    warning = (
                        f"{function.__name__}: unsupported perf path "
                        "(does not use the PerfConfig/perf_report interface)"
                    )
                    module_result["warnings"].append(warning)
                    all_warnings.append(f"{relative_path}: {warning}")
                    continue

                capture_start = len(collector.captures)
                error_counts: Counter[str] = Counter()
                error_examples: dict[str, list[int]] = {}
                runtime_skip_reasons: Counter[str] = Counter()
                for index, params in enumerate(rows):
                    kwargs = dict(params)
                    kwargs["perf_report"] = object()
                    missing = [
                        name
                        for name, parameter in signature.parameters.items()
                        if name not in kwargs
                        and parameter.default is inspect.Parameter.empty
                    ]
                    if missing:
                        message = f"unresolved fixture(s): {missing}"
                        error_counts[message] += 1
                        examples = error_examples.setdefault(message, [])
                        if len(examples) < 3:
                            examples.append(index)
                        continue

                    case = {name: _canonical(value) for name, value in params.items()}
                    collector.current_case = {
                        "module": relative_path,
                        "function": function.__name__,
                        "case_index": index,
                        "case": case,
                    }
                    try:
                        function(**kwargs)
                    except pytest.skip.Exception as exc:
                        function_result["skipped_cases"] += 1
                        runtime_skip_reasons[str(exc)] += 1
                    except BaseException as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        error_counts[message] += 1
                        examples = error_examples.setdefault(message, [])
                        if len(examples) < 3:
                            examples.append(index)
                    finally:
                        collector.current_case = None

                function_result["captured_configs"] = (
                    len(collector.captures) - capture_start
                )
                function_result["runtime_skip_reasons"] = [
                    {"reason": reason, "case_count": count}
                    for reason, count in runtime_skip_reasons.most_common()
                ]
                function_result["errors"] = [
                    {
                        "message": message,
                        "case_count": count,
                        "example_case_indices": error_examples[message],
                    }
                    for message, count in error_counts.most_common()
                ]
                error_case_count = sum(error_counts.values())
                if error_case_count:
                    all_warnings.append(
                        f"{relative_path}:{function.__name__}: "
                        f"{error_case_count} case error(s)"
                    )

            module_captures = collector.captures[module_capture_start:]
            module_result["schemas"] = _schema_summary(module_captures)
            if len(module_result["schemas"]) > 1:
                warning = (
                    f"{len(module_result['schemas'])} distinct expected CSV schemas "
                    "would share this module's CSV"
                )
                module_result["warnings"].append(warning)
                all_warnings.append(f"{relative_path}: {warning}")
            for schema in module_result["schemas"]:
                if schema["duplicate_columns"]:
                    warning = "duplicate sweep columns: " + ", ".join(
                        schema["duplicate_columns"]
                    )
                    module_result["warnings"].append(warning)
                    all_warnings.append(f"{relative_path}: {warning}")
    finally:
        PerfConfig.run = original_run
        TestConfig.INFRA_TESTING = original_infra_testing

    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "architecture": arch,
        "scope": "**/perf_*.py",
        "header_limitations": (
            "Expected headers include configuration, marker, profiler timing, and "
            "TEXT_SIZE columns. Hardware-counter columns remain runtime-derived; "
            "std timing columns are expected when run_count > 1."
        ),
        "summary": {
            "module_count": len(modules),
            "function_count": sum(len(module["functions"]) for module in modules),
            "collected_case_count": total_cases,
            "captured_config_count": len(collector.captures),
            "schema_count": sum(len(module["schemas"]) for module in modules),
            "warning_count": len(all_warnings),
        },
        "warnings": all_warnings,
        "modules": modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        default=os.environ.get("CHIP_ARCH", "quasar"),
        help="architecture used while resolving test sweeps (default: CHIP_ARCH or quasar)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("perf_catalog.json"),
        help="JSON catalog path (default: perf_catalog.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero when catalog warnings or collection errors are found",
    )
    args = parser.parse_args()

    catalog = generate_catalog(args.arch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2) + "\n")

    summary = catalog["summary"]
    print(
        f"Wrote {args.output}: {summary['module_count']} modules, "
        f"{summary['collected_case_count']} cases, "
        f"{summary['schema_count']} schemas, "
        f"{summary['warning_count']} warnings"
    )
    for warning in catalog["warnings"]:
        print(f"WARNING: {warning}")

    return 1 if args.strict and catalog["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
