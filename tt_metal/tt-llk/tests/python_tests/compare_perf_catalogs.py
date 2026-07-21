# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare CSV headers in two catalogs produced by generate_perf_catalog.py.

Only output-affecting data is compared: active module paths and their ordered
expected CSV headers. Generation timestamps, parameter values, case counts,
skips, errors, warnings, examples, and source paths are diagnostic metadata and
do not cause a mismatch.

Examples:

    python compare_perf_catalogs.py \
        perf_catalog.wormhole.json perf_catalog.wormhole.after.json

    python compare_perf_catalogs.py \
        perf_catalog.blackhole.json perf_catalog.blackhole.after.json \
        --max-differences 500

Exit status is 0 for equal catalogs, 1 for catalog differences, and 2 for
invalid input or incompatible architectures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CatalogComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class Difference:
    path: str
    before: Any
    after: Any


def _load_catalog(path: Path) -> dict[str, Any]:
    try:
        catalog = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CatalogComparisonError(f"catalog does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogComparisonError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(catalog, dict):
        raise CatalogComparisonError(f"catalog root must be an object: {path}")

    required = {"architecture", "summary", "modules"}
    missing = sorted(required - catalog.keys())
    if missing:
        raise CatalogComparisonError(
            f"{path} is missing required catalog field(s): {', '.join(missing)}"
        )
    if not isinstance(catalog["modules"], list):
        raise CatalogComparisonError(f"{path}: 'modules' must be a list")
    return catalog


def _schema_id(schema: dict[str, Any]) -> str:
    header = schema.get("expected_header")
    if not isinstance(header, list):
        raise CatalogComparisonError("schema is missing list field 'expected_header'")
    # Column order is part of CSV output and therefore part of schema identity.
    canonical = "\0".join(header).encode()
    return hashlib.sha256(canonical).hexdigest()[:12]


def _index_unique(
    items: list[dict[str, Any]], key_name: str, context: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for item in items:
        if not isinstance(item, dict) or key_name not in item:
            raise CatalogComparisonError(
                f"{context}: every item must contain {key_name!r}"
            )
        key = str(item[key_name])
        if key in result:
            raise CatalogComparisonError(f"{context}: duplicate {key_name} {key!r}")
        result[key] = item
    return result


def _normalize(catalog: dict[str, Any]) -> dict[str, Any]:
    modules = _index_unique(catalog["modules"], "path", "modules")

    normalized_modules = {}
    for path, module in modules.items():
        schemas = {}
        for schema in module.get("schemas", []):
            schema_id = _schema_id(schema)
            if schema_id in schemas:
                raise CatalogComparisonError(
                    f"{path}.schemas: duplicate schema identity {schema_id}"
                )
            schemas[schema_id] = {"expected_header": schema["expected_header"]}
        # A fully skipped/unsupported module emits no CSV and is not part of the
        # architecture's active header contract.
        if schemas:
            normalized_modules[path] = {"schemas": schemas}

    return {
        "architecture": catalog["architecture"],
        "modules": normalized_modules,
    }


def _compare(before: Any, after: Any, path: str = "$") -> list[Difference]:
    if type(before) is not type(after):
        return [Difference(path, before, after)]

    if isinstance(before, dict):
        differences = []
        for key in sorted(before.keys() | after.keys()):
            child_path = f"{path}.{key}"
            if key not in before:
                differences.append(Difference(child_path, "<missing>", after[key]))
            elif key not in after:
                differences.append(Difference(child_path, before[key], "<missing>"))
            else:
                differences.extend(_compare(before[key], after[key], child_path))
        return differences

    if isinstance(before, list):
        if before == after:
            return []
        # Lists are deliberately atomic. In particular, sweep_header ordering is
        # part of the CSV contract and should be shown as one understandable change.
        return [Difference(path, before, after)]

    return [] if before == after else [Difference(path, before, after)]


def compare_catalogs(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[Difference]:
    baseline_arch = baseline["architecture"]
    candidate_arch = candidate["architecture"]
    if baseline_arch != candidate_arch:
        raise CatalogComparisonError(
            "cannot compare catalogs for different architectures: "
            f"{baseline_arch!r} vs {candidate_arch!r}"
        )
    return _compare(_normalize(baseline), _normalize(candidate))


def _format_value(value: Any, limit: int) -> str:
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="saved default catalog")
    parser.add_argument("candidate", type=Path, help="newly generated catalog")
    parser.add_argument(
        "--max-differences",
        type=int,
        default=200,
        help="maximum differences to print (default: 200; 0 means unlimited)",
    )
    parser.add_argument(
        "--value-width",
        type=int,
        default=500,
        help="maximum rendered width for each old/new value (default: 500)",
    )
    args = parser.parse_args()

    if args.max_differences < 0:
        parser.error("--max-differences must be non-negative")
    if args.value_width < 20:
        parser.error("--value-width must be at least 20")

    try:
        baseline = _load_catalog(args.baseline)
        candidate = _load_catalog(args.candidate)
        differences = compare_catalogs(baseline, candidate)
    except CatalogComparisonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    architecture = baseline["architecture"]
    if not differences:
        print(
            f"Catalogs match ({architecture}): " f"{args.baseline} == {args.candidate}"
        )
        return 0

    print(
        f"Catalogs differ ({architecture}): {len(differences)} difference(s)\n"
        f"  baseline : {args.baseline}\n"
        f"  candidate: {args.candidate}"
    )
    shown = (
        differences
        if args.max_differences == 0
        else differences[: args.max_differences]
    )
    for difference in shown:
        print(f"\n{difference.path}")
        print(f"  - {_format_value(difference.before, args.value_width)}")
        print(f"  + {_format_value(difference.after, args.value_width)}")

    omitted = len(differences) - len(shown)
    if omitted:
        print(
            f"\n... {omitted} more difference(s) omitted; "
            "increase --max-differences to display them."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
