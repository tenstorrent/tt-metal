#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TensorShape coverage parser.

Reads loguru test logs (test_run.log, test_run_gw*.log) and extracts DEVICE_PRINT
coverage lines of the form:

  '[<fn>] tensor_shape: face_r_dim=N face_c_dim=N num_faces_r_dim=N num_faces_c_dim=N'

Accumulates per-function shape sets, then regenerates the TRISC-specific coverage
headers under tt_metal/tt-llk/common/.

Typical workflow (from tests/python_tests):

    # Optional: bootstrap coverage.json from the checked-in headers
    python3 helpers/tensor_shape_coverage_parser.py seed

    # Discover unobserved shapes (asserts disabled so DPRINT still emits):
    TT_LLK_DISABLE_ASSERTS=1 pytest --logging-level=DEBUG <tests>

    # Harvest worker logs into coverage.json, then rewrite headers:
    python3 helpers/tensor_shape_coverage_parser.py harvest <label>
    python3 helpers/tensor_shape_coverage_parser.py emit
    python3 helpers/tensor_shape_coverage_parser.py summary

Harvest state defaults to tests/python_tests/tensor_shape_coverage/coverage.json
(override with --coverage-json). That path is gitignored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from helpers.tile_constants import FACE_C_DIM

HELPERS_DIR = Path(__file__).resolve().parent
PYTHON_TESTS_DIR = HELPERS_DIR.parent
LLK_ROOT = PYTHON_TESTS_DIR.parent.parent
COMMON_DIR = LLK_ROOT / "common"

DEFAULT_COVERAGE_JSON = PYTHON_TESTS_DIR / "tensor_shape_coverage" / "coverage.json"
DEFAULT_MATH_HEADER = COMMON_DIR / "tensor_shape_coverage_math.h"
DEFAULT_UNPACK_HEADER = COMMON_DIR / "tensor_shape_coverage_unpack.h"
DEFAULT_PACK_HEADER = COMMON_DIR / "tensor_shape_coverage_pack.h"

DPRINT_RE = re.compile(
    r"\[(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\] tensor_shape: "
    r"face_r_dim=(?P<fr>\d+) face_c_dim=(?P<fc>\d+) "
    r"num_faces_r_dim=(?P<nfr>\d+) num_faces_c_dim=(?P<nfc>\d+)"
)

SHAPE_CONST_RE = re.compile(r"TENSOR_SHAPE_FR(?P<fr>\d+)_NF(?P<nfr>\d+)x(?P<nfc>\d+)")
PREDICATE_BODY_RE = re.compile(
    r"(?:constexpr|inline) bool (?P<name>\w+)\(const TensorShape& tensor_shape\)\s*\{(?P<body>.*?)^\}",
    re.MULTILINE | re.DOTALL,
)

# (fr, nfr, nfc) -> constexpr name for shapes defined in tensor_shape_coverage.h.
# face_c_dim is always FACE_C_DIM by HW. Keep in sync with the named constants there.
SHAPE_NAMES = {
    (1, 1, 1): "TENSOR_SHAPE_FR1_NF1x1",
    (1, 1, 2): "TENSOR_SHAPE_FR1_NF1x2",
    (2, 1, 1): "TENSOR_SHAPE_FR2_NF1x1",
    (2, 1, 2): "TENSOR_SHAPE_FR2_NF1x2",
    (4, 1, 1): "TENSOR_SHAPE_FR4_NF1x1",
    (4, 1, 2): "TENSOR_SHAPE_FR4_NF1x2",
    (8, 1, 1): "TENSOR_SHAPE_FR8_NF1x1",
    (8, 1, 2): "TENSOR_SHAPE_FR8_NF1x2",
    (16, 1, 1): "TENSOR_SHAPE_FR16_NF1x1",
    (16, 1, 2): "TENSOR_SHAPE_FR16_NF1x2",
    (16, 2, 1): "TENSOR_SHAPE_FR16_NF2x1",
    (16, 2, 2): "TENSOR_SHAPE_FR16_NF2x2",
}

# DPRINT function-name prefixes that contribute to each TRISC coverage table.
# Coverage checkers are TRISC-scoped (no per-API enum); these lists only decide
# which harvested log tags feed which header on emit/seed.
MATH_FUNCTIONS = (
    "_llk_math_eltwise_binary_standard_",
    "_llk_math_eltwise_binary_standard_init_",
    "eltwise_binary_configure_mop_standard",
    "_llk_math_eltwise_binary_with_dest_reuse_",
    "_llk_math_eltwise_binary_with_dest_reuse_init_",
    "eltwise_binary_configure_mop_with_dest_reuse",
    "_llk_math_reduce_",
    "_llk_math_reduce_init_",
)

UNPACK_FUNCTIONS = (
    "_llk_unpack_AB_init_",
    "_llk_unpack_AB_mop_config_",
    "_llk_unpack_A_init_",
    "_llk_unpack_A_mop_config_",
    "_llk_unpack_AB_reduce_init_",
    "_llk_unpack_reduce_init_",
    "_llk_unpack_AB_reduce_mop_config_",
)

# Predicate name used when seeding from a checked-in TRISC header.
TRISC_CHECKERS = {
    "math": "is_math_tensor_shape_covered",
    "unpack": "is_unpack_tensor_shape_covered",
}


def _load_coverage(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {"tests": {}, "functions": {}}
    data.setdefault("functions", {})
    data.setdefault("tests", {})
    return data


def _save_coverage(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _log_paths(log_dir: Path) -> list[Path]:
    paths: list[Path] = []
    single = log_dir / "test_run.log"
    if single.exists():
        paths.append(single)
    paths.extend(sorted(log_dir.glob("test_run_gw*.log")))
    return paths


def _shape_name(fr: int, fc: int, nfr: int, nfc: int) -> str | None:
    if fc != FACE_C_DIM:
        return None
    return SHAPE_NAMES.get((fr, nfr, nfc))


def _union_shapes_for_fns(
    functions: dict[str, list], fn_names: tuple[str, ...]
) -> list[tuple[int, int, int, int]]:
    shapes: set[tuple[int, int, int, int]] = set()
    for fn in fn_names:
        for shape in functions.get(fn, []):
            shapes.add(tuple(shape))  # type: ignore[arg-type]
    return sorted(shapes)


def _render_predicate(name: str, shapes: list[tuple[int, int, int, int]]) -> list[str]:
    lines = [
        f"inline bool {name}(const TensorShape& tensor_shape)",
        "{",
    ]
    if not shapes:
        lines.append("    return false;")
        lines.append("}")
        return lines

    unknowns: list[tuple[int, int, int, int]] = []
    known: list[str] = []
    for fr, fc, nfr, nfc in shapes:
        const = _shape_name(fr, fc, nfr, nfc)
        if const is None:
            unknowns.append((fr, fc, nfr, nfc))
        else:
            known.append(const)

    if unknowns:
        lines.append(
            "    // WARNING: shapes outside the named TENSOR_SHAPE_FR* set were observed:"
        )
        for fr, fc, nfr, nfc in unknowns:
            lines.append(f"    //   TensorShape{{{fr}, {fc}, {nfr}, {nfc}}}")

    if not known:
        lines.append("    return false;")
        lines.append("}")
        return lines

    # One comparison per source line; clang-format may later pack them denser.
    parts = [f"tensor_shape_eq(tensor_shape, {name})" for name in known]
    lines.append(f"    return {parts[0]}" + (" ||" if len(parts) > 1 else ";"))
    for i, part in enumerate(parts[1:], start=1):
        suffix = " ||" if i < len(parts) - 1 else ";"
        indent = "           "
        lines.append(f"{indent}{part}{suffix}")
    lines.append("}")
    return lines


def _header_banner(kind: str, detail: str) -> list[str]:
    return [
        "// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "#pragma once",
        "",
        '#include "tensor_shape_coverage.h"',
        "",
        f"// TensorShape coverage table for {kind} LLKs. Lists TensorShape values observed",
        f"// across {detail}. Validation is TRISC-scoped: any {kind} call site shares",
        f"// this set, so new {kind} APIs do not need a central enum entry.",
        "//",
        "// Match tensor_shape_coverage.h's gate so production kernel builds do not see this table.",
        "// Regenerated by helpers/tensor_shape_coverage_parser.py.",
        "",
        "#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)",
        "",
        "namespace ckernel::coverage",
        "{",
        "",
    ]


def _shapes_from_predicate_body(body: str) -> list[tuple[int, int, int, int]]:
    shapes: set[tuple[int, int, int, int]] = set()
    for match in SHAPE_CONST_RE.finditer(body):
        fr = int(match.group("fr"))
        nfr = int(match.group("nfr"))
        nfc = int(match.group("nfc"))
        shapes.add((fr, FACE_C_DIM, nfr, nfc))
    return sorted(shapes)


def _seed_from_header(
    header_path: Path,
    checker_name: str,
    fn_names: tuple[str, ...],
    functions: dict[str, set],
) -> int:
    """Populate per-function shape sets from a TRISC coverage checker body."""
    if not header_path.exists():
        return 0
    text = header_path.read_text()
    added = 0
    for match in PREDICATE_BODY_RE.finditer(text):
        if match.group("name") != checker_name:
            continue
        shapes = _shapes_from_predicate_body(match.group("body"))
        for fn in fn_names:
            bucket = functions.setdefault(fn, set())
            before = len(bucket)
            bucket.update(shapes)
            added += len(bucket) - before
    return added


def seed(
    coverage_json: Path,
    math_header: Path,
    unpack_header: Path,
) -> tuple[int, int]:
    """Bootstrap coverage.json from the checked-in math/unpack headers."""
    coverage = _load_coverage(coverage_json)
    functions: dict[str, set[tuple[int, int, int, int]]] = {
        fn: {tuple(s) for s in shapes}  # type: ignore[misc]
        for fn, shapes in coverage["functions"].items()
    }
    added = 0
    added += _seed_from_header(
        math_header, TRISC_CHECKERS["math"], MATH_FUNCTIONS, functions
    )
    added += _seed_from_header(
        unpack_header, TRISC_CHECKERS["unpack"], UNPACK_FUNCTIONS, functions
    )
    coverage["functions"] = {
        fn: sorted(shapes) for fn, shapes in sorted(functions.items())
    }
    coverage["tests"]["seed_from_headers"] = {
        "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dprint_lines_seen": 0,
        "unique_fn_shape_pairs": sum(len(v) for v in functions.values()),
        "functions": sorted(functions),
        "math_header": str(math_header),
        "unpack_header": str(unpack_header),
    }
    _save_coverage(coverage_json, coverage)
    return added, len(functions)


def _header_footer() -> list[str]:
    return [
        "} // namespace ckernel::coverage",
        "",
        "#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)",
        "",
    ]


def harvest(test_name: str, log_dir: Path, coverage_json: Path) -> tuple[int, int, int]:
    """Read logs, extract DPRINT lines, persist into coverage.json.

    Returns (lines_seen, shapes_added, function_count_after).
    """
    coverage = _load_coverage(coverage_json)
    by_fn: dict[str, set[tuple[int, int, int, int]]] = {
        fn: {tuple(s) for s in shapes}  # type: ignore[misc]
        for fn, shapes in coverage["functions"].items()
    }

    test_entries: set[tuple[str, tuple[int, int, int, int]]] = set()
    lines_seen = 0
    for log_path in _log_paths(log_dir):
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        for match in DPRINT_RE.finditer(text):
            lines_seen += 1
            fn = match.group("fn")
            shape = (
                int(match.group("fr")),
                int(match.group("fc")),
                int(match.group("nfr")),
                int(match.group("nfc")),
            )
            test_entries.add((fn, shape))
            by_fn.setdefault(fn, set()).add(shape)

    shapes_added = 0
    for fn, shapes in by_fn.items():
        prev = {tuple(s) for s in coverage["functions"].get(fn, [])}
        shapes_added += len(shapes - prev)

    coverage["functions"] = {fn: sorted(shapes) for fn, shapes in sorted(by_fn.items())}
    coverage["tests"][test_name] = {
        "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dprint_lines_seen": lines_seen,
        "unique_fn_shape_pairs": len(test_entries),
        "functions": sorted({fn for fn, _ in test_entries}),
        "log_dir": str(log_dir),
    }
    _save_coverage(coverage_json, coverage)
    return lines_seen, shapes_added, len(by_fn)


def emit_math_header(coverage_json: Path, out_path: Path) -> None:
    coverage = _load_coverage(coverage_json)
    shapes = _union_shapes_for_fns(coverage["functions"], MATH_FUNCTIONS)
    lines = _header_banner("math", "math init / execute / MOP configuration paths")
    lines.extend(_render_predicate("is_math_tensor_shape_covered", shapes))
    lines.append("")
    lines.extend(_header_footer())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def emit_unpack_header(coverage_json: Path, out_path: Path) -> None:
    coverage = _load_coverage(coverage_json)
    shapes = _union_shapes_for_fns(coverage["functions"], UNPACK_FUNCTIONS)
    lines = _header_banner("unpack", "unpack init / MOP configuration paths")
    lines.extend(_render_predicate("is_unpack_tensor_shape_covered", shapes))
    lines.append("")
    lines.extend(_header_footer())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def emit_pack_header(out_path: Path) -> None:
    """Pack probes are not defined yet; keep the stub header regeneratable."""
    text = """// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_shape_coverage.h"

// TensorShape coverage hooks for pack LLKs. Pack probes are not currently
// defined, so the validation helper returns false until pack TensorShape
// coverage tables are added.
//
// Match tensor_shape_coverage.h's gate so production kernel builds do not see this table.
// Regenerated by helpers/tensor_shape_coverage_parser.py.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

namespace ckernel::coverage
{

// No pack TensorShape coverage probes are currently defined.

__attribute__((noinline)) inline bool is_pack_tensor_shape_covered(const TensorShape&)
{
    return false;
}

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)


def summary(coverage_json: Path) -> None:
    coverage = _load_coverage(coverage_json)
    functions = coverage["functions"]
    tests = coverage["tests"]
    print(f"coverage json: {coverage_json}")
    print(f"tests harvested: {len(tests)}")
    for name in sorted(tests):
        rec = tests[name]
        print(
            f"  {name}: {rec.get('dprint_lines_seen', 0)} dprint lines, "
            f"{rec.get('unique_fn_shape_pairs', 0)} unique (fn, shape) pairs"
        )
    print(f"\nfunctions seen: {len(functions)}")
    for fn in sorted(functions):
        shapes = sorted(tuple(s) for s in functions[fn])
        formatted = ", ".join(
            _shape_name(fr, fc, nfr, nfc) or f"{{{fr},{fc},{nfr},{nfc}}}"
            for fr, fc, nfr, nfc in shapes
        )
        print(f"  {fn}: {len(shapes)} -> {formatted}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=DEFAULT_COVERAGE_JSON,
        help=f"harvest state file (default: {DEFAULT_COVERAGE_JSON})",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PYTHON_TESTS_DIR,
        help=f"directory containing test_run*.log (default: {PYTHON_TESTS_DIR})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    seed_p = sub.add_parser(
        "seed",
        help="bootstrap coverage.json from TENSOR_SHAPE_FR* refs in checked-in headers",
    )
    seed_p.add_argument("--math", type=Path, default=DEFAULT_MATH_HEADER)
    seed_p.add_argument("--unpack", type=Path, default=DEFAULT_UNPACK_HEADER)

    harvest_p = sub.add_parser("harvest", help="parse test_run*.log into coverage.json")
    harvest_p.add_argument(
        "test_name", help="label stored under tests[] in coverage.json"
    )

    emit_p = sub.add_parser(
        "emit",
        help="rewrite tensor_shape_coverage_{math,unpack,pack}.h from coverage.json",
    )
    emit_p.add_argument("--math", type=Path, default=DEFAULT_MATH_HEADER)
    emit_p.add_argument("--unpack", type=Path, default=DEFAULT_UNPACK_HEADER)
    emit_p.add_argument("--pack", type=Path, default=DEFAULT_PACK_HEADER)
    emit_p.add_argument(
        "--skip-pack",
        action="store_true",
        help="do not rewrite the pack stub header",
    )

    sub.add_parser("summary", help="print harvested coverage summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "seed":
        added, fn_count = seed(args.coverage_json, args.math, args.unpack)
        print(f"seeded {added} shape entries across {fn_count} functions")
        print(f"wrote {args.coverage_json}")
        return 0

    if args.command == "harvest":
        lines, added, fn_count = harvest(
            args.test_name, args.log_dir, args.coverage_json
        )
        print(
            f"[{args.test_name}] dprint lines: {lines}, new shapes: {added}, "
            f"functions tracked: {fn_count}"
        )
        print(f"wrote {args.coverage_json}")
        return 0

    if args.command == "emit":
        if not args.coverage_json.exists():
            print(
                f"error: {args.coverage_json} does not exist; run `seed` or `harvest` first",
                file=sys.stderr,
            )
            return 1
        emit_math_header(args.coverage_json, args.math)
        emit_unpack_header(args.coverage_json, args.unpack)
        print(f"wrote {args.math}")
        print(f"wrote {args.unpack}")
        if not args.skip_pack:
            emit_pack_header(args.pack)
            print(f"wrote {args.pack}")
        return 0

    if args.command == "summary":
        summary(args.coverage_json)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
