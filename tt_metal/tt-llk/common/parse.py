#!/usr/bin/env python3
"""TensorShape coverage parser.

Reads loguru per-worker logs (test_run_gw*.log) and extracts
'[<fn>] tensor_shape: face_r_dim=N face_c_dim=N num_faces_r_dim=N num_faces_c_dim=N'
lines. Accumulates per-function shape sets into coverage.json.

Usage:
    parse.py harvest   <test_name>           # consume *.log files into coverage.json
    parse.py emit      <out_header_path>     # regenerate monolithic coverage header
    parse.py emit-pack <out_header_path>     # regenerate tensor_shape_coverage_pack.h
    parse.py summary                         # print per-function summary
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

TT_LLK_COMMON_DIR = Path(__file__).resolve().parent
PYTHON_TESTS_DIR = TT_LLK_COMMON_DIR.parent / "tests" / "python_tests"
COVERAGE_JSON = Path("/tmp/ts-coverage/parsed/coverage.json")

# Pack functions registered in tensor_shape_coverage.h TensorShapeFunctionCoverage.
PACK_FUNCTIONS = (
    "_llk_pack_hw_configure_",
    "_llk_pack_init_",
    "_llk_pack_mop_config_",
    "_llk_pack_reconfig_data_format_",
)

DPRINT_RE = re.compile(
    r"\[(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\] tensor_shape: "
    r"face_r_dim=(?P<fr>\d+) face_c_dim=(?P<fc>\d+) "
    r"num_faces_r_dim=(?P<nfr>\d+) num_faces_c_dim=(?P<nfc>\d+)"
)
# DEVICE_PRINT emits TensorShapeFunctionCoverage as uint32_t (see tensor_shape.h).
DPRINT_NUMERIC_FN_RE = re.compile(
    r"\[(?P<fn_id>\d+)\] tensor_shape: "
    r"face_r_dim=(?P<fr>\d+) face_c_dim=(?P<fc>\d+) "
    r"num_faces_r_dim=(?P<nfr>\d+) num_faces_c_dim=(?P<nfc>\d+)"
)

# Must match TensorShapeFunctionCoverage declaration order in tensor_shape_coverage.h.
FN_BY_ENUM_ID: dict[int, str] = {
    0: "_llk_math_eltwise_binary_standard_",
    1: "_llk_math_eltwise_binary_standard_init_",
    2: "_llk_math_eltwise_binary_with_dest_reuse_",
    3: "_llk_math_eltwise_binary_with_dest_reuse_init_",
    4: "_llk_math_reduce_",
    5: "_llk_math_reduce_init_",
    6: "_llk_unpack_AB_init_",
    7: "_llk_unpack_AB_mop_config_",
    8: "_llk_unpack_AB_reduce_init_",
    9: "_llk_unpack_reduce_init_",
    10: "_llk_unpack_AB_reduce_mop_config_",
    11: "_llk_unpack_A_init_",
    12: "_llk_unpack_A_mop_config_",
    13: "_llk_pack_hw_configure_",
    14: "_llk_pack_init_",
    15: "_llk_pack_mop_config_",
    16: "_llk_pack_reconfig_data_format_",
    17: "eltwise_binary_configure_mop_standard",
    18: "eltwise_binary_configure_mop_with_dest_reuse",
}

# (fr, nfr, nfc) -> constexpr name. face_c_dim is always 16 by HW.
SHAPE_NAMES = {
    (fr, nfr, nfc): f"TENSOR_SHAPE_FR{fr}_NF{nfr}x{nfc}"
    for fr in (1, 2, 4, 8, 16)
    for nfr in (1, 2)
    for nfc in (1, 2)
}


def _load_coverage() -> dict:
    if COVERAGE_JSON.exists():
        return json.loads(COVERAGE_JSON.read_text())
    return {"tests": {}, "functions": {}}


def _save_coverage(data: dict) -> None:
    COVERAGE_JSON.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_JSON.write_text(json.dumps(data, indent=2, sort_keys=True))


def _log_paths() -> list[Path]:
    """Per-worker logs (xdist) and single-process test_run.log."""
    paths = sorted(PYTHON_TESTS_DIR.glob("test_run_gw*.log"))
    single = PYTHON_TESTS_DIR / "test_run.log"
    if single.exists():
        paths.append(single)
    return paths


def _parse_dprint_line(
    m: re.Match, by_fn: dict[str, set[tuple[int, int, int, int]]]
) -> tuple[str, tuple[int, int, int, int]] | None:
    if "fn_id" in m.groupdict() and m.group("fn_id") is not None:
        fn_id = int(m.group("fn_id"))
        fn = FN_BY_ENUM_ID.get(fn_id)
        if fn is None:
            return None
    else:
        fn = m.group("fn")
    shape = (
        int(m.group("fr")),
        int(m.group("fc")),
        int(m.group("nfr")),
        int(m.group("nfc")),
    )
    by_fn.setdefault(fn, set()).add(shape)
    return fn, shape


def _harvest(test_name: str) -> tuple[int, int, int]:
    """Read worker logs, extract DPRINT lines, persist into coverage.json.

    Returns (lines_seen, shapes_added, function_count_after).
    """
    coverage = _load_coverage()
    by_fn: dict[str, set[tuple[int, int, int, int]]] = {
        fn: set(tuple(s) for s in shapes)
        for fn, shapes in coverage["functions"].items()
    }

    test_lines: set[tuple[str, tuple[int, int, int, int]]] = set()
    lines_seen = 0
    for log_path in _log_paths():
        try:
            text = log_path.read_text(errors="replace")
        except Exception:
            continue
        for m in DPRINT_RE.finditer(text):
            lines_seen += 1
            parsed = _parse_dprint_line(m, by_fn)
            if parsed:
                test_lines.add(parsed)
        for m in DPRINT_NUMERIC_FN_RE.finditer(text):
            lines_seen += 1
            parsed = _parse_dprint_line(m, by_fn)
            if parsed:
                test_lines.add(parsed)

    shapes_added = 0
    for fn, shapes in by_fn.items():
        prev = set(tuple(s) for s in coverage["functions"].get(fn, []))
        if shapes - prev:
            shapes_added += len(shapes - prev)

    coverage["functions"] = {fn: sorted(shapes) for fn, shapes in sorted(by_fn.items())}
    coverage["tests"][test_name] = {
        "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dprint_lines_seen": lines_seen,
        "unique_fn_shape_pairs": len(test_lines),
        "functions": sorted(set(fn for fn, _ in test_lines)),
    }
    _save_coverage(coverage)
    return lines_seen, shapes_added, len(by_fn)


def _shape_to_literal(fr: int, fc: int, nfr: int, nfc: int) -> tuple[str, bool]:
    """Return (cpp_literal, is_known_constant)."""
    if fc == 16 and (fr, nfr, nfc) in SHAPE_NAMES:
        return SHAPE_NAMES[(fr, nfr, nfc)], True
    return f"TensorShape{{{fr}, {fc}, {nfr}, {nfc}}}", False


def _array_name(fn: str) -> str:
    """Build a portable C++ identifier from a function name.

    LLK 'internal' functions like _llk_unpack_A_init_ have leading and trailing
    underscores. Concatenating "<fn>_covered_shapes" would produce reserved
    identifiers (double underscore anywhere is reserved for the implementation).
    Strip the leading/trailing underscores and prefix the array consistently.
    """
    cleaned = fn.strip("_")
    return f"covered_shapes_{cleaned}"


def _emit(out_path: Path) -> None:
    coverage = _load_coverage()
    fns = coverage["functions"]
    tests_done = sorted(coverage["tests"].keys())

    lines: list[str] = []
    lines.append("// SPDX-FileCopyrightText: \u00a9 2026 Tenstorrent AI ULC")
    lines.append("//")
    lines.append("// SPDX-License-Identifier: Apache-2.0")
    lines.append("//")
    lines.append(
        "// AUTO-GENERATED: per-function TensorShape coverage observed across LLK pytests."
    )
    lines.append(
        "// Sourced from DPRINT lines emitted by LLK_DPRINT_TENSOR_SHAPE in front of"
    )
    lines.append(
        "// LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(...)) call sites."
    )
    lines.append("//")
    lines.append(
        "// Regenerate by running the BH functional pytests with --logging-level=DEBUG"
    )
    lines.append(
        "// and feeding the per-worker test_run_gw*.log files through /tmp/ts-coverage/parse.py."
    )
    lines.append("//")
    lines.append(
        f"// Last update : {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append(f"// Tests run   : {len(tests_done)}")
    lines.append(f"// Architecture: blackhole")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <array>")
    lines.append("")
    lines.append('#include "tensor_shape.h"')
    lines.append("")
    lines.append(
        "// The TENSOR_SHAPE_FR*_NF*x* constants this manifest references are themselves"
    )
    lines.append(
        "// gated to ENABLE_LLK_ASSERT or DEBUG_PRINT_ENABLED builds (see tensor_shape.h)."
    )
    lines.append(
        "// Mirror the same gate here so this header stays a no-op when included from a"
    )
    lines.append("// production kernel build that defines neither flag.")
    lines.append("#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)")
    lines.append("")
    lines.append("namespace ckernel::coverage")
    lines.append("{")
    lines.append("")

    for fn in sorted(fns.keys()):
        shapes = sorted([tuple(s) for s in fns[fn]])
        arr = _array_name(fn)
        lines.append(f"// {fn}: {len(shapes)} unique TensorShape(s)")
        if not shapes:
            lines.append(f"inline constexpr std::array<TensorShape, 0> {arr} = {{}};")
            lines.append("")
            continue
        rendered = []
        any_unknown = False
        for fr, fc, nfr, nfc in shapes:
            literal, known = _shape_to_literal(fr, fc, nfr, nfc)
            tag = (
                ""
                if known
                else "  // out-of-bounds (validate_tensor_shape_tile_dependent_ops_ rejects)"
            )
            if not known:
                any_unknown = True
            rendered.append((literal, tag, (fr, fc, nfr, nfc)))
        lines.append(
            f"inline constexpr std::array<TensorShape, {len(shapes)}> {arr} = {{{{"
        )
        for literal, tag, dims in rendered:
            lines.append(f"    {literal},{tag}")
        lines.append("}};")
        if any_unknown:
            lines.append(
                "// Note: this function observed shapes outside the validated set; investigate."
            )
        lines.append("")

    if not fns:
        lines.append("// No coverage observed yet. Run the BH functional pytests with")
        lines.append(
            "// --logging-level=DEBUG and re-run /tmp/ts-coverage/parse.py emit."
        )
        lines.append("")

    lines.append("} // namespace ckernel::coverage")
    lines.append("")
    lines.append("#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def _emit_shape_arrays(
    lines: list[str], fn: str, shapes: list[tuple[int, int, int, int]]
) -> bool:
    """Append covered_shapes_* array for fn. Returns True if any out-of-bounds shape."""
    arr = _array_name(fn)
    lines.append(f"// {fn}: {len(shapes)} unique TensorShape(s)")
    if not shapes:
        lines.append(f"inline constexpr std::array<TensorShape, 0> {arr} = {{}};")
        lines.append("")
        return False

    any_unknown = False
    rendered: list[tuple[str, str]] = []
    for fr, fc, nfr, nfc in shapes:
        literal, known = _shape_to_literal(fr, fc, nfr, nfc)
        tag = (
            ""
            if known
            else "  // out-of-bounds (validate_tensor_shape_tile_dependent_ops_ rejects)"
        )
        if not known:
            any_unknown = True
        rendered.append((literal, tag))

    lines.append(
        f"inline constexpr std::array<TensorShape, {len(shapes)}> {arr} = {{{{"
    )
    for literal, tag in rendered:
        lines.append(f"    {literal},{tag}")
    lines.append("}};")
    if any_unknown:
        lines.append(
            "// Note: this function observed shapes outside the validated set; investigate."
        )
    lines.append("")
    return any_unknown


def _emit_pack(out_path: Path) -> None:
    """Write tensor_shape_coverage_pack.h with is_pack_tensor_shape_covered()."""
    coverage = _load_coverage()
    all_fns = coverage["functions"]
    pack_fns = {
        fn: sorted([tuple(s) for s in all_fns[fn]])
        for fn in PACK_FUNCTIONS
        if fn in all_fns
    }
    tests_done = sorted(coverage["tests"].keys())

    lines: list[str] = []
    lines.append("// SPDX-FileCopyrightText: \u00a9 2026 Tenstorrent AI ULC")
    lines.append("//")
    lines.append("// SPDX-License-Identifier: Apache-2.0")
    lines.append("//")
    lines.append(
        "// AUTO-GENERATED: pack TensorShape coverage observed across LLK pytests."
    )
    lines.append(
        "// Sourced from DPRINT lines emitted by LLK_VALIDATE_TENSOR_SHAPE_PACK before"
    )
    lines.append("// assert_tensor_shape_unobserved_() on uncovered shapes.")
    lines.append("//")
    lines.append("// Regenerate: run WH pack pytests with TT_LLK_DISABLE_ASSERTS=1 and")
    lines.append(
        "// --logging-level=debug, then parse.py harvest <name> and parse.py emit-pack."
    )
    lines.append("//")
    lines.append(
        f"// Last update : {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append(f"// Tests run   : {len(tests_done)}")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append(
        "// Match tensor_shape.h's gate so production kernel builds do not see this table."
    )
    lines.append("#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)")
    lines.append("")
    lines.append("#include <array>")
    lines.append("")
    lines.append('#include "tensor_shape_coverage.h"')
    lines.append("")
    lines.append("namespace ckernel::coverage")
    lines.append("{")
    lines.append("")

    for fn in PACK_FUNCTIONS:
        shapes = pack_fns.get(fn, [])
        _emit_shape_arrays(lines, fn, shapes)

    if not any(pack_fns.values()):
        lines.append("// No pack coverage observed yet. Run WH pack pytests with")
        lines.append(
            "// TT_LLK_DISABLE_ASSERTS=1 --logging-level=debug and re-harvest."
        )
        lines.append("")

    lines.append("constexpr bool is_pack_tensor_shape_covered(")
    lines.append(
        "    const TensorShapeFunctionCoverage fn, const TensorShape& tensor_shape)"
    )
    lines.append("{")
    lines.append("    using Function = TensorShapeFunctionCoverage;")
    lines.append("    switch (fn)")
    lines.append("    {")
    for fn in PACK_FUNCTIONS:
        arr = _array_name(fn)
        enum_case = f"Function::{fn}"
        shapes = pack_fns.get(fn, [])
        if shapes:
            lines.append(f"        case {enum_case}:")
            lines.append(
                f"            return contains_tensor_shape({arr}, tensor_shape);"
            )
    lines.append("        default:")
    lines.append("            return false;")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace ckernel::coverage")
    lines.append("")
    lines.append("#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def _summary() -> None:
    coverage = _load_coverage()
    fns = coverage["functions"]
    tests = coverage["tests"]
    print(f"tests harvested: {len(tests)}")
    for t in sorted(tests):
        rec = tests[t]
        print(
            f"  {t}: {rec['dprint_lines_seen']} dprint lines, "
            f"{rec['unique_fn_shape_pairs']} unique (fn, shape) pairs"
        )
    print(f"\nfunctions seen: {len(fns)}")
    for fn in sorted(fns):
        shapes = sorted([tuple(s) for s in fns[fn]])
        formatted = ", ".join(f"({fr},{fc},{nfr},{nfc})" for fr, fc, nfr, nfc in shapes)
        print(f"  {fn}: {len(shapes)} shape(s) -> {formatted}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd = argv[1]
    if cmd == "harvest":
        if len(argv) < 3:
            print("usage: parse.py harvest <test_name>")
            return 1
        lines, added, fn_count = _harvest(argv[2])
        print(
            f"[{argv[2]}] dprint lines: {lines}, new shapes added: {added}, "
            f"total functions tracked: {fn_count}"
        )
        return 0
    if cmd == "emit":
        if len(argv) < 3:
            print("usage: parse.py emit <out_header_path>")
            return 1
        _emit(Path(argv[2]))
        print(f"wrote {argv[2]}")
        return 0
    if cmd == "emit-pack":
        if len(argv) < 3:
            print("usage: parse.py emit-pack <out_header_path>")
            return 1
        _emit_pack(Path(argv[2]))
        print(f"wrote {argv[2]}")
        return 0
    if cmd == "summary":
        _summary()
        return 0
    print(f"unknown cmd: {cmd}")
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
