#!/usr/bin/env python3
"""TensorShape coverage parser.

Reads loguru test logs (test_run.log, test_run_gw*.log) and extracts DPRINT
coverage lines into coverage.json.

Single-shape:
  '[<fn>] tensor_shape: face_r_dim=N face_c_dim=N num_faces_r_dim=N num_faces_c_dim=N'

Pairwise (matmul):
  '[<fn>] tensor_shape_pair: in0_face_r_dim=... in0_face_c_dim=... in0_num_faces_r_dim=... in0_num_faces_c_dim=...'
  ' in1_face_r_dim=... in1_face_c_dim=... in1_num_faces_r_dim=... in1_num_faces_c_dim=...'

Usage:
    parse.py harvest  <test_name>            # consume *.log files into coverage.json
    parse.py emit     <out_header_path>      # regenerate single-shape coverage header
    parse.py emit-math <out_header_path>     # regenerate matmul pair coverage header
    parse.py summary                         # print per-function summary
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

LLK_COMMON_DIR = Path(__file__).resolve().parent
LLK_ROOT = LLK_COMMON_DIR.parent
PYTHON_TESTS_DIR = LLK_ROOT / "tests" / "python_tests"
COVERAGE_JSON = Path("/tmp/ts-coverage/parsed/coverage.json")

MATMUL_PAIR_FUNCTIONS = (
    "matmul_configure_addrmod",
    "matmul_configure_mop",
    "matmul_configure_mop_throttled",
    "_llk_math_matmul_init_",
)

DPRINT_RE = re.compile(
    r"\[(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\] tensor_shape: "
    r"face_r_dim=(?P<fr>\d+) face_c_dim=(?P<fc>\d+) "
    r"num_faces_r_dim=(?P<nfr>\d+) num_faces_c_dim=(?P<nfc>\d+)"
)

PAIR_DPRINT_RE = re.compile(
    r"\[(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\] tensor_shape_pair: "
    r"in0_face_r_dim=(?P<in0_fr>\d+) in0_face_c_dim=(?P<in0_fc>\d+) "
    r"in0_num_faces_r_dim=(?P<in0_nfr>\d+) in0_num_faces_c_dim=(?P<in0_nfc>\d+) "
    r"in1_face_r_dim=(?P<in1_fr>\d+) in1_face_c_dim=(?P<in1_fc>\d+) "
    r"in1_num_faces_r_dim=(?P<in1_nfr>\d+) in1_num_faces_c_dim=(?P<in1_nfc>\d+)"
)

# (fr, nfr, nfc) -> constexpr name. face_c_dim is always 16 by HW.
SHAPE_NAMES = {
    (fr, nfr, nfc): f"TENSOR_SHAPE_FR{fr}_NF{nfr}x{nfc}"
    for fr in (1, 2, 4, 8, 16)
    for nfr in (1, 2)
    for nfc in (1, 2)
}


def _load_coverage() -> dict:
    if COVERAGE_JSON.exists():
        data = json.loads(COVERAGE_JSON.read_text())
    else:
        data = {"tests": {}, "functions": {}, "function_pairs": {}}
    data.setdefault("functions", {})
    data.setdefault("function_pairs", {})
    data.setdefault("tests", {})
    return data


def _save_coverage(data: dict) -> None:
    COVERAGE_JSON.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_JSON.write_text(json.dumps(data, indent=2, sort_keys=True))


def _log_paths() -> list[Path]:
    paths: list[Path] = []
    single = PYTHON_TESTS_DIR / "test_run.log"
    if single.exists():
        paths.append(single)
    paths.extend(sorted(PYTHON_TESTS_DIR.glob("test_run_gw*.log")))
    return paths


def _parse_shape(match: re.Match[str], prefix: str = "") -> tuple[int, int, int, int]:
    return (
        int(match.group(f"{prefix}fr")),
        int(match.group(f"{prefix}fc")),
        int(match.group(f"{prefix}nfr")),
        int(match.group(f"{prefix}nfc")),
    )


def _harvest(test_name: str) -> tuple[int, int, int, int, int]:
    """Read worker logs, extract DPRINT lines, persist into coverage.json.

    Returns (single_lines, pair_lines, shapes_added, pairs_added, function_count_after).
    """
    coverage = _load_coverage()
    by_fn: dict[str, set[tuple[int, int, int, int]]] = {
        fn: set(tuple(s) for s in shapes)
        for fn, shapes in coverage["functions"].items()
    }
    by_fn_pairs: dict[
        str, set[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]
    ] = {
        fn: {tuple(tuple(part) for part in pair) for pair in pairs}
        for fn, pairs in coverage["function_pairs"].items()
    }

    test_single: set[tuple[str, tuple[int, int, int, int]]] = set()
    test_pairs: set[
        tuple[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]
    ] = set()
    single_lines = 0
    pair_lines = 0

    for log_path in _log_paths():
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        for m in DPRINT_RE.finditer(text):
            single_lines += 1
            fn = m.group("fn")
            shape = _parse_shape(m)
            test_single.add((fn, shape))
            by_fn.setdefault(fn, set()).add(shape)
        for m in PAIR_DPRINT_RE.finditer(text):
            pair_lines += 1
            fn = m.group("fn")
            in0 = _parse_shape(m, prefix="in0_")
            in1 = _parse_shape(m, prefix="in1_")
            pair = (in0, in1)
            test_pairs.add((fn, pair))
            by_fn_pairs.setdefault(fn, set()).add(pair)

    shapes_added = 0
    for fn, shapes in by_fn.items():
        prev = set(tuple(s) for s in coverage["functions"].get(fn, []))
        if shapes - prev:
            shapes_added += len(shapes - prev)

    pairs_added = 0
    for fn, pairs in by_fn_pairs.items():
        prev = {
            tuple(tuple(part) for part in pair)
            for pair in coverage["function_pairs"].get(fn, [])
        }
        if pairs - prev:
            pairs_added += len(pairs - prev)

    coverage["functions"] = {fn: sorted(shapes) for fn, shapes in sorted(by_fn.items())}
    coverage["function_pairs"] = {
        fn: [[list(in0), list(in1)] for in0, in1 in sorted(pairs)]
        for fn, pairs in sorted(by_fn_pairs.items())
    }
    coverage["tests"][test_name] = {
        "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dprint_lines_seen": single_lines + pair_lines,
        "single_dprint_lines_seen": single_lines,
        "pair_dprint_lines_seen": pair_lines,
        "unique_fn_shape_pairs": len(test_single) + len(test_pairs),
        "functions": sorted(set(fn for fn, _ in test_single)),
        "pair_functions": sorted(set(fn for fn, _ in test_pairs)),
    }
    _save_coverage(coverage)
    fn_count = len(set(by_fn.keys()) | set(by_fn_pairs.keys()))
    return single_lines, pair_lines, shapes_added, pairs_added, fn_count


def _shape_to_literal(fr: int, fc: int, nfr: int, nfc: int) -> tuple[str, bool]:
    """Return (cpp_literal, is_known_constant)."""
    if fc == 16 and (fr, nfr, nfc) in SHAPE_NAMES:
        return SHAPE_NAMES[(fr, nfr, nfc)], True
    return f"TensorShape{{{fr}, {fc}, {nfr}, {nfc}}}", False


def _array_name(fn: str) -> str:
    cleaned = fn.strip("_")
    return f"covered_shapes_{cleaned}"


def _pair_array_name(fn: str) -> str:
    cleaned = fn.strip("_")
    return f"covered_shape_pairs_{cleaned}"


def _emit_single_shape_header(out_path: Path) -> None:
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
    lines.append("//")
    lines.append(
        "// Regenerate by running functional pytests with --logging-level=DEBUG"
    )
    lines.append("// and harvesting logs through common/parse.py.")
    lines.append("//")
    lines.append(
        f"// Last update : {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append(f"// Tests run   : {len(tests_done)}")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <array>")
    lines.append("")
    lines.append('#include "tensor_shape.h"')
    lines.append("")
    lines.append("#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)")
    lines.append("")
    lines.append("namespace ckernel::coverage")
    lines.append("{")
    lines.append("")

    for fn in sorted(fns.keys()):
        shapes = sorted(tuple(s) for s in fns[fn])
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
            tag = "" if known else "  // out-of-bounds"
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
                "// Note: observed shapes outside the validated set; investigate."
            )
        lines.append("")

    lines.append("} // namespace ckernel::coverage")
    lines.append("")
    lines.append("#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def _emit_math_pair_header(out_path: Path) -> None:
    coverage = _load_coverage()
    pair_fns = coverage["function_pairs"]
    tests_done = sorted(coverage["tests"].keys())

    lines: list[str] = []
    lines.append("// SPDX-FileCopyrightText: \u00a9 2026 Tenstorrent AI ULC")
    lines.append("//")
    lines.append("// SPDX-License-Identifier: Apache-2.0")
    lines.append("//")
    lines.append(
        "// AUTO-GENERATED: per-function TensorShape pair coverage for matmul helpers."
    )
    lines.append("//")
    lines.append("// Regenerate by running matmul pytests with --logging-level=DEBUG,")
    lines.append(
        "// TT_LLK_DISABLE_ASSERTS=1, then: parse.py harvest && parse.py emit-math"
    )
    lines.append("//")
    lines.append(
        f"// Last update : {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append(f"// Tests run   : {len(tests_done)}")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)")
    lines.append("")
    lines.append("#include <array>")
    lines.append("")
    lines.append('#include "tensor_shape_coverage.h"')
    lines.append("")
    lines.append("namespace ckernel::coverage")
    lines.append("{")
    lines.append("")

    emit_fns = sorted(set(MATMUL_PAIR_FUNCTIONS) | set(pair_fns.keys()))
    for fn in emit_fns:
        raw_pairs = pair_fns.get(fn, [])
        pairs = sorted(tuple(tuple(part) for part in pair) for pair in raw_pairs)
        arr = _pair_array_name(fn)
        lines.append(f"// {fn}: {len(pairs)} unique TensorShape pair(s)")
        if not pairs:
            lines.append(
                f"inline constexpr std::array<TensorShapePair, 0> {arr} = {{}};"
            )
            lines.append("")
            continue
        any_unknown = False
        lines.append(
            f"inline constexpr std::array<TensorShapePair, {len(pairs)}> {arr} = {{{{"
        )
        for in0, in1 in pairs:
            in0_lit, in0_known = _shape_to_literal(*in0)
            in1_lit, in1_known = _shape_to_literal(*in1)
            if not in0_known or not in1_known:
                any_unknown = True
            tag = ""
            if not in0_known or not in1_known:
                tag = "  // out-of-bounds"
            lines.append(f"    {{{in0_lit}, {in1_lit}}},{tag}")
        lines.append("}};")
        if any_unknown:
            lines.append(
                "// Note: observed pair shapes outside the validated set; investigate."
            )
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
    pair_fns = coverage["function_pairs"]
    tests = coverage["tests"]
    print(f"tests harvested: {len(tests)}")
    for t in sorted(tests):
        rec = tests[t]
        print(
            f"  {t}: {rec.get('dprint_lines_seen', 0)} dprint lines "
            f"({rec.get('single_dprint_lines_seen', 0)} single, "
            f"{rec.get('pair_dprint_lines_seen', 0)} pair), "
            f"{rec.get('unique_fn_shape_pairs', 0)} unique entries"
        )
    print(f"\nsingle-shape functions seen: {len(fns)}")
    for fn in sorted(fns):
        shapes = sorted(tuple(s) for s in fns[fn])
        formatted = ", ".join(f"({fr},{fc},{nfr},{nfc})" for fr, fc, nfr, nfc in shapes)
        print(f"  {fn}: {len(shapes)} shape(s) -> {formatted}")
    print(f"\npairwise functions seen: {len(pair_fns)}")
    for fn in sorted(pair_fns):
        pairs = sorted(tuple(tuple(part) for part in pair) for pair in pair_fns[fn])
        formatted = ", ".join(
            f"(({i0[0]},{i0[1]},{i0[2]},{i0[3]}),({i1[0]},{i1[1]},{i1[2]},{i1[3]}))"
            for i0, i1 in pairs
        )
        print(f"  {fn}: {len(pairs)} pair(s) -> {formatted}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd = argv[1]
    if cmd == "harvest":
        if len(argv) < 3:
            print("usage: parse.py harvest <test_name>")
            return 1
        single, pair, shapes_added, pairs_added, fn_count = _harvest(argv[2])
        print(
            f"[{argv[2]}] single dprint lines: {single}, pair dprint lines: {pair}, "
            f"new shapes: {shapes_added}, new pairs: {pairs_added}, "
            f"total functions tracked: {fn_count}"
        )
        return 0
    if cmd == "emit":
        if len(argv) < 3:
            print("usage: parse.py emit <out_header_path>")
            return 1
        _emit_single_shape_header(Path(argv[2]))
        print(f"wrote {argv[2]}")
        return 0
    if cmd == "emit-math":
        if len(argv) < 3:
            print("usage: parse.py emit-math <out_header_path>")
            return 1
        _emit_math_pair_header(Path(argv[2]))
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
