#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
CI validation script: ensures all C++ operations bound via bind_function<>
have TT_OP_SCOPE / FunctionScope instrumentation in their implementation.

Scans all nanobind source files (.cpp and .hpp) for bind_function<"name">
calls, then verifies that a matching TT_OP_SCOPE("fqn") exists in the
codebase.

Usage: python scripts/check_composite_tracing.py
Exit code 0 = all operations traced, 1 = missing traces found.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "ttnn" / "cpp"
EXPECTED_MISSING = {
    "ttnn::test_hang_device_operation",
}

BIND_RE = re.compile(r'bind_function<"([^"]+)"(?:\s*,\s*"([^"]+)")?\s*>')
MACRO_INVOCATION_RE = re.compile(r"(?:DEFINE_UNARY_NG_OP\w*|TTNN_BINARY_OP_\w+)\(\s*(\w+)")
MACRO_STRINGIFY_RE = re.compile(r'"ttnn::"\s*#')


def iter_source_files():
    for pattern in ("*.cpp", "*.hpp"):
        yield from SCAN_ROOT.rglob(pattern)


def parse_nanobind_files():
    """Find every bind_function<"name"[,"namespace"]> occurrence and return {fqn: op_name}."""
    ops = {}
    for path in iter_source_files():
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "bind_function<" not in text:
            continue
        for m in BIND_RE.finditer(text):
            op_name = m.group(1)
            namespace = m.group(2) or "ttnn."
            fqn = (namespace + op_name).replace(".", "::")
            ops.setdefault(fqn, op_name)
    return ops


def find_trace_literals():
    """Scan once and return the set of trace name string literals found in the codebase."""
    trace_literal_re = re.compile(r'"((?:ttnn|tt)::[A-Za-z0-9_:]+)"')
    found = set()
    for path in iter_source_files():
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in trace_literal_re.finditer(text):
            found.add(m.group(1))
    return found


def find_macro_traced_ops():
    """
    Find ops covered indirectly via a macro that stringifies its argument into the
    trace name (e.g. `"ttnn::" #NAME`). Returns the set of op_name tokens passed to
    such macros.
    """
    macro_files = []
    for path in iter_source_files():
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if MACRO_STRINGIFY_RE.search(text):
            macro_files.append((path, text))

    op_names = set()
    for _, text in macro_files:
        op_names.update(MACRO_INVOCATION_RE.findall(text))
    return op_names


def main():
    ops = parse_nanobind_files()
    print(f"Found {len(ops)} unique operations bound via bind_function<>")

    trace_literals = find_trace_literals()
    macro_traced = find_macro_traced_ops()

    missing = []
    for fqn, op_name in sorted(ops.items()):
        if fqn in EXPECTED_MISSING:
            continue
        if fqn in trace_literals:
            continue
        if op_name in macro_traced:
            continue
        missing.append(fqn)

    if missing:
        print(f"\nERROR: {len(missing)} operation(s) missing TT_OP_SCOPE:\n")
        for fqn in missing:
            print(f"  {fqn}")
        print(
            "\nEvery C++ operation bound via bind_function<> must include a\n"
            "TT_OP_SCOPE guard at the top of its function body:\n"
            "\n"
            '    #include "ttnn/graph/composite_trace.hpp"\n'
            "\n"
            "    Tensor my_op(const Tensor& input) {\n"
            '        TT_OP_SCOPE("ttnn::my_op");\n'
            "        ...\n"
            "    }\n"
            "\n"
            "The trace name must match the fully qualified name from bind_function<>.\n"
            "See docs/source/ttnn/ttnn/adding_new_ttnn_operation.rst for details."
        )
        return 1

    print("All operations have TT_OP_SCOPE instrumentation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
