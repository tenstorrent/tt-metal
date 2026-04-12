#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
CI validation script: ensures all C++ operations bound via bind_function<>
have TT_OP_SCOPE / FunctionScope instrumentation in their implementation.

Scans all nanobind .cpp files for bind_function<"name"> calls, then verifies
that a matching TT_OP_SCOPE("fqn") exists in the codebase.

Usage: python scripts/check_composite_tracing.py
Exit code 0 = all operations traced, 1 = missing traces found.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MISSING = {
    "ttnn::test_hang_device_operation",
}


def parse_nanobind_files():
    result = subprocess.run(
        ["grep", "-rn", 'bind_function<"', "ttnn/cpp/", "--include=*.cpp", "-A", "15"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    ops = {}
    for block in result.stdout.split("--\n"):
        m = re.search(r'bind_function<"([^"]+)"(?:\s*,\s*"([^"]+)")?\s*>', block)
        if not m:
            continue
        op_name = m.group(1)
        namespace = m.group(2) or "ttnn."
        fqn = (namespace + op_name).replace(".", "::")
        if fqn not in ops:
            ops[fqn] = op_name
    return ops


def find_trace_in_codebase(fqn):
    pattern = f'"{fqn}"'
    result = subprocess.run(
        ["grep", "-rl", pattern, "ttnn/cpp/", "--include=*.cpp"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return bool(result.stdout.strip())


def check_macro_traces(fqn, op_name):
    """Check if the operation is covered by a macro that generates traces via #NAME stringification."""
    result = subprocess.run(
        ["grep", "-r", '"ttnn::" #', "ttnn/cpp/", "--include=*.cpp"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if not result.stdout:
        return False

    macro_files = set()
    for line in result.stdout.strip().split("\n"):
        filepath = line.split(":")[0]
        macro_files.add(filepath)

    for filepath in macro_files:
        content = Path(REPO_ROOT / filepath).read_text()
        macro_invocations = re.findall(r"(?:DEFINE_UNARY_NG_OP\w*|TTNN_BINARY_OP_\w+)\(\s*(\w+)", content)
        if op_name in macro_invocations:
            return True

    return False


def main():
    ops = parse_nanobind_files()
    print(f"Found {len(ops)} unique operations bound via bind_function<>")

    missing = []
    for fqn, op_name in sorted(ops.items()):
        if fqn in EXPECTED_MISSING:
            continue
        if find_trace_in_codebase(fqn):
            continue
        if check_macro_traces(fqn, op_name):
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
