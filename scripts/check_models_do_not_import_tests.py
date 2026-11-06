#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reject imports of the tests package from models/.

models may depend on tt_py_test_utils_common. Importing tests reverses tests -> models.
"""

import ast
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _REPO_ROOT / "models"


def _tests_import(node):
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "tests" or alias.name.startswith("tests."):
                return f"import {alias.name}"
    elif (
        isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module
        and (node.module == "tests" or node.module.startswith("tests."))
    ):
        # Relative imports such as ``from ..tests.test_factory`` name a model package, not the top-level tests tree.
        return f"from {node.module} import ..."
    return None


def violations_in(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno or 0}: failed to parse ({exc.msg})"]
    found = []
    for node in ast.walk(tree):
        statement = _tests_import(node)
        if statement:
            found.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {statement}")
    return found


def paths_to_check(argv):
    if not argv:
        return sorted(_MODELS.rglob("*.py"))
    selected = []
    for arg in argv:
        path = Path(arg)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        try:
            path.resolve().relative_to(_MODELS.resolve())
        except ValueError:
            continue
        if path.is_file() and path.suffix == ".py":
            selected.append(path)
    return selected


def main(argv):
    violations = []
    for path in paths_to_check(argv):
        violations.extend(violations_in(path))
    if violations:
        print("models/ must not import the tests package:")
        for violation in violations:
            print(violation)
        return 1
    print("No tests imports found under models/.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
