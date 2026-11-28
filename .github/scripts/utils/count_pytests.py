#!/usr/bin/env python3
"""Counts total number of pytests including parametrize expansions.

The count follows pytest's collection rules instead of every `test_*` def in the
file: only test files are read (`test_*.py` and `*_test.py`, both pytest
defaults), and only module-level functions plus methods of `Test*` classes are
counted. A `test_` helper nested inside another function, or a method of a class
pytest does not consider a test class, never runs as a test.
"""

import ast
import sys
from pathlib import Path

TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


def _iter_test_files(directory):
    """Every file pytest treats as a test module, once, in a deterministic order."""
    root = Path(directory)
    seen = set()
    files = []
    for pattern in TEST_FILE_PATTERNS:
        for filepath in root.rglob(pattern):
            key = filepath.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(filepath)
    return sorted(files)


def _test_functions(tree):
    """Test defs pytest would collect: module-level, or a method of a `Test*` class."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield item


def _parametrize_multiplier(node):
    mult = 1
    for dec in node.decorator_list:
        # @pytest.mark.parametrize(...) with an explicit list/tuple of values.
        if (
            isinstance(dec, ast.Call)
            and hasattr(dec.func, "attr")
            and dec.func.attr == "parametrize"
            and len(dec.args) >= 2
        ):
            val = dec.args[1]
            if isinstance(val, (ast.List, ast.Tuple)):
                mult *= len(val.elts)
    return mult


def count_tests_in_directory(directory):
    total = 0

    for filepath in _iter_test_files(directory):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Parse the Python file into an Abstract Syntax Tree (AST) which converts the code into a tree structure we can loop
                tree = ast.parse(f.read(), filename=str(filepath))

            for node in _test_functions(tree):
                if not node.name.startswith("test_"):
                    continue
                # Start with 1 test, then multiply by parameter counts
                total += _parametrize_multiplier(node)

        except Exception:
            # Skip files that can't be parsed
            pass

    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: count_pytests.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    count = count_tests_in_directory(directory)
    print(count)
