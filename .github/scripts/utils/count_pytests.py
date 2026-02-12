#!/usr/bin/env python3
"""Counts total number of pytests including parametrize expansions."""

import ast
import sys
from pathlib import Path


def count_tests_in_directory(directory):
    total = 0

    for filepath in Path(directory).rglob("test_*.py"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Parse the Python file into an Abstract Syntax Tree (AST) which converts the code into a tree structure we can loop
                tree = ast.parse(f.read(), filename=str(filepath))

            # Walk through every element in the tree (functions, classes, variables, etc.)
            for node in ast.walk(tree):
                # Check if this element is a test function and then check if its name starts with 'test_'
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                    # Start with 1 test, then multiply by parameter counts
                    mult = 1

                    # Loop through all decorators on this function (like @pytest.mark.parametrize)
                    for dec in node.decorator_list:
                        # Check if this decorator is @pytest.mark.parametrize(...) with values
                        # First check if the decorator has parentheses => it is a function call
                        # and then check, does the function call have a name attribute
                        # and is the attribute name "parametrize"
                        # and does it have at least 2 arguments (param names and values)
                        if (
                            isinstance(dec, ast.Call)
                            and hasattr(dec.func, "attr")
                            and dec.func.attr == "parametrize"
                            and len(dec.args) >= 2
                        ):
                            # Get the second argument (the list/tuple of parameter values)
                            val = dec.args[1]
                            # If it's a list or tuple, multiply test count by number of values
                            if isinstance(val, (ast.List, ast.Tuple)):
                                mult *= len(val.elts)

                    # Add the total tests from this function to the overall count
                    total += mult

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
