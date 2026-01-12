#!/usr/bin/env python3
"""Count test functions including parametrize expansions."""

import ast
import sys
from pathlib import Path


def count_tests_in_directory(directory):
    """Count all test cases in a directory, including parametrized expansions."""
    total = 0
    
    for filepath in Path(directory).rglob("test_*.py"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(filepath))
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith('test_'):
                        # Count parametrize decorators
                        mult = 1
                        for dec in node.decorator_list:
                            if isinstance(dec, ast.Call):
                                if (hasattr(dec.func, 'attr') and 
                                    dec.func.attr == 'parametrize' and 
                                    len(dec.args) >= 2):
                                    val = dec.args[1]
                                    if isinstance(val, (ast.List, ast.Tuple)):
                                        mult *= len(val.elts)
                        total += mult
        except Exception:
            # Skip files that can't be parsed
            pass
    
    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: count_tests.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    count = count_tests_in_directory(directory)
    print(count)
