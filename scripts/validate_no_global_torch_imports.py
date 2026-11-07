#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reject a module-level torch import under ttnn/ttnn.

With filenames, only those files are checked. This is what pre-commit passes.
With no filenames, or when this script is among them, the whole ttnn/ttnn tree is scanned.
"""

import ast
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
TTNN_DIR = SCRIPT.parent.parent / "ttnn" / "ttnn"
# Allowed to import torch at module scope, relative to TTNN_DIR.
EXCEPTION_DIRS = ("examples",)
EXCEPTION_FILES = (
    "model_preprocessing.py",
    "torch_tracer.py",
)


class TorchImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.global_torch_imports = []
        self.scope_level = 0  # Track scope level: we care about imports in scope 0 (global)

    def visit_Import(self, node):
        if self.scope_level == 0:
            for alias in node.names:
                if alias.name == "torch":
                    self.global_torch_imports.append((node.lineno, f"import {alias.name}"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if self.scope_level == 0 and node.module == "torch":
            self.global_torch_imports.append((node.lineno, f"from {node.module} import ..."))
        self.generic_visit(node)

    # Track function/class/lambda definitions to update scope level

    def visit_FunctionDef(self, node):
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1

    def visit_ClassDef(self, node):
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1

    def visit_Lambda(self, node):
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1


def ttnn_relative(path):
    """The path relative to TTNN_DIR, or None when it lies outside."""
    try:
        return path.resolve().relative_to(TTNN_DIR)
    except ValueError:
        return None


def is_exception(rel):
    if rel.parts and rel.parts[0] in EXCEPTION_DIRS:
        return True
    return rel.as_posix() in EXCEPTION_FILES


def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        try:
            code = file.read()
            tree = ast.parse(code)
            visitor = TorchImportVisitor()
            visitor.visit(tree)
            return visitor.global_torch_imports
        except SyntaxError:
            return [(0, f"Failed to parse {filepath}")]


def iter_files(filenames):
    if filenames:
        for name in filenames:
            path = Path(name)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.suffix != ".py" or not path.is_file():
                continue
            rel = ttnn_relative(path)
            if rel is not None and not is_exception(rel):
                yield path
        return
    for path in TTNN_DIR.rglob("*.py"):
        rel = ttnn_relative(path)
        if path.is_file() and rel is not None and not is_exception(rel):
            yield path


def main(argv=None):
    if not TTNN_DIR.is_dir():
        print(f"Error: ttnn directory not found at {TTNN_DIR}")
        return 1

    filenames = sys.argv[1:] if argv is None else argv
    if any(Path(name).resolve() == SCRIPT for name in filenames):
        filenames = []
    violations = []
    for filepath in iter_files(filenames):
        for line, import_stmt in check_file(filepath):
            violations.append(f"{filepath}:{line}: {import_stmt}")

    if violations:
        print("Global torch imports found:")
        for violation in violations:
            print(violation)
        return 1

    if filenames:
        print("No global torch imports found in the given files.")
    else:
        print("No global torch imports found in ttnn/ directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
