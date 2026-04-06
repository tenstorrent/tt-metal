#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import sys
from pathlib import Path


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


def main():
    ttnn_dir = Path(__file__).parent.parent / "ttnn" / "ttnn"
    if not ttnn_dir.exists():
        print(f"Error: ttnn directory not found at {ttnn_dir}")
        sys.exit(1)

    exceptions = [
        "examples",
        "model_preprocessing.py",
        "torch_tracer.py",
    ]
    exceptions_paths = [os.path.join(ttnn_dir, path) for path in exceptions]

    violations = []

    for root, _, files in os.walk(ttnn_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)

                # Check if this specific file is in exceptions list
                if any(filepath.startswith(wp) for wp in exceptions_paths):
                    continue

                imports = check_file(filepath)
                if imports:
                    for line, import_stmt in imports:
                        violations.append(f"{filepath}:{line}: {import_stmt}")

    if violations:
        print("Global torch imports found:")
        for violation in violations:
            print(violation)
        sys.exit(1)
    else:
        print("No global torch imports found in ttnn/ directory.")
        sys.exit(0)


if __name__ == "__main__":
    main()
