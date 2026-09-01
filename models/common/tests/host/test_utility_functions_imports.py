# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Import-dependency tests for models.common.utility_functions."""

import ast
from pathlib import Path


def test_pytest_is_not_imported_at_module_scope():
    utility_functions_path = Path(__file__).parents[2] / "utility_functions.py"
    syntax_tree = ast.parse(utility_functions_path.read_text())

    top_level_pytest_imports = [
        node
        for node in syntax_tree.body
        if (isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "pytest")
    ]

    assert not top_level_pytest_imports, "pytest must remain an optional test-only dependency"


def test_ti_skip_imports_pytest_lazily():
    utility_functions_path = Path(__file__).parents[2] / "utility_functions.py"
    syntax_tree = ast.parse(utility_functions_path.read_text())
    ti_skip = next(
        node for node in syntax_tree.body if isinstance(node, ast.FunctionDef) and node.name == "ti_skip"
    )

    assert any(
        isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names) for node in ti_skip.body
    ), "ti_skip must import pytest when the test helper is used"
