# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every KDA op test module under the standard (one that declares NOT_APPLICABLE) follows README.md.

Reads the modules' source, so it needs no device and no ttnn.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.ttnn.nightly.unit_tests.operations.experimental.kda.kda_op_checklist import (
    CHECKLIST,
    DEVICE_TIME_LEAVES,
    LEAVES,
    binding_of,
    item_for,
)

pytestmark = pytest.mark.host

OP_TEST_DIR = Path(__file__).parent
PERFORMANCE_DECORATOR = "checks.performance_test"


def _declares_not_applicable(tree: ast.Module) -> bool:
    return any(_assigned_name(node) == "NOT_APPLICABLE" for node in tree.body)


def _assigned_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _standard_modules() -> dict[str, ast.Module]:
    modules = {}
    for path in sorted(OP_TEST_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        if _declares_not_applicable(tree):
            modules[path.stem] = tree
    return modules


STANDARD_MODULES = _standard_modules()
standard_module = pytest.mark.parametrize("name", list(STANDARD_MODULES))


def _tests(tree: ast.Module) -> list[ast.FunctionDef]:
    return [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]


def _not_applicable(tree: ast.Module) -> dict[str, str]:
    node = next(node for node in tree.body if _assigned_name(node) == "NOT_APPLICABLE")
    return ast.literal_eval(node.value)


def _tested_leaves(name: str) -> set[str]:
    binding = binding_of(name)
    return {item.leaf for test in _tests(STANDARD_MODULES[name]) if (item := item_for(test.name, binding))}


def test_at_least_one_module_follows_the_standard() -> None:
    assert STANDARD_MODULES, "no op test module declares NOT_APPLICABLE; the checks below would pass vacuously"


def test_each_checklist_name_matches_exactly_one_item() -> None:
    """A name with its patterns' word parts filled in belongs to its own item only, so groups cannot overlap."""
    for item in CHECKLIST:
        example = re.sub(r"\\w\+", "x", item.suffix)
        owners = [other.leaf for other in CHECKLIST if other.matches(example)]
        assert owners == [item.leaf], f"{example!r} matches {owners}"


@standard_module
def test_every_test_name_is_in_the_vocabulary(name: str) -> None:
    binding = binding_of(name)
    outside = [test.name for test in _tests(STANDARD_MODULES[name]) if item_for(test.name, binding) is None]
    assert not outside, f"{name}: rename to test_{binding}_<checklist suffix> (kda_op_checklist.py): {outside}"


@standard_module
def test_every_enforced_item_is_tested_or_declared_not_applicable(name: str) -> None:
    enforced = {item.leaf for item in CHECKLIST if item.enforced}
    missing = enforced - _tested_leaves(name) - _not_applicable(STANDARD_MODULES[name]).keys()
    assert not missing, f"{name}: add tests for {sorted(missing)} or declare them in NOT_APPLICABLE with a reason"


@standard_module
def test_not_applicable_declarations_name_untested_items_with_reasons(name: str) -> None:
    declared = _not_applicable(STANDARD_MODULES[name])
    unknown = declared.keys() - LEAVES.keys()
    assert not unknown, f"{name}: NOT_APPLICABLE names unknown checklist items {sorted(unknown)}"
    blank = [leaf for leaf, reason in declared.items() if not isinstance(reason, str) or not reason.strip()]
    assert not blank, f"{name}: NOT_APPLICABLE needs a reason for {blank}"
    contradicted = declared.keys() & _tested_leaves(name)
    assert not contradicted, f"{name}: {sorted(contradicted)} are declared not applicable but tested"


@standard_module
def test_device_time_tests_carry_the_performance_decorator(name: str) -> None:
    binding = binding_of(name)
    undecorated = [
        test.name
        for test in _tests(STANDARD_MODULES[name])
        if (item := item_for(test.name, binding)) is not None
        and item.leaf in DEVICE_TIME_LEAVES
        and PERFORMANCE_DECORATOR not in {ast.unparse(decorator) for decorator in test.decorator_list}
    ]
    assert not undecorated, f"{name}: decorate with @{PERFORMANCE_DECORATOR}: {undecorated}"


@standard_module
def test_no_test_module_imports_another(name: str) -> None:
    """A test module is not an API: shared code lives in a utilities module."""
    imported = []
    for node in ast.walk(STANDARD_MODULES[name]):
        if isinstance(node, ast.ImportFrom):
            modules = [node.module or ""] + [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        imported += [module for module in modules if module.rsplit(".", 1)[-1].startswith("test_")]
    assert not imported, f"{name}: imports test modules {imported}"
