# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

PYTHON_TESTS = Path(__file__).parents[1] / "python_tests"


def _perf_backed_functional_modules():
    for perf_module in PYTHON_TESTS.glob("perf_*.py"):
        functional_module = perf_module.with_name(
            perf_module.name.removeprefix("perf_")
        )
        if functional_module.exists():
            yield functional_module


def _imports_name(tree, name):
    return any(
        (
            isinstance(node, ast.ImportFrom)
            and any((alias.asname or alias.name) == name for alias in node.names)
        )
        or (
            isinstance(node, ast.Import)
            and any((alias.asname or alias.name) == name for alias in node.names)
        )
        for node in ast.walk(tree)
    )


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _is_not_name(node, name):
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and _is_name(node.operand, name)
    )


def _contains_return(statements):
    return any(
        isinstance(node, ast.Return)
        for statement in statements
        for node in ast.walk(statement)
    )


def _result_accesses_guarded(function):
    early_return_lines = [
        statement.lineno
        for statement in function.body
        if isinstance(statement, ast.If)
        and _is_name(statement.test, "is_perf")
        and not statement.orelse
        and _contains_return(statement.body)
    ]

    guarded_accesses = set()
    for node in ast.walk(function):
        if isinstance(node, ast.If) and _is_not_name(node.test, "is_perf"):
            guarded_accesses.update(
                child.lineno
                for statement in node.body
                for child in ast.walk(statement)
                if isinstance(child, ast.Attribute) and child.attr == "test_results"
            )

    accesses = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute) and node.attr == "test_results"
    ]
    return all(
        line in guarded_accesses or any(guard < line for guard in early_return_lines)
        for line in accesses
    )


def test_perf_backed_runners_have_no_stale_configuration_references():
    failures = []
    for path in _perf_backed_functional_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        stale = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "variant_stimuli"
            and _is_name(node.value, "configuration")
        ]
        if stale:
            failures.append(f"{path.name}:{','.join(map(str, stale))}")
    assert not failures, "stale configuration.variant_stimuli access: " + "; ".join(
        failures
    )


def test_perf_backed_runners_import_test_config_when_referenced():
    failures = []
    for path in _perf_backed_functional_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        referenced = any(
            _is_name(node, "TestConfig") and isinstance(node.ctx, ast.Load)
            for node in ast.walk(tree)
        )
        if referenced and not _imports_name(tree, "TestConfig"):
            failures.append(path.name)
    assert not failures, "TestConfig referenced without import: " + ", ".join(failures)


def test_perf_backed_runners_guard_functional_results():
    failures = []
    for path in _perf_backed_functional_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            has_is_perf = any(
                argument.arg == "is_perf"
                for argument in (*function.args.args, *function.args.kwonlyargs)
            )
            has_results = any(
                isinstance(node, ast.Attribute) and node.attr == "test_results"
                for node in ast.walk(function)
            )
            if has_is_perf and has_results and not _result_accesses_guarded(function):
                failures.append(f"{path.name}:{function.name}")
    assert (
        not failures
    ), "functional result access is not guarded from perf runs: " + ", ".join(failures)


def test_perf_modules_do_not_export_imported_functional_tests():
    failures = []
    for path in PYTHON_TESTS.glob("perf_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not (
                node.module or ""
            ).startswith("test_"):
                continue
            for alias in node.names:
                local_name = alias.asname or alias.name
                if local_name.startswith("test_"):
                    failures.append(f"{path.name}:{local_name}")
    assert not failures, (
        "pytest would collect imported functional tests from perf modules: "
        + ", ".join(failures)
    )
