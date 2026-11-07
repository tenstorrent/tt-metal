# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

PYTHON_TESTS = Path(__file__).parents[1] / "python_tests"


def _perf_backed_functional_modules():
    for perf_module in PYTHON_TESTS.glob("perf_*.py"):
        functional_module = perf_module.with_name(
            perf_module.name.replace("perf_", "test_", 1)
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


def _implies_is_perf(node):
    return _is_name(node, "is_perf") or (
        isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.And)
        and any(_implies_is_perf(value) for value in node.values)
    )


def _result_accesses_guarded(function):
    early_return_lines = [
        statement.lineno
        for statement in function.body
        if isinstance(statement, ast.If)
        and _implies_is_perf(statement.test)
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
                if isinstance(child, ast.Attribute) and child.attr == "result"
            )

    accesses = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute) and node.attr == "result"
    ]
    return all(
        line in guarded_accesses or any(guard < line for guard in early_return_lines)
        for line in accesses
    )


def _undefined_configuration_accesses(function):
    accesses = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and node.attr == "variant_stimuli"
        and _is_name(node.value, "configuration")
    ]
    configuration_is_bound = any(
        _is_name(node, "configuration") and isinstance(node.ctx, ast.Store)
        for node in ast.walk(function)
    )
    return [] if configuration_is_bound else accesses


def test_perf_backed_functional_module_discovery_is_not_empty():
    modules = list(_perf_backed_functional_modules())
    assert modules
    assert all(path.name.startswith("test_") for path in modules)


def test_guard_analysis_handles_blaze_runner_shapes():
    safe = ast.parse(
        """
def run(is_perf, collect_result):
    if is_perf and not collect_result:
        return None
    return configuration.run().result
"""
    ).body[0]
    unsafe = ast.parse(
        """
def run(is_perf):
    return configuration.run().result
"""
    ).body[0]
    assert _result_accesses_guarded(safe)
    assert not _result_accesses_guarded(unsafe)


def test_stale_configuration_analysis_requires_a_local_binding():
    bound = ast.parse(
        """
def run():
    configuration = build()
    return configuration.variant_stimuli
"""
    ).body[0]
    stale = ast.parse(
        """
def run():
    outcome = execute_llk()
    return configuration.variant_stimuli
"""
    ).body[0]
    assert not _undefined_configuration_accesses(bound)
    assert _undefined_configuration_accesses(stale)


def test_perf_backed_runners_have_no_stale_configuration_references():
    failures = []
    for path in _perf_backed_functional_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            accesses = _undefined_configuration_accesses(function)
            if accesses:
                failures.append(
                    f"{path.name}:{function.name}:{','.join(map(str, accesses))}"
                )
    assert not failures, "undefined configuration.variant_stimuli access: " + "; ".join(
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
                isinstance(node, ast.Attribute) and node.attr == "result"
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


def test_execute_llk_canonicalizes_loop_factor(monkeypatch):
    from helpers.perf import blaze
    from helpers.test_variant_parameters import LOOP_FACTOR, NUM_FACES

    captured = {}

    class Configuration:
        def run(self):
            return "outcome"

    def create_configuration(**kwargs):
        captured.update(kwargs["test_config_kwargs"])
        return Configuration()

    monkeypatch.setattr(blaze, "create_test_or_perf_config", create_configuration)
    outcome = blaze.execute_llk(
        loop_factor=7,
        test_config_kwargs={
            "templates": [NUM_FACES(), LOOP_FACTOR(2)],
            "runtimes": [LOOP_FACTOR(3)],
        },
    )

    assert outcome == "outcome"
    assert not any(isinstance(param, LOOP_FACTOR) for param in captured["templates"])
    loop_factors = [
        param for param in captured["runtimes"] if isinstance(param, LOOP_FACTOR)
    ]
    assert [param.loop_factor for param in loop_factors] == [7]
