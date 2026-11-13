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


def _eval_bool(node, state):
    if isinstance(node, ast.Name):
        return state.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _eval_bool(node.operand, state)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_eval_bool(value, state) for value in node.values]
        if isinstance(node.op, ast.And):
            return False if False in values else (True if all(values) else None)
        return (
            True
            if True in values
            else (False if all(value is False for value in values) else None)
        )
    return None


def _refine_bool(state, node, expected):
    state = dict(state)
    value = _eval_bool(node, state)
    if value is not None:
        return state if value == expected else None
    if isinstance(node, ast.Name):
        state[node.id] = expected
        return state
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _refine_bool(state, node.operand, not expected)
    if isinstance(node, ast.BoolOp):
        refine_all = (isinstance(node.op, ast.And) and expected) or (
            isinstance(node.op, ast.Or) and not expected
        )
        if refine_all:
            for child in node.values:
                state = _refine_bool(state, child, expected)
                if state is None:
                    return None
            return state
        unresolved = [
            child for child in node.values if _eval_bool(child, state) is None
        ]
        decisive = isinstance(node.op, ast.Or)
        if len(unresolved) == 1:
            return _refine_bool(state, unresolved[0], decisive)
    return state


def _contains_result_access(node):
    return any(
        isinstance(child, ast.Attribute) and child.attr == "result"
        for child in ast.walk(node)
    )


def _perf_paths_reach_result(statements, states):
    states = list(states)
    for statement in statements:
        if not states:
            break
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(statement, ast.If):
            if _contains_result_access(statement.test):
                return True, []
            remaining = []
            for state in states:
                true_state = _refine_bool(state, statement.test, True)
                false_state = _refine_bool(state, statement.test, False)
                for branch, branch_state in (
                    (statement.body, true_state),
                    (statement.orelse, false_state),
                ):
                    if branch_state is None:
                        continue
                    failed, paths = _perf_paths_reach_result(branch, [branch_state])
                    if failed:
                        return True, []
                    remaining.extend(paths)
            states = remaining
            continue
        if _contains_result_access(statement):
            return True, []
        if isinstance(statement, (ast.Return, ast.Raise)):
            states = []
    return False, states


def _result_accesses_guarded(function):
    reaches_result, _ = _perf_paths_reach_result(function.body, [{"is_perf": True}])
    return not reaches_result


def _configuration_accesses(node):
    return [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr == "variant_stimuli"
        and _is_name(child.value, "configuration")
    ]


def _binds_configuration(statement):
    def binds_name(target):
        if _is_name(target, "configuration"):
            return True
        if isinstance(target, (ast.List, ast.Tuple)):
            return any(binds_name(element) for element in target.elts)
        if isinstance(target, ast.Starred):
            return binds_name(target.value)
        return False

    targets = []
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
    return any(binds_name(target) for target in targets)


def _configuration_accesses_in_block(statements, bound=False):
    failures = []
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(statement, ast.If):
            if not bound:
                failures.extend(_configuration_accesses(statement.test))
            body_failures, body_bound = _configuration_accesses_in_block(
                statement.body, bound
            )
            else_failures, else_bound = _configuration_accesses_in_block(
                statement.orelse, bound
            )
            failures.extend(body_failures)
            failures.extend(else_failures)
            continuing = [
                branch_bound
                for branch_bound in (body_bound, else_bound)
                if branch_bound is not None
            ]
            if not continuing:
                return failures, None
            bound = all(continuing)
            continue
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            failures.extend(_configuration_accesses_in_block(statement.body, bound)[0])
            failures.extend(
                _configuration_accesses_in_block(statement.orelse, bound)[0]
            )
            continue
        if isinstance(statement, ast.Try):
            failures.extend(_configuration_accesses_in_block(statement.body, bound)[0])
            for handler in statement.handlers:
                failures.extend(
                    _configuration_accesses_in_block(handler.body, bound)[0]
                )
            failures.extend(
                _configuration_accesses_in_block(statement.orelse, bound)[0]
            )
            failures.extend(
                _configuration_accesses_in_block(statement.finalbody, bound)[0]
            )
            continue
        if not bound:
            failures.extend(_configuration_accesses(statement))
        if _binds_configuration(statement):
            bound = True
        if isinstance(statement, (ast.Return, ast.Raise)):
            return failures, None
    return failures, bound


def _undefined_configuration_accesses(function):
    return _configuration_accesses_in_block(function.body)[0]


def test_perf_backed_functional_module_discovery_is_not_empty():
    modules = list(_perf_backed_functional_modules())
    assert modules
    assert all(path.name.startswith("test_") for path in modules)


def test_guard_analysis_handles_blaze_runner_shapes():
    safe = ast.parse(
        """
def run(is_perf):
    if is_perf:
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
    partial = ast.parse(
        """
def run(is_perf, collect_result):
    if is_perf:
        if not collect_result:
            return None
    return configuration.run().result
"""
    ).body[0]
    assert _result_accesses_guarded(safe)
    assert not _result_accesses_guarded(unsafe)
    assert not _result_accesses_guarded(partial)


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
    late = ast.parse(
        """
def run():
    result = configuration.variant_stimuli
    configuration = build()
    return result
"""
    ).body[0]
    conditional = ast.parse(
        """
def run(use_new):
    if use_new:
        configuration = build()
    return configuration.variant_stimuli
"""
    ).body[0]
    attribute_target = ast.parse(
        """
def run():
    configuration.runtimes = []
    return configuration.variant_stimuli
"""
    ).body[0]
    terminating_branch = ast.parse(
        """
def run(use_existing):
    if use_existing:
        return None
    configuration = build()
    return configuration.variant_stimuli
"""
    ).body[0]
    assert not _undefined_configuration_accesses(bound)
    assert _undefined_configuration_accesses(stale)
    assert _undefined_configuration_accesses(late)
    assert _undefined_configuration_accesses(conditional)
    assert _undefined_configuration_accesses(attribute_target)
    assert not _undefined_configuration_accesses(terminating_branch)


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
