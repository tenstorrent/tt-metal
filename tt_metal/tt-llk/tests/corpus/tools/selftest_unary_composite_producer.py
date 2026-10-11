#!/usr/bin/env python3
"""Static contract checks for multi-configuration unary admission nodes."""

import ast
import copy
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "python_tests" / "test_sfpu_unary.py"


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def call_name(call: ast.Call) -> str:
    return call.func.id if isinstance(call.func, ast.Name) else ""


def main() -> None:
    tree = ast.parse(SOURCE.read_text())
    prepare_helper = function(tree, "_producer_prepare_only")
    executable = ast.Module(body=[copy.deepcopy(prepare_helper)], type_ignores=[])
    ast.fix_missing_locations(executable)

    class FakeBuildMode:
        PRODUCE = object()
        DEFAULT = object()
        CONSUME = object()

    class FakeTestConfig:
        BUILD_MODE = FakeBuildMode.DEFAULT

    namespace = {"BuildMode": FakeBuildMode, "TestConfig": FakeTestConfig}
    exec(compile(executable, str(SOURCE), "exec"), namespace)
    prepare = namespace["_producer_prepare_only"]

    class Configuration:
        def __init__(self):
            self.prepared = 0

        def prepare(self):
            self.prepared += 1

    for mode in (FakeBuildMode.DEFAULT, FakeBuildMode.CONSUME):
        FakeTestConfig.BUILD_MODE = mode
        configuration = Configuration()
        assert prepare(configuration, False) is False
        assert configuration.prepared == 0
        try:
            prepare(configuration, True)
        except RuntimeError as error:
            assert "compile-producer" in str(error)
        else:
            raise AssertionError("prepare_only was accepted outside producer mode")
        assert configuration.prepared == 0

    FakeTestConfig.BUILD_MODE = FakeBuildMode.PRODUCE
    configuration = Configuration()
    assert prepare(configuration, True) is True
    assert configuration.prepared == 1

    helper = function(tree, "eltwise_unary_sfpu")
    defaults = dict(
        zip(
            [arg.arg for arg in helper.args.args[-len(helper.args.defaults) :]],
            helper.args.defaults,
        )
    )
    assert isinstance(defaults.get("prepare_only"), ast.Constant)
    assert defaults["prepare_only"].value is False

    prepare_calls = [
        (index, statement)
        for index, statement in enumerate(helper.body)
        if "_producer_prepare_only(configuration, prepare_only)"
        in ast.unparse(statement)
    ]
    assert len(prepare_calls) == 1
    guard_index, guard = prepare_calls[0]
    assert isinstance(guard, ast.If)
    assert isinstance(guard.body[-1], ast.Return)
    run_indices = [
        index
        for index, statement in enumerate(helper.body)
        if "configuration.run()" in ast.unparse(statement)
    ]
    assert run_indices and min(run_indices) > guard_index
    raw_hook_indices = [
        index
        for index, statement in enumerate(helper.body)
        if "SFPU_RAW_A" in ast.unparse(statement)
        or "SFPU_STREAM" in ast.unparse(statement)
    ]
    assert raw_hook_indices and min(raw_hook_indices) > guard_index

    expected_calls = {
        "test_gelu_appx_licensed_cpp": 3,
        "test_relu_fresh_cpp_admission": 2,
        "test_hardshrink_fresh_cpp_admission": 3,
    }
    for name, count in expected_calls.items():
        test = function(tree, name)
        calls = [
            node
            for node in ast.walk(test)
            if isinstance(node, ast.Call) and call_name(node) == "eltwise_unary_sfpu"
        ]
        assert len(calls) == count
        for call in calls:
            keyword = next(
                (kw for kw in call.keywords if kw.arg == "prepare_only"), None
            )
            assert keyword is not None
            assert isinstance(keyword.value, ast.Name)
            assert keyword.value.id == "prepare_only"
        assignments = [
            node
            for node in test.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "prepare_only"
                for target in node.targets
            )
        ]
        assert len(assignments) == 1
        assert ast.unparse(assignments[0].value) == (
            "TestConfig.BUILD_MODE == BuildMode.PRODUCE"
        )

    print("unary composite producer selftest: PASS")


if __name__ == "__main__":
    main()
