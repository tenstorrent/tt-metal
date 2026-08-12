# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from conftest import (
    _collapse_runtime_only_variants,
    _group_runtime_variants_by_compile_key,
    pytest_collection_modifyitems,
)
from helpers.param_config import (
    RUNTIME_AXES_MARK,
    CircularDependencyError,
    UnknownDependenciesError,
    _compute_dependency_map,
    _compute_dependency_matrix,
    _compute_resolution_order,
    _param_dependencies,
    _params_solve_dependencies,
    _verify_dependency_map,
    compile_item_key,
    parametrize,
    runtime,
)
from helpers.test_config import BuildMode, TestConfig


class _FakeItem:
    def __init__(self, nodeid, params=None, marker=None):
        self.nodeid = nodeid
        self.callspec = SimpleNamespace(params=params) if params is not None else None
        self._marker = marker

    def get_closest_marker(self, name):
        if self._marker is not None and self._marker.name == name:
            return self._marker
        return None


def _runtime_axes_marker(**params):
    def test_case(**kwargs):
        pass

    decorated = parametrize(**params)(test_case)
    return next(
        marker for marker in decorated.pytestmark if marker.name == RUNTIME_AXES_MARK
    )


def test_param_dependencies_constant():
    """
    Parameters can have constant values attached to them.
    Since these are constants, the parameter has no dependencies.
    """
    constraint = 1
    dependencies = _param_dependencies("constraint", constraint)
    assert len(dependencies) == 0
    assert dependencies == []


def test_param_dependencies_list():
    """
    Parameters can have a list of values attached to them.
    This is used to represent a parameter that takes multiple values.
    Since this is a list of constants, the parameter has no dependencies.
    """
    constraint = [1, 2, 3]
    assert _param_dependencies("constraint", constraint) == []


def test_param_dependencies_empty_lambda():
    """
    Parameters can have a lambda function attached to them.
    The parameters of the lambda function are treated as dependencies of the parameter.
    """
    constraint = lambda: []
    assert _param_dependencies("constraint", constraint) == []


def test_param_dependencies_multiple_lambda():
    """
    Parameters can have a lambda function attached to them.
    The parameters of the lambda function are treated as dependencies of the parameter.
    """
    constraint = lambda arg0, arg1, arg2: []
    dependencies = _param_dependencies("constraint", constraint)

    assert len(dependencies) == 3
    assert dependencies[0] == "arg0"
    assert dependencies[1] == "arg1"
    assert dependencies[2] == "arg2"


def test_verify_dependency_map_pass():
    """
    The dependency map is valid if all dependencies are known parameters.
    """
    dependency_map = {
        "exist1": ["exist2", "exist3"],
        "exist2": ["exist3"],
        "exist3": [],
        "exist4": ["exist1"],
    }

    try:
        # no exception should be raised
        _verify_dependency_map(dependency_map)
    except Exception as e:
        assert False, f"Unexpected exception raised: {e}"


def test_verify_dependency_map_fail():
    """
    The dependency map is invalid if a dependency is not a known parameter.
    """
    dependency_map = {
        "exist1": ["exist2", "missing1"],
        "exist2": ["missing2", "exist3"],
        "exist3": [],
        "exist4": ["missing1", "missing2"],
    }

    expected_missing = {
        "exist1": {"missing1"},
        "exist2": {"missing2"},
        "exist4": {"missing1", "missing2"},
    }

    with pytest.raises(UnknownDependenciesError) as error:
        _verify_dependency_map(dependency_map)

    assert error.value.missing == expected_missing


def test_compute_dependency_matrix_pass():
    """
    The dependency matrix is valid if all dependencies are known parameters.
    """

    expected = {
        "exist1": ["exist2", "exist3"],
        "exist2": ["exist3"],
        "exist3": [],
        "exist4": ["exist1"],
        "exist5": [],
    }

    dependency_map = _compute_dependency_map(
        exist1=lambda exist2, exist3: [],
        exist2=lambda exist3: [],
        exist3="value",
        exist4=lambda exist1: [],
        exist5=["value", "value"],
    )

    assert dependency_map == expected


def test_compute_dependency_matrix_fail():
    """
    The dependency matrix is invalid if a dependency is not a known parameter.
    """

    expected_missing = {
        "exist1": {"missing1"},
        "exist2": {"missing2"},
        "exist4": {"missing1", "missing2"},
    }

    with pytest.raises(UnknownDependenciesError) as error:
        _compute_dependency_map(
            exist1=lambda exist2, missing1: [],
            exist2=lambda missing2, exist3: [],
            exist3="value",
            exist4=lambda missing1, missing2: [],
            exist5=["value", "value2"],
        )

    assert error.value.missing == expected_missing


def test_compute_dependency_matrix():
    """
    The dependency matrix is computed correctly.
    """

    expected = [
        [1, 2],
        [2],
        [],
        [0],
        [],
    ]

    matrix = _compute_dependency_matrix(
        exist1=lambda exist2, exist3: [],
        exist2=lambda exist3: [],
        exist3="value",
        exist4=lambda exist1: [],
        exist5=["value", "value"],
    )

    assert matrix == expected


def test_compute_resolution_order_pass():
    """
    The resolution order is computed correctly.
    """

    def verify_resolution_order(matrix: list[list[int]], order: list[int]):
        resolved = [False] * len(matrix)
        for i in order:
            if not all(resolved[j] for j in matrix[i]):
                return False
            resolved[i] = True
        return True

    kwargs = {
        "exist1": lambda exist2, exist3: [],
        "exist2": lambda exist3: [],
        "exist3": "value",
        "exist4": lambda exist1: [],
        "exist5": ["value", "value"],
    }

    parameters = list(kwargs.keys())
    matrix = _compute_dependency_matrix(**kwargs)

    order = _compute_resolution_order(parameters, matrix)

    assert verify_resolution_order(matrix, order)


def test_compute_resolution_order_fail_circular():
    """
    The resolution order is invalid if there is a circular dependency
    """

    kwargs = {
        "exist1": lambda exist2, exist3: [],
        "exist2": lambda exist3, exist4: [],
        "exist3": "value",
        "exist4": lambda exist1: [],
        "exist5": ["value", "value"],
    }

    parameters = list(kwargs.keys())
    matrix = _compute_dependency_matrix(**kwargs)

    with pytest.raises(CircularDependencyError) as error:
        _compute_resolution_order(parameters, matrix)

    expected_cycle = ["exist1", "exist2", "exist4"]
    cycle = error.value.cycle

    assert len(cycle) == len(expected_cycle)
    assert set(cycle) == set(expected_cycle)


def test_params_solve_dependencies_no_constraints_values():
    """
    Test case where no constrain functions are provided.
    Parameters are values, not lists.
    Result should be a single value.
    """

    result = _params_solve_dependencies(
        b=1,
        c=10,
        d=100,
    )

    expected = [
        (1, 10, 100),
    ]

    assert len(result) == len(expected)
    assert set(result) == set(expected)


def test_params_solve_dependencies_no_constraints_lists():
    """
    Test case where no constrain functions are provided.
    Result should be a full cartesian product of the parameters.
    """

    result = _params_solve_dependencies(b=[1, 2], c=[10, 20], d=[100, 200])

    expected = [
        (1, 10, 100),
        (1, 10, 200),
        (1, 20, 100),
        (1, 20, 200),
        (2, 10, 100),
        (2, 10, 200),
        (2, 20, 100),
        (2, 20, 200),
    ]

    assert len(result) == len(expected)
    assert set(result) == set(expected)


def test_params_solve_dependencies_single_constraint_empty():
    """
    Test with a simple constraint where b returns an empty list.
    """
    result = _params_solve_dependencies(
        a=[1, 2, 3],
        b=lambda a: [],
    )

    assert len(result) == 0


def test_params_solve_dependencies_single_constraint_value():
    """
    Test with a simple constraint where b depends on a.
    Constraint function returns a single value.
    """
    result = _params_solve_dependencies(
        a=[1, 2, 3],
        b=lambda a: a * a,
    )

    expected = [
        (1, 1),
        (2, 4),
        (3, 9),
    ]

    assert len(result) == len(expected)
    assert set(result) == set(expected)


def test_params_solve_dependencies_single_constraint_list():
    """
    Test with a simple constraint where b depends on a.
    Constraint function returns a list of values.
    """
    result = _params_solve_dependencies(
        a=[1, 2, 3],
        b=lambda a: [i * i for i in range(1, a + 1)],
    )

    expected = [
        (1, 1),
        (2, 1),
        (2, 4),
        (3, 1),
        (3, 4),
        (3, 9),
    ]

    assert len(result) == len(expected)
    assert set(result) == set(expected)


def test_params_solve_dependencies_single_constraint_multiple_dependencies():
    """
    Test constraint that test if constraint function is called with the correct dependencies.
    """
    result = _params_solve_dependencies(
        a=[1, 2, 3], b=[1, 2, 3, 4], c=lambda b, a: [b - a]
    )

    # Verify completeness - count valid combinations
    expected = [
        (1, 1, 0),
        (1, 2, 1),
        (1, 3, 2),
        (1, 4, 3),
        (2, 1, -1),
        (2, 2, 0),
        (2, 3, 1),
        (2, 4, 2),
        (3, 1, -2),
        (3, 2, -1),
        (3, 3, 0),
        (3, 4, 1),
    ]

    assert len(result) == len(expected)
    assert set(result) == set(expected)


def test_params_solve_dependencies_multiple_chain_constraints_ordered():
    """
    Test with multiple chain dependencies: z depends on y, y depends on x.
    """
    result = _params_solve_dependencies(
        x=[1, 2],
        y=lambda x: [x * 2, x * 3],
        z=lambda y: [y * 5, y * 7],
    )

    expected = [
        (1, 2, 10),
        (1, 2, 14),
        (1, 3, 15),
        (1, 3, 21),
        (2, 4, 20),
        (2, 4, 28),
        (2, 6, 30),
        (2, 6, 42),
    ]

    assert len(result) == len(expected)
    assert set(result) == set(expected)


def test_params_solve_dependencies_multiple_chain_constraints_shuffled():
    """
    Test with multiple chain dependencies: z depends on y, y depends on x.
    """
    result = _params_solve_dependencies(
        z=lambda y: [y + 1, y + 2],
        x=[1, 2],
        y=lambda x: [x * 10, x * 20],
    )

    expected = [
        (11, 1, 10),
        (12, 1, 10),
        (21, 1, 20),
        (22, 1, 20),
        (21, 2, 20),
        (22, 2, 20),
        (41, 2, 40),
        (42, 2, 40),
    ]

    assert len(result) == len(expected)
    assert set(result) == set(expected)


def test_compile_item_key_keeps_normal_axes_and_drops_runtime_axes():
    marker = _runtime_axes_marker(
        compile_axis=["first", "second"], runtime_axis=runtime([10, 20])
    )

    first = _FakeItem(
        "test_module.py::test_case[first-10]",
        {"compile_axis": "first", "runtime_axis": 10},
        marker,
    )
    same_compile = _FakeItem(
        "test_module.py::test_case[first-20]",
        {"compile_axis": "first", "runtime_axis": 20},
        marker,
    )
    other_compile = _FakeItem(
        "test_module.py::test_case[second-10]",
        {"compile_axis": "second", "runtime_axis": 10},
        marker,
    )

    assert compile_item_key(first) == compile_item_key(same_compile)
    assert compile_item_key(first) != compile_item_key(other_compile)


def test_compile_item_key_uses_resolved_callable_dependencies():
    marker = _runtime_axes_marker(
        source=[1, 2],
        derived=lambda source: source * 10,
        sample=runtime(["a", "b"]),
    )

    first = _FakeItem(
        "test_module.py::test_case[first]",
        {"source": 1, "derived": 10, "sample": "a"},
        marker,
    )
    same_compile = _FakeItem(
        "test_module.py::test_case[second]",
        {"source": 1, "derived": 10, "sample": "b"},
        marker,
    )
    other_compile = _FakeItem(
        "test_module.py::test_case[third]",
        {"source": 2, "derived": 20, "sample": "a"},
        marker,
    )

    assert compile_item_key(first) == compile_item_key(same_compile)
    assert compile_item_key(first) != compile_item_key(other_compile)


def test_compile_item_key_drops_only_runtime_parts_of_tuple_axis():
    marker = _runtime_axes_marker(
        combo=[
            ("compile-a", runtime("run-1")),
            ("compile-a", runtime("run-2")),
            ("compile-b", runtime("run-1")),
        ]
    )

    first = _FakeItem(
        "test_module.py::test_case[first]", {"combo": ("compile-a", "run-1")}, marker
    )
    same_compile = _FakeItem(
        "test_module.py::test_case[second]",
        {"combo": ("compile-a", "run-2")},
        marker,
    )
    other_compile = _FakeItem(
        "test_module.py::test_case[third]", {"combo": ("compile-b", "run-1")}, marker
    )

    assert compile_item_key(first) == compile_item_key(same_compile)
    assert compile_item_key(first) != compile_item_key(other_compile)


def test_compile_item_key_ignores_items_without_runtime_metadata():
    item = _FakeItem("test_module.py::test_case[value]", {"axis": "value"})

    assert compile_item_key(item) is None


def test_producer_collapse_keeps_one_item_per_compile_key():
    marker = _runtime_axes_marker(compile_axis=["a", "b"], runtime_axis=runtime([1, 2]))
    items = [
        _FakeItem(
            "test_module.py::test_case[a-1]",
            {"compile_axis": "a", "runtime_axis": 1},
            marker,
        ),
        _FakeItem(
            "test_module.py::test_case[a-2]",
            {"compile_axis": "a", "runtime_axis": 2},
            marker,
        ),
        _FakeItem(
            "test_module.py::test_case[b-1]",
            {"compile_axis": "b", "runtime_axis": 1},
            marker,
        ),
    ]
    deselected = []
    config = SimpleNamespace(
        hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items))
    )

    _collapse_runtime_only_variants(config, items)

    assert [item.nodeid for item in items] == [
        "test_module.py::test_case[a-1]",
        "test_module.py::test_case[b-1]",
    ]
    assert [item.nodeid for item in deselected] == ["test_module.py::test_case[a-2]"]


def test_consumer_grouping_is_stable_within_each_test_function():
    marker = _runtime_axes_marker(compile_axis=["a", "b"], runtime_axis=runtime([1, 2]))

    def item(function, item_id, compile_axis, runtime_axis):
        return _FakeItem(
            f"test_module.py::{function}[{item_id}]",
            {"compile_axis": compile_axis, "runtime_axis": runtime_axis},
            marker,
        )

    items = [
        item("test_first", "a-1", "a", 1),
        item("test_first", "b-1", "b", 1),
        item("test_first", "a-2", "a", 2),
        _FakeItem("test_module.py::test_unmarked[value]", {"axis": "value"}),
        item("test_second", "b-1", "b", 1),
        item("test_second", "a-1", "a", 1),
        item("test_second", "b-2", "b", 2),
    ]

    _group_runtime_variants_by_compile_key(items)

    assert [item.nodeid for item in items] == [
        "test_module.py::test_first[a-1]",
        "test_module.py::test_first[a-2]",
        "test_module.py::test_first[b-1]",
        "test_module.py::test_unmarked[value]",
        "test_module.py::test_second[b-1]",
        "test_module.py::test_second[b-2]",
        "test_module.py::test_second[a-1]",
    ]


def test_explicit_order_file_takes_precedence_over_consumer_grouping(
    monkeypatch, tmp_path
):
    marker = _runtime_axes_marker(compile_axis=["a", "b"], runtime_axis=runtime([1, 2]))
    items = [
        _FakeItem(
            "test_module.py::test_case[a-1]",
            {"compile_axis": "a", "runtime_axis": 1},
            marker,
        ),
        _FakeItem(
            "test_module.py::test_case[b-1]",
            {"compile_axis": "b", "runtime_axis": 1},
            marker,
        ),
        _FakeItem(
            "test_module.py::test_case[a-2]",
            {"compile_axis": "a", "runtime_axis": 2},
            marker,
        ),
    ]
    requested_order = [
        "test_module.py::test_case[b-1]",
        "test_module.py::test_case[a-2]",
        "test_module.py::test_case[a-1]",
    ]
    order_file = tmp_path / "order.json"
    order_file.write_text(
        json.dumps({"runner": [{"test": nodeid} for nodeid in requested_order]})
    )

    options = {
        "--op": [],
        "--test-order-file": str(order_file),
        "--rewind-runner": "runner",
    }
    config = SimpleNamespace(getoption=lambda name, **kwargs: options[name])
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)

    pytest_collection_modifyitems(config, items)

    assert [item.nodeid for item in items] == requested_order
