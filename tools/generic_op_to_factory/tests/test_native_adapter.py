# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only adapter tests using synthetic modules; no production identities."""

import sys
import types

import pytest

from tools.generic_op_to_factory.native_adapter import Route, resolve


@pytest.fixture
def modules(monkeypatch):
    source = types.ModuleType("synthetic_source")
    native = types.ModuleType("synthetic_native")
    runtime = types.ModuleType("ttnn")

    def original(value):
        return value + 1

    original.__module__ = source.__name__
    source.operation = original
    native.operation = lambda value: value * 2
    native.operation.is_cpp_operation = True
    runtime.generic_op = lambda: "original"
    for module in (source, native, runtime):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return source, native, runtime


@pytest.mark.parametrize("mode,expected", [("source", 4), ("native", 6)])
def test_explicit_routing_and_restore(modules, mode, expected):
    source, _, runtime = modules
    original, generic = source.operation, runtime.generic_op
    route = Route(
        {
            "source": "synthetic_source:operation",
            "native": "synthetic_native:operation",
            "mode": mode,
        }
    )
    route.install()
    try:
        assert source.operation(3) == expected
        assert route.calls == 1
        if mode == "native":
            with pytest.raises(  # allow-pytest.raises: host-only route guard
                AssertionError, match="must not fall back"
            ):
                runtime.generic_op()
    finally:
        route.restore()
    assert source.operation is original
    assert runtime.generic_op is generic


def test_package_and_defining_module_are_both_routed(modules, monkeypatch):
    source, _, _ = modules
    package = types.ModuleType("synthetic_package")
    package.operation = source.operation
    monkeypatch.setitem(sys.modules, package.__name__, package)
    route = Route(
        {
            "source": "synthetic_package:operation",
            "native": "synthetic_native:operation",
            "mode": "native",
        }
    )
    route.install()
    try:
        assert package.operation(3) == source.operation(3) == 6
        assert route.calls == 2
    finally:
        route.restore()
    assert package.operation is source.operation


@pytest.mark.parametrize("spec", ["missing_colon", "invalid-path:operation", "name:invalid.symbol"])
def test_invalid_entry_points(spec):
    with pytest.raises(ValueError):  # allow-pytest.raises: host-only parser validation
        resolve(spec)


def test_python_callable_cannot_be_certified_as_native(modules):
    source, native, _ = modules
    original = source.operation
    del native.operation.is_cpp_operation
    route = Route({"source": "synthetic_source:operation", "native": "synthetic_native:operation", "mode": "native"})
    with pytest.raises(ValueError, match="registered C\\+\\+ operation"):  # allow-pytest.raises: host-only route guard
        route.install()
    assert source.operation is original


@pytest.mark.parametrize("mode,expected", [("source", 4), ("native", 6)])
def test_explicit_top_level_alias_routes_and_restores(modules, mode, expected):
    source, _, runtime = modules
    original = source.operation
    runtime.public_alias = original
    route = Route(
        {
            "source": "synthetic_source:operation",
            "native": "synthetic_native:operation",
            "mode": mode,
            "aliases": ["ttnn:public_alias", "synthetic_source:operation"],
        }
    )
    route.install()
    try:
        assert runtime.public_alias is source.operation
        assert runtime.public_alias(3) == expected
        assert route.calls == 1
    finally:
        route.restore()
    assert runtime.public_alias is source.operation is original


def test_unrelated_alias_refused_before_any_route_mutation(modules):
    source, native, runtime = modules
    original, generic = source.operation, runtime.generic_op
    runtime.good_alias = original
    runtime.unrelated = native.operation
    route = Route(
        {
            "source": "synthetic_source:operation",
            "native": "synthetic_native:operation",
            "mode": "native",
            "aliases": ["ttnn:good_alias", "ttnn:unrelated"],
        }
    )
    with pytest.raises(ValueError, match="frozen source"):  # allow-pytest.raises: host-only alias guard
        route.install()
    assert runtime.good_alias is source.operation is original
    assert runtime.unrelated is native.operation
    assert runtime.generic_op is generic
    assert not route.restores


@pytest.mark.parametrize("aliases", ["not-a-list", [None], ["ttnn:alias", "ttnn:alias"]])
def test_invalid_alias_lists_rejected(aliases):
    with pytest.raises(ValueError, match="aliases"):  # allow-pytest.raises: host-only route config
        Route(
            {
                "source": "synthetic_source:operation",
                "native": "synthetic_native:operation",
                "mode": "native",
                "aliases": aliases,
            }
        )
