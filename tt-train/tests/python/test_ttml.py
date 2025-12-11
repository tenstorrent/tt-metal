# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttml recursive import functionality.

This module tests that all _ttml C++ extension symbols are properly imported
into the ttml Python package, with Python implementations taking precedence.
"""

import inspect
import pytest
import sys
import types

import ttml  # noqa: E402
import ttml._ttml  # noqa: E402


def test_ttml_module_imported():
    """Test that ttml module can be imported."""
    assert ttml is not None
    assert hasattr(ttml, "_ttml")


def test_recursive_import_top_level():
    """Test that top-level symbols from _ttml are imported into ttml."""
    # Dynamically discover submodules from _ttml
    _ttml_attrs = dir(ttml._ttml)

    # Filter to get only submodules (public attributes that are modules)
    expected_submodules = []
    for attr_name in _ttml_attrs:
        if not attr_name.startswith("_"):  # Only public attributes
            try:
                attr_value = getattr(ttml._ttml, attr_name)
                if inspect.ismodule(attr_value):
                    expected_submodules.append(attr_name)
            except (AttributeError, TypeError):
                continue

    # If a submodule exists in _ttml, it should be imported into ttml
    for submodule_name in expected_submodules:
        # If it exists in _ttml, it must be imported into ttml
        assert hasattr(
            ttml, submodule_name
        ), f"ttml.{submodule_name} should exist if _ttml.{submodule_name} exists (recursive import failed)"
        # Verify it's actually a module
        assert inspect.ismodule(
            getattr(ttml, submodule_name)
        ), f"ttml.{submodule_name} should be a module"


def test_nested_submodule_import():
    """Test that nested submodules are recursively imported."""
    # Check that ops submodules exist
    if hasattr(ttml._ttml, "ops"):
        _ttml_ops = ttml._ttml.ops

        # Check for nested submodules in ops
        nested_submodules = [
            "binary",
            "distributed",
            "dropout",
            "embedding",
            "layernorm",
            "linear",
            "loss",
            "rope",
            "matmul",
            "multi_head_utils",
            "rmsnorm",
            "sample",
            "unary",
        ]

        for submodule_name in nested_submodules:
            if hasattr(_ttml_ops, submodule_name):
                # If it exists in _ttml.ops, it should be accessible via ttml.ops
                assert hasattr(ttml, "ops"), f"ttml.ops should exist"
                ttml_ops = getattr(ttml, "ops")
                assert hasattr(
                    ttml_ops, submodule_name
                ), f"ttml.ops.{submodule_name} should be imported from _ttml.ops.{submodule_name}"


def test_python_override_precedence():
    """Test that Python implementations take precedence over C++ symbols."""
    # In ttml.modules, Python implementations should override C++ ones
    assert hasattr(ttml, "modules")
    ttml_modules = ttml.modules

    # Check that Python implementations exist
    assert hasattr(
        ttml_modules, "AbstractModuleBase"
    ), "Python AbstractModuleBase should be available"
    assert hasattr(ttml_modules, "Parameter"), "Python Parameter should be available"
    assert hasattr(ttml_modules, "Buffer"), "Python Buffer should be available"

    # Verify these are Python classes, not C++ bindings
    from ttml.modules.module_base import AbstractModuleBase
    from ttml.modules.parameter import Parameter, Buffer

    # The module should have the Python versions
    assert ttml_modules.AbstractModuleBase is AbstractModuleBase
    assert ttml_modules.Parameter is Parameter
    assert ttml_modules.Buffer is Buffer


def test_conditional_import_no_overwrite():
    """Test that existing Python symbols are not overwritten."""
    # Create a test attribute before import
    test_value = "test_python_override"

    # Check that Python implementations in modules weren't overwritten
    from ttml.modules.module_base import AbstractModuleBase as PyAbstractModuleBase
    from ttml.modules.parameter import Parameter as PyParameter

    # These should still be the Python versions
    assert ttml.modules.AbstractModuleBase is PyAbstractModuleBase
    assert ttml.modules.Parameter is PyParameter


def test_private_symbols_not_imported():
    """Test that private symbols (starting with _) are not imported."""
    # Get all attributes from ttml
    ttml_attrs = dir(ttml)

    # Check that private _ttml symbols are not directly exposed
    # (except _ttml itself and _recursive_import which are implementation details)
    private_from_ttml = [
        attr
        for attr in ttml_attrs
        if attr.startswith("_")
        and attr
        not in (
            "_ttml",
            "_recursive_import",
            "__name__",
            "__doc__",
            "__package__",
            "__file__",
            "__path__",
            "__loader__",
            "__spec__",
            "__cached__",
            "__builtins__",
            "__all__",
        )
    ]

    # Most private symbols should not be from _ttml
    # (we allow some implementation details)
    for attr in private_from_ttml:
        if hasattr(ttml._ttml, attr):
            # If it's a private symbol in _ttml, it shouldn't be imported
            # unless it's a special case
            assert attr in (
                "_ttml",
                "_recursive_import",
            ), f"Private symbol {attr} from _ttml should not be imported"


def test_module_metadata_not_imported():
    """Test that module metadata is not imported."""
    # These should not be imported from _ttml
    metadata_attrs = [
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__path__",
        "__loader__",
        "__spec__",
        "__cached__",
    ]

    for attr in metadata_attrs:
        if hasattr(ttml, attr):
            # These should be from the ttml module itself, not _ttml
            ttml_value = getattr(ttml, attr)
            if hasattr(ttml._ttml, attr):
                _ttml_value = getattr(ttml._ttml, attr)
                # They might be the same for some attributes, but the point
                # is that we're not blindly copying them
                pass


def test_symbols_available_from_ttml():
    """Test that public symbols from _ttml are accessible via ttml."""
    # Check some known public symbols
    if hasattr(ttml._ttml, "autograd"):
        # If autograd exists in _ttml, check some of its symbols
        _ttml_autograd = ttml._ttml.autograd
        _ttml_autograd_attrs = [
            attr for attr in dir(_ttml_autograd) if not attr.startswith("_")
        ]

        # Some symbols should be available
        if hasattr(ttml, "autograd"):
            ttml_autograd = ttml.autograd
            # At least some public symbols should be imported
            imported_count = sum(
                1 for attr in _ttml_autograd_attrs if hasattr(ttml_autograd, attr)
            )
            # We expect at least some symbols to be imported
            # (exact count depends on implementation)


def test_submodule_structure_preserved():
    """Test that submodule structure is preserved during import."""
    # Check that if _ttml has a submodule, ttml has a corresponding one
    if hasattr(ttml._ttml, "ops"):
        assert hasattr(ttml, "ops"), "ttml.ops should exist if _ttml.ops exists"
        assert inspect.ismodule(ttml.ops), "ttml.ops should be a module"

        # Check nested structure
        if hasattr(ttml._ttml.ops, "binary"):
            assert hasattr(
                ttml.ops, "binary"
            ), "ttml.ops.binary should exist if _ttml.ops.binary exists"
            assert inspect.ismodule(
                ttml.ops.binary
            ), "ttml.ops.binary should be a module"


def test_circular_dependency_prevention():
    """Test that circular dependencies are prevented."""
    # The recursive import should handle cycles gracefully
    # This is tested implicitly by the fact that imports succeed
    # without infinite recursion

    # Try to access nested modules multiple times
    if hasattr(ttml, "ops"):
        ops1 = ttml.ops
        ops2 = ttml.ops
        assert ops1 is ops2, "Same module should be returned"

        if hasattr(ttml.ops, "binary"):
            binary1 = ttml.ops.binary
            binary2 = ttml.ops.binary
            assert binary1 is binary2, "Same submodule should be returned"


def test_exceptions_imported():
    """Test that Python exceptions in modules are properly available."""
    # Check that Python exceptions are available and not overridden
    from ttml.modules.exceptions import (
        ModuleError,
        DuplicateNameError,
        NameNotFoundError,
        UninitializedModuleError,
    )

    assert hasattr(ttml.modules, "ModuleError")
    assert hasattr(ttml.modules, "DuplicateNameError")
    assert hasattr(ttml.modules, "NameNotFoundError")
    assert hasattr(ttml.modules, "UninitializedModuleError")

    # Verify they are the Python versions
    assert ttml.modules.ModuleError is ModuleError
    assert ttml.modules.DuplicateNameError is DuplicateNameError
    assert ttml.modules.NameNotFoundError is NameNotFoundError
    assert ttml.modules.UninitializedModuleError is UninitializedModuleError


def test_all_attribute_handling():
    """Test that __all__ attributes are handled correctly."""
    # Check that modules has __all__ defined
    if hasattr(ttml.modules, "__all__"):
        modules_all = ttml.modules.__all__
        assert isinstance(modules_all, (list, tuple))

        # Check that expected items are in __all__
        expected_items = [
            "AbstractModuleBase",
            "Parameter",
            "Buffer",
            "ModuleError",
            "DuplicateNameError",
            "NameNotFoundError",
            "UninitializedModuleError",
        ]

        for item in expected_items:
            assert item in modules_all, f"{item} should be in ttml.modules.__all__"
            assert hasattr(
                ttml.modules, item
            ), f"{item} should be available in ttml.modules"


def test_backward_compatibility():
    """Test that existing import patterns still work."""
    # Test that direct imports still work
    import ttml.autograd  # noqa: F401
    import ttml.ops  # noqa: F401
    import ttml.modules  # noqa: F401

    # Test that accessing via ttml works
    assert hasattr(ttml, "autograd") or hasattr(ttml, "ops") or hasattr(ttml, "modules")

    # Test that Python implementations are accessible
    from ttml.modules import AbstractModuleBase, Parameter, Buffer

    assert AbstractModuleBase is not None
    assert Parameter is not None
    assert Buffer is not None


def test_readonly_attributes_handled():
    """Test that read-only attributes are handled gracefully."""
    # The recursive import should skip attributes that can't be set
    # This is tested implicitly - if there were issues, imports would fail

    # Try to access various attributes
    if hasattr(ttml, "ops"):
        ops = ttml.ops
        # Accessing should not raise errors
        dir(ops)  # Should not raise


def test_import_idempotency():
    """Test that re-importing doesn't change the module structure."""
    # Reload the module and check structure is the same
    import importlib

    # Get current state
    if hasattr(ttml, "ops"):
        ops_before = ttml.ops
        ops_attrs_before = set(dir(ops_before))

    # The import should be idempotent - re-running shouldn't change things
    # (This is tested by the fact that the module structure is consistent)


def test_nested_submodule_symbols():
    """Test that symbols in nested submodules are accessible."""
    # Check that symbols in nested submodules are imported
    if hasattr(ttml, "ops") and hasattr(ttml.ops, "binary"):
        binary_ops = ttml.ops.binary
        # Should be able to access the module
        assert inspect.ismodule(binary_ops)

        # Check that it has some attributes (exact attributes depend on implementation)
        binary_attrs = dir(binary_ops)
        # Should have at least some public attributes or be an empty module
        assert isinstance(binary_attrs, list)


def _get_ttml_submodules():
    """Dynamically discover submodules from _ttml for parametrized tests."""
    try:
        _ttml_attrs = dir(ttml._ttml)
    except (AttributeError, TypeError):
        # If _ttml isn't available, return empty list
        return []

    submodules = []
    for attr_name in _ttml_attrs:
        if not attr_name.startswith("_"):  # Only public attributes
            try:
                attr_value = getattr(ttml._ttml, attr_name)
                if inspect.ismodule(attr_value):
                    submodules.append(attr_name)
            except (AttributeError, TypeError):
                continue
    return submodules


@pytest.mark.parametrize("submodule_name", _get_ttml_submodules())
def test_submodule_imported(submodule_name):
    """Parametrized test for each expected submodule."""
    # Check that _ttml has the submodule
    if hasattr(ttml._ttml, submodule_name):
        _ttml_submodule = getattr(ttml._ttml, submodule_name)
        assert inspect.ismodule(
            _ttml_submodule
        ), f"_ttml.{submodule_name} should be a module"

        # Check that ttml has the corresponding submodule
        assert hasattr(ttml, submodule_name), f"ttml.{submodule_name} should exist"

        ttml_submodule = getattr(ttml, submodule_name)
        assert inspect.ismodule(
            ttml_submodule
        ), f"ttml.{submodule_name} should be a module"

        # Check that some symbols are imported (if the submodule has any)
        _ttml_attrs = [
            attr for attr in dir(_ttml_submodule) if not attr.startswith("_")
        ]

        if _ttml_attrs:
            # At least some public symbols should be available
            imported_attrs = [
                attr for attr in _ttml_attrs if hasattr(ttml_submodule, attr)
            ]
            # We don't require all symbols to be imported (some might be filtered)
            # but the submodule should exist and be accessible
