# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttml module imports.

These tests verify that the ttml package and its submodules can be imported
correctly, ensuring the Python bindings are properly configured.
"""

import inspect

import pytest


@pytest.mark.smoke
def test_ttml_import():
    """Verify ttml module can be imported."""
    import ttml

    assert ttml is not None
    assert hasattr(ttml, "_ttml")


@pytest.mark.smoke
def test_ttml_core_submodules():
    """Verify core submodules are accessible."""
    import ttml

    assert hasattr(ttml, "autograd"), "ttml.autograd should be accessible"
    assert hasattr(ttml, "ops"), "ttml.ops should be accessible"
    assert hasattr(ttml, "modules"), "ttml.modules should be accessible"
    assert hasattr(ttml, "optimizers"), "ttml.optimizers should be accessible"
    assert hasattr(ttml, "core"), "ttml.core should be accessible"


@pytest.mark.smoke
def test_ttml_ops_submodules():
    """Verify ops submodules are accessible."""
    import ttml

    assert hasattr(ttml.ops, "binary"), "ttml.ops.binary should be accessible"
    assert hasattr(ttml.ops, "loss"), "ttml.ops.loss should be accessible"
    assert hasattr(ttml.ops, "unary"), "ttml.ops.unary should be accessible"
    assert hasattr(ttml.ops, "matmul"), "ttml.ops.matmul should be accessible"


@pytest.mark.smoke
def test_ttml_modules_exports():
    """Verify key module classes are exported."""
    from ttml.modules import AbstractModuleBase, ModuleBase, Parameter, Buffer

    assert AbstractModuleBase is not None
    assert ModuleBase is not None
    assert Parameter is not None
    assert Buffer is not None


@pytest.mark.smoke
def test_ttml_autograd_exports():
    """Verify key autograd classes are exported."""
    import ttml

    assert hasattr(ttml.autograd, "Tensor"), "ttml.autograd.Tensor should exist"
    assert hasattr(ttml.autograd, "AutoContext"), "ttml.autograd.AutoContext should exist"
    assert hasattr(ttml.autograd, "GradMode"), "ttml.autograd.GradMode should exist"


@pytest.mark.smoke
def test_ttml_optimizers_exports():
    """Verify optimizer classes are exported."""
    import ttml

    assert hasattr(ttml.optimizers, "SGD"), "ttml.optimizers.SGD should exist"
    assert hasattr(ttml.optimizers, "SGDConfig"), "ttml.optimizers.SGDConfig should exist"
    assert hasattr(ttml.optimizers, "AdamW"), "ttml.optimizers.AdamW should exist"
    assert hasattr(ttml.optimizers, "AdamWConfig"), "ttml.optimizers.AdamWConfig should exist"


@pytest.mark.smoke
def test_ttml_models_accessible():
    """Verify models submodule is accessible."""
    import ttml

    assert hasattr(ttml, "models"), "ttml.models should be accessible"
    assert inspect.ismodule(ttml.models), "ttml.models should be a module"


@pytest.mark.smoke
def test_ttnn_dependency():
    """Verify ttnn (required dependency) can be imported."""
    import ttnn

    assert ttnn is not None
    assert hasattr(ttnn, "DataType")
    assert hasattr(ttnn, "Layout")
