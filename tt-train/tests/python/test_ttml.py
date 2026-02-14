# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttml recursive import functionality.

This module tests that all _ttml C++ extension symbols are properly imported
into the ttml Python package, with Python implementations taking precedence.
"""

import inspect
import pytest
import numpy as np

import ttml  # noqa: E402
import ttml._ttml  # noqa: E402
import ttnn


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


def test_symbols_available_from_ttml():
    """Test that public symbols from _ttml are accessible via ttml."""
    # Check some known public symbols
    assert hasattr(ttml._ttml, "autograd")
    # If autograd exists in _ttml, check some of its symbols
    _ttml_autograd = ttml._ttml.autograd
    _ttml_autograd_attrs = [
        attr for attr in dir(_ttml_autograd) if not attr.startswith("_")
    ]

    # Some symbols should be available
    assert hasattr(ttml, "autograd")
    ttml_autograd = ttml.autograd
    # At least some public symbols should be imported
    imported_count = sum(
        1 for attr in _ttml_autograd_attrs if hasattr(ttml_autograd, attr)
    )
    # We expect at least some symbols to be imported
    # (exact count depends on implementation)

    assert imported_count > 0


def test_submodule_structure_preserved():
    """Test that submodule structure is preserved during import."""
    # Check that if _ttml has a submodule, ttml has a corresponding one
    assert hasattr(ttml._ttml, "ops")
    assert hasattr(ttml, "ops"), "ttml.ops should exist if _ttml.ops exists"
    assert inspect.ismodule(ttml.ops), "ttml.ops should be a module"

    # Check nested structure
    assert hasattr(ttml._ttml.ops, "binary")
    assert hasattr(
        ttml.ops, "binary"
    ), "ttml.ops.binary should exist if _ttml.ops.binary exists"
    assert inspect.ismodule(ttml.ops.binary), "ttml.ops.binary should be a module"


def test_circular_dependency_prevention():
    """Test that circular dependencies are prevented."""
    # The recursive import should handle cycles gracefully
    # This is tested implicitly by the fact that imports succeed
    # without infinite recursion

    # Try to access nested modules multiple times
    assert hasattr(ttml, "ops")
    ops1 = ttml.ops
    ops2 = ttml.ops
    assert ops1 is ops2, "Same module should be returned"

    assert hasattr(ttml.ops, "binary")
    binary1 = ttml.ops.binary
    binary2 = ttml.ops.binary
    assert binary1 is binary2, "Same submodule should be returned"


def test_all_attribute_handling():
    """Test that __all__ attributes are handled correctly."""
    # Check that modules has __all__ defined
    assert hasattr(ttml.modules, "__all__")
    modules_all = ttml.modules.__all__
    assert isinstance(modules_all, (list, tuple))

    # Check that expected items are in __all__
    expected_items = [
        "AbstractModuleBase",
        "ModuleBase",
        "ModuleDict",
        "ModuleList",
        "Parameter",
        "Buffer",
        "RunMode",
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
    assert hasattr(ttml, "ops")
    ops = ttml.ops
    # Accessing should not raise errors
    dir(ops)  # Should not raise


def test_nested_submodule_symbols():
    """Test that symbols in nested submodules are accessible."""
    # Check that symbols in nested submodules are imported
    assert hasattr(ttml, "ops")
    assert hasattr(ttml.ops, "binary")
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
    assert hasattr(ttml._ttml, submodule_name)
    _ttml_submodule = getattr(ttml._ttml, submodule_name)
    assert inspect.ismodule(
        _ttml_submodule
    ), f"_ttml.{submodule_name} should be a module"

    # Check that ttml has the corresponding submodule
    assert hasattr(ttml, submodule_name), f"ttml.{submodule_name} should exist"

    ttml_submodule = getattr(ttml, submodule_name)
    assert inspect.ismodule(ttml_submodule), f"ttml.{submodule_name} should be a module"

    # Check that some symbols are imported (if the submodule has any)
    _ttml_attrs = [attr for attr in dir(_ttml_submodule) if not attr.startswith("_")]

    assert _ttml_attrs
    # At least some public symbols should be available
    imported_attrs = [attr for attr in _ttml_attrs if hasattr(ttml_submodule, attr)]

    assert imported_attrs, (
        f"Expected ttml.{submodule_name} to import at least one public symbol from "
        f"_ttml.{submodule_name}"
    )


class TestCppOptimizersWithPythonModules:
    """Test that C++ optimizers work with Python module parameter registration."""

    def test_sgd_optimizer_with_python_module(self):
        """Test SGD optimizer updates parameters registered via Python AbstractModuleBase."""
        from ttml.modules import AbstractModuleBase, Parameter

        # Define a simple Python module
        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                # Create a parameter tensor and register it
                weight_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                weight_tensor = ttml.autograd.Tensor.from_numpy(weight_np)
                self.weight = Parameter(weight_tensor)

            def __call__(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        # Create module and verify parameter registration
        model = SimpleModule()
        params = model.parameters()
        assert len(params) > 0, "Module should have registered parameters"
        assert any(
            "weight" in k for k in params.keys()
        ), "Should have 'weight' parameter"

        # Get initial weight values
        weight_key = [k for k in params.keys() if "weight" in k][0]
        weight_before = params[weight_key].to_numpy(ttnn.DataType.FLOAT32).copy()

        # Create C++ SGD optimizer with the Python module's parameters
        sgd_config = ttml.optimizers.SGDConfig.make(
            lr=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False
        )
        optimizer = ttml.optimizers.SGD(params, sgd_config)

        # Run a training step
        model.train()
        optimizer.zero_grad()

        # Forward pass
        x_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
        x = ttml.autograd.Tensor.from_numpy(x_np)
        output = model(x)

        # Create a simple loss (sum of outputs)
        target_np = np.zeros((1, 1, 32, 32), dtype=np.float32)
        target = ttml.autograd.Tensor.from_numpy(target_np)
        loss = ttml.ops.loss.mse_loss(output, target, ttml.ops.ReduceType.MEAN)

        # Backward pass
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        # Optimizer step
        optimizer.step()

        # Verify weights changed
        params_after = model.parameters()
        weight_after = params_after[weight_key].to_numpy(ttnn.DataType.FLOAT32)

        assert not np.allclose(
            weight_before, weight_after, atol=1e-6
        ), "SGD optimizer should have updated the weights"

    def test_adamw_optimizer_with_python_module(self):
        """Test AdamW optimizer updates parameters registered via Python AbstractModuleBase."""
        from ttml.modules import AbstractModuleBase, Parameter

        # Define a simple Python module with multiple parameters
        class TwoLayerModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w1_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                w2_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                self.weight1 = Parameter(ttml.autograd.Tensor.from_numpy(w1_np))
                self.weight2 = Parameter(ttml.autograd.Tensor.from_numpy(w2_np))

            def __call__(self, x):
                h = ttml.ops.binary.mul(x, self.weight1.tensor)
                return ttml.ops.binary.mul(h, self.weight2.tensor)

        model = TwoLayerModule()
        params = model.parameters()

        # Verify both parameters registered
        assert len(params) >= 2, "Module should have at least 2 parameters"
        weight_keys = [k for k in params.keys() if "weight" in k]
        assert len(weight_keys) >= 2, "Should have both weight1 and weight2"

        # Store initial values
        initial_weights = {
            k: params[k].to_numpy(ttnn.DataType.FLOAT32).copy() for k in weight_keys
        }

        # Create C++ AdamW optimizer
        adamw_config = ttml.optimizers.AdamWConfig.make(
            lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01
        )
        optimizer = ttml.optimizers.AdamW(params, adamw_config)

        # Training loop
        model.train()
        for _ in range(3):
            optimizer.zero_grad()

            x_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            x = ttml.autograd.Tensor.from_numpy(x_np)
            output = model(x)

            target_np = np.ones((1, 1, 32, 32), dtype=np.float32)
            target = ttml.autograd.Tensor.from_numpy(target_np)
            loss = ttml.ops.loss.mse_loss(output, target, ttml.ops.ReduceType.MEAN)

            loss.backward(False)
            ttml.autograd.AutoContext.get_instance().reset_graph()
            optimizer.step()

        # Verify all weights changed
        params_after = model.parameters()
        for key in weight_keys:
            weight_after = params_after[key].to_numpy(ttnn.DataType.FLOAT32)
            assert not np.allclose(
                initial_weights[key], weight_after, atol=1e-6
            ), f"AdamW optimizer should have updated {key}"

    def test_optimizer_with_nested_python_modules(self):
        """Test optimizer works with nested Python modules (submodules)."""
        from ttml.modules import AbstractModuleBase, Parameter

        class InnerModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                self.inner_weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def __call__(self, x):
                return ttml.ops.binary.mul(x, self.inner_weight.tensor)

        class OuterModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                self.outer_weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))
                self.inner = InnerModule()  # Nested submodule

            def __call__(self, x):
                x = ttml.ops.binary.mul(x, self.outer_weight.tensor)
                return self.inner(x)

        model = OuterModule()
        params = model.parameters()

        # Should have parameters from both outer and inner modules
        param_names = list(params.keys())
        assert any(
            "outer" in k.lower() for k in param_names
        ), "Should have outer_weight"
        assert any(
            "inner" in k.lower() for k in param_names
        ), "Should have inner_weight"

        # Store initial values
        initial_weights = {
            k: params[k].to_numpy(ttnn.DataType.FLOAT32).copy() for k in param_names
        }

        # Create optimizer and train
        sgd_config = ttml.optimizers.SGDConfig.make(0.1, 0.0, 0.0, 0.0, False)
        optimizer = ttml.optimizers.SGD(params, sgd_config)

        model.train()
        optimizer.zero_grad()

        x = ttml.autograd.Tensor.from_numpy(
            np.random.randn(1, 1, 32, 32).astype(np.float32)
        )
        output = model(x)
        target = ttml.autograd.Tensor.from_numpy(
            np.zeros((1, 1, 32, 32), dtype=np.float32)
        )
        loss = ttml.ops.loss.mse_loss(output, target, ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        # Verify all weights (both outer and inner) were updated
        params_after = model.parameters()
        for key in param_names:
            weight_after = params_after[key].to_numpy(ttnn.DataType.FLOAT32)
            assert not np.allclose(
                initial_weights[key], weight_after, atol=1e-6
            ), f"Optimizer should have updated nested parameter {key}"

    def test_optimizer_lr_adjustment(self):
        """Test that optimizer learning rate can be adjusted via set_lr."""
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def __call__(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        model = SimpleModule()
        params = model.parameters()

        # Create optimizer with initial LR
        adamw_config = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.0)
        optimizer = ttml.optimizers.AdamW(params, adamw_config)

        # Verify initial LR
        assert abs(optimizer.get_lr() - 0.001) < 1e-6, "Initial LR should be 0.001"

        # Adjust LR
        optimizer.set_lr(0.01)
        assert abs(optimizer.get_lr() - 0.01) < 1e-6, "LR should be updated to 0.01"

        # Verify optimizer still works after LR change
        model.train()
        optimizer.zero_grad()

        x = ttml.autograd.Tensor.from_numpy(
            np.random.randn(1, 1, 32, 32).astype(np.float32)
        )
        output = model(x)
        target = ttml.autograd.Tensor.from_numpy(
            np.zeros((1, 1, 32, 32), dtype=np.float32)
        )
        loss = ttml.ops.loss.mse_loss(output, target, ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        # Should complete without error
        assert True, "Optimizer should work after LR adjustment"


class TestModuleList:
    """Tests for the PyTorch-compatible ModuleList container."""

    def test_module_list_basic(self):
        """Test basic ModuleList creation and indexing."""
        from ttml.modules import AbstractModuleBase, ModuleList, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self, idx):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32) * idx
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        # Create ModuleList
        modules = ModuleList([SimpleModule(i) for i in range(3)])

        # Test length
        assert len(modules) == 3

        # Test indexing
        assert modules[0] is not None
        assert modules[1] is not None
        assert modules[-1] is modules[2]

    def test_module_list_parameter_tracking(self):
        """Test that ModuleList properly tracks parameters of contained modules."""
        from ttml.modules import AbstractModuleBase, ModuleList, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self, idx):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32) * idx
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        class OuterModule(AbstractModuleBase):
            def __init__(self, n_layers):
                super().__init__()
                self.layers = ModuleList([SimpleModule(i) for i in range(n_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = OuterModule(n_layers=4)
        params = model.parameters()

        # Should have 4 parameters (one weight per layer)
        # Parameter names should include layer indices
        param_names = list(params.keys())
        assert (
            len(param_names) == 4
        ), f"Should have 4 parameters, got {len(param_names)}: {param_names}"

    def test_module_list_iteration(self):
        """Test that ModuleList supports iteration."""
        from ttml.modules import AbstractModuleBase, ModuleList, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        modules = ModuleList([SimpleModule(i) for i in range(5)])

        # Test iteration
        count = 0
        for module in modules:
            assert module is not None
            count += 1
        assert count == 5

        # Test list conversion
        module_list = list(modules)
        assert len(module_list) == 5

    def test_module_list_append(self):
        """Test ModuleList append method."""
        from ttml.modules import AbstractModuleBase, ModuleList, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        modules = ModuleList()
        assert len(modules) == 0

        modules.append(SimpleModule())
        assert len(modules) == 1

        modules.append(SimpleModule())
        assert len(modules) == 2

        # Parameters should be tracked
        class OuterModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                self.layers = ModuleList()
                self.layers.append(SimpleModule())
                self.layers.append(SimpleModule())

            def forward(self, x):
                return x

        model = OuterModule()
        params = model.parameters()
        param_names = list(params.keys())
        assert (
            len(param_names) == 2
        ), f"Should have 2 parameters after append, got {param_names}"

    def test_module_list_extend(self):
        """Test ModuleList extend method."""
        from ttml.modules import AbstractModuleBase, ModuleList, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        modules = ModuleList([SimpleModule()])
        assert len(modules) == 1

        modules.extend([SimpleModule(), SimpleModule()])
        assert len(modules) == 3

    def test_module_list_slicing(self):
        """Test ModuleList slicing returns a new ModuleList."""
        from ttml.modules import AbstractModuleBase, ModuleList, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        modules = ModuleList([SimpleModule(i) for i in range(5)])
        sliced = modules[1:4]

        assert isinstance(sliced, ModuleList)
        assert len(sliced) == 3

    def test_module_list_with_optimizer(self):
        """Test that ModuleList parameters work with optimizer."""
        from ttml.modules import AbstractModuleBase, ModuleList, Parameter

        class SimpleLayer(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        class MultiLayerModel(AbstractModuleBase):
            def __init__(self, n_layers):
                super().__init__()
                self.layers = ModuleList([SimpleLayer() for _ in range(n_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = MultiLayerModel(n_layers=3)
        params = model.parameters()

        # Store initial values
        initial_weights = {
            k: params[k].to_numpy(ttnn.DataType.FLOAT32).copy() for k in params.keys()
        }

        # Create optimizer and train
        sgd_config = ttml.optimizers.SGDConfig.make(0.1, 0.0, 0.0, 0.0, False)
        optimizer = ttml.optimizers.SGD(params, sgd_config)

        model.train()
        optimizer.zero_grad()

        x = ttml.autograd.Tensor.from_numpy(
            np.random.randn(1, 1, 32, 32).astype(np.float32)
        )
        output = model(x)
        target = ttml.autograd.Tensor.from_numpy(
            np.zeros((1, 1, 32, 32), dtype=np.float32)
        )
        loss = ttml.ops.loss.mse_loss(output, target, ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        # Verify all weights were updated
        params_after = model.parameters()
        for key in initial_weights:
            weight_after = params_after[key].to_numpy(ttnn.DataType.FLOAT32)
            assert not np.allclose(
                initial_weights[key], weight_after, atol=1e-6
            ), f"Optimizer should have updated parameter {key}"


class TestModuleDict:
    """Tests for the PyTorch-compatible ModuleDict container."""

    def test_module_dict_basic(self):
        """Test basic ModuleDict creation and access."""
        from ttml.modules import AbstractModuleBase, ModuleDict, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self, name):
                super().__init__()
                self.module_name = name
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        # Create ModuleDict
        modules = ModuleDict(
            {
                "encoder": SimpleModule("enc"),
                "decoder": SimpleModule("dec"),
            }
        )

        assert len(modules) == 2
        assert "encoder" in modules
        assert "decoder" in modules
        assert modules["encoder"] is not None

    def test_module_dict_parameter_tracking(self):
        """Test that ModuleDict properly tracks parameters of contained modules."""
        from ttml.modules import AbstractModuleBase, ModuleDict, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        class OuterModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                self.layers = ModuleDict(
                    {
                        "first": SimpleModule(),
                        "second": SimpleModule(),
                        "third": SimpleModule(),
                    }
                )

            def forward(self, x, layer_name):
                return self.layers[layer_name](x)

        model = OuterModule()
        params = model.parameters()

        # Should have 3 parameters
        param_names = list(params.keys())
        assert (
            len(param_names) == 3
        ), f"Should have 3 parameters, got {len(param_names)}: {param_names}"

    def test_module_dict_iteration(self):
        """Test ModuleDict iteration methods."""
        from ttml.modules import AbstractModuleBase, ModuleDict, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self, name):
                super().__init__()
                self.name_tag = name
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        modules = ModuleDict(
            {
                "a": SimpleModule("a"),
                "b": SimpleModule("b"),
                "c": SimpleModule("c"),
            }
        )

        # Test keys
        keys = list(modules.keys())
        assert set(keys) == {"a", "b", "c"}

        # Test values
        values = list(modules.values())
        assert len(values) == 3

        # Test items
        items = list(modules.items())
        assert len(items) == 3
        for key, val in items:
            assert key in {"a", "b", "c"}
            assert val is not None

    def test_module_dict_update(self):
        """Test ModuleDict update method."""
        from ttml.modules import AbstractModuleBase, ModuleDict, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        modules = ModuleDict({"first": SimpleModule()})
        assert len(modules) == 1

        modules.update({"second": SimpleModule(), "third": SimpleModule()})
        assert len(modules) == 3
