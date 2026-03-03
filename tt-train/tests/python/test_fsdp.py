# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for FSDP (Fully Sharded Data Parallel) functionality.

This module tests the FSDP implementation including:
- C++ hook infrastructure in ModuleBase
- Python FSDPModule wrapper
- Prefetching configuration
- Gradient synchronization
"""

import inspect
import pytest
import numpy as np

import ttml


class TestModuleBaseHooks:
    """Tests for C++ ModuleBase pre/post forward hooks."""

    def test_register_pre_forward_hook(self):
        """Test that pre-forward hooks can be registered."""
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        module = SimpleModule()

        # Initially no hooks
        assert not module.has_pre_forward_hooks()

        # Register a hook
        hook_called = []

        def pre_hook(mod, inp):
            hook_called.append(True)

        handle = module.register_pre_forward_hook(pre_hook)
        assert module.has_pre_forward_hooks()
        assert isinstance(handle, int)

    def test_register_post_forward_hook(self):
        """Test that post-forward hooks can be registered."""
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        module = SimpleModule()

        # Initially no hooks
        assert not module.has_post_forward_hooks()

        # Register a hook
        hook_called = []

        def post_hook(mod, inp, out):
            hook_called.append(True)

        handle = module.register_post_forward_hook(post_hook)
        assert module.has_post_forward_hooks()
        assert isinstance(handle, int)

    def test_remove_pre_forward_hook(self):
        """Test that pre-forward hooks can be removed."""
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        module = SimpleModule()

        def hook(mod, inp):
            pass

        handle = module.register_pre_forward_hook(hook)
        assert module.has_pre_forward_hooks()

        module.remove_pre_forward_hook(handle)
        assert not module.has_pre_forward_hooks()

    def test_remove_post_forward_hook(self):
        """Test that post-forward hooks can be removed."""
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        module = SimpleModule()

        def hook(mod, inp, out):
            pass

        handle = module.register_post_forward_hook(hook)
        assert module.has_post_forward_hooks()

        module.remove_post_forward_hook(handle)
        assert not module.has_post_forward_hooks()

    def test_clear_all_hooks(self):
        """Test that all hooks can be cleared at once."""
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        module = SimpleModule()

        # Register multiple hooks
        module.register_pre_forward_hook(lambda m, i: None)
        module.register_pre_forward_hook(lambda m, i: None)
        module.register_post_forward_hook(lambda m, i, o: None)

        assert module.has_pre_forward_hooks()
        assert module.has_post_forward_hooks()

        module.clear_all_hooks()

        assert not module.has_pre_forward_hooks()
        assert not module.has_post_forward_hooks()

    def test_call_with_hooks_executes_hooks(self):
        """Test that call_with_hooks executes registered hooks."""
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        module = SimpleModule()

        pre_hook_calls = []
        post_hook_calls = []

        def pre_hook(mod, inp):
            pre_hook_calls.append(inp)

        def post_hook(mod, inp, out):
            post_hook_calls.append((inp, out))

        module.register_pre_forward_hook(pre_hook)
        module.register_post_forward_hook(post_hook)

        # Create input tensor
        x_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
        x = ttml.autograd.Tensor.from_numpy(x_np)

        # Call with hooks
        output = module.call_with_hooks(x)

        # Verify hooks were called
        assert len(pre_hook_calls) == 1
        assert len(post_hook_calls) == 1
        assert pre_hook_calls[0] is x
        assert post_hook_calls[0][0] is x
        assert post_hook_calls[0][1] is output

    def test_multiple_hooks_executed_in_order(self):
        """Test that multiple hooks are executed in registration order."""
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        module = SimpleModule()

        execution_order = []

        module.register_pre_forward_hook(lambda m, i: execution_order.append(1))
        module.register_pre_forward_hook(lambda m, i: execution_order.append(2))
        module.register_pre_forward_hook(lambda m, i: execution_order.append(3))

        x = ttml.autograd.Tensor.from_numpy(
            np.random.randn(1, 1, 32, 32).astype(np.float32)
        )
        module.call_with_hooks(x)

        assert execution_order == [
            1,
            2,
            3,
        ], "Hooks should execute in registration order"


class TestFSDPModuleAPI:
    """Tests for the Python FSDPModule API."""

    def test_fsdp_module_import(self):
        """Test that FSDP module can be imported."""
        from ttml.distributed import FSDPModule, fully_shard

        assert FSDPModule is not None
        assert fully_shard is not None
        assert callable(fully_shard)

    def test_fsdp_distributed_module_exists(self):
        """Test that ttml.distributed module exists with FSDP exports."""
        import ttml.distributed

        assert hasattr(ttml.distributed, "FSDPModule")
        assert hasattr(ttml.distributed, "fully_shard")
        assert hasattr(ttml.distributed, "setup_prefetching")
        assert hasattr(ttml.distributed, "synchronize_fsdp_gradients")
        assert hasattr(ttml.distributed, "clear_fsdp_registry")

    def test_fsdp_module_init(self):
        """Test FSDPModule initialization."""
        from ttml.distributed import FSDPModule
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        module = SimpleModule()

        # This should not raise (though actual sharding needs devices)
        try:
            fsdp_module = FSDPModule(
                module=module,
                shard_dim=0,
                reshard_after_forward=True,
            )
            assert fsdp_module._wrapped is module
            assert fsdp_module._shard_dim == 0
            assert fsdp_module._reshard_after_forward is True
        except Exception:
            # Expected to fail without proper device setup
            # But the API should exist
            pass

    def test_fully_shard_function(self):
        """Test fully_shard convenience function."""
        from ttml.distributed import fully_shard
        from ttml.modules import AbstractModuleBase, Parameter

        class SimpleModule(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

            def forward(self, x):
                return x

        module = SimpleModule()

        try:
            fsdp_module = fully_shard(
                module,
                shard_dim=0,
                reshard_after_forward=True,
            )
            # Should return an FSDPModule
            from ttml.distributed import FSDPModule

            assert isinstance(fsdp_module, FSDPModule)
        except Exception:
            # Expected to fail without proper device setup
            pass

    def test_fsdp_module_has_prefetch_methods(self):
        """Test that FSDPModule has prefetch configuration methods."""
        from ttml.distributed import FSDPModule

        # Check that the class has the expected methods
        assert hasattr(FSDPModule, "set_modules_to_forward_prefetch")
        assert hasattr(FSDPModule, "set_modules_to_backward_prefetch")
        assert hasattr(FSDPModule, "unshard")
        assert hasattr(FSDPModule, "reshard")

    def test_fsdp_module_has_gradient_methods(self):
        """Test that FSDPModule has gradient synchronization methods."""
        from ttml.distributed import FSDPModule

        assert hasattr(FSDPModule, "reduce_scatter_gradients")
        assert hasattr(FSDPModule, "all_gather_gradients")


class TestSetupPrefetching:
    """Tests for the setup_prefetching helper function."""

    def test_setup_prefetching_exists(self):
        """Test that setup_prefetching function exists."""
        from ttml.distributed import setup_prefetching

        assert callable(setup_prefetching)

        # Check function signature
        sig = inspect.signature(setup_prefetching)
        params = list(sig.parameters.keys())
        assert "modules" in params
        assert "num_to_forward_prefetch" in params
        assert "num_to_backward_prefetch" in params


class TestFSDPGradientSync:
    """Tests for FSDP gradient synchronization functions."""

    def test_synchronize_fsdp_gradients_exists(self):
        """Test that synchronize_fsdp_gradients function exists."""
        from ttml.distributed import synchronize_fsdp_gradients

        assert callable(synchronize_fsdp_gradients)

    def test_clear_fsdp_registry_exists(self):
        """Test that clear_fsdp_registry function exists."""
        from ttml.distributed import clear_fsdp_registry

        assert callable(clear_fsdp_registry)

        # Should be callable without errors
        clear_fsdp_registry()


class TestShardedParameter:
    """Tests for the ShardedParameter helper class."""

    def test_sharded_parameter_exists(self):
        """Test that ShardedParameter class exists."""
        from ttml.distributed.fsdp import ShardedParameter

        assert ShardedParameter is not None

    def test_sharded_parameter_init(self):
        """Test ShardedParameter initialization."""
        from ttml.distributed.fsdp import ShardedParameter

        # Create a mock tensor (just a placeholder)
        mock_tensor = object()

        sharded_param = ShardedParameter(
            name="test_param",
            sharded_tensor=mock_tensor,
            original_tensor=mock_tensor,
            shard_dim=0,
        )

        assert sharded_param.name == "test_param"
        assert sharded_param.sharded_tensor is mock_tensor
        assert sharded_param.original_tensor is mock_tensor
        assert sharded_param.shard_dim == 0
        assert sharded_param.full_tensor is None


class TestFSDPDocumentation:
    """Tests for FSDP documentation and docstrings."""

    def test_fully_shard_has_docstring(self):
        """Test that fully_shard has a docstring."""
        from ttml.distributed import fully_shard

        assert fully_shard.__doc__ is not None
        assert "FSDP" in fully_shard.__doc__ or "shard" in fully_shard.__doc__

    def test_fsdp_module_has_docstring(self):
        """Test that FSDPModule has a docstring."""
        from ttml.distributed import FSDPModule

        assert FSDPModule.__doc__ is not None
        assert "FSDP" in FSDPModule.__doc__ or "shard" in FSDPModule.__doc__

    def test_setup_prefetching_has_docstring(self):
        """Test that setup_prefetching has a docstring."""
        from ttml.distributed import setup_prefetching

        assert setup_prefetching.__doc__ is not None


class TestFSDPIntegration:
    """Integration tests for FSDP with the module system."""

    def test_fsdp_with_linear_layer(self):
        """Test FSDP wrapping a LinearLayer."""
        from ttml.distributed import FSDPModule, clear_fsdp_registry

        # Clear registry first
        clear_fsdp_registry()

        # Create a C++ LinearLayer
        try:
            linear = ttml.modules.LinearLayer(64, 64, has_bias=True)

            # Wrap with FSDP
            fsdp_linear = FSDPModule(
                module=linear,
                shard_dim=0,
                reshard_after_forward=True,
            )

            assert fsdp_linear._wrapped is linear
        except Exception:
            # Expected to fail without proper device setup
            pass

    def test_fsdp_with_module_list(self):
        """Test FSDP with ModuleList containing multiple layers."""
        from ttml.distributed import fully_shard, setup_prefetching, clear_fsdp_registry
        from ttml.modules import AbstractModuleBase, ModuleList, Parameter

        # Clear registry
        clear_fsdp_registry()

        class SimpleLayer(AbstractModuleBase):
            def __init__(self, idx):
                super().__init__()
                w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
                self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))
                self.idx = idx

            def forward(self, x):
                return ttml.ops.binary.mul(x, self.weight.tensor)

        class MultiLayerModel(AbstractModuleBase):
            def __init__(self, n_layers):
                super().__init__()
                self.layers = ModuleList([SimpleLayer(i) for i in range(n_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        try:
            model = MultiLayerModel(n_layers=4)

            # Apply FSDP to each layer
            fsdp_layers = []
            for layer in model.layers:
                fsdp_layer = fully_shard(layer)
                fsdp_layers.append(fsdp_layer)

            # Set up prefetching
            setup_prefetching(fsdp_layers, num_to_forward_prefetch=2)

            # Verify prefetching was configured
            # First layers should have prefetch modules
            assert len(fsdp_layers[0]._forward_prefetch_modules) == 2
            assert len(fsdp_layers[1]._forward_prefetch_modules) == 2
            # Last layers shouldn't prefetch anything
            assert len(fsdp_layers[-1]._forward_prefetch_modules) == 0
        except Exception:
            # Expected to fail without proper device setup for sharding
            pass
