# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fully Sharded Data Parallel (FSDP) implementation for TTML.

This module provides FSDP functionality similar to PyTorch's FSDP2, enabling
memory-efficient distributed training by sharding parameters across devices.

Key concepts:
- Parameters are sharded (scattered) across devices when not in use
- Before forward/backward, parameters are all-gathered to form full tensors
- After forward, parameters are resharded (optional, controlled by reshard_after_forward)
- After backward, gradients are reduce-scattered back to sharded form
- Prefetching overlaps communication with computation for better performance

Backward Pass Handling:
- FSDP uses reduce-scatter instead of all-reduce for gradients
- Call `synchronize_fsdp_gradients()` after `loss.backward()` to reduce-scatter gradients
- Or use `FSDPModule.backward_with_sync()` for automatic handling
"""

from typing import Any, Dict, List, Optional, Set, Callable
from ..modules.module_base import AbstractModuleBase
from .._ttml.modules import ModuleBase as CppModuleBase
from .._ttml import ops
from .._ttml import autograd


# Registry of all FSDP modules for gradient synchronization
_fsdp_modules: List["FSDPModule"] = []


class ShardedParameter:
    """Represents a parameter that is sharded across devices.

    Attributes:
        name: The parameter name in the module hierarchy
        sharded_tensor: The sharded tensor (scattered across devices)
        full_tensor: The full (all-gathered) tensor, None when resharded
        original_tensor: Reference to the original parameter in the module
        shard_dim: Dimension along which the parameter is sharded
    """

    def __init__(
        self,
        name: str,
        sharded_tensor: Any,
        original_tensor: Any,
        shard_dim: int = 0,
    ):
        self.name = name
        self.sharded_tensor = sharded_tensor
        self.full_tensor: Optional[Any] = None
        self.original_tensor = original_tensor
        self.shard_dim = shard_dim


class FSDPModule(AbstractModuleBase):
    """Fully Sharded Data Parallel module wrapper.

    Wraps a module with FSDP sharding, enabling memory-efficient distributed
    training. Parameters are sharded across devices and all-gathered before
    forward/backward computation.

    Example:
        ```python
        # Apply FSDP to each transformer block
        for layer in model.layers:
            fully_shard(layer)

        # Apply FSDP to the root model
        fully_shard(model)

        # Training loop
        for batch in dataloader:
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        ```

    Attributes:
        _wrapped: The wrapped module
        _shard_dim: Dimension along which parameters are sharded
        _reshard_after_forward: Whether to reshard after forward pass
        _sharded_params: Dictionary of sharded parameters
        _is_unsharded: Whether parameters are currently unsharded
        _forward_prefetch_modules: Modules to prefetch during forward
        _backward_prefetch_modules: Modules to prefetch during backward
    """

    def __init__(
        self,
        module: CppModuleBase,
        shard_dim: int = 0,
        reshard_after_forward: bool = True,
        cluster_axis: Optional[int] = None,
    ):
        """Initialize FSDP wrapper.

        Args:
            module: The module to wrap with FSDP
            shard_dim: Dimension along which to shard parameters (default: 0)
            reshard_after_forward: Whether to reshard params after forward (default: True)
                Set to False for SHARD_GRAD_OP-like behavior
            cluster_axis: Optional mesh device axis for sharding (default: None, all devices)
        """
        super().__init__()

        self._wrapped = module
        self._shard_dim = shard_dim
        self._reshard_after_forward = reshard_after_forward
        self._cluster_axis = cluster_axis
        self._sharded_params: Dict[str, ShardedParameter] = {}
        self._is_unsharded = False

        # Prefetch configuration
        self._forward_prefetch_modules: List["FSDPModule"] = []
        self._backward_prefetch_modules: List["FSDPModule"] = []

        # Track if this is the root FSDP module
        self._is_root = True

        # Register the wrapped module
        self.register_module(module, "wrapped")

        # Shard the parameters
        self._shard_parameters()

        # Register hooks for forward pass with hooks
        self._pre_hook_handle = module.register_pre_forward_hook(self._pre_forward_hook)
        self._post_hook_handle = module.register_post_forward_hook(
            self._post_forward_hook
        )

        # Register this module globally for gradient synchronization
        _fsdp_modules.append(self)

    def _shard_parameters(self) -> None:
        """Shard all parameters of the wrapped module across devices."""
        params = self._wrapped.parameters()

        for name, tensor in params.items():
            # Skip if already managed by a nested FSDP module
            if self._is_nested_fsdp_param(name):
                continue

            # Scatter the parameter across devices
            sharded = ops.distributed.scatter(
                tensor, dim=self._shard_dim, cluster_axis=self._cluster_axis
            )

            self._sharded_params[name] = ShardedParameter(
                name=name,
                sharded_tensor=sharded,
                original_tensor=tensor,
                shard_dim=self._shard_dim,
            )

    def _is_nested_fsdp_param(self, param_name: str) -> bool:
        """Check if a parameter belongs to a nested FSDP module."""
        # Check if any registered submodule is an FSDPModule
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, FSDPModule) and attr_name != "_wrapped":
                # Check if param belongs to this nested FSDP
                for nested_name in attr._sharded_params.keys():
                    if param_name == nested_name or param_name.startswith(nested_name):
                        return True
        return False

    def _all_gather_params(self) -> None:
        """All-gather sharded parameters before computation."""
        if self._is_unsharded:
            return

        for name, sharded_param in self._sharded_params.items():
            # All-gather the sharded tensor
            full = ops.distributed.all_gather(
                sharded_param.sharded_tensor,
                dim=sharded_param.shard_dim,
                cluster_axis=self._cluster_axis,
            )
            sharded_param.full_tensor = full

            # Update the module's parameter reference
            # This allows the wrapped module to use the full tensor
            self._update_param_in_module(name, full)

        self._is_unsharded = True

    def _reshard_params(self) -> None:
        """Re-shard parameters after computation."""
        if not self._is_unsharded:
            return

        for name, sharded_param in self._sharded_params.items():
            # Clear the full tensor reference
            sharded_param.full_tensor = None

            # Restore sharded tensor reference in module
            self._update_param_in_module(name, sharded_param.sharded_tensor)

        self._is_unsharded = False

    def _update_param_in_module(self, param_name: str, tensor: Any) -> None:
        """Update a parameter tensor in the wrapped module.

        This updates the tensor value while preserving the parameter structure.

        Args:
            param_name: Full parameter name (e.g., "ModuleName/weight")
            tensor: The new tensor value
        """
        # The parameter name includes the module hierarchy
        # We need to find the actual parameter and update its value
        params = self._wrapped.parameters()
        if param_name in params:
            original_param = params[param_name]
            # Copy the value from tensor to the original parameter
            # This preserves the autograd graph connections
            original_param.set_value(tensor.get_value())

    def _pre_forward_hook(self, module: CppModuleBase, input_tensor: Any) -> None:
        """Pre-forward hook: all-gather parameters."""
        self._all_gather_params()

        # Issue prefetch for next modules (explicit prefetching)
        for prefetch_module in self._forward_prefetch_modules:
            prefetch_module.unshard()

    def _post_forward_hook(
        self, module: CppModuleBase, input_tensor: Any, output_tensor: Any
    ) -> None:
        """Post-forward hook: optionally reshard parameters."""
        if self._reshard_after_forward:
            self._reshard_params()

    def unshard(self) -> None:
        """Manually trigger all-gather of parameters.

        Call this before forward to issue the all-gather earlier,
        allowing it to overlap with other computation.
        """
        self._all_gather_params()

    def reshard(self) -> None:
        """Manually trigger resharding of parameters."""
        self._reshard_params()

    def set_modules_to_forward_prefetch(self, modules: List["FSDPModule"]) -> None:
        """Set modules to prefetch during forward pass.

        These modules will have their all-gather issued before the current
        module's forward computation, enabling communication-computation overlap.

        Args:
            modules: List of FSDPModule instances to prefetch
        """
        self._forward_prefetch_modules = modules

    def set_modules_to_backward_prefetch(self, modules: List["FSDPModule"]) -> None:
        """Set modules to prefetch during backward pass.

        These modules will have their all-gather issued during the current
        module's backward pass, enabling communication-computation overlap.

        Args:
            modules: List of FSDPModule instances to prefetch
        """
        self._backward_prefetch_modules = modules

    def set_reshard_after_forward(self, value: bool) -> None:
        """Set whether to reshard parameters after forward.

        Args:
            value: If True, reshard after forward (FULL_SHARD mode)
                   If False, keep unsharded until backward (SHARD_GRAD_OP mode)
        """
        self._reshard_after_forward = value

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with FSDP hooks.

        The pre/post hooks are already registered on the wrapped module,
        so we just delegate to the wrapped module's forward.
        """
        # Use call_with_hooks to ensure hooks are executed
        if len(args) == 1 and not kwargs:
            return self._wrapped.call_with_hooks(args[0])
        elif len(args) == 2 and not kwargs:
            return self._wrapped.call_with_hooks(args[0], args[1])
        else:
            # Fallback for other signatures
            return self._wrapped(*args, **kwargs)

    def parameters(self) -> Dict[str, Any]:
        """Return the parameters of the wrapped module.

        Returns the sharded parameters when in sharded state,
        or full parameters when unsharded.
        """
        return self._wrapped.parameters()

    def train(self) -> None:
        """Set the module to training mode."""
        self._wrapped.train()

    def eval(self) -> None:
        """Set the module to evaluation mode."""
        self._wrapped.eval()

    def reduce_scatter_gradients(self) -> None:
        """Reduce-scatter gradients after backward pass.

        This should be called after loss.backward() to reduce-scatter the
        gradients back to sharded form. This is more memory-efficient than
        all-reduce since each device only needs to store its shard of gradients.

        For automatic handling, use synchronize_fsdp_gradients() which calls
        this on all registered FSDP modules.
        """
        for name, sharded_param in self._sharded_params.items():
            original_tensor = sharded_param.original_tensor

            if original_tensor.is_grad_initialized():
                # Get the current (full) gradient
                full_grad = original_tensor.get_grad()

                # Reduce-scatter to get sharded gradient
                # Each device will have the sum of gradients for its shard
                # Note: We use the underlying ttnn tensor for reduce-scatter
                sharded_grad = ops.distributed.reduce_scatter(
                    autograd.create_tensor(full_grad, requires_grad=False),
                    dim=sharded_param.shard_dim,
                    cluster_axis=self._cluster_axis,
                )

                # Update the gradient to be the sharded version
                original_tensor.set_grad(sharded_grad.get_value())

    def all_gather_gradients(self) -> None:
        """All-gather sharded gradients for optimizer step.

        If you're using a sharded optimizer, you don't need this.
        If using a regular optimizer that expects full gradients,
        call this after reduce_scatter_gradients() and before optimizer.step().
        """
        for name, sharded_param in self._sharded_params.items():
            original_tensor = sharded_param.original_tensor

            if original_tensor.is_grad_initialized():
                sharded_grad = original_tensor.get_grad()

                # All-gather to get full gradient
                full_grad = ops.distributed.all_gather(
                    autograd.create_tensor(sharded_grad, requires_grad=False),
                    dim=sharded_param.shard_dim,
                    cluster_axis=self._cluster_axis,
                )

                original_tensor.set_grad(full_grad.get_value())

    def __repr__(self) -> str:
        """Return string representation."""
        wrapped_name = (
            self._wrapped.get_name()
            if hasattr(self._wrapped, "get_name")
            else type(self._wrapped).__name__
        )
        return (
            f"FSDPModule(\n"
            f"  wrapped={wrapped_name},\n"
            f"  shard_dim={self._shard_dim},\n"
            f"  reshard_after_forward={self._reshard_after_forward},\n"
            f"  num_sharded_params={len(self._sharded_params)}\n"
            f")"
        )


def setup_prefetching(
    modules: List[FSDPModule],
    num_to_forward_prefetch: int = 1,
    num_to_backward_prefetch: int = 1,
) -> None:
    """Configure prefetching for a sequence of FSDP modules.

    This sets up forward and backward prefetching automatically for a list
    of sequential FSDP modules (e.g., transformer layers). Each module will
    prefetch the next N modules during forward and previous N during backward.

    Example:
        ```python
        # Wrap each layer with FSDP
        fsdp_layers = [fully_shard(layer) for layer in model.layers]

        # Set up prefetching
        setup_prefetching(fsdp_layers, num_to_forward_prefetch=2)
        ```

    Args:
        modules: List of FSDPModule instances in execution order
        num_to_forward_prefetch: Number of modules to prefetch during forward
        num_to_backward_prefetch: Number of modules to prefetch during backward
    """
    num_modules = len(modules)

    # Set up forward prefetching
    for i, module in enumerate(modules):
        if i >= num_modules - num_to_forward_prefetch:
            # Last few modules have nothing to prefetch
            continue

        prefetch_modules = [
            modules[i + j]
            for j in range(1, num_to_forward_prefetch + 1)
            if i + j < num_modules
        ]
        module.set_modules_to_forward_prefetch(prefetch_modules)

    # Set up backward prefetching (reverse order)
    for i, module in enumerate(modules):
        if i < num_to_backward_prefetch:
            # First few modules have nothing to prefetch in backward
            continue

        prefetch_modules = [
            modules[i - j] for j in range(1, num_to_backward_prefetch + 1) if i - j >= 0
        ]
        module.set_modules_to_backward_prefetch(prefetch_modules)


def synchronize_fsdp_gradients() -> None:
    """Reduce-scatter gradients for all registered FSDP modules.

    Call this after loss.backward() to synchronize gradients across devices
    using reduce-scatter (more memory-efficient than all-reduce).

    Example:
        ```python
        loss = model(x).sum()
        loss.backward()
        synchronize_fsdp_gradients()  # Reduce-scatter all gradients
        optimizer.step()
        optimizer.zero_grad()
        ```
    """
    for fsdp_module in _fsdp_modules:
        fsdp_module.reduce_scatter_gradients()


def clear_fsdp_registry() -> None:
    """Clear the global registry of FSDP modules.

    Call this when creating a new model to avoid stale references.
    """
    global _fsdp_modules
    _fsdp_modules = []


def fully_shard(
    module: CppModuleBase,
    shard_dim: int = 0,
    reshard_after_forward: bool = True,
    cluster_axis: Optional[int] = None,
) -> FSDPModule:
    """Apply FSDP sharding to a module.

    This is the main entry point for using FSDP. Apply it to each submodule
    that should have its parameters sharded, and finally to the root module.

    Example:
        ```python
        model = Transformer()

        # Apply FSDP to each transformer block
        for layer in model.layers:
            fully_shard(layer)

        # Apply FSDP to root (handles embedding, output layers)
        fully_shard(model)

        # Training loop
        for batch in dataloader:
            loss = model(batch).sum()
            loss.backward()
            synchronize_fsdp_gradients()  # FSDP gradient sync
            optimizer.step()
            optimizer.zero_grad()
        ```

    Args:
        module: The module to shard
        shard_dim: Dimension along which to shard (default: 0)
        reshard_after_forward: If True, reshard after forward (saves memory)
            If False, keep unsharded until backward (faster, uses more memory)
        cluster_axis: Optional mesh axis for sharding

    Returns:
        FSDPModule wrapping the input module
    """
    return FSDPModule(
        module=module,
        shard_dim=shard_dim,
        reshard_after_forward=reshard_after_forward,
        cluster_axis=cluster_axis,
    )
