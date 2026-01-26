# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Module system for roofline modeling with auto-registration.

This module provides MockModule, MockParameter, and MockModuleList classes
that mirror ttml.modules with PyTorch-like auto-registration via __setattr__.
"""

from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockParameter:
    """Wrapper marking a MockTensor as a trainable parameter.

    This class is used to explicitly mark tensors as parameters
    during module construction.

    Example:
        >>> class MyModule(MockModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = MockParameter(MockTensor((64, 32)))
    """

    def __init__(self, tensor: MockTensor):
        """Initialize a parameter wrapper.

        Args:
            tensor: The MockTensor to wrap as a parameter
        """
        self.tensor = tensor


class MockModule:
    """Base class for roofline-aware modules with auto-registration.

    This class mirrors PyTorch's nn.Module and ttml's module system,
    providing automatic registration of parameters and submodules
    via __setattr__.

    Subclasses should implement forward() to define the computation.

    Example:
        >>> class MyLayer(MockModule):
        ...     def __init__(self, in_features, out_features):
        ...         super().__init__()
        ...         self.weight = MockParameter(MockTensor((1, 1, out_features, in_features)))
        ...         self.bias = MockParameter(MockTensor((1, 1, 1, out_features)))
        ...
        ...     def forward(self, ctx: RooflineContext, x: MockTensor) -> MockTensor:
        ...         # Perform roofline estimation
        ...         pass
    """

    def __init__(self):
        """Initialize module with empty parameter and submodule dicts."""
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-register parameters and submodules on attribute assignment."""
        # First set the attribute
        object.__setattr__(self, name, value)

        # Auto-register submodules
        if isinstance(value, MockModule):
            self._modules[name] = value
            return

        # Auto-register parameters (either wrapped or requires_grad tensors)
        if isinstance(value, MockParameter):
            self._parameters[name] = value.tensor
        elif isinstance(value, MockTensor) and value.requires_grad:
            self._parameters[name] = value

    def parameters(self) -> Dict[str, MockTensor]:
        """Return all parameters (own + submodules) as a flat dict.

        Returns:
            Dict mapping parameter names to MockTensors.
            Submodule parameters are prefixed with "{submodule_name}.".
        """
        params = dict(self._parameters)
        for name, module in self._modules.items():
            for param_name, param in module.parameters().items():
                params[f"{name}.{param_name}"] = param
        return params

    def named_parameters(self) -> Iterator[Tuple[str, MockTensor]]:
        """Iterate over all named parameters.

        Yields:
            (name, tensor) tuples for each parameter
        """
        yield from self.parameters().items()

    def modules(self) -> Iterator["MockModule"]:
        """Iterate over all submodules (recursively).

        Yields:
            Each submodule (including self)
        """
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "MockModule"]]:
        """Iterate over all named submodules (recursively).

        Args:
            prefix: Prefix to prepend to names

        Yields:
            (name, module) tuples
        """
        yield (prefix, self) if prefix else ("", self)
        for name, module in self._modules.items():
            submodule_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(submodule_prefix)

    def forward(self, ctx: "RooflineContext", *args, **kwargs) -> Any:
        """Forward pass - must be implemented by subclasses.

        Args:
            ctx: RooflineContext for accumulating estimates
            *args: Input arguments
            **kwargs: Keyword arguments

        Returns:
            Output MockTensor(s)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, ctx: "RooflineContext", *args, **kwargs) -> Any:
        """Make module callable - delegates to forward().

        Args:
            ctx: RooflineContext for accumulating estimates
            *args: Input arguments
            **kwargs: Keyword arguments

        Returns:
            Output from forward()
        """
        return self.forward(ctx, *args, **kwargs)

    def __repr__(self) -> str:
        """String representation showing module structure."""
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            module_str = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {module_str}")
        for name, param in self._parameters.items():
            lines.append(f"  ({name}): Parameter{param.shape}")
        lines.append(")")
        return "\n".join(lines)


class MockModuleList(MockModule):
    """List container for modules with auto-registration.

    This class provides list-like access to a sequence of modules
    while automatically registering them as submodules.

    Example:
        >>> layers = MockModuleList([
        ...     MockLinearLayer(64, 128),
        ...     MockLinearLayer(128, 64),
        ... ])
        >>> for layer in layers:
        ...     x = layer(ctx, x)
    """

    def __init__(self, modules: Optional[List[MockModule]] = None):
        """Initialize with optional list of modules.

        Args:
            modules: List of modules to add
        """
        super().__init__()
        self._module_list: List[MockModule] = []
        if modules:
            for m in modules:
                self.append(m)

    def append(self, module: MockModule) -> None:
        """Add a module to the list.

        Args:
            module: Module to append
        """
        idx = len(self._module_list)
        self._module_list.append(module)
        self._modules[str(idx)] = module

    def __iter__(self) -> Iterator[MockModule]:
        """Iterate over modules."""
        return iter(self._module_list)

    def __len__(self) -> int:
        """Return number of modules."""
        return len(self._module_list)

    def __getitem__(self, idx: int) -> MockModule:
        """Get module by index."""
        return self._module_list[idx]

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Sequential forward through all modules.

        Args:
            ctx: RooflineContext for estimates
            x: Input tensor

        Returns:
            Output after passing through all modules
        """
        for module in self._module_list:
            x = module(ctx, x)
        return x

    def __repr__(self) -> str:
        """String representation showing contained modules."""
        lines = [f"{self.__class__.__name__}("]
        for i, module in enumerate(self._module_list):
            module_str = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({i}): {module_str}")
        lines.append(")")
        return "\n".join(lines)


class MockModuleDict(MockModule):
    """Dict container for modules with auto-registration.

    Similar to MockModuleList but with string keys.

    Example:
        >>> layers = MockModuleDict({
        ...     'encoder': EncoderModule(),
        ...     'decoder': DecoderModule(),
        ... })
        >>> enc_out = layers['encoder'](ctx, x)
    """

    def __init__(self, modules: Optional[Dict[str, MockModule]] = None):
        """Initialize with optional dict of modules.

        Args:
            modules: Dict mapping names to modules
        """
        super().__init__()
        self._module_dict: Dict[str, MockModule] = {}
        if modules:
            for name, m in modules.items():
                self[name] = m

    def __setitem__(self, key: str, module: MockModule) -> None:
        """Add a module with given key."""
        self._module_dict[key] = module
        self._modules[key] = module

    def __getitem__(self, key: str) -> MockModule:
        """Get module by key."""
        return self._module_dict[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._module_dict)

    def __len__(self) -> int:
        """Return number of modules."""
        return len(self._module_dict)

    def keys(self):
        """Return module keys."""
        return self._module_dict.keys()

    def values(self):
        """Return modules."""
        return self._module_dict.values()

    def items(self):
        """Return (key, module) pairs."""
        return self._module_dict.items()

    def __repr__(self) -> str:
        """String representation showing contained modules."""
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._module_dict.items():
            module_str = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {module_str}")
        lines.append(")")
        return "\n".join(lines)
