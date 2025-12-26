# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for Python modules that mirror ttml's ModuleBase interface.

This module provides AbstractModuleBase, a Python abstract base class that defines
the interface for future Python modules that implement functionality similar to
ttml's C++ ModuleBase class. This ensures consistency between Python and C++ module
implementations.

The interface is designed to be PyTorch-like, with automatic parameter and module
tracking via attribute assignment, eliminating the need for manual registration.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterator, Tuple

from .._ttml.modules import RunMode

from .exceptions import (
    DuplicateNameError,
    NameNotFoundError,
    UninitializedModuleError,
)
from .parameter import Buffer, Parameter


class AbstractModuleBase(ABC):
    """Abstract base class for Python modules mirroring ttml's ModuleBase.

    This class defines the common interface for Python modules that need to
    integrate with or mirror the behavior of ttml's C++ ModuleBase. It provides:

    - PyTorch-like automatic parameter and module tracking
    - Name management for module identification
    - Run mode management (TRAIN/EVAL) with propagation to submodules
    - Parameter registration and retrieval with hierarchical naming
    - Support for nested modules and weight tying
    - Abstract forward pass methods that must be implemented by subclasses

    The interface is designed to be PyTorch-like while maintaining compatibility
    with the C++ ModuleBase. Parameters and modules are automatically registered
    when assigned as attributes, eliminating the need for manual registration.

    Example (PyTorch-like automatic registration):
        >>> class MyModule(AbstractModuleBase):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = create_tensor(...)  # Auto-registered
        ...         self.submodule = SubModule()     # Auto-registered
        ...
        ...     def __call__(self, tensor):
        ...         # Implement forward pass
        ...         return tensor

    Example (Manual registration - still supported):
        >>> class MyModule(AbstractModuleBase):
        ...     def __init__(self):
        ...         super().__init__()
        ...         weight = create_tensor(...)
        ...         self._register_tensor(weight, "weight")
    """

    def __init__(self) -> None:
        """Initialize the abstract module base.

        Sets up internal state for name, run mode, and parameter tracking.
        Automatically sets the module name to the subclass name if not already set.
        """
        # Set these directly to avoid triggering __setattr__ during initialization
        object.__setattr__(self, "_name", None)
        object.__setattr__(self, "_run_mode", RunMode.TRAIN)
        # Use OrderedDict to match C++ std::map behavior (ordered iteration for serialization)
        object.__setattr__(self, "_named_tensors", OrderedDict())
        object.__setattr__(self, "_named_modules", OrderedDict())
        object.__setattr__(self, "_named_buffers", OrderedDict())

        # Automatically set module name to subclass name if not explicitly set
        # This provides a convenient default while still allowing manual override
        # Subclasses can call self._create_name("custom_name") to override
        if self._name is None:
            subclass_name = self.__class__.__name__
            self._create_name(subclass_name)

    def _is_tensor(self, value: Any) -> bool:
        """Check if a value is a tensor-like object.

        This method uses duck typing to detect tensors by checking for
        common tensor attributes like 'shape' and 'dtype'.

        Args:
            value: The value to check.

        Returns:
            True if the value appears to be a tensor, False otherwise.
        """
        # Check for common tensor attributes
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return True
        # Could add more specific type checks here if needed
        return False

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, automatically registering modules and parameters.

        This method automatically detects and registers:
        - AbstractModuleBase instances as submodules
        - Parameter instances as parameters
        - Buffer instances as buffers
        - Tensor-like objects as parameters (if not wrapped)

        Args:
            name: The attribute name.
            value: The value to assign.
        """
        # Skip during initialization (before _named_tensors is set)
        if "_named_tensors" not in self.__dict__:
            object.__setattr__(self, name, value)
            return

        # Skip private attributes
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        # Remove from old registrations if reassigning
        self._named_tensors.pop(name, None)
        self._named_modules.pop(name, None)
        self._named_buffers.pop(name, None)

        # Auto-register modules
        if isinstance(value, AbstractModuleBase):
            self._named_modules[name] = value
            object.__setattr__(self, name, value)
            return

        # Auto-register parameters
        if isinstance(value, Parameter):
            self._named_tensors[name] = value.tensor
            object.__setattr__(self, name, value)
            return

        # Auto-register buffers
        if isinstance(value, Buffer):
            self._named_buffers[name] = value.tensor
            object.__setattr__(self, name, value)
            return

        # Auto-register tensor-like objects as parameters
        if self._is_tensor(value):
            self._named_tensors[name] = value
            object.__setattr__(self, name, value)
            return

        # Regular attribute
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute and remove it from registrations.

        Args:
            name: The attribute name to delete.
        """
        # Remove from registrations
        self._named_tensors.pop(name, None)
        self._named_modules.pop(name, None)
        self._named_buffers.pop(name, None)
        object.__delattr__(self, name)

    def get_name(self) -> str:
        """Get the module's name.

        Returns:
            The module's name.

        Raises:
            ValueError: If the module name has not been set.
        """
        if self._name is None:
            raise ValueError("Module name has not been set. Call _create_name() first.")
        return self._name

    def _create_name(self, name: str) -> None:
        """Set the module's name.

        This is a protected method intended for use by subclasses during initialization.

        Args:
            name: The name to assign to this module.
        """
        if not isinstance(name, str):
            raise TypeError(f"Module name must be a string, got {type(name)}")
        self._name = name

    def train(self) -> None:
        """Set the module to training mode.

        This sets the run mode to TRAIN and propagates the change to all
        registered submodules.
        """
        self.set_run_mode(RunMode.TRAIN)

    def eval(self) -> None:
        """Set the module to evaluation mode.

        This sets the run mode to EVAL and propagates the change to all
        registered submodules.
        """
        self.set_run_mode(RunMode.EVAL)

    def set_run_mode(self, mode: RunMode) -> None:
        """Set the run mode for this module and all submodules.

        Args:
            mode: The run mode to set (TRAIN or EVAL).

        Raises:
            TypeError: If mode is not a RunMode enum value.
        """
        # Check if mode is a valid RunMode enum value
        # Check if it's the same type as known enum values (works for both Python and nanobind enums)
        if type(mode) is not type(RunMode.TRAIN):
            raise TypeError(
                f"mode must be a RunMode enum value, got {mode} (type: {type(mode)})"
            )
        self._run_mode = mode
        # Propagate to all submodules
        for module in self._named_modules.values():
            module.set_run_mode(mode)

    def get_run_mode(self) -> RunMode:
        """Get the current run mode.

        Returns:
            The current run mode (TRAIN or EVAL).
        """
        return self._run_mode

    def parameters_dict(
        self, separator: str = "/", use_dot: bool = False
    ) -> Dict[str, Any]:
        """Get all parameters from this module and all submodules as a dictionary.

        Returns a flat dictionary mapping parameter names to tensors. The names
        use hierarchical naming with the specified separator.

        Weight tying is handled by tracking tensor identity (using id()) to avoid
        duplicates, similar to how C++ uses tensor addresses.

        This method maintains backward compatibility with the C++ ModuleBase interface.
        For PyTorch-like iteration, use `parameters()` or `named_parameters()`.

        Args:
            separator: Separator to use for hierarchical names (default: "/" for C++ compatibility).
            use_dot: If True, use "." as separator (PyTorch convention). Overrides separator.

        Returns:
            A dictionary mapping hierarchical parameter names to tensors.
        """
        sep = "." if use_dot else separator
        params: Dict[str, Any] = {}
        name_prefix = self.get_name() + sep if self._name else ""

        # Track tensor identities to handle weight tying (avoid duplicates)
        tensors_in_params: set[int] = set()

        # Queue of (module, name_prefix) pairs to process
        modules_to_process: list[tuple["AbstractModuleBase", str]] = [
            (self, name_prefix)
        ]
        modules_in_queue: set[str] = {name_prefix}

        while modules_to_process:
            module_ptr, current_prefix = modules_to_process.pop(0)
            # Process all tensors in this module
            for tensor_name, tensor_ptr in module_ptr._named_tensors.items():
                tensor_id = id(tensor_ptr)
                if tensor_id not in tensors_in_params:
                    tensors_in_params.add(tensor_id)
                    params[current_prefix + tensor_name] = tensor_ptr

            # Add submodules to the queue
            for module_name, next_module_ptr in module_ptr._named_modules.items():
                module_name_with_prefix = current_prefix + module_name
                if module_name_with_prefix not in modules_in_queue:
                    modules_to_process.append(
                        (next_module_ptr, module_name_with_prefix + sep)
                    )
                    modules_in_queue.add(module_name_with_prefix)

        return params

    # Alias for backward compatibility with C++ interface
    def parameters(self, separator: str = "/", use_dot: bool = False) -> Dict[str, Any]:
        """Get all parameters as a dictionary (backward compatibility).

        This method is an alias for `parameters_dict()` to maintain compatibility
        with the C++ ModuleBase interface. For PyTorch-like iteration, use the
        iterator version or `named_parameters()`.

        Args:
            separator: Separator to use for hierarchical names (default: "/").
            use_dot: If True, use "." as separator. Overrides separator.

        Returns:
            A dictionary mapping hierarchical parameter names to tensors.
        """
        return self.parameters_dict(separator=separator, use_dot=use_dot)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Any]]:
        """Yield (name, parameter) pairs from this module and submodules.

        Args:
            prefix: Prefix to prepend to all parameter names.
            recurse: If True, recursively yield parameters from submodules.

        Yields:
            (name, parameter) tuples where name uses "." as separator.
        """
        for name, tensor in self._named_tensors.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield (full_name, tensor)

        if recurse:
            for child_name, child in self._named_modules.items():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                yield from child.named_parameters(prefix=child_prefix, recurse=True)

    def named_modules(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, "AbstractModuleBase"]]:
        """Yield (name, module) pairs from this module and submodules.

        Args:
            prefix: Prefix to prepend to all module names.
            recurse: If True, recursively yield modules from submodules.

        Yields:
            (name, module) tuples where name uses "." as separator.
        """
        if prefix:
            yield (prefix, self)
        else:
            yield ("", self)

        if recurse:
            for child_name, child in self._named_modules.items():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                yield from child.named_modules(prefix=child_prefix, recurse=True)

    def named_children(self) -> Iterator[Tuple[str, "AbstractModuleBase"]]:
        """Yield (name, module) pairs for direct children only.

        Yields:
            (name, module) tuples for direct submodules only.
        """
        yield from self._named_modules.items()

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Any]]:
        """Yield (name, buffer) pairs from this module and submodules.

        Args:
            prefix: Prefix to prepend to all buffer names.
            recurse: If True, recursively yield buffers from submodules.

        Yields:
            (name, buffer) tuples where name uses "." as separator.
        """
        for name, buffer in self._named_buffers.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield (full_name, buffer)

        if recurse:
            for child_name, child in self._named_modules.items():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                yield from child.named_buffers(prefix=child_prefix, recurse=True)

    def modules(self, recurse: bool = True) -> Iterator["AbstractModuleBase"]:
        """Yield modules from this module and submodules.

        Args:
            recurse: If True, recursively yield modules from submodules.

        Yields:
            Module instances.
        """
        for _, module in self.named_modules(recurse=recurse):
            yield module

    def children(self) -> Iterator["AbstractModuleBase"]:
        """Yield direct child modules only.

        Yields:
            Direct submodule instances.
        """
        for _, child in self.named_children():
            yield child

    def buffers(self, recurse: bool = True) -> Iterator[Any]:
        """Yield buffers from this module and submodules.

        Args:
            recurse: If True, recursively yield buffers from submodules.

        Yields:
            Buffer tensors.
        """
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def state_dict(self, prefix: str = "", keep_vars: bool = False) -> Dict[str, Any]:
        """Return a dictionary containing the module's state.

        Args:
            prefix: Prefix to prepend to all parameter names.
            keep_vars: If False, return plain tensors. If True, return Parameter/Buffer objects.

        Returns:
            A dictionary mapping parameter names to their values.
        """
        state_dict: Dict[str, Any] = {}

        # Add parameters
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if keep_vars:
                # Wrap in Parameter if it was originally a Parameter
                state_dict[name] = Parameter(param)
            else:
                state_dict[name] = param

        # Add buffers
        for name, buffer in self.named_buffers(prefix=prefix, recurse=True):
            if keep_vars:
                state_dict[name] = Buffer(buffer)
            else:
                state_dict[name] = buffer

        return state_dict

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True
    ) -> Tuple[list[str], list[str]]:
        """Load parameters and buffers from a state dictionary.

        Args:
            state_dict: Dictionary mapping parameter names to values.
            strict: If True, raise an error if keys are missing or unexpected.

        Returns:
            Tuple of (missing_keys, unexpected_keys).

        Raises:
            RuntimeError: If strict=True and there are missing or unexpected keys.
        """
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []

        # Build a map of all named parameters and buffers
        all_params = dict(self.named_parameters(recurse=True))
        all_buffers = dict(self.named_buffers(recurse=True))

        # Track which keys we've used
        used_keys: set[str] = set()

        # Load parameters
        for name, value in state_dict.items():
            if name in all_params:
                # Extract tensor from Parameter wrapper if needed
                if isinstance(value, Parameter):
                    tensor_value = value.tensor
                elif isinstance(value, Buffer):
                    tensor_value = value.tensor
                else:
                    tensor_value = value

                # Find the module and attribute name
                parts = name.split(".")
                if len(parts) == 1:
                    # Direct parameter
                    self._named_tensors[parts[0]] = tensor_value
                else:
                    # Nested parameter - find the module
                    module = self
                    for part in parts[:-1]:
                        if part not in module._named_modules:
                            missing_keys.append(name)
                            break
                        module = module._named_modules[part]
                    else:
                        attr_name = parts[-1]
                        module._named_tensors[attr_name] = tensor_value

                used_keys.add(name)
            elif name in all_buffers:
                # Similar logic for buffers
                if isinstance(value, Buffer):
                    tensor_value = value.tensor
                elif isinstance(value, Parameter):
                    tensor_value = value.tensor
                else:
                    tensor_value = value

                parts = name.split(".")
                if len(parts) == 1:
                    self._named_buffers[parts[0]] = tensor_value
                else:
                    module = self
                    for part in parts[:-1]:
                        if part not in module._named_modules:
                            missing_keys.append(name)
                            break
                        module = module._named_modules[part]
                    else:
                        attr_name = parts[-1]
                        module._named_buffers[attr_name] = tensor_value

                used_keys.add(name)
            else:
                unexpected_keys.append(name)

        # Check for missing keys
        for name in all_params:
            if name not in used_keys:
                missing_keys.append(name)
        for name in all_buffers:
            if name not in used_keys:
                missing_keys.append(name)

        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys in state_dict: {unexpected_keys}")

        return (missing_keys, unexpected_keys)

    def _register_tensor(self, tensor: Any, name: str) -> None:
        """Register a tensor with a name.

        This is a protected method intended for use by subclasses during initialization.

        Args:
            tensor: The tensor to register.
            name: The name to assign to the tensor.

        Raises:
            TypeError: If name is not a string.
            DuplicateNameError: If a tensor with this name already exists.
        """
        if not isinstance(name, str):
            raise TypeError(f"Tensor name must be a string, got {type(name)}")
        if name in self._named_tensors:
            raise DuplicateNameError(
                f"Tensor with name '{name}' already exists in module '{self._name or 'unnamed'}'"
            )
        self._named_tensors[name] = tensor

    def _register_module(self, module: "AbstractModuleBase", name: str) -> None:
        """Register a submodule with a name.

        This is a protected method intended for use by subclasses during initialization.

        Args:
            module: The submodule to register.
            name: The name to assign to the submodule.

        Raises:
            TypeError: If name is not a string.
            UninitializedModuleError: If module is None.
            DuplicateNameError: If a module with this name already exists.
        """
        if not isinstance(name, str):
            raise TypeError(f"Module name must be a string, got {type(name)}")
        if module is None:
            raise UninitializedModuleError("Cannot register uninitialized module")
        if name in self._named_modules:
            raise DuplicateNameError(
                f"Module with name '{name}' already exists in module '{self._name or 'unnamed'}'"
            )
        self._named_modules[name] = module

    def _override_tensor(self, tensor: Any, name: str) -> None:
        """Override an existing tensor with a new one.

        This is a protected method intended for use by subclasses.

        Args:
            tensor: The new tensor to use.
            name: The name of the tensor to override.

        Raises:
            TypeError: If name is not a string.
            NameNotFoundError: If no tensor with this name exists.
        """
        if not isinstance(name, str):
            raise TypeError(f"Tensor name must be a string, got {type(name)}")
        if name not in self._named_tensors:
            raise NameNotFoundError(
                f"Tensor with name '{name}' does not exist in module '{self._name or 'unnamed'}'"
            )
        self._named_tensors[name] = tensor

    def _override_module(self, module: "AbstractModuleBase", name: str) -> None:
        """Override an existing submodule with a new one.

        This is a protected method intended for use by subclasses.

        Args:
            module: The new submodule to use.
            name: The name of the submodule to override.

        Raises:
            TypeError: If name is not a string.
            UninitializedModuleError: If module is None.
            NameNotFoundError: If no module with this name exists.
        """
        if not isinstance(name, str):
            raise TypeError(f"Module name must be a string, got {type(name)}")
        if module is None:
            raise UninitializedModuleError(
                f"Cannot override with None module for name '{name}'"
            )
        if name not in self._named_modules:
            raise NameNotFoundError(
                f"Module with name '{name}' does not exist in module '{self._name or 'unnamed'}'"
            )
        self._named_modules[name] = module

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass for the module.

        This is an abstract method that must be implemented by subclasses.
        The implementation should define the forward pass computation.

        The method signature is flexible to support different module types:
        - Single tensor input: `__call__(self, tensor) -> tensor`
        - Dual tensor input: `__call__(self, tensor, other) -> tensor`
        - Additional arguments: `__call__(self, *args, **kwargs) -> Any`

        Args:
            *args: Variable positional arguments (typically one or two tensors).
            **kwargs: Variable keyword arguments for additional configuration.

        Returns:
            The output of the forward pass (typically a tensor or tuple of tensors).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            f"Subclasses of {self.__class__.__name__} must implement the __call__ method"
        )
