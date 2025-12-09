# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for Python modules that mirror ttml's ModuleBase interface.

This module provides AbstractModuleBase, a Python abstract base class that defines
the interface for future Python modules that implement functionality similar to
ttml's C++ ModuleBase class. This ensures consistency between Python and C++ module
implementations.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Optional

from .._ttml import RunMode

from .exceptions import (
    DuplicateNameError,
    NameNotFoundError,
    UninitializedModuleError,
)


class AbstractModuleBase(ABC):
    """Abstract base class for Python modules mirroring ttml's ModuleBase.

    This class defines the common interface for Python modules that need to
    integrate with or mirror the behavior of ttml's C++ ModuleBase. It provides:

    - Name management for module identification
    - Run mode management (TRAIN/EVAL) with propagation to submodules
    - Parameter registration and retrieval with hierarchical naming
    - Support for nested modules and weight tying
    - Abstract forward pass methods that must be implemented by subclasses

    The interface is designed to match the C++ ModuleBase as closely as possible
    while following Python conventions (e.g., using OrderedDict instead of std::map).

    Example:
        >>> class MyModule(AbstractModuleBase):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._create_name("my_module")
        ...         # Register tensors and submodules...
        ...
        ...     def __call__(self, tensor):
        ...         # Implement forward pass
        ...         return tensor
    """

    def __init__(self) -> None:
        """Initialize the abstract module base.

        Sets up internal state for name, run mode, and parameter tracking.
        """
        self._name: Optional[str] = None
        self._run_mode: RunMode = RunMode.TRAIN
        # Use OrderedDict to match C++ std::map behavior (ordered iteration for serialization)
        self._named_tensors: OrderedDict[str, Any] = OrderedDict()
        self._named_modules: OrderedDict[str, "AbstractModuleBase"] = OrderedDict()

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

    def parameters(self) -> Dict[str, Any]:
        """Get all parameters from this module and all submodules.

        Returns a flat dictionary mapping parameter names to tensors. The names
        use hierarchical naming with "/" as a separator (e.g., "module/submodule/weight").
        This matches the C++ ModuleBase behavior.

        Weight tying is handled by tracking tensor identity (using id()) to avoid
        duplicates, similar to how C++ uses tensor addresses.

        Returns:
            A dictionary mapping hierarchical parameter names to tensors.
        """
        params: Dict[str, Any] = {}
        name_prefix = self.get_name() + "/" if self._name else ""

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
                        (next_module_ptr, module_name_with_prefix + "/")
                    )
                    modules_in_queue.add(module_name_with_prefix)

        return params

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
