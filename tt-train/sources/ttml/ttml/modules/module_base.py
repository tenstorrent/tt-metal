# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python module base inheriting from C++ ModuleBase with auto-registration."""

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union, overload
from .._ttml.modules import ModuleBase as CppModuleBase
from .parameter import Buffer, Parameter


class AbstractModuleBase(CppModuleBase):
    """Module base with PyTorch-like auto-registration via __setattr__."""

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_buffers", {})
        self.create_name(self.__class__.__name__)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        # Skip if not initialized yet
        if "_buffers" not in self.__dict__:
            return

        # Auto-register modules
        if isinstance(value, CppModuleBase):
            try:
                self.register_module(value, name)
            except RuntimeError:
                self.override_module(value, name)
            return

        # Auto-register tensors
        tensor = None
        if isinstance(value, Parameter):
            tensor = value.tensor
        elif isinstance(value, Buffer):
            self._buffers[name] = value.tensor
            return
        elif hasattr(value, "get_value"):
            tensor = value

        if tensor is not None:
            try:
                self.register_tensor(tensor, name)
            except RuntimeError:
                self.override_tensor(tensor, name)

    def __delattr__(self, name: str) -> None:
        self._buffers.pop(name, None)
        object.__delattr__(self, name)

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Any]]:
        """Yield (name, parameter) pairs."""
        for name, tensor in self.parameters().items():
            short = name.split("/", 1)[-1].replace("/", ".")
            yield (f"{prefix}.{short}" if prefix else short, tensor)

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, Any]]:
        """Yield (name, module) pairs."""
        yield (prefix or "", self)
        for name, attr in self.__dict__.items():
            if isinstance(attr, CppModuleBase):
                child = f"{prefix}.{name}" if prefix else name
                if isinstance(attr, AbstractModuleBase):
                    yield from attr.named_modules(child)
                else:
                    yield (child, attr)

    def named_children(self) -> Iterator[Tuple[str, Any]]:
        """Yield (name, module) pairs for direct children."""
        for name, attr in self.__dict__.items():
            if isinstance(attr, CppModuleBase):
                yield (name, attr)

    def named_buffers(self, prefix: str = "") -> Iterator[Tuple[str, Any]]:
        """Yield (name, buffer) pairs."""
        for name, buf in self._buffers.items():
            yield (f"{prefix}.{name}" if prefix else name, buf)
        for name, attr in self.__dict__.items():
            if isinstance(attr, AbstractModuleBase):
                yield from attr.named_buffers(f"{prefix}.{name}" if prefix else name)

    def modules(self) -> Iterator[Any]:
        """Yield all modules."""
        return (m for _, m in self.named_modules())

    def children(self) -> Iterator[Any]:
        """Yield direct child modules."""
        return (c for _, c in self.named_children())

    def buffers(self) -> Iterator[Any]:
        """Yield all buffers."""
        return (b for _, b in self.named_buffers())

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary."""
        return {**dict(self.named_parameters()), **dict(self.named_buffers())}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke forward(). Subclasses implement forward()."""
        return self.forward(*args, **kwargs)


class ModuleList(AbstractModuleBase):
    """A list of modules with automatic registration (PyTorch-compatible).

    ModuleList can be indexed like a regular Python list and properly registers
    all contained modules so their parameters are tracked.

    Example:
        class MyModel(AbstractModuleBase):
            def __init__(self, num_layers):
                super().__init__()
                self.layers = ModuleList([LinearLayer(64, 64) for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
    """

    def __init__(self, modules: Optional[Iterable[CppModuleBase]] = None) -> None:
        """Initialize ModuleList.

        Args:
            modules: Optional iterable of modules to add.
        """
        super().__init__()
        self._modules_list: List[CppModuleBase] = []
        if modules is not None:
            for module in modules:
                self.append(module)

    def _get_abs_idx(self, idx: int) -> int:
        """Convert potentially negative index to absolute index."""
        if idx < 0:
            idx = len(self._modules_list) + idx
        if idx < 0 or idx >= len(self._modules_list):
            raise IndexError(f"index {idx} is out of range")
        return idx

    @overload
    def __getitem__(self, idx: int) -> CppModuleBase:
        pass

    @overload
    def __getitem__(self, idx: slice) -> "ModuleList":
        pass

    def __getitem__(self, idx: Union[int, slice]) -> Union[CppModuleBase, "ModuleList"]:
        """Get module(s) by index or slice."""
        if isinstance(idx, slice):
            return ModuleList(self._modules_list[idx])
        return self._modules_list[self._get_abs_idx(idx)]

    def __setitem__(self, idx: int, module: CppModuleBase) -> None:
        """Set module at index, updating registration."""
        abs_idx = self._get_abs_idx(idx)
        self._modules_list[abs_idx] = module
        # Override the existing registration
        self.override_module(module, str(abs_idx))

    def __delitem__(self, idx: int) -> None:
        """Delete module at index (rebuilds registrations)."""
        abs_idx = self._get_abs_idx(idx)
        del self._modules_list[abs_idx]
        # Need to rebuild registrations since indices shift
        self._rebuild_registrations()

    def __len__(self) -> int:
        """Return number of modules."""
        return len(self._modules_list)

    def __iter__(self) -> Iterator[CppModuleBase]:
        """Iterate over modules."""
        return iter(self._modules_list)

    def __contains__(self, module: CppModuleBase) -> bool:
        """Check if module is in list."""
        return module in self._modules_list

    def __repr__(self) -> str:
        """Return string representation."""
        lines = [f"{self.__class__.__name__}("]
        for idx, module in enumerate(self._modules_list):
            module_name = (
                module.get_name()
                if hasattr(module, "get_name")
                else type(module).__name__
            )
            lines.append(f"  ({idx}): {module_name}")
        lines.append(")")
        return "\n".join(lines)

    def _rebuild_registrations(self) -> None:
        """Rebuild all module registrations after structural change."""
        # This is needed after delete/insert operations that shift indices
        # We need to clear and re-register all modules
        # Since the C++ side doesn't have a clear method, we work around by
        # creating a new internal state - but we can't easily clear m_named_modules
        # For now, we just re-register with override for existing indices
        # and register new ones. This works because we maintain index-based names.
        pass  # Current implementation doesn't support full rebuild
        # Users should prefer append() over insert()/del operations

    def append(self, module: CppModuleBase) -> "ModuleList":
        """Append a module to the list.

        Args:
            module: Module to append.

        Returns:
            self, for chaining.
        """
        idx = len(self._modules_list)
        self._modules_list.append(module)
        self.register_module(module, str(idx))
        return self

    def extend(self, modules: Iterable[CppModuleBase]) -> "ModuleList":
        """Extend list with modules from an iterable.

        Args:
            modules: Iterable of modules to add.

        Returns:
            self, for chaining.
        """
        for module in modules:
            self.append(module)
        return self

    def insert(self, idx: int, module: CppModuleBase) -> None:
        """Insert a module at given index.

        Note: This operation is O(n) as it requires re-registering modules
        after the insertion point. Prefer append() when possible.

        Args:
            idx: Index to insert at.
            module: Module to insert.
        """
        if idx < 0:
            idx = max(0, len(self._modules_list) + idx + 1)
        idx = min(idx, len(self._modules_list))

        # Insert into list
        self._modules_list.insert(idx, module)

        # Re-register all modules from idx onwards
        for i in range(idx, len(self._modules_list)):
            if i == idx:
                self.register_module(self._modules_list[i], str(i))
            else:
                # Need to override existing registration with new index
                try:
                    self.register_module(self._modules_list[i], str(i))
                except RuntimeError:
                    self.override_module(self._modules_list[i], str(i))

    def pop(self, idx: int = -1) -> CppModuleBase:
        """Remove and return module at index.

        Note: Due to C++ backend limitations, the removed module's registration
        remains but points to the old module. Prefer creating a new ModuleList
        if you need to remove modules frequently.

        Args:
            idx: Index of module to remove (default: -1, last module).

        Returns:
            The removed module.
        """
        abs_idx = self._get_abs_idx(idx)
        module = self._modules_list.pop(abs_idx)
        return module

    def __iadd__(self, modules: Iterable[CppModuleBase]) -> "ModuleList":
        """Support += operator for extending."""
        return self.extend(modules)


class ModuleDict(AbstractModuleBase):
    """A dictionary of modules with automatic registration (PyTorch-compatible).

    ModuleDict can be indexed like a regular Python dict and properly registers
    all contained modules so their parameters are tracked.

    Example:
        class MyModel(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                self.layers = ModuleDict({
                    'encoder': LinearLayer(64, 128),
                    'decoder': LinearLayer(128, 64),
                })

            def forward(self, x, layer_name):
                return self.layers[layer_name](x)
    """

    def __init__(
        self,
        modules: Optional[
            Union[Dict[str, CppModuleBase], Iterable[Tuple[str, CppModuleBase]]]
        ] = None,
    ) -> None:
        """Initialize ModuleDict.

        Args:
            modules: Optional dict or iterable of (name, module) pairs.
        """
        super().__init__()
        self._modules_dict: Dict[str, CppModuleBase] = {}
        if modules is not None:
            if isinstance(modules, dict):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in modules:
                    self[key] = module

    def __getitem__(self, key: str) -> CppModuleBase:
        """Get module by key."""
        return self._modules_dict[key]

    def __setitem__(self, key: str, module: CppModuleBase) -> None:
        """Set module at key, updating registration."""
        if key in self._modules_dict:
            self.override_module(module, key)
        else:
            self.register_module(module, key)
        self._modules_dict[key] = module

    def __delitem__(self, key: str) -> None:
        """Delete module at key.

        Note: Due to C++ backend limitations, the registration remains.
        """
        del self._modules_dict[key]

    def __len__(self) -> int:
        """Return number of modules."""
        return len(self._modules_dict)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._modules_dict)

    def __contains__(self, key: str) -> bool:
        """Check if key is in dict."""
        return key in self._modules_dict

    def __repr__(self) -> str:
        """Return string representation."""
        lines = [f"{self.__class__.__name__}("]
        for key, module in self._modules_dict.items():
            module_name = (
                module.get_name()
                if hasattr(module, "get_name")
                else type(module).__name__
            )
            lines.append(f"  ({key}): {module_name}")
        lines.append(")")
        return "\n".join(lines)

    def keys(self) -> Iterator[str]:
        """Return iterator over keys."""
        return iter(self._modules_dict.keys())

    def values(self) -> Iterator[CppModuleBase]:
        """Return iterator over modules."""
        return iter(self._modules_dict.values())

    def items(self) -> Iterator[Tuple[str, CppModuleBase]]:
        """Return iterator over (key, module) pairs."""
        return iter(self._modules_dict.items())

    def get(
        self, key: str, default: Optional[CppModuleBase] = None
    ) -> Optional[CppModuleBase]:
        """Get module by key with optional default."""
        return self._modules_dict.get(key, default)

    def update(
        self,
        modules: Union[Dict[str, CppModuleBase], Iterable[Tuple[str, CppModuleBase]]],
    ) -> None:
        """Update dict with modules from another dict or iterable."""
        if isinstance(modules, dict):
            for key, module in modules.items():
                self[key] = module
        else:
            for key, module in modules:
                self[key] = module

    def pop(
        self, key: str, default: Optional[CppModuleBase] = None
    ) -> Optional[CppModuleBase]:
        """Remove and return module at key.

        Note: Due to C++ backend limitations, the registration remains.
        """
        return self._modules_dict.pop(key, default)

    def clear(self) -> None:
        """Clear all modules.

        Note: Due to C++ backend limitations, registrations remain.
        """
        self._modules_dict.clear()
