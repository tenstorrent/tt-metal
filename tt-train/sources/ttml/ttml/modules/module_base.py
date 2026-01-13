# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python module base inheriting from C++ ModuleBase with auto-registration."""

from typing import Any, Dict, Iterator, Tuple
from .._ttml.modules import ModuleBase as CppModuleBase
from .parameter import Buffer, Parameter


class AbstractModuleBase(CppModuleBase):
    """Module base with PyTorch-like auto-registration via __setattr__."""

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_registered", set())
        self.create_name(self.__class__.__name__)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        # Skip if not initialized yet
        if "_registered" not in self.__dict__:
            return

        # Auto-register modules
        if isinstance(value, CppModuleBase):
            try:
                self.register_module(value, name)
            except Exception:
                try:
                    self.override_module(value, name)
                except Exception:
                    pass
            self._registered.add(name)
            return

        # Auto-register tensors
        tensor = None
        if isinstance(value, Parameter):
            tensor = value.tensor
        elif isinstance(value, Buffer):
            self._buffers[name] = value.tensor
            self._registered.add(name)
            return
        elif hasattr(value, "get_value"):
            tensor = value

        if tensor is not None:
            try:
                self.register_tensor(tensor, name)
            except Exception:
                try:
                    self.override_tensor(tensor, name)
                except Exception:
                    pass
            self._registered.add(name)

    def __delattr__(self, name: str) -> None:
        self._registered.discard(name)
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
        for name in self._registered:
            attr = getattr(self, name, None)
            if isinstance(attr, CppModuleBase):
                child = f"{prefix}.{name}" if prefix else name
                if isinstance(attr, AbstractModuleBase):
                    yield from attr.named_modules(child)
                else:
                    yield (child, attr)

    def named_children(self) -> Iterator[Tuple[str, Any]]:
        """Yield (name, module) pairs for direct children."""
        for name in self._registered:
            attr = getattr(self, name, None)
            if isinstance(attr, CppModuleBase):
                yield (name, attr)

    def named_buffers(self, prefix: str = "") -> Iterator[Tuple[str, Any]]:
        """Yield (name, buffer) pairs."""
        for name, buf in self._buffers.items():
            yield (f"{prefix}.{name}" if prefix else name, buf)
        for name in self._registered:
            attr = getattr(self, name, None)
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
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__")
