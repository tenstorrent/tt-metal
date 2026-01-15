# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, overload

import ttnn
from typing_extensions import deprecated

from ..utils import tensor
from ..utils.substate import pop_substate

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, MutableSequence, Sequence
    from typing import Any

    import torch


class IncompatibleKeys(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]


class LoadingError(Exception):
    pass


class Module(ABC):
    def __init__(self) -> None:
        self._children = {}
        self._parameters = {}
        self._is_loaded = False

    def named_children(self) -> Iterator[tuple[str, Module]]:
        yield from self._children.items()

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        yield from self._parameters.items()

    def add_module(self, name: str, module: Module) -> None:
        self._children[name] = module

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        super().__setattr__(name, value)

        if name in ("_children", "_parameters"):
            return

        children = self.__dict__.get("_children")
        parameters = self.__dict__.get("_parameters")

        if isinstance(value, Module):
            if children is None:
                msg = "cannot assign child module before Module.__init__() call"
                raise AttributeError(msg)
            self._children[name] = value
        elif isinstance(value, Parameter):
            if parameters is None:
                msg = "cannot assign parameter before Module.__init__() call"
                raise AttributeError(msg)
            self._parameters[name] = value
        else:
            if children is not None:
                children.pop(name, None)
            if parameters is not None:
                parameters.pop(name, None)

    def __delattr__(self, name: str) -> None:
        children = self.__dict__.get("_children")
        parameters = self.__dict__.get("_parameters")

        if children is not None:
            children.pop(name, None)
        if parameters is not None:
            parameters.pop(name, None)

        super().__delattr__(name)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:  # noqa: B027
        """Prepare a PyTorch `state_dict` in place before loading.

        Override this method to adjust entries before loading them into submodules and parameters.

        This method should modify `state` in place and, where possible, avoid raising exceptions for
        missing keys; skip them instead. This way, missing keys can be collected and returned by
        `load_torch_state_dict`.
        """

    def _load_torch_state_dict_inner(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        module_key_prefix: str,
        missing_keys: MutableSequence[str],
        unexpected_keys: MutableSequence[str],
    ) -> None:
        state_dict = dict(state_dict)
        self._prepare_torch_state(state_dict)

        for name, child in self.named_children():
            child_state = pop_substate(state_dict, name)

            try:
                child._load_torch_state_dict_inner(  # noqa: SLF001
                    child_state,
                    module_key_prefix=f"{module_key_prefix}{name}.",
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys,
                )
            except LoadingError:
                raise
            except Exception as err:
                msg = f"an exception occurred while loading '{module_key_prefix}{name}'"
                raise LoadingError(msg) from err

        for name, parameter in self.named_parameters():
            if name in state_dict:
                try:
                    parameter.load_torch_tensor(state_dict.pop(name))
                except LoadingError as err:
                    msg = f"while loading '{module_key_prefix}{name}': {err}"
                    raise LoadingError(msg) from err
            else:
                missing_keys.append(f"{module_key_prefix}{name}")

        for name in state_dict:
            unexpected_keys.append(f"{module_key_prefix}{name}")

    def load_torch_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True) -> IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []

        self._load_torch_state_dict_inner(
            state_dict, module_key_prefix="", missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

        if strict and (missing_keys or unexpected_keys):
            parts = []
            if missing_keys:
                parts.append("missing Torch state keys: " + ", ".join(missing_keys))
            if unexpected_keys:
                parts.append("unexpected Torch state keys: " + ", ".join(unexpected_keys))
            raise ValueError("; ".join(parts))

        self._is_loaded = True
        return IncompatibleKeys(missing_keys, unexpected_keys)

    @deprecated("Use load_torch_state_dict instead")
    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.load_torch_state_dict(state_dict)

    def save(self, directory: str | Path, /) -> None:
        directory = Path(directory)
        directory.mkdir(exist_ok=True)

        for name, child in self.named_children():
            child.save(directory / name)

        for name, parameter in self.named_parameters():
            parameter.save(directory / f"{name}.tensorbin")

    def load(self, directory: str | Path, /) -> None:
        directory = Path(directory)
        if not directory.exists():
            msg = f"directory does not exist: {directory}"
            raise RuntimeError(msg)

        for name, child in self.named_children():
            child.load(directory / name)

        for name, parameter in self.named_parameters():
            path = directory / f"{name}.tensorbin"
            try:
                parameter.load(path)
            except LoadingError as err:
                msg = f"{err} while loading '{path}'"
                raise LoadingError(msg) from err

        self._is_loaded = True

    def to_cached_state_dict(self, path_prefix: str) -> dict[str, str]:
        cache_dict = {}

        for name, child in self.named_children():
            child_cache_dict = child.to_cached_state_dict(f"{path_prefix}{name}.")
            cache_dict.update({f"{name}.{k}": v for k, v in child_cache_dict.items()})

        for name, parameter in self.named_parameters():
            cache_dict[name] = path = f"{path_prefix}{name}.tensorbin"
            parameter.save(path)

        return cache_dict

    def from_cached_state_dict(self, cache_dict: Mapping[str, str]) -> None:
        def substate(state: Mapping[str, str], key: str) -> dict[str, str]:
            prefix = f"{key}."
            return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}

        for name, child in self.named_children():
            child.from_cached_state_dict(substate(cache_dict, name))

        for name, parameter in self.named_parameters():
            path = cache_dict[name]
            try:
                parameter.load(path)
            except LoadingError as err:
                msg = f"{err} while loading '{path}'"
                raise LoadingError(msg) from err

        self._is_loaded = True

    def deallocate_weights(self) -> None:
        """Deallocate all parameter weights from device memory recursively."""
        for _, child in self.named_children():
            child.deallocate_weights()

        for _, parameter in self.named_parameters():
            parameter.deallocate()

        self._is_loaded = False

    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = ()) -> None:
        super().__init__()

        for i, m in enumerate(modules):
            self.add_module(str(i), m)

    def forward(self) -> None:
        msg = "forward() should not be called on ModuleList. Iterate over the modules instead."
        raise RuntimeError(msg)

    def append(self, module: Module) -> None:
        self.add_module(str(len(self)), module)

    def __len__(self) -> int:
        return len(self._children)

    @overload
    def __getitem__(self, key: int) -> Module:
        ...

    @overload
    def __getitem__(self, key: slice) -> ModuleList:
        ...

    def __getitem__(self, key: int | slice) -> Module | ModuleList:
        n = len(self._children)

        if isinstance(key, slice):
            start, stop, step = key.indices(n)
            return ModuleList(self._children[str(i)] for i in range(start, stop, step))

        if isinstance(key, int):
            if key < 0:
                key += n
            if key < 0 or key >= n:
                raise IndexError
            return self._children[str(key)]

        msg = f"expected int or slice argument, got {key}"
        raise ValueError(msg)


class UnregisteredModule:
    """A wrapper for Module instances that prevents automatic registration in parent modules.

    This class provides a way to hold references to Module instances without having them
    automatically registered as child modules when assigned as attributes to another Module. This is
    useful when you need to store a module reference but don't want it to appear in the module
    hierarchy or participate in operations like parameter loading.

    The UnregisteredModule acts as a transparent proxy, forwarding all attribute access and method
    calls to the wrapped module.

    Args:
        module: The Module instance to wrap and keep unregistered.

    Example:
        >>> class MyModule(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         # This will be registered as a child module
        ...         self.registered_child = SomeModule()
        ...         # This will NOT be registered as a child module
        ...         self.unregistered_child = UnregisteredModule(SomeModule())
        ...
        ...     def forward(self, x):
        ...         return self.registered_child(x) + self.unregistered_child(x)
        ...
        >>> my_module = MyModule()
        >>> list(my_module.named_children())
        [('registered_child', <SomeModule instance>)]
    """

    def __init__(self, module: Module) -> None:
        self.module = module

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self.module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.module(*args, **kwargs)


class Parameter:
    def __init__(
        self,
        *,
        total_shape: Sequence[int],
        device: ttnn.MeshDevice,
        layout: ttnn.Layout = ttnn.Layout.TILE,
        dtype: ttnn.DataType = ttnn.bfloat16,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        pad_value: float | None = None,
        mesh_axes: Sequence[int | None] | None = None,
        on_host: bool = False,
    ) -> None:
        """Initialize a Parameter for use in a Module.

        The parameter is initially uninitialized. It is typically populated via the parent module's
        `load_torch_state_dict()`. Alternatively, call `load_torch_tensor()` or assign the `data`
        property directly with a correctly shaped and distributed `ttnn.Tensor`.

        Args:
            total_shape: The global shape of the parameter tensor across all mesh devices.
            device: The mesh device on which the parameter is stored. If `on_host` is
                `True`, this is used only to create the mesh mapper for distributing the tensor.
            layout: See `ttnn.from_torch()`. Defaults to `ttnn.Layout.TILE`.
            dtype: See `ttnn.from_torch()`. Defaults to `ttnn.bfloat16`.
            memory_config: See `ttnn.from_torch()`. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.
            pad_value: See `ttnn.from_torch()`. Defaults to `None`.
            mesh_axes: Maps tensor dimensions to mesh device axes for distribution.
                For a rank-3 tensor whose second and third dimensions are sharded on mesh axes 0 and
                1, respectively, use `[None, 0, 1]`.
            on_host: If `True`, keep the tensor in host memory instead of device memory.
        """
        total_shape = tuple(total_shape)
        mesh_axes = tuple(mesh_axes) if mesh_axes is not None else (None,) * len(total_shape)

        tensor.verify_tensor_mesh_axes(mesh_axes, tensor_rank=len(total_shape), mesh_rank=len(list(device.shape)))

        local_shape = list(total_shape)
        for tensor_dim, mesh_axis in enumerate(mesh_axes):
            if mesh_axis is not None:
                n = device.shape[mesh_axis]
                if local_shape[tensor_dim] % n != 0:
                    msg = (
                        f"tensor with shape {total_shape} cannot be evenly distributed over mesh with shape "
                        f"{tuple(device.shape)} along mesh axis {mesh_axis} and tensor dimension {tensor_dim} "
                    )
                    raise ValueError(msg)
                local_shape[tensor_dim] //= n
        local_shape = tuple(local_shape)

        self.total_shape = total_shape
        self.local_shape = local_shape
        self.device = device
        self.layout = layout
        self.dtype = dtype
        self.memory_config = memory_config
        self.pad_value = pad_value
        self.mesh_axes = mesh_axes
        self.on_host = on_host
        self._data = None

    def load_torch_tensor(self, torch_tensor: torch.Tensor, /) -> None:
        shape = tuple(torch_tensor.shape)
        if shape != self.total_shape:
            msg = f"expected tensor shape {self.total_shape}, got {shape}"
            raise LoadingError(msg)

        self.data = tensor.from_torch(
            torch_tensor,
            device=self.device,
            layout=self.layout,
            dtype=self.dtype,
            memory_config=self.memory_config,
            pad_value=self.pad_value,
            mesh_axes=self.mesh_axes,
            on_host=self.on_host,
        )

    def save(self, path: str | Path, /) -> None:
        ttnn.dump_tensor(path, self.data)

    def load(self, path: str | Path, /) -> None:
        self.data = ttnn.load_tensor(path, device=None if self.on_host else self.device)

    @property
    def data(self) -> ttnn.Tensor:
        if self._data is None:
            msg = "parameter has no data"
            raise RuntimeError(msg)
        return self._data

    @data.setter
    def data(self, value: ttnn.Tensor) -> None:
        self._check_data(value)
        self._data = value

    def deallocate(self) -> None:
        """Deallocate the parameter's device memory."""
        if self._data is not None:
            ttnn.deallocate(self._data)
            self._data = None

    def _check_data(self, value: ttnn.Tensor) -> None:
        if self.on_host:
            if value.device() is not None:
                msg = "expected host tensor, got device tensor"
                raise LoadingError(msg)
        elif value.device() != self.device:
            msg = "device mismatch"
            raise LoadingError(msg)

        if value.dtype != self.dtype:
            msg = f"dtype mismatch: expected {self.dtype}, got {value.dtype}"
            raise LoadingError(msg)

        if value.layout != self.layout:
            msg = f"layout mismatch: expected {self.layout}, got {value.layout}"
            raise LoadingError(msg)

        if value.memory_config() != self.memory_config:
            msg = f"memory config mismatch: expected {self.memory_config}, got {value.memory_config()}"
            raise LoadingError(msg)

        if value.shape != self.local_shape:
            msg = f"shape mismatch: expected {self.local_shape}, got {tuple(value.shape)}"
            raise LoadingError(msg)
