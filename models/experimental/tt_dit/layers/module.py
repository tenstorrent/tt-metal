# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import ttnn

from ..utils import tensor
from ..utils.substate import pop_substate

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, MutableSequence
    from typing import Any

    import torch


class IncompatibleKeys(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]


class ParameterLoadingError(Exception):
    pass


class Module:
    def __init__(self) -> None:
        self._children = {}
        self._parameters = {}

    # TODO: change "Any" to "Module" as soon as all modules are migrated
    def named_children(self) -> Iterator[tuple[str, Any]]:
        yield from self._children.items()

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        yield from self._parameters.items()

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        super().__setattr__(name, value)

        if name in ("_children", "_parameters"):
            return

        children = self.__dict__.get("_children")
        parameters = self.__dict__.get("_parameters")

        if isinstance(value, Module) or hasattr(value, "load_state_dict"):
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

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Prepare Torch state dict before loading."""

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

            if isinstance(child, Module):
                child._load_torch_state_dict_inner(  # noqa: SLF001
                    child_state,
                    module_key_prefix=f"{module_key_prefix}{name}.",
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys,
                )
            else:  # legacy
                child.load_state_dict(child_state)

        for name, parameter in self.named_parameters():
            if name in state_dict:
                try:
                    parameter.load_torch_tensor(state_dict.pop(name))
                except ParameterLoadingError as err:
                    msg = f"while loading {module_key_prefix}{name}: {err}"
                    raise ParameterLoadingError(msg) from err
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

        error_msg = ""
        if strict and missing_keys:
            error_msg += "missing Torch state keys: " + ", ".join(missing_keys) + "; "
        if strict and unexpected_keys:
            error_msg += "unexpected Torch state keys: " + ", ".join(unexpected_keys) + "\n"
        if error_msg:
            raise ValueError(error_msg)

        return IncompatibleKeys(missing_keys, unexpected_keys)

    # deprecated
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
            except ParameterLoadingError as err:
                msg = f"{err} while loading {path}"
                raise ParameterLoadingError(msg) from err

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
            except ParameterLoadingError as err:
                msg = f"{err} while loading {path}"
                raise ParameterLoadingError(msg) from err

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = ()) -> None:
        super().__init__()

        for i, m in enumerate(modules):
            self._children[str(i)] = m

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, idx: int) -> Module:
        if idx < 0:
            idx += len(self._children)
        if idx < 0 or idx >= len(self._children):
            raise IndexError
        return self._children[str(idx)]


class Parameter:
    def __init__(
        self,
        *,
        shape: Iterable[int],
        device: ttnn.MeshDevice,
        layout: ttnn.Layout = ttnn.Layout.TILE,
        dtype: ttnn.DataType = ttnn.bfloat16,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapping: Mapping[int, int] | None = None,
        to_host: bool = False,
    ) -> None:
        self.shape = tuple(shape)
        self.device = device
        self.layout = layout
        self.dtype = dtype
        self.memory_config = memory_config
        self.mesh_mapping = dict(mesh_mapping) if mesh_mapping else {}
        self.to_host = to_host
        self._data = None

        local_shape = list(self.shape)
        for k, v in self.mesh_mapping.items():
            if k is not None:
                local_shape[v] //= self.device.shape[k]
        self.local_shape = tuple(local_shape)

    def load_torch_tensor(self, torch_tensor: torch.Tensor, /) -> None:
        shape = tuple(torch_tensor.shape)
        if shape != self.shape:
            msg = f"expected tensor shape {self.shape}, got {shape}"
            raise ParameterLoadingError(msg)

        self.data = tensor.from_torch(
            torch_tensor,
            device=self.device,
            layout=self.layout,
            dtype=self.dtype,
            memory_config=self.memory_config,
            mesh_mapping=self.mesh_mapping,
            to_host=self.to_host,
        )

    def save(self, path: str | Path, /) -> None:
        ttnn.dump_tensor(path, self.data)

    def load(self, path: str | Path, /) -> None:
        self.data = ttnn.load_tensor(path, device=None if self.to_host else self.device)

    @property
    def data(self) -> ttnn.Tensor:
        assert self._data is not None, "parameter has no data"
        return self._data

    @data.setter
    def data(self, value: ttnn.Tensor) -> None:
        self._check_data(value)
        self._data = value

    def _check_data(self, value: ttnn.Tensor) -> None:
        if self.to_host:
            if value.device() is not None:
                msg = "expected host tensor, got device tensor"
                raise ParameterLoadingError(msg)
        elif value.device() != self.device:
            msg = "device mismatch"
            raise ParameterLoadingError(msg)

        if value.dtype != self.dtype:
            msg = f"dtype mismatch: expected {self.dtype}, got {value.dtype}"
            raise ParameterLoadingError(msg)

        if value.layout != self.layout:
            msg = f"layout mismatch: expected {self.layout}, got {value.layout}"
            raise ParameterLoadingError(msg)

        if value.memory_config() != self.memory_config:
            msg = f"memory config mismatch: expected {self.memory_config}, got {value.memory_config()}"
            raise ParameterLoadingError(msg)

        if value.shape != self.local_shape:
            msg = f"shape mismatch: expected {self.local_shape}, got {tuple(value.shape)}"
            raise ParameterLoadingError(msg)
