# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A small module system for TT weights.

A `Module` registers child modules and `Parameter`s by attribute assignment. Naming them like the
torch reference makes `load_torch_state_dict(reference.state_dict())` load the whole tree: each
module first adjusts its slice of the state dict in `_prepare_torch_state` (transposes, reshapes,
renames), then hands each child its prefix and loads its own parameters. `save`/`load` write and
read one `.tensorbin` per parameter, and `load_cached` uses them as a weight cache.

Parameters are replicated across the mesh: MoonViT is sequence parallel only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch

import ttnn


class LoadingError(Exception):
    pass


def pop_substate(state: MutableMapping[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    """Remove the `key.`-prefixed entries from `state` and return them with the prefix stripped."""
    prefix = f"{key}."
    return {k.removeprefix(prefix): state.pop(k) for k in list(state) if k.startswith(prefix)}


def rename_substate(state: MutableMapping[str, torch.Tensor], key_from: str, key_to: str) -> None:
    """Move the `key_from.`-prefixed entries of `state` under `key_to.`."""
    for k, v in pop_substate(state, key_from).items():
        state[f"{key_to}.{k}"] = v


class Parameter:
    """A replicated device tensor with a fixed shape, filled by `load_torch_tensor`."""

    def __init__(
        self,
        *,
        shape: Sequence[int],
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        layout: ttnn.Layout = ttnn.TILE_LAYOUT,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> None:
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self.layout = layout
        self.memory_config = memory_config
        self._data: ttnn.Tensor | None = None

    @property
    def data(self) -> ttnn.Tensor:
        if self._data is None:
            raise RuntimeError("parameter has no data")
        return self._data

    def load_torch_tensor(self, value: torch.Tensor) -> None:
        if tuple(value.shape) != self.shape:
            raise LoadingError(f"expected shape {self.shape}, got {tuple(value.shape)}")
        self._data = ttnn.from_torch(
            value,
            device=self.device,
            dtype=self.dtype,
            layout=self.layout,
            memory_config=self.memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def save(self, path: Path) -> None:
        ttnn.dump_tensor(path, self.data)

    def load(self, path: Path) -> None:
        self._data = ttnn.load_tensor(path, device=self.device)


class Module:
    def __init__(self) -> None:
        self._children: dict[str, Module] = {}
        self._parameters: dict[str, Parameter] = {}
        self._is_loaded = False

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if isinstance(value, Module):
            self.__dict__["_children"][name] = value
        elif isinstance(value, Parameter):
            self.__dict__["_parameters"][name] = value

    def named_children(self) -> Iterator[tuple[str, Module]]:
        yield from self._children.items()

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        yield from self._parameters.items()

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Adjust this module's slice of the torch state dict in place before it loads."""

    def load_torch_state_dict(self, state_dict: MutableMapping[str, torch.Tensor]) -> None:
        """Load a torch state dict keyed like this tree. Missing or unexpected keys raise."""
        missing: list[str] = []
        unexpected: list[str] = []
        self._load_inner(dict(state_dict), "", missing, unexpected)
        if missing or unexpected:
            raise LoadingError(f"missing keys {missing}, unexpected keys {unexpected}")
        self._mark_loaded()

    def _load_inner(self, state: dict[str, torch.Tensor], prefix: str, missing: list, unexpected: list) -> None:
        self._prepare_torch_state(state)
        for name, child in self.named_children():
            child._load_inner(pop_substate(state, name), f"{prefix}{name}.", missing, unexpected)
        for name, parameter in self.named_parameters():
            if name not in state:
                missing.append(f"{prefix}{name}")
                continue
            try:
                parameter.load_torch_tensor(state.pop(name))
            except LoadingError as err:
                raise LoadingError(f"{prefix}{name}: {err}") from err
        unexpected.extend(f"{prefix}{name}" for name in state)

    def _mark_loaded(self) -> None:
        self._is_loaded = True
        for _, child in self.named_children():
            child._mark_loaded()

    def is_loaded(self) -> bool:
        return self._is_loaded

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, child in self.named_children():
            child.save(directory, prefix=f"{prefix}{name}.")
        for name, parameter in self.named_parameters():
            parameter.save(directory / f"{prefix}{name}.tensorbin")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        directory = Path(directory)
        for name, child in self.named_children():
            child.load(directory, prefix=f"{prefix}{name}.")
        for name, parameter in self.named_parameters():
            parameter.load(directory / f"{prefix}{name}.tensorbin")
        self._is_loaded = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.forward")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__()
        for i, module in enumerate(modules):
            self._children[str(i)] = module

    def __getitem__(self, index: int) -> Module:
        return self._children[str(index)]

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._children.values())


CACHE_COMPLETE = "cache_complete"


def load_cached(
    module: Module, cache_dir: str | Path | None, get_torch_state_dict: Callable[[], MutableMapping[str, torch.Tensor]]
) -> None:
    """Load `module` from `cache_dir` if a complete cache is there, else from torch and write it.

    With `cache_dir=None` it always loads from torch. The cache is only valid for the mesh shape
    and dtype it was written with; put those in the directory name.
    """
    if cache_dir is None:
        module.load_torch_state_dict(get_torch_state_dict())
        return
    cache_dir = Path(cache_dir)
    if (cache_dir / CACHE_COMPLETE).is_file():
        module.load(cache_dir)
        return
    module.load_torch_state_dict(get_torch_state_dict())
    module.save(cache_dir)
    (cache_dir / CACHE_COMPLETE).touch()
