# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, overload

from loguru import logger
from typing_extensions import deprecated

import ttnn

from ..utils import tensor
from ..utils.progress import Watchdog as _Watchdog
from ..utils.substate import pop_substate

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, MutableSequence, Sequence
    from typing import Any

    import torch


class IncompatibleKeys(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]


class LoadingError(Exception):
    pass


class _LoadProgress:
    """Heartbeat for weight loading. A 46 GB load+convert is silent for minutes and
    reads as a hang; this emits a timed progress line so it never looks dead."""

    def __init__(self, total: int, what: str) -> None:
        self._total = total
        self._done = 0
        self._what = what
        self._t0 = time.monotonic()
        self._last = 0.0

    def tick(self) -> None:
        self._done += 1
        elapsed = time.monotonic() - self._t0
        # Throttle to ~5s so fast loads stay quiet and slow ones still show life.
        if elapsed - self._last >= 5.0 or self._done == self._total:
            self._last = elapsed
            pct = 100 * self._done / self._total if self._total else 100
            logger.info(f"{self._what}: {self._done}/{self._total} tensors ({pct:.0f}%), {elapsed:.0f}s")


def release_device_cache(cache: dict) -> None:
    """Free the device tensors a module's lazy per-shape cache holds and drop those entries.

    `Module.deallocate_weights` frees parameters only; a module that also memoises device
    constants (pad blocks, masks, filter taps) keyed by input shape would otherwise keep them
    resident after its eviction, for the life of the process. Entries without a device tensor
    (compute configs, recorded op choices) are kept: they cost nothing and are expensive to redo.
    """

    def has_device_tensor(value) -> bool:
        if isinstance(value, ttnn.Tensor):
            try:
                return value.device() is not None
            except Exception:  # noqa: BLE001 - a dead tensor is not on device
                return False
        if isinstance(value, (list, tuple)):
            return any(has_device_tensor(v) for v in value)
        return False

    def free(value) -> None:
        if isinstance(value, ttnn.Tensor):
            if has_device_tensor(value) and value.is_allocated():
                ttnn.deallocate(value)
        elif isinstance(value, (list, tuple)):
            for v in value:
                free(v)

    for key in [k for k, v in cache.items() if has_device_tensor(v)]:
        free(cache.pop(key))


class Module(ABC):
    def __init__(self) -> None:
        self._children = {}
        self._parameters = {}
        self._is_loaded = False
        self.coresident_exclusions = None  # modules that cannot be resident in memory at the same time as this module. They should be deallocated before this module is loaded.
        self._coresident_peers: list[Module] = []
        self._eviction_hooks: list[Callable[[], None]] = []

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

    def _num_parameters(self) -> int:
        return len(self._parameters) + sum(
            child._num_parameters() for _, child in self.named_children()
        )  # noqa: SLF001

    def _load_torch_state_dict_inner(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        module_key_prefix: str,
        missing_keys: MutableSequence[str],
        unexpected_keys: MutableSequence[str],
        progress: _LoadProgress | None = None,
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
                    progress=progress,
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
                if progress is not None:
                    progress.tick()
            else:
                missing_keys.append(f"{module_key_prefix}{name}")

        for name in state_dict:
            unexpected_keys.append(f"{module_key_prefix}{name}")

    def _mark_loaded(self) -> None:
        """Recursively mark this module and all descendants as loaded."""
        self._is_loaded = True
        for _, child in self.named_children():
            child._mark_loaded()  # noqa: SLF001

    def load_torch_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True) -> IncompatibleKeys:
        """Load PyTorch state dict into module parameters.

        Args:
            state_dict: Mapping of parameter names to PyTorch tensors.
            strict: If `True`, raises ValueError on missing or unexpected keys.

        Returns:
            `IncompatibleKeys` containing lists of missing and unexpected keys.
        """
        missing_keys = []
        unexpected_keys = []
        self.evict_coresident_exclusions()
        with _Watchdog(f"convert {type(self).__name__}"):
            self._load_torch_state_dict_inner(
                state_dict,
                module_key_prefix="",
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
                progress=_LoadProgress(self._num_parameters(), "converting weights to device"),
            )

        if strict and (missing_keys or unexpected_keys):
            parts = []
            if missing_keys:
                parts.append("missing Torch state keys: " + ", ".join(missing_keys))
            if unexpected_keys:
                parts.append("unexpected Torch state keys: " + ", ".join(unexpected_keys))
            raise ValueError("; ".join(parts))

        self._mark_loaded()
        return IncompatibleKeys(missing_keys, unexpected_keys)

    @deprecated("Use load_torch_state_dict instead")
    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.load_torch_state_dict(state_dict)

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        for name, child in self.named_children():
            child.save(directory, prefix=f"{prefix}{name}.")

        for name, parameter in self.named_parameters():
            parameter.save(directory / f"{prefix}{name}.tensorbin")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        directory = Path(directory)

        # Top-level only: announce the cache load and arm a timer watchdog so a slow/stalled
        # module load is never silent. Signature is left untouched — subclasses (Mochi/Wan)
        # override this method.
        watchdog = None
        if prefix == "":
            logger.info(f"loading {self._num_parameters()} cached weight tensors from '{directory}'...")
            watchdog = _Watchdog(f"load-cache {type(self).__name__}").__enter__()
        try:
            self.evict_coresident_exclusions()

            for name, child in self.named_children():
                child.load(directory, prefix=f"{prefix}{name}.")

            for name, parameter in self.named_parameters():
                path = directory / f"{prefix}{name}.tensorbin"
                try:
                    parameter.load(path)
                except LoadingError as err:
                    msg = f"{err} while loading '{path}'"
                    raise LoadingError(msg) from err

            self._is_loaded = True
        finally:
            if watchdog is not None:
                watchdog.__exit__()

    def deallocate_weights(self) -> None:
        """Deallocate all parameter weights from device memory recursively, then run the eviction hooks."""
        for _, child in self.named_children():
            child.deallocate_weights()

        for _, parameter in self.named_parameters():
            parameter.deallocate()

        self._set_evicted()

    def shard(self, mesh_axis: int) -> None:
        """Shard the module's parameters along the mesh_axis. This is a transient sharding; you cannot save
        or run the __call__/forward functions on a transiently_sharded module. You need to call gather first. 

        The module counts as unloaded while sharded.
        
        When calling 'shard' the module (and it's children) call their eviction hooks. If a module overrides
        `deallocate_weights`, we cannot safely shard, so for now we will raise in that situation. TODO: fix.
        """
        modules = list(self._all_modules())
        for module in modules:
            if type(module).deallocate_weights is not Module.deallocate_weights:
                msg = (
                    f"{type(module).__name__} overrides deallocate_weights, which shard does not run; free its "
                    "other device state from a hook (register_eviction_hook) instead"
                )
                raise TypeError(msg)
        parameters = list(self._all_parameters())
        for parameter in parameters:
            parameter.check_shard_axis(mesh_axis)

        defer_free: list[ttnn.Tensor] = []
        for parameter in parameters:
            parameter.shard(mesh_axis, defer_free=defer_free)
        for t in defer_free:
            ttnn.deallocate(t)
        for module in modules:
            module._set_evicted()  # noqa: SLF001

    def gather(self, ccl_manager: Any) -> None:  # noqa: ANN401
        """Undo `shard`: all-gather every sliced parameter; the module is loaded again."""
        defer_free: list[ttnn.Tensor] = []
        for parameter in self._all_parameters():
            parameter.gather(ccl_manager, defer_free=defer_free)
        for t in defer_free:
            ttnn.deallocate(t)
        self._mark_loaded()

    def is_transiently_sharded(self) -> bool:
        """Whether any parameter in any submodule is transiently sharded."""
        return any(p.is_transiently_sharded() for p in self._all_parameters())

    def _all_modules(self) -> Iterator[Module]:
        for _, child in self.named_children():
            yield from child._all_modules()  # noqa: SLF001
        yield self

    def _all_parameters(self) -> Iterator[Parameter]:
        for module in self._all_modules():
            for _, parameter in module.named_parameters():
                yield parameter

    def _set_evicted(self) -> None:
        self._is_loaded = False
        for hook in self._eviction_hooks:
            hook()

    def register_eviction_hook(self, hook: Callable[[], None]) -> None:
        """Run `hook` whenever this module's weights are deallocated.

        For device state that belongs with the module's residency but lives outside its parameters
        and children -- an owner's per-geometry caches, for instance -- so that evicting the module
        frees it too and it is rebuilt lazily on the next use.
        """
        self._eviction_hooks.append(hook)

    def is_loaded(self) -> bool:
        return self._is_loaded

    def register_coresident_exclusions(self, *args: Module) -> None:
        """
        Register modules that cannot be resident in memory at the same time as this module.
        They should be deallocated before this module is loaded. See `evict_coresident_exclusions` .
        Args:
            *args: Modules that cannot be co-resident in memory with this module.
        """
        if self.coresident_exclusions is None:
            self.coresident_exclusions = set()
        self.coresident_exclusions.update(args)

    def evict_coresident_exclusions(self) -> None:
        """Evict the modules that cannot be resident in memory at the same time as this module."""
        if self.coresident_exclusions is not None:
            for module in self.coresident_exclusions:
                module.deallocate_weights()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        # only walk the current module's params, not the full tree! Sub-modules should check
        # their own parameters when they get called.
        if any(p.is_transiently_sharded() for _, p in self.named_parameters()):
            msg = f"{type(self).__name__} has weights with a transient split; gather it first"
            raise RuntimeError(msg)
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

        for tensor_dim, mesh_axis in enumerate(mesh_axes):
            if mesh_axis is not None and total_shape[tensor_dim] % device.shape[mesh_axis] != 0:
                msg = (
                    f"tensor with shape {total_shape} cannot be evenly distributed over mesh with shape "
                    f"{tuple(device.shape)} along mesh axis {mesh_axis} and tensor dimension {tensor_dim} "
                )
                raise ValueError(msg)

        self.total_shape = total_shape
        self.device = device
        self.layout = layout
        self.dtype = dtype
        self.memory_config = memory_config
        self.pad_value = pad_value

        # This sets the durable/frozen layout, which is how the parameter tensor will always be loaded, saved and cached.
        # The Module which owns this Parameter expects it to be in the durable_mesh_axes layout when calling forward or __call__.
        self.durable_mesh_axes = mesh_axes

        # self.mesh_axes is the current layout, which can be changed via `shard` and `gather`.
        self.mesh_axes = mesh_axes
        self.on_host = on_host
        self._data = None

    @property
    def local_shape(self) -> tuple[int, ...]:
        """Each device's shape of the tensor in its current layout (`mesh_axes`)."""
        return tuple(
            size if axis is None else size // self.device.shape[axis]
            for size, axis in zip(self.total_shape, self.mesh_axes, strict=True)
        )

    def load_torch_tensor(self, torch_tensor: torch.Tensor, /) -> None:
        if self.is_transiently_sharded():
            self.deallocate()
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
            mesh_axes=self.durable_mesh_axes,
            on_host=self.on_host,
        )

    def save(self, path: str | Path, /) -> None:
        ttnn.dump_tensor(path, self.data, mode=ttnn.DumpTensorMode.LOCAL)

    def load(self, path: str | Path, /) -> None:
        if self.is_transiently_sharded():
            self.deallocate()
        try:
            tensor = ttnn.load_tensor(path, device=None if self.on_host else self.device)
        except RuntimeError as err:
            msg = f"TT-NN error «{err}»"
            raise LoadingError(msg) from err
        self.data = tensor

    @property
    def data(self) -> ttnn.Tensor:
        if self._data is None:
            msg = "parameter has no data"
            raise RuntimeError(msg)
        if self.is_transiently_sharded():
            axes = [self.mesh_axes[d] for d in self.transient_dims()]
            msg = f"parameter is transiently sharded over mesh axes {axes}; gather it first"
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
        self.mesh_axes = self.durable_mesh_axes

    def is_transiently_sharded(self) -> bool:
        """Whether the parameter has a transient split e.g. via `shard`."""
        return self.mesh_axes != self.durable_mesh_axes

    def transient_dims(self) -> list[int]:
        """The dims split beyond the durable layout, by `shard`."""
        return [d for d, (a, b) in enumerate(zip(self.mesh_axes, self.durable_mesh_axes, strict=True)) if a != b]

    def check_shard_axis(self, mesh_axis: int) -> None:
        """Raise if `mesh_axis` already splits this parameter: in its durable layout (the owning module's forward is
        written for that split, so it cannot be gathered away) or transiently, by an earlier `shard`."""
        if mesh_axis in self.durable_mesh_axes:
            msg = (
                f"mesh axis {mesh_axis} is part of this parameter's durable layout (durable_mesh_axes "
                f"{self.durable_mesh_axes}); shard can only split over another mesh axis"
            )
            raise ValueError(msg)
        if mesh_axis in self.mesh_axes:
            msg = f"parameter already has a transient split over mesh axis {mesh_axis}"
            raise ValueError(msg)

    def shard(self, mesh_axis: int, /, *, defer_free: list[ttnn.Tensor] | None = None) -> None:
        """Shard data along the mesh_axis. This sharding is transient, meaning the Parameter cannot be used
        in compute/save and must be `gather`d before use. Attempting to use a transiently sharded parameter 
        will result in an error.
        
        The full tensor it replaces is either deallocated immediately or appended to `defer_free` (if not None).
        Defer free is used to allow a caller sharding many parameters to call `free` at the end to reduce memory 
        fragmentation.
        """
        self.check_shard_axis(mesh_axis)
        if self._data is None or self.on_host:
            return
        n = self.device.shape[mesh_axis]
        dim = self._shard_dim(n) if n > 1 and len(self.total_shape) <= 4 else None
        if dim is None:
            return
        full = self._data
        offset = 4 - len(self.total_shape)
        piece = ttnn.mesh_partition(
            ttnn.reshape(full, (1,) * offset + self.local_shape),
            dim + offset,
            cluster_axis=mesh_axis,
            memory_config=self.memory_config,
        )
        self._set_mesh_axis(dim, mesh_axis)
        self.data = ttnn.reshape(piece, self.local_shape)
        self._free(full, defer_free)

    def gather(self, ccl_manager: Any, /, *, defer_free: list[ttnn.Tensor] | None = None) -> None:  # noqa: ANN401
        """Undo `shard`: all-gather every transiently split dim, back to the durable layout (defined by self.durable_mesh_axes)

        If defer_free is provided, the sharded ref is given to the caller to dealloc, else it gets dealloc'd immediately.
        """
        dims = self.transient_dims()
        if not dims or self._data is None:
            return
        piece = current = self._data
        offset = 4 - len(self.total_shape)
        for dim in dims:
            gathered = ccl_manager.all_gather(
                ttnn.reshape(current, (1,) * offset + self.local_shape),
                dim=dim + offset,
                mesh_axis=self.mesh_axes[dim],
                use_hyperparams=False,
            )
            if current is not piece:
                ttnn.deallocate(current)
            self._set_mesh_axis(dim, self.durable_mesh_axes[dim])
            current = ttnn.reshape(gathered, self.local_shape)
        if current.memory_config() != self.memory_config:
            current = ttnn.to_memory_config(current, self.memory_config)
        self.data = current
        self._free(piece, defer_free)

    def _shard_dim(self, n: int) -> int | None:
        """The first unsplit dim that splits n ways: whole tiles for the last two dims of a tiled tensor, never the
        last dim of a row-major one."""
        rank = len(self.total_shape)
        tiled = self.layout == ttnn.Layout.TILE
        for d, (size, axis) in enumerate(zip(self.local_shape, self.mesh_axes, strict=True)):
            if axis is not None or (d == rank - 1 and not tiled):
                continue
            if size % (n * ttnn.TILE_SIZE if tiled and d >= rank - 2 else n) == 0:
                return d
        return None

    def _set_mesh_axis(self, dim: int, mesh_axis: int | None) -> None:
        axes = list(self.mesh_axes)
        axes[dim] = mesh_axis
        self.mesh_axes = tuple(axes)

    @staticmethod
    def _free(t: ttnn.Tensor, defer_free: list[ttnn.Tensor] | None) -> None:
        if defer_free is None:
            ttnn.deallocate(t)
        else:
            defer_free.append(t)

    def _check_data(self, value: ttnn.Tensor) -> None:
        if self.on_host:
            if value.device() is not None:
                msg = "expected host tensor, got device tensor"
                raise LoadingError(msg)
        elif value.device() is None:
            msg = "expected device tensor, got host tensor"
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
