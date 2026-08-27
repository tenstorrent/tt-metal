# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle owner for Wormhole Galaxy 2D weight prefetch resources."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import torch

import ttnn

PrefetcherMode = Literal["prefill", "decode"]
WeightCompatibilityValidator = Callable[[str, Any, tuple[Any, ...]], None]
GlobalCBSizeDeriver = Callable[[tuple[Any, ...]], int]
DramPrefetchStart = Callable[["Prefetcher2DContext"], Any]
DramPrefetchStop = Callable[[Any, Any], Any]


def _validate_wh_galaxy(mesh_device: Any, mesh_shape: tuple[int, int], architecture: Any) -> None:
    if architecture != ttnn.device.Arch.WORMHOLE_B0:
        raise ValueError(f"Prefetcher2D requires Wormhole B0, got {architecture}")
    if mesh_shape != (8, 4):
        raise ValueError(f"Prefetcher2D requires logical mesh shape (8, 4), got {mesh_shape}")
    if tuple(mesh_device.shape) != mesh_shape:
        raise ValueError(f"mesh device shape {tuple(mesh_device.shape)} does not match resolved shape {mesh_shape}")
    if mesh_device.get_num_devices() != 32:
        raise ValueError(f"Prefetcher2D requires exactly 32 devices, got {mesh_device.get_num_devices()}")
    if mesh_device.arch() != architecture:
        raise ValueError("mesh device architecture does not match the resolved architecture")


@dataclass(frozen=True)
class Prefetcher2DModeConfig:
    mode: PrefetcherMode
    sub_devices: tuple[Any, ...]
    worker_sub_device_id: Any
    stall_group: tuple[Any, ...]
    local_l1_size: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sub_devices", tuple(self.sub_devices))
        object.__setattr__(self, "stall_group", tuple(self.stall_group))
        if self.mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported prefetcher mode: {self.mode}")
        if not self.sub_devices:
            raise ValueError("at least one subdevice must be configured")
        if self.worker_sub_device_id is None:
            raise ValueError("worker_sub_device_id must be resolved")
        if not self.stall_group:
            raise ValueError("stall_group must be resolved")
        if self.local_l1_size < 0:
            raise ValueError("local_l1_size cannot be negative")


@dataclass(frozen=True)
class Prefetcher2DConfig:
    """Frozen, fully resolved prefetcher construction policy."""

    mesh_device: Any
    architecture: Any
    prefill: Prefetcher2DModeConfig
    decode: Prefetcher2DModeConfig
    sender_receiver_mapping: tuple[tuple[Any, Any], ...]
    global_cb_size: int | None
    expected_weight_count: int
    address_repeat_count: int
    address_memory_config: Any
    address_mesh_mapper: Any
    prefetch_num_layers: int = 1
    mesh_shape: tuple[int, int] = (8, 4)
    #: Allocate the global circular buffer on the first ``activate("decode")``
    #: instead of in ``seal()``.
    #:
    #: The global CB is ~774 kB of L1 on every sender/receiver core and nothing
    #: can free it, so with the default ``False`` a *prefill* program that needs
    #: static circular buffers on those cores cannot be placed at all:
    #:
    #:     TT_THROW ... Statically allocated circular buffers in program 100
    #:     clash with L1 buffers on core range [0-0 - 0-3]. L1 buffer allocated
    #:     at 579104 and static circular buffer region ends at 630080
    #:                                              (from ttnn.embedding, prefill)
    #:
    #: Prefill never reads the buffer - ``seal()`` already hands the prefill
    #: context ``global_cb=None`` - so holding it through prefill buys nothing.
    #: The production Galaxy prefetcher makes the same choice and says so:
    #: ``self.global_circular_buffer = None  # Global CB will only be allocated
    #: before decode runs`` in ``models/demos/llama3_70b_galaxy/tt/
    #: prefetcher_common.py``, with allocation in its own ``create_global_cb()``.
    #:
    #: Defaults to ``False`` so the Milestone A qualification of this module is
    #: bit-for-bit unchanged unless a caller asks for the deferral.
    defer_global_cb: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sender_receiver_mapping",
            tuple(tuple(pair) for pair in self.sender_receiver_mapping),
        )
        _validate_wh_galaxy(self.mesh_device, self.mesh_shape, self.architecture)
        if self.prefill.mode != "prefill" or self.decode.mode != "decode":
            raise ValueError("prefill and decode subdevice configs must match their declared modes")
        if not self.sender_receiver_mapping or any(len(pair) != 2 for pair in self.sender_receiver_mapping):
            raise ValueError("sender_receiver_mapping must contain resolved sender/receiver pairs")
        if self.address_repeat_count > len(self.sender_receiver_mapping):
            raise ValueError("address_repeat_count cannot exceed the sender/receiver mapping count")
        if self.global_cb_size is not None and self.global_cb_size <= 0:
            raise ValueError("global_cb_size must be positive when specified")
        if self.expected_weight_count <= 0:
            raise ValueError("expected_weight_count must be positive")
        if self.address_repeat_count <= 0:
            raise ValueError("address_repeat_count must be positive")
        if self.address_memory_config is None or self.address_mesh_mapper is None:
            raise ValueError("address tensor placement must be fully resolved")
        if self.prefetch_num_layers <= 0:
            raise ValueError("prefetch_num_layers must be positive")
        if self.expected_weight_count % self.prefetch_num_layers != 0:
            raise ValueError("expected_weight_count must be divisible by prefetch_num_layers")


@dataclass(frozen=True)
class Prefetcher2DContext:
    """Immutable resources borrowed by modules and model-owned executors."""

    mode: PrefetcherMode
    mesh_device: Any
    sub_device_manager_id: Any
    worker_sub_device_id: Any
    stall_group: tuple[Any, ...]
    global_cb: Any
    weights: tuple[Any, ...]
    weight_addresses: Any
    weight_address_metadata: Any

    @property
    def sub_device_id(self) -> Any:
        return self.worker_sub_device_id


@runtime_checkable
class Prefetcher2DResourceOwner(Protocol):
    """Structural owner API consumed by model-level resource collaborators."""

    @property
    def mesh_device(self) -> Any:
        ...

    def borrow_context(
        self,
        mode: PrefetcherMode,
        *,
        sub_devices: tuple[Any, ...],
        worker_sub_device_id: Any,
        stall_group: tuple[Any, ...],
        local_l1_size: int,
    ) -> Prefetcher2DContext:
        ...

    def activate(self, mode: PrefetcherMode) -> Prefetcher2DContext:
        ...


class Prefetcher2D:
    """Own managers, packed addresses, global CB, and running prefetch results."""

    def __init__(
        self,
        config: Prefetcher2DConfig,
        *,
        create_global_cb: Callable[[Any, list[tuple[Any, Any]], int], Any] | None = None,
        create_address_metadata: Callable[..., Any] | None = None,
        deallocate: Callable[[Any], None] | None = None,
        validate_weight_compatibility: WeightCompatibilityValidator | None = None,
        derive_global_cb_size: GlobalCBSizeDeriver | None = None,
        dram_prefetch_start: DramPrefetchStart | None = None,
        dram_prefetch_stop: DramPrefetchStop | None = None,
    ):
        self.config = config
        self._create_global_cb = create_global_cb or ttnn.create_global_circular_buffer
        self._create_address_metadata = create_address_metadata or ttnn.as_tensor
        self._deallocate = deallocate or ttnn.deallocate
        self._validate_weight_compatibility = validate_weight_compatibility or self._default_validate_weight
        self._derive_global_cb_size = derive_global_cb_size or self._default_derive_global_cb_size
        self._dram_prefetch_start = dram_prefetch_start or self._default_dram_prefetch_start
        self._dram_prefetch_stop = dram_prefetch_stop or self._default_dram_prefetch_stop
        self._managers: dict[PrefetcherMode, Any] = {}
        self._registered_weights: OrderedDict[str, Any] = OrderedDict()
        self._global_cb: Any = None
        self._resolved_global_cb_size: int | None = None
        self._weight_address_metadata: Any = None
        self._contexts: dict[PrefetcherMode, Prefetcher2DContext] = {}
        self._active_mode: PrefetcherMode | None = None
        self._loaded_mode: PrefetcherMode | None = None
        self._prefetch_result: Any = None
        self._retained_prefetch_resources: list[Any] = []
        self._initialized = False
        self._sealed = False
        self._cleaned = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def active_mode(self) -> PrefetcherMode | None:
        return self._active_mode

    @property
    def mesh_device(self) -> Any:
        return self.config.mesh_device

    @property
    def prefetch_result(self) -> Any:
        return self._prefetch_result

    @property
    def resolved_global_cb_size(self) -> int | None:
        return self._resolved_global_cb_size

    @property
    def borrowed_weights(self) -> tuple[Any, ...]:
        return tuple(self._registered_weights.values())

    @property
    def owned_resources(self) -> tuple[Any, ...]:
        resources = list(self._managers.values())
        resources.extend(
            resource for resource in (self._global_cb, self._weight_address_metadata) if resource is not None
        )
        resources.extend(self._retained_prefetch_resources)
        if self._prefetch_result is not None:
            resources.append(self._prefetch_result)
        return tuple(resources)

    def initialize(self) -> None:
        self._ensure_open()
        if self._initialized:
            return

        mesh = self.config.mesh_device
        created: list[Any] = []
        try:
            for mode_config in (self.config.prefill, self.config.decode):
                manager = mesh.create_sub_device_manager(list(mode_config.sub_devices), mode_config.local_l1_size)
                self._managers[mode_config.mode] = manager
                created.append(manager)
        except Exception:
            for manager in reversed(created):
                mesh.remove_sub_device_manager(manager)
            self._managers.clear()
            raise

        self._initialized = True

    def register_weight(self, name: str, tensor: Any) -> None:
        self._ensure_open()
        if not self._initialized:
            raise RuntimeError("Prefetcher2D must be initialized before weight registration")
        if self._sealed:
            raise RuntimeError("weight registration is sealed")
        if not name:
            raise ValueError("registered weight name cannot be empty")
        if name in self._registered_weights:
            raise ValueError(f"weight is already registered: {name}")
        if len(self._registered_weights) >= self.config.expected_weight_count:
            raise ValueError("registered weight count exceeds the resolved configuration")
        if not callable(getattr(tensor, "buffer_address", None)):
            raise TypeError("registered weights must be materialized device tensors")

        tensor_device = getattr(tensor, "device", None)
        tensor_device = tensor_device() if callable(tensor_device) else tensor_device
        if tensor_device is not None and tensor_device != self.config.mesh_device:
            raise ValueError("registered weight belongs to a different mesh")
        self._validate_weight_compatibility(name, tensor, tuple(self._registered_weights.values()))
        self._registered_weights[name] = tensor

    def seal(self) -> tuple[Prefetcher2DContext, Prefetcher2DContext]:
        self._ensure_open()
        if not self._initialized:
            raise RuntimeError("Prefetcher2D must be initialized before sealing")
        if self._sealed:
            return self.context("prefill"), self.context("decode")
        if len(self._registered_weights) != self.config.expected_weight_count:
            raise RuntimeError(
                f"expected {self.config.expected_weight_count} registered weights, "
                f"got {len(self._registered_weights)}"
            )

        weights = tuple(self._registered_weights.values())
        configured_cb_size = self.config.global_cb_size
        resolved_cb_size = configured_cb_size or self._derive_global_cb_size(weights)
        if resolved_cb_size <= 0:
            raise ValueError("resolved global CB size must be positive")
        weight_addresses = MappingProxyType(
            {name: tensor.buffer_address() for name, tensor in self._registered_weights.items()}
        )
        addresses = torch.tensor(tuple(weight_addresses.values()), dtype=torch.int64)
        addresses = addresses.repeat(self.config.address_repeat_count, 1)

        global_cb = None
        metadata = None
        try:
            self._configure_mode_resources(self.config.decode)
            if not self.config.defer_global_cb:
                global_cb = self._create_global_cb(
                    self.config.mesh_device,
                    list(self.config.sender_receiver_mapping),
                    resolved_cb_size,
                )
            metadata = self._create_address_metadata(
                addresses,
                device=self.config.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self.config.address_memory_config,
                mesh_mapper=self.config.address_mesh_mapper,
            )
        except Exception:
            rollback_error = None
            for resource in (metadata,):
                if resource is not None:
                    try:
                        self._deallocate(resource)
                    except Exception as exc:
                        if rollback_error is None:
                            rollback_error = exc
            if rollback_error is not None:
                raise RuntimeError("Prefetcher2D sealing and rollback both failed") from rollback_error
            raise

        self._global_cb = global_cb
        self._weight_address_metadata = metadata
        self._resolved_global_cb_size = resolved_cb_size
        self._contexts = {
            "prefill": self._make_context(
                self.config.prefill,
                global_cb=None,
                weights=(),
                weight_addresses=weight_addresses,
            ),
            "decode": self._make_context(
                self.config.decode,
                global_cb=global_cb,
                weights=weights,
                weight_addresses=weight_addresses,
            ),
        }
        self._sealed = True
        return self._contexts["prefill"], self._contexts["decode"]

    def context(self, mode: PrefetcherMode) -> Prefetcher2DContext:
        self._ensure_open()
        if not self._sealed:
            raise RuntimeError("Prefetcher2D contexts are unavailable until registration is sealed")
        try:
            return self._contexts[mode]
        except KeyError as exc:
            raise ValueError(f"unsupported prefetcher mode: {mode}") from exc

    def borrow_context(
        self,
        mode: PrefetcherMode,
        *,
        sub_devices: tuple[Any, ...],
        worker_sub_device_id: Any,
        stall_group: tuple[Any, ...],
        local_l1_size: int,
    ) -> Prefetcher2DContext:
        """Return a sealed context after exact subdevice-policy validation."""

        context = self.context(mode)
        mode_config = self.config.prefill if mode == "prefill" else self.config.decode
        expected = (
            mode_config.sub_devices,
            mode_config.worker_sub_device_id,
            mode_config.stall_group,
            mode_config.local_l1_size,
        )
        requested = (tuple(sub_devices), worker_sub_device_id, tuple(stall_group), local_l1_size)
        if requested != expected:
            raise ValueError(f"{mode} Galaxy resources do not match the Prefetcher2D subdevice policy")
        return context

    def activate(self, mode: PrefetcherMode) -> Prefetcher2DContext:
        self._ensure_open()
        context = self.context(mode)
        previous_mode = self._active_mode
        previous_was_prefetching = self._prefetch_result is not None
        if previous_was_prefetching:
            self._stop_prefetch()

        try:
            self._configure_mode(context)
            if mode == "decode":
                self._ensure_global_cb(context)
                self._start_prefetch(context)
        except Exception as activation_error:
            if self._prefetch_result is not None:
                self._stop_prefetch(suppress_errors=True)
            try:
                if previous_mode is None:
                    self.config.mesh_device.reset_sub_device_stall_group()
                    self.config.mesh_device.clear_loaded_sub_device_manager()
                    self._loaded_mode = None
                else:
                    previous = self.context(previous_mode)
                    self._configure_mode(previous)
                    if previous_was_prefetching:
                        self._start_prefetch(previous)
                self._active_mode = previous_mode
            except Exception as rollback_error:
                self._active_mode = None
                raise RuntimeError("Prefetcher2D activation and rollback both failed") from rollback_error
            raise activation_error

        self._active_mode = mode
        return context

    def _ensure_global_cb(self, context: Prefetcher2DContext) -> None:
        """Allocate the deferred global circular buffer and bind it to `context`.

        Called from ``activate("decode")`` before the prefetch program is
        enqueued, which is the first moment anything reads the buffer.

        The binding is an ``object.__setattr__`` on a frozen dataclass, and that
        is deliberate rather than lazy typing. Module configs capture the
        *context object* at construction (`MLP2DConfig.decode_prefetch_context`,
        read as ``getattr(context, "global_cb", None)`` at call time), so
        replacing the entry in ``self._contexts`` would leave every already-built
        module holding a context whose ``global_cb`` is still ``None``. The field
        is bound exactly once, from ``None`` to the buffer, and never rebound.
        """

        if not self.config.defer_global_cb or self._global_cb is not None:
            return
        if self._resolved_global_cb_size is None:
            raise RuntimeError("global CB size was not resolved during sealing")
        global_cb = self._create_global_cb(
            self.config.mesh_device,
            list(self.config.sender_receiver_mapping),
            self._resolved_global_cb_size,
        )
        self._global_cb = global_cb
        object.__setattr__(context, "global_cb", global_cb)

    def cleanup(self) -> None:
        if self._cleaned:
            return

        first_error: Exception | None = None

        def attempt(action: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                action()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if self._prefetch_result is not None:
            attempt(self._stop_prefetch)
            if self._prefetch_result is not None:
                self._retained_prefetch_resources.append(self._prefetch_result)
                self._prefetch_result = None
        mesh = self.config.mesh_device
        if self._loaded_mode is not None:
            attempt(mesh.reset_sub_device_stall_group)
            attempt(mesh.clear_loaded_sub_device_manager)
            self._active_mode = None
            self._loaded_mode = None

        resources = list(reversed(self._retained_prefetch_resources))
        self._retained_prefetch_resources.clear()
        if self._weight_address_metadata is not None:
            resources.append(self._weight_address_metadata)
        seen: set[int] = set()
        for resource in resources:
            if id(resource) not in seen:
                attempt(lambda resource=resource: self._deallocate(resource))
                seen.add(id(resource))
        self._weight_address_metadata = None
        self._global_cb = None

        for mode in ("decode", "prefill"):
            manager = self._managers.pop(mode, None)
            if manager is not None:
                attempt(lambda manager=manager: mesh.remove_sub_device_manager(manager))

        self._contexts.clear()
        self._cleaned = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "Prefetcher2D":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()

    def _make_context(
        self,
        mode_config: Prefetcher2DModeConfig,
        *,
        global_cb: Any,
        weights: tuple[Any, ...],
        weight_addresses: Any,
    ) -> Prefetcher2DContext:
        return Prefetcher2DContext(
            mode=mode_config.mode,
            mesh_device=self.config.mesh_device,
            sub_device_manager_id=self._managers[mode_config.mode],
            worker_sub_device_id=mode_config.worker_sub_device_id,
            stall_group=mode_config.stall_group,
            global_cb=global_cb,
            weights=weights,
            weight_addresses=weight_addresses,
            weight_address_metadata=self._weight_address_metadata,
        )

    def _configure_mode(self, context: Prefetcher2DContext) -> None:
        mode_config = self.config.prefill if context.mode == "prefill" else self.config.decode
        self._configure_mode_resources(mode_config)

    def _configure_mode_resources(self, mode_config: Prefetcher2DModeConfig) -> None:
        mesh = self.config.mesh_device
        mesh.load_sub_device_manager(self._managers[mode_config.mode])
        if mode_config.mode == "prefill":
            mesh.set_sub_device_stall_group(list(mode_config.stall_group))
        else:
            mesh.set_sub_device_stall_group(
                [ttnn.SubDeviceId(index) for index in range(len(self.config.decode.sub_devices))]
            )
        self._loaded_mode = mode_config.mode

    def _start_prefetch(self, context: Prefetcher2DContext) -> None:
        result = self._dram_prefetch_start(context)
        if result is None:
            raise RuntimeError("dram prefetch start must return an owned result")
        self._prefetch_result = result
        self.config.mesh_device.set_sub_device_stall_group(list(context.stall_group))

    def _stop_prefetch(self, *, suppress_errors: bool = False) -> None:
        result = self._prefetch_result
        if result is None:
            return
        try:
            sync_result = self._dram_prefetch_stop(self.config.mesh_device, result)
        except Exception:
            if suppress_errors:
                self._prefetch_result = None
                self._retained_prefetch_resources.append(result)
                return
            raise
        self._prefetch_result = None
        if sync_result is None:
            return
        self._retained_prefetch_resources.append(sync_result)

    def _default_validate_weight(self, name: str, tensor: Any, existing: tuple[Any, ...]) -> None:
        del name
        address = tensor.buffer_address()
        if not isinstance(address, int) or address < 0:
            raise ValueError("registered weight buffer address must be a non-negative integer")
        if any(other is tensor or other.buffer_address() == address for other in existing):
            raise ValueError("registered weights must refer to distinct device buffers")

    def _default_derive_global_cb_size(self, weights: tuple[Any, ...]) -> int:
        sizes: list[int] = []
        for weight in weights:
            buffer_size = getattr(weight, "buffer_size", None)
            if callable(buffer_size):
                sizes.append(int(buffer_size()))
        if sizes:
            return 2 * max(sizes)
        if self.config.global_cb_size is None:
            raise ValueError("global_cb_size requires an injected deriver when weights do not expose buffer_size()")
        return self.config.global_cb_size

    def _default_dram_prefetch_start(self, context: Prefetcher2DContext) -> Any:
        return ttnn.dram_prefetcher(
            list(context.weights) + [context.weight_address_metadata],
            num_layers=self.config.prefetch_num_layers,
            global_cb=context.global_cb,
        )

    @staticmethod
    def _default_dram_prefetch_stop(mesh_device: Any, result: Any) -> Any:
        del mesh_device
        ttnn.deallocate(result)
        return None

    def _ensure_open(self) -> None:
        if self._cleaned:
            raise RuntimeError("Prefetcher2D has been cleaned up")
